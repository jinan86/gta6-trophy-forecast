#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per *.in file (schema makefun/gta6-trophies-case/v1) and writes
one forecast per case: five quantiles of each trophy's Exophase earn rate at
the case's target, as {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.

Model, in one paragraph
-----------------------
An Exophase earn rate is a ratio, rate = earners / players_tracked, and the
two sides move for different reasons, so they are forecast separately and
only divided at the end.  The page itself only changes when Exophase
refreshes its counts (hours apart): between refreshes every read is
identical, so the history is first collapsed to its distinct snapshots and
every change is treated as one refresh event.

* Players tracked P(t): fitted as log P linear in log(1 + hours since
  release) over the distinct snapshots (power-law growth, which already
  decelerates), anchored at the last read, with the slope shrunk toward the
  recent-window slope and damped for extrapolation past the observed range.
* Earners C_i(t) per trophy: fitted as log C = a + e_i * log P + g_i * u
  (u = log(1 + hours since release)) by ridge-regularised least squares.
  e_i is the dilution elasticity -- how much of a player influx has already
  earned the trophy (near 0 when a refresh floods in fresh players who hold
  almost nothing, near 1 when new players catch up fast) -- and g_i is the
  residual progress of the existing population through the game.  When the
  history shows no refresh at all, e_i and g_i are unidentified and fall back
  to difficulty priors built from the trophy's tier and description.
* The median rate is 100 * C_i(target) / P(target).  Quantiles live on the
  logit scale: spread comes from the fit residual, parameter uncertainty,
  and a random-walk term growing with sqrt(horizon in days), because the
  final horizon (6,510 minutes, ~4.5 days) is far longer than the practice
  horizon (90 minutes).
* Flat history (no refresh ever seen): a two-component mixture.  With
  probability p_r a refresh lands inside the horizon, multiplying players by
  ~4x while earners grow only by g^e, which drags every rate down by roughly
  a logit; with probability 1 - p_r nothing changes.  The mixture's
  quantiles are computed numerically, which gives the honest left-skewed
  band such a bimodal future deserves: a low 5% quantile reaching down to
  the refreshed rates, a median near the last read.
* A trophy at 100% in every read is forecast near 100% (median 99.99);
  the platinum is capped at the rarest other trophy's median, and the
  chapter trophies are kept non-increasing in story order.

Everything is deterministic (no randomness, no clock, no network, stdlib
only).  Any failure on a case falls back to a wide persistence forecast so
the output is still valid.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

RELEASE_FALLBACK = "2026-11-19T00:00:00Z"
# Prior mean interval between Exophase refreshes, hours (first refresh came
# ~4.5 h after the first probe; refreshes are hours apart, not continuous).
REFRESH_INTERVAL_H = 5.0
# Prior player jump at one refresh while the base is still small: the first
# refresh took players tracked 638 -> 3,868 (6.06x) while total trophies
# earned went 943 -> 4,276, i.e. the incoming players held almost nothing
# (mean non-100% count per trophy grew only ~1.34x) -- an elasticity of
# ~0.16.  A refresh therefore drags rates down by roughly a logit.
REFRESH_JUMP_PRIOR = 6.0


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    """logit of a percent, with the scorer's 0.01..99.99 clamp."""
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    """Inverse logit back to a percent."""
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 99.99)


def round4(x):
    """Round to 4 decimals, half up (all values here are >= 0)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_time(s):
    """ISO-8601 (with Z or offset, or date-only) -> aware datetime (UTC)."""
    if not isinstance(s, str) or not s:
        return None
    txt = s.strip()
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def least_squares(xs, ys):
    """Ordinary least squares y ~ a + b*x. Returns (a, b, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid, n


def ridge2(xs1, xs2, ys, prior1, prior2, lam):
    """Ridge fit y ~ a + b1*x1 + b2*x2, shrinking (b1, b2) toward priors.

    Returns (a, b1, b2, resid_std, n).  Solved via the 3x3 normal equations
    with a tiny Gaussian elimination; falls back to the priors on failure.
    """
    n = len(ys)
    if n == 0:
        return 0.0, prior1, prior2, 0.0, 0
    m1 = sum(xs1) / n
    m2 = sum(xs2) / n
    my = sum(ys) / n
    c1 = [x - m1 for x in xs1]
    c2 = [x - m2 for x in xs2]
    cy = [y - my for y in ys]
    s11 = sum(v * v for v in c1) + lam
    s22 = sum(v * v for v in c2) + lam
    s12 = sum(a * b for a, b in zip(c1, c2))
    t1 = sum(a * b for a, b in zip(c1, cy)) + lam * prior1
    t2 = sum(a * b for a, b in zip(c2, cy)) + lam * prior2
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return my, prior1, prior2, 0.0, n
    b1 = (t1 * s22 - t2 * s12) / det
    b2 = (s11 * t2 - s12 * t1) / det
    a = my - b1 * m1 - b2 * m2
    resid = math.sqrt(
        sum((y - (a + b1 * x1 + b2 * x2)) ** 2
            for x1, x2, y in zip(xs1, xs2, ys)) / n)
    return a, b1, b2, resid, n


# ---------------------------------------------------------------------------
# difficulty prior from trophy metadata
# ---------------------------------------------------------------------------

def difficulty_prior(trophy):
    """Prior (elasticity e, progress g, logit spread) for a trophy.

    Easy early story beats: new players catch up fast (high e) and the
    existing population earns them quickly (high g).  Grinds and
    collections: new players hold almost none (low e) and progress is slow.
    """
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    tier = (trophy.get("tier") or "bronze").lower()
    tier_e = {"platinum": 0.05, "gold": 0.10, "silver": 0.15,
              "bronze": 0.25}.get(tier, 0.25)
    tier_g = {"platinum": 0.05, "gold": 0.10, "silver": 0.15,
              "bronze": 0.25}.get(tier, 0.25)
    e, g = tier_e, tier_g
    # Story chapters and one-off missions: fast.
    if "chapter" in text or "complete the final mission" in text:
        e, g = max(e, 0.55), max(g, 0.55)
    if "complete" in text and "mission" in text:
        e, g = max(e, 0.45), max(g, 0.45)
    # Grinds: explicit counts of repeated actions stay rare for a long time.
    grind_words = ("100% completion", "250 stunt", "100 locations",
                   "100 vehicles", "50 challenges", "20 different races",
                   "3 collectible card sets", "every spot", "all of leonida",
                   "gold medal in all")
    if any(w in text for w in grind_words):
        e, g = min(e, 0.08), min(g, 0.08)
    if "earn $" in text or "spend $" in text or "purchase all" in text:
        e, g = min(e, 0.12), min(g, 0.12)
    # First, trivial beats: the fishing opener sits at 100% from launch.
    if "gone fishing" in text:
        e, g = 1.0, 1.0
    return e, g


# ---------------------------------------------------------------------------
# case parsing
# ---------------------------------------------------------------------------

def distinct_snapshots(case, release_dt):
    """Collapse history to distinct snapshots (Exophase only changes on
    refresh).  Returns list of (u, t_hours, players, rates) with u the
    log-time since release, oldest first.  Reads with no usable player
    count are skipped."""
    snaps = []
    prev_key = None
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        at = parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        t_h = (at - release_dt).total_seconds() / 3600.0
        if t_h < -1.0:
            continue
        u = math.log1p(max(t_h, 0.0))
        snaps.append((u, max(t_h, 0.0), float(players), rates))
    return snaps


def fit_players(snaps, u_target):
    """Forecast players tracked at u_target.

    Log-log linear fit (all distinct snapshots, plus a recent-window fit);
    the conservative slope wins, damped for extrapolation.  Returns
    (p_med, p_lo, p_hi, n_snaps, saw_refresh, resid).  With a single
    snapshot there is no fit: fall back to the launch-night prior slope.
    """
    p_last = snaps[-1][2]
    u_last = snaps[-1][0]
    xs = [s[0] for s in snaps]
    ys = [math.log(s[2]) for s in snaps]
    _, b_all, resid_all, _ = least_squares(xs, ys)
    # Recent window: distinct snapshots from the last 24 h of history.
    t_last = snaps[-1][1]
    recent = [s for s in snaps if t_last - s[1] <= 24.0]
    b_recent = None
    if len(recent) >= 2 and len(set(s[2] for s in recent)) >= 2:
        _, b_recent, _, _ = least_squares(
            [s[0] for s in recent], [math.log(s[2]) for s in recent])
    saw_refresh = len(set(s[2] for s in snaps)) >= 2
    if not saw_refresh:
        # No refresh ever seen: prior power-law slope from launch night
        # (638 -> 3,868 players over the first ~4.5 h is slope ~1.06).
        b = 1.0
        resid = 0.30
    else:
        slopes = [b for b in (b_all, b_recent) if b is not None]
        b = min(slopes) if slopes else 0.0
        resid = resid_all
    b = clamp(b, 0.0, 2.5)
    # Damp the slope past the observed range: growth decelerates.
    b_eff = b * 0.90
    du = max(u_target - u_last, 0.0)
    log_p = math.log(p_last) + b_eff * du
    p_med = math.exp(log_p)
    # Sanity caps: players never fall, and never explode absurdly.
    p_med = clamp(p_med, p_last, p_last * 60.0)
    # Log-space uncertainty for the players forecast (used for the spread).
    sig_p = 0.20 + 0.30 * math.sqrt(max(du, 0.0)) + resid
    return p_med, p_last * math.exp(-sig_p), p_last * math.exp(sig_p), \
        len(snaps), saw_refresh, resid


def fit_trophy(snaps, idx, prior_e, prior_g):
    """Fit one trophy's earners: log C ~ a + e*log P + g*u (ridge).

    Returns (e, g, resid, n_points).  With no usable variation the priors
    are returned.  Counts use a +0.5 continuity correction so a 0% rate
    still fits.
    """
    xs_p, xs_u, ys = [], [], []
    for (u, _t, players, rates) in snaps:
        if idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        count = r * players / 100.0
        xs_p.append(math.log(players))
        xs_u.append(u)
        ys.append(math.log(count + 0.5))
    n = len(ys)
    if n < 3 or len(set(xs_p)) < 2:
        return prior_e, prior_g, 0.30, n
    # Ridge toward the priors; stronger shrinkage with few refreshes.
    lam = 0.35 if n >= 6 else 1.0
    _a, e, g, resid, _ = ridge2(xs_p, xs_u, ys, prior_e, prior_g, lam)
    e = clamp(e, -0.2, 1.6)
    g = clamp(g, -0.3, 1.2)
    return e, g, resid, n


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logistic_rate_cap(snaps, idx, u_target):
    """Saturating logistic-in-log-time fit to one trophy's observed rate
    series: r(u) = A * sigmoid(a + b*u).  Returns the predicted rate at
    u_target, or None when the fit is not trustworthy.

    Earn fractions saturate -- a story trophy cannot keep growing
    log-linearly forever -- while the count/players decomposition above
    extrapolates without a ceiling.  The logistic prediction is therefore
    used only as a cap on upward extrapolation, never to push a forecast
    down: a bad cap can only trim an explosive median, not invent a fall.
    """
    pts = []
    for (u, _t, _p, rates) in snaps:
        if idx >= len(rates) or rates[idx] is None:
            continue
        try:
            r = float(rates[idx])
        except (TypeError, ValueError):
            continue
        pts.append((u, clamp(r, 0.005, 99.99)))
    if len(pts) < 4 or len(set(r for _u, r in pts)) < 2:
        return None
    last_rate = pts[-1][1]
    max_rate = max(r for _u, r in pts)
    a0_init = clamp(last_rate * 4.0, max_rate * 1.5, 99.0)
    # Initialise (a, b) from a linear fit of logit(r / A0) on u.
    xs, ys = [], []
    for (u, r) in pts:
        p = clamp(r / a0_init, 0.001, 0.999)
        xs.append(u)
        ys.append(math.log(p / (1.0 - p)))
    a_lin, b_lin, _r, _n = least_squares(xs, ys)
    theta = [a0_init, a_lin, clamp(b_lin, 0.02, 5.0)]

    def sse(th):
        A, a, b = th
        s = 0.0
        for (u, r) in pts:
            d = r - A * _sigmoid(a + b * u)
            s += d * d
        return s

    cur = sse(theta)
    for _ in range(25):
        # Numerical Jacobian (central differences), 3 params.
        jac = []
        for j in range(3):
            step = 1e-6 * max(1.0, abs(theta[j]))
            up = list(theta)
            dn = list(theta)
            up[j] += step
            dn[j] -= step
            col = []
            for (u, r) in pts:
                mu = theta[0] * _sigmoid(theta[1] + theta[2] * u)
                mp = up[0] * _sigmoid(up[1] + up[2] * u)
                mm = dn[0] * _sigmoid(dn[1] + dn[2] * u)
                col.append((mp - mm) / (2.0 * step))
            jac.append(col)
        # Normal equations (J^T J + lam I) d = J^T resid.
        resid = [r - theta[0] * _sigmoid(theta[1] + theta[2] * u)
                 for (u, r) in pts]
        m = [[0.0] * 3 for _ in range(3)]
        v = [0.0] * 3
        for i in range(3):
            for j in range(3):
                m[i][j] = sum(jac[i][k] * jac[j][k] for k in range(len(pts)))
            v[i] = sum(jac[i][k] * resid[k] for k in range(len(pts)))
        for i in range(3):
            m[i][i] += 1e-6 * (m[i][i] + 1.0)
        # 3x3 solve by Cramer.
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        if abs(det) < 1e-18:
            break
        d = [0.0] * 3
        for j in range(3):
            mj = [row[:] for row in m]
            for i in range(3):
                mj[i][j] = v[i]
            d[j] = (mj[0][0] * (mj[1][1] * mj[2][2] - mj[1][2] * mj[2][1])
                    - mj[0][1] * (mj[1][0] * mj[2][2] - mj[1][2] * mj[2][0])
                    + mj[0][2] * (mj[1][0] * mj[2][1] - mj[1][1] * mj[2][0])) / det
        # Step halving until the fit improves.
        scale = 1.0
        improved = False
        for _try in range(12):
            cand = [theta[j] + scale * d[j] for j in range(3)]
            if 0.0 < cand[0] <= 100.0 and -20.0 < cand[1] < 20.0 \
                    and -2.0 < cand[2] < 10.0:
                val = sse(cand)
                if val < cur:
                    theta, cur = cand, val
                    improved = True
                    break
            scale *= 0.5
        if not improved:
            break
    A, a, b = theta
    if not (0.0 < A <= 100.0) or b < 0.0:
        return None
    pred = A * _sigmoid(a + b * u_target)
    if not math.isfinite(pred):
        return None
    return clamp(pred, 0.005, 99.99)


# ---------------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------------

def normal_quantiles(mu_logit, sigma):
    """Plain five quantiles of N(mu, sigma^2) on the logit scale."""
    out = []
    for z in Z:
        out.append(round4(inv_logit_pct(mu_logit + z * sigma)))
    return out


def mixture_quantiles(mu_logit, sigma, delta_logit, p_refresh):
    """Quantiles of a two-component logit-normal mixture:
    (1-p) * N(mu, sigma^2) + p * N(mu + delta, sigma^2).
    Inverted numerically on a grid; exact enough for 4-decimal output."""
    lo = mu_logit + min(0.0, delta_logit) - 6.0 * sigma - 0.5
    hi = mu_logit + max(0.0, delta_logit) + 6.0 * sigma + 0.5
    steps = 600
    grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]

    def cdf(x):
        def ncdf(v):
            return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
        return ((1.0 - p_refresh) * ncdf((x - mu_logit) / sigma)
                + p_refresh * ncdf((x - mu_logit - delta_logit) / sigma))

    out = []
    gi = 0
    for level in LEVELS:
        while gi < steps and cdf(grid[gi]) < level:
            gi += 1
        if gi == 0:
            x = grid[0]
        elif gi > steps:
            x = grid[-1]
        else:
            c0, c1 = cdf(grid[gi - 1]), cdf(grid[gi])
            if c1 > c0:
                f = (level - c0) / (c1 - c0)
            else:
                f = 0.0
            x = grid[gi - 1] + f * (grid[gi] - grid[gi - 1])
        out.append(round4(inv_logit_pct(x)))
    # Enforce non-decreasing after rounding.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    if not trophies:
        return {"trophies": []}
    horizon_min = case.get("horizon_minutes")
    try:
        horizon_min = float(horizon_min)
    except (TypeError, ValueError):
        horizon_min = 0.0
    horizon_days = max(horizon_min, 0.0) / 1440.0
    horizon_h = max(horizon_min, 0.0) / 60.0

    release_dt = parse_time((case.get("game") or {}).get("release_date"))
    if release_dt is None:
        release_dt = parse_time(RELEASE_FALLBACK)
    at_dt = parse_time(case.get("at")) or parse_time(case.get("target"))
    target_dt = parse_time(case.get("target")) or at_dt
    if at_dt is None or target_dt is None:
        raise ValueError("case has no usable timestamps")
    u_target = math.log1p(max((target_dt - release_dt).total_seconds()
                              / 3600.0, 0.0))

    snaps = distinct_snapshots(case, release_dt)
    if not snaps:
        # No history at all: flat priors by tier, very wide spread.
        rows = []
        for tr in trophies:
            tier = (tr.get("tier") or "bronze").lower()
            med = {"platinum": 0.10, "gold": 0.20, "silver": 0.30,
                   "bronze": 0.50}.get(tier, 0.50)
            rows.append((tr.get("id"), med, 1.8, None, False))
        return finish_rows(rows, trophies)

    p_med, _p_lo, _p_hi, _n, saw_refresh, _resid = fit_players(snaps, u_target)
    p_last = snaps[-1][2]
    u_last = snaps[-1][0]
    du = max(u_target - u_last, 0.0)
    last_rates = snaps[-1][3]

    # Refresh probability inside the horizon, used only when no refresh
    # has ever been seen (otherwise the fitted growth already prices it).
    p_refresh = 0.0
    if not saw_refresh and horizon_h > 0:
        p_refresh = clamp(1.0 - math.exp(-horizon_h / REFRESH_INTERVAL_H),
                          0.05, 0.90)

    rows = []  # (id, median_pct, sigma_logit, delta_logit_or_None)
    for k, tr in enumerate(trophies):
        tid = tr.get("id")
        prior_e, prior_g = difficulty_prior(tr)
        last = None
        if k < len(last_rates) and last_rates[k] is not None:
            try:
                last = float(last_rates[k])
            except (TypeError, ValueError):
                last = None
        if last is None:
            last = 0.5
        if last >= 99.995:
            # Sat at 100% through every read (incl. a 6x player influx):
            # stays ~100%; only the low tail admits a small slip.
            rows.append((tid, 99.99, 0.0, None, True))
            continue
        e, g, resid, npts = fit_trophy(snaps, k, prior_e, prior_g)
        c_last = last * p_last / 100.0
        if saw_refresh:
            c_med = c_last * (p_med / p_last) ** e * math.exp(g * du)
            r_med = 100.0 * c_med / p_med
            # Saturating cap: the earn fraction cannot keep growing
            # log-linearly; the logistic fit to the observed rate series
            # bounds the upside.  Only applied when it sits above the
            # last read, so it never invents a fall.
            cap = logistic_rate_cap(snaps, k, u_target)
            if cap is not None and cap > last:
                r_med = min(r_med, cap)
            delta = None
        else:
            # Flat history: counts have not moved at all, so the
            # no-refresh median is just the last rate; the refresh
            # scenario lives in the mixture shift, not the median.
            # At a launch-night refresh the incoming players hold almost
            # no trophies yet, so the refresh elasticity is far below
            # the trophy's long-run elasticity: about half the prior,
            # floored for grinds, capped for story beats.
            r_med = last
            e_refresh = clamp(0.5 * prior_e, 0.08, 0.30)
            delta = -(1.0 - e_refresh) * math.log(REFRESH_JUMP_PRIOR)
        r_med = clamp(r_med, 0.01, 99.99)
        # Spread: fit residual + parameter uncertainty + random walk in
        # horizon + (flat history) refresh/systemic uncertainty.
        sigma = math.sqrt(
            max(resid, 0.15) ** 2
            + (0.20 + 0.25 * math.sqrt(horizon_days)) ** 2
            + (0.45 * math.sqrt(horizon_days)) ** 2
            + (0.0 if saw_refresh else 0.35 ** 2))
        sigma = clamp(sigma, 0.35, 2.2)
        rows.append((tid, r_med, sigma, delta, False))
    return finish_rows(rows, trophies, p_refresh)


def finish_rows(rows, trophies, p_refresh=0.0):
    """Turn (id, median, sigma, delta, is_100pct) rows into output rows,
    applying the cross-sectional guards (story order, platinum rarest)."""
    # Median guards, applied on the medians only; quantiles follow.
    med = {}
    for row in rows:
        med[row[0]] = row[1]
    by_id = {tr.get("id"): tr for tr in trophies}
    # Chapter trophies (ids 3..7: Ch1, Ch2, Ch3, Ch4, final mission) must
    # be non-increasing in story order.
    chain = [i for i in (3, 4, 5, 6, 7) if i in med]
    run = None
    for i in chain:
        if run is None or med[i] <= run:
            run = med[i]
        else:
            med[i] = run
    # The platinum is earned only by holding (nearly) everything else.
    if 1 in med and len(med) > 1:
        others = [v for k, v in med.items() if k != 1]
        cap = min(others)
        if med[1] > cap:
            med[1] = cap
    out = []
    for row in rows:
        tid, _r, sigma, delta, is_100 = row
        m = med[tid]
        if is_100:
            q = [99.0, 99.9, 99.99, 99.99, 99.99]
        elif delta is not None and p_refresh > 0.0:
            q = mixture_quantiles(logit_pct(m), sigma, delta, p_refresh)
        else:
            q = normal_quantiles(logit_pct(m), sigma)
        out.append({"id": tid, "q": q})
    out.sort(key=lambda r: (r["id"] is None, r["id"]))
    return {"trophies": out}


def fallback_forecast(case):
    """Wide persistence forecast: last rate, honest spread.  Never fails."""
    trophies = case.get("trophies") or []
    horizon_min = case.get("horizon_minutes")
    try:
        horizon_days = max(float(horizon_min), 0.0) / 1440.0
    except (TypeError, ValueError):
        horizon_days = 0.0
    sigma = clamp(0.5 * math.sqrt(horizon_days) + 0.6, 0.6, 2.2)
    last = {}
    hist = case.get("history") or []
    if hist:
        rates = hist[-1].get("rates") or []
        for k, tr in enumerate(trophies):
            if k < len(rates) and rates[k] is not None:
                try:
                    last[tr.get("id")] = float(rates[k])
                except (TypeError, ValueError):
                    pass
    out = []
    for tr in trophies:
        tid = tr.get("id")
        r = last.get(tid, 0.5)
        if r >= 99.995:
            q = [99.0, 99.9, 99.99, 99.99, 99.99]
        else:
            q = normal_quantiles(logit_pct(r), sigma)
        out.append({"id": tid, "q": q})
    out.sort(key=lambda r: (r["id"] is None, r["id"]))
    return {"trophies": out}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(in_dir))
    except OSError:
        names = []
    for name in names:
        if not name.endswith(".in"):
            continue
        path = os.path.join(in_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except (OSError, ValueError):
            continue
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_forecast(case)
        # Final shape validation: one row per trophy, five non-decreasing
        # finite quantiles in [0, 100]; else fall back.
        try:
            validate(result, case)
        except Exception:
            try:
                result = fallback_forecast(case)
                validate(result, case)
            except Exception:
                result = {"trophies": [
                    {"id": tr.get("id"), "q": [0.5, 0.5, 0.5, 0.5, 0.5]}
                    for tr in (case.get("trophies") or [])]}
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


def validate(result, case):
    if not isinstance(result, dict) or set(result.keys()) != {"trophies"}:
        raise ValueError("bad top-level shape")
    rows = result["trophies"]
    want = len(case.get("trophies") or [])
    if len(rows) != want:
        raise ValueError("wrong row count")
    seen = set()
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            raise ValueError("bad row shape")
        if row["id"] in seen:
            raise ValueError("duplicate id")
        seen.add(row["id"])
        q = row["q"]
        if len(q) != 5:
            raise ValueError("wrong quantile count")
        prev = None
        for v in q:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("non-finite quantile")
            if v < 0.0 or v > 100.0:
                raise ValueError("quantile out of range")
            if prev is not None and v < prev:
                raise ValueError("quantiles decrease")
            prev = v


if __name__ == "__main__":
    sys.exit(main())
