# GTA VI Trophy Rarity Forecaster

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at a case's target time — as five quantiles per trophy —
from the house's recorded reads of the game's trophy page up to the case's
checkpoint.

Entry for the make.fun bounty
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
(`gta6-trophy-rarity-forecast-01`).

## Run

```sh
python3 main.py
```

The program reads `$MAKEFUN_INPUT_DIR` (default `/input`) and writes to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). It processes every `*.in` file
it finds, in name order, and writes one `/output/<id>.out` per case:

```json
{"trophies": [{"id": 1, "q": [0.0761, 0.2287, 0.4945, 0.7897, 1.406]}, ...]}
```

One row per trophy, five quantiles at the 5%, 25%, 50%, 75% and 95%
levels, in percent, non-decreasing. Python 3.13, standard library only,
no network, deterministic (two runs on the same case are byte-identical),
well under the 8-second limit per case.

## Why the model looks the way it does

An Exophase earn rate is a ratio — earners divided by players tracked —
and the two sides move for different reasons. Launch night shows why
that matters: the page sat unchanged at 638 players tracked for hours,
then refreshed to 3,868 players while total trophies earned went from
943 to 4,276. The incoming players held almost nothing, so nearly every
rate fell to about a fifth of its launch-night value. A forecaster that
only extrapolates rates misses that entirely.

So the program forecasts the two sides separately and divides at the end:

- **Players tracked** is fitted as log-players linear in log(1 + hours
  since release) — power-law growth that already decelerates — anchored
  at the last read, with the slope shrunk toward the recent window and
  damped for extrapolation.
- **Earners** per trophy are fitted as log-count linear in log-players
  and log-time (ridge-regularised): a *dilution elasticity* (how much of
  a player influx has already earned the trophy) plus a *progress* term
  for the existing population working through the game. Trophy tier and
  description (story chapter, grind, collection) supply priors where the
  history cannot identify the parameters.
- **Saturation.** Earn fractions saturate; a log-linear count model does
  not. A logistic curve in log-time, fitted to each trophy's observed
  rate series, caps upward extrapolation — it can only trim an explosive
  median, never invent a fall.
- **Quantiles** live on the logit scale. Spread combines the fit
  residual, parameter uncertainty, and a random-walk term growing with
  the square root of the horizon in days: the final horizon (6,510
  minutes, ~4.5 days) is far longer than the 90-minute practice horizon,
  so its bands are far wider.
- **Flat history.** When no refresh has ever been seen, the future is
  bimodal: either nothing changes, or a refresh lands and drags every
  rate down by roughly a logit. The forecast is the two-component
  mixture's quantiles, computed numerically — a low 5% quantile reaching
  down to the refreshed rates, a median near the last read.
- A trophy at 100% in every read is forecast near 100%; the platinum
  is capped at the rarest other trophy's median, and the chapter
  trophies are kept non-increasing in story order.

Any failure on a case falls back to a wide persistence forecast, so the
output is still valid.

## Local checks

Practice cases from the kit (inputs in the kit's `input/`):

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Both practice outputs are valid (55 rows, non-decreasing quantiles),
byte-identical across runs, and run in ~0.15 s for both cases. Against
the only answer shape consistent with the published refresh aggregate
(3,868 players, 4,276 trophies earned — mean non-100% rate ratio
≈ 0.22), the indicative pinball loss is ≈ 0.65, versus ≈ 1.40 for the
house persistence baseline (the baseline's published practice loss is
≈ 1.51). The exact practice answers are not in the kit, so that figure
is indicative, not a measured score.

A synthetic final-like case (120 reads over two days with refreshes,
target 4.5 days out) scores ≈ 0.28 against its known truth with 98%
90%-coverage, and edge cases (single read, null rates, empty history,
zero horizon) all produce valid output.
