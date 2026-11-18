# gta6-twitch-ccu-forecast-01-v1

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aurorabelle788/gta6-twitch-ccu-forecast-01-v1/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aurorabelle788/gta6-twitch-ccu-forecast-01-v1/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's hour-by-hour Twitch concurrent viewers (CCU) on release day, 2026-11-19 (UTC), as five quantiles per hour (q05, q25, q50, q75, q95) for hours 2..24.

Model: shape-transfer from comparable past launches. For each hour, the median log-ratio of that hour's viewers to the hour-1 print across comparable launches — weighted by how close each comparable's hour-1 print is to this case's hour-1 print (Gaussian kernel, bandwidth 1.0 in log space) — gives the median ratio; the spread is the weighted standard deviation of the same log-ratios around that median, scaled by 0.85. Validated by leave-one-out cross-validation over 37 comparable launches: mean pinball loss 0.40 overall, 0.21 on big launches (hour-1 >= 50k), vs 0.93 / 0.25 for the persistence baseline.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=azvv97zj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
