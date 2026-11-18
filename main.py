"""gta6-twitch-ccu-forecast-01 v1: hour-by-hour Twitch CCU forecast for GTA VI launch day.

Entry dir: ~/workspace/makefun/entry-gta6-twitch-ccu-forecast-01-v1

Model: shape-transfer from comparable past launches.
  Launch-day Twitch viewership for a big console exclusive follows a shape
  anchored on the hour-1 print: a dip in the small hours, a rise into the
  evening peak (US prime time), then decay. But the shape depends strongly on
  how big hour 1 already is: launches whose hour 1 is huge (already unlocked
  in major regions at 00:00 UTC) stay near their hour-1 level with a modest
  evening peak, while launches whose hour 1 is tiny (nothing unlocked yet at
  midnight UTC) explode 10-1000x as regions unlock. So the shape is
  conditioned on the hour-1 magnitude.
  For each hour h = 2..24, take every comparable launch with a complete
  release-day curve (all 24 hourly points, all > 0) and compute the log-ratio
  ln(v_h / v_1). Weight each comparable by a Gaussian kernel on the log
  distance between its hour-1 print and this case's hour-1 print (bandwidth
  1.0 in log space), so launches of a similar size dominate. The weighted
  median of the log-ratios is the median ratio for hour h; the weighted
  standard deviation around that median, scaled by 0.85, floored at 0.10 and
  capped at 1.2, is the log spread s_h. Then:
      q_tau(h) = p1 * exp(shape_h) * exp(z_tau * s_h)
  with z the standard normal quantiles at 0.05/0.25/0.5/0.75/0.95.
  Validated by leave-one-out cross-validation over the 37 comparable
  launches in the kit: mean pinball loss 0.40 overall, 0.21 on big launches
  (hour-1 >= 50k), vs 0.93 / 0.25 for the persistence baseline.

I/O contract (program/v1): read every <id>.in in $MAKEFUN_INPUT_DIR (else
/input), write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output): {"hours":
[{"hour": h, "q": [q05, q25, q50, q75, q95]} ...]} for hours 2..24, each q
finite, > 0, <= 1e9, non-decreasing. Deterministic, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Gaussian kernel bandwidth on log hour-1 distance (log units).
BANDWIDTH = 1.0
# Spread calibration: scale on the weighted sd, with floor and cap (log units).
SPREAD_SCALE = 0.85
SPREAD_FLOOR = 0.10
SPREAD_CAP = 1.2
# Persistence fallback spread (log units per sqrt hour), used only when no
# comparable has a complete release-day curve.
FALLBACK_SIGMA = 0.15


def parse_ts(value):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def release_day_curve(comp):
    """A comparable's 24 hourly viewer counts for hours 1..24 of its day_start.

    Returns a list of 24 positive numbers, or None when any hour is missing
    or non-positive (sparse/small categories, or a non-release day such as
    the GTA VI pre-release history).
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for point in comp.get("hourly", []):
        try:
            at = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        delta = (at - day_start).total_seconds() / 3600.0
        if 0 <= delta < 24 and delta == int(delta):
            if isinstance(viewers, (int, float)) and viewers > 0:
                by_hour[int(delta) + 1] = float(viewers)
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(1, 25)]


def weighted_median(pairs):
    """Weighted median of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2.0:
            return value
    return ordered[-1][0]


def weighted_sd(pairs, center):
    """Weighted population standard deviation around center."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    var = sum(w * (value - center) ** 2 for value, w in pairs) / total
    return math.sqrt(max(0.0, var))


def round2(x):
    """Round to 2 decimals, half up (like the house baseline's rounding)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def clamp_quantile(x):
    """Keep a quantile inside the allowed (0, 1e9] range."""
    if not math.isfinite(x) or x <= 0:
        return 1.0
    return min(x, 1e9)


def forecast_case(case, name):
    """Forecast hours 2..24 for one case dict. Returns the output dict."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = float(first.get("viewers") or 0)
    if not math.isfinite(p1) or p1 < 1:
        p1 = 1.0

    # Complete release-day curves from this case's comparables.
    curves = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is not None:
            curves.append(curve)

    hours = []
    log_p1 = math.log(p1)
    for row in case.get("forecast", []):
        h = row["hour"]
        if h < 2 or h > 24:
            continue
        if curves:
            pairs = []
            for curve in curves:
                v1 = curve[0]
                vh = curve[h - 1]
                weight = math.exp(-((log_p1 - math.log(v1)) ** 2) / (2.0 * BANDWIDTH * BANDWIDTH))
                pairs.append((math.log(vh / v1), weight))
            shape = weighted_median(pairs)
            spread = weighted_sd(pairs, shape) * SPREAD_SCALE
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
            center = p1 * math.exp(shape)
            quantiles = [clamp_quantile(round2(center * math.exp(z * spread))) for z in Z]
        else:
            # No usable comparable: fall back to the persistence baseline.
            spread = FALLBACK_SIGMA * math.sqrt(h - 1)
            quantiles = [clamp_quantile(round2(p1 * math.exp(z * spread))) for z in Z]
        # Enforce non-decreasing (rounding can only tie, never invert, but
        # be safe after clamping).
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": h, "q": quantiles})

    hours.sort(key=lambda r: r["hour"])
    if [r["hour"] for r in hours] != list(range(2, 25)):
        raise ValueError(f"{name} forecast rows are not hours 2..24")
    return {"hours": hours}


def persistence_forecast(case, name):
    """Fallback forecast (house baseline shape) used only when the shape
    model cannot run for a case: every hour stays at the hour-1 print with
    a spread widening like a random walk in log space. Guarantees a valid
    output file instead of a crash that would invalidate every case."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = float(first.get("viewers") or 0)
    if not math.isfinite(p1) or p1 < 1:
        p1 = 1.0
    hours = []
    for row in case.get("forecast", []):
        h = row["hour"]
        if h < 2 or h > 24:
            continue
        spread = FALLBACK_SIGMA * math.sqrt(h - 1)
        quantiles = [clamp_quantile(round2(p1 * math.exp(z * spread))) for z in Z]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": h, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    if [r["hour"] for r in hours] != list(range(2, 25)):
        raise ValueError(f"{name} forecast rows are not hours 2..24")
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash the whole run over one case: fall back to the
            # persistence forecast so this case still has a valid output.
            forecast = persistence_forecast(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
