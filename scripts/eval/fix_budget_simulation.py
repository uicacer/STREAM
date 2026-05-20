#!/usr/bin/env python3
"""
Re-run the budget simulation with fully normalized (0-1) values.

Both axes are deployment-agnostic:
  - time:  period_fraction = (day - 1) / (total_days - 1)  →  [0.0, 1.0]
  - spend: spend_fraction  = cumulative_spend / B           →  [0.0, 1.0]
  - theta: already [0.0, 1.0]

Budget B = total fixed spend over 30 days, so fixed routing reaches spend=1.0
at period=1.0. Adaptive routing activates when spend_fraction crosses theta_base (0.5).
"""

import json
from pathlib import Path

SIM_PATH = Path("scripts/eval/results/budget_simulation.json")
SWEEP_PATH = Path("scripts/eval/results/theta_sweep.json")

sweep_data = json.loads(SWEEP_PATH.read_text())
sim_data = json.loads(SIM_PATH.read_text())

# Build theta → cloud_routing_rate and high_recall lookup (sweep grid: 0.00 to 1.00, step 0.01)
sweep_by_theta = {}
for entry in sweep_data["sweep"]:
    t = round(entry["theta"], 2)
    sweep_by_theta[t] = entry


def lookup(theta: float) -> tuple[float, float]:
    """Return (cloud_routing_rate, high_recall) for the nearest sweep theta."""
    t = round(round(theta * 100) / 100, 2)
    t = max(0.0, min(1.0, t))
    if t not in sweep_by_theta:
        # Snap to nearest available key
        t = min(sweep_by_theta.keys(), key=lambda k: abs(k - theta))
    e = sweep_by_theta[t]
    return e["cloud_routing_rate"], e["high_recall"]


# --- Fixed simulation (unchanged, re-expressed as fractions) ---
fixed_raw = sim_data["fixed"]
cost_per_usd = sim_data["cost_per_query_usd"]
T = len(fixed_raw)  # total days (30)

# Reconstruct B from cloud_n counts (works regardless of old/new format)
B = sum(d["cloud_n"] * cost_per_usd for d in fixed_raw)

fixed_norm = []
cumul = 0.0
for d in fixed_raw:
    cumul += d["cloud_n"] * cost_per_usd
    period_frac = round((d["day"] - 1) / (T - 1), 4)  # day 1 → 0.0, day 30 → 1.0
    fixed_norm.append(
        {
            "day": d["day"],
            "period_fraction": period_frac,
            "theta_eff": round(d.get("theta_eff", 0.5), 3),
            "cloud_n": d["cloud_n"],
            "total_n": d["total_n"],
            "spend_fraction": round(cumul / B, 4),
            "high_recall": round(d["high_recall"], 4),
        }
    )


# --- Adaptive simulation (re-run with dynamic theta_eff) ---
# Use same daily total_n as fixed (same daily query volumes)
theta_base = 0.5
adaptive_norm = []
cumulative_spend = 0.0

for _i, d in enumerate(fixed_raw):
    total_n = d["total_n"]

    # theta_eff is based on spend fraction AT START of this day (from previous day)
    spend_frac = cumulative_spend / B
    theta_eff = max(theta_base, min(spend_frac, 1.0))

    # Snap to nearest 0.01 for sweep lookup
    cloud_rate, high_recall = lookup(theta_eff)

    cloud_n = max(0, round(total_n * cloud_rate))
    day_cost = cloud_n * cost_per_usd
    cumulative_spend += day_cost

    period_frac = round((d["day"] - 1) / (T - 1), 4)
    adaptive_norm.append(
        {
            "day": d["day"],
            "period_fraction": period_frac,
            "theta_eff": round(theta_eff, 3),
            "cloud_n": cloud_n,
            "total_n": total_n,
            "spend_fraction": round(cumulative_spend / B, 4),
            "high_recall": round(high_recall, 4),
        }
    )


# --- Summary stats ---
fixed_final_frac = fixed_norm[-1]["spend_fraction"]  # ≈ 1.0 by construction
adap_final_frac = adaptive_norm[-1]["spend_fraction"]

spend_reduction_pct = (fixed_final_frac - adap_final_frac) / fixed_final_frac * 100

# Quality: compare average HIGH recall (adaptive vs fixed)
fixed_avg_recall = sum(d["high_recall"] for d in fixed_norm) / len(fixed_norm)
adap_avg_recall = sum(d["high_recall"] for d in adaptive_norm) / len(adaptive_norm)
quality_preserved_pct = adap_avg_recall / fixed_avg_recall * 100

# First day adaptive theta rises above theta_base
activation_day = None
for d in adaptive_norm:
    if d["theta_eff"] > theta_base:
        activation_day = d["day"]
        break

activation_frac = round((activation_day - 1) / (T - 1), 4) if activation_day else None

summary = {
    "spend_reduction_pct": round(spend_reduction_pct, 1),
    "quality_preserved_pct": round(quality_preserved_pct, 1),
    "fixed_final_spend_fraction": round(fixed_final_frac, 4),
    "adaptive_final_spend_fraction": round(adap_final_frac, 4),
    "theta_activation_day": activation_day,
    "theta_activation_period_frac": activation_frac,
    "fixed_avg_high_recall": round(fixed_avg_recall, 4),
    "adaptive_avg_high_recall": round(adap_avg_recall, 4),
}

output = {
    "theta_base": theta_base,
    "budget_normalized": 1.0,
    "period_normalized": 1.0,
    "budget_usd": B,
    "total_days": T,
    "cost_per_query_usd": cost_per_usd,
    "cost_per_query_fraction": round(cost_per_usd / B, 6),
    "fixed": fixed_norm,
    "adaptive": adaptive_norm,
    "summary": summary,
}

SIM_PATH.write_text(json.dumps(output, indent=2))
print("Updated budget_simulation.json")
print(json.dumps(summary, indent=2))

# Print daily adaptive data to verify adaptive kicks in
print("\nAdaptive daily theta and spend:")
for d in adaptive_norm:
    marker = " <-- kicks in" if d["theta_eff"] > theta_base else ""
    print(
        f"  day {d['day']:2d}  theta_eff={d['theta_eff']:.3f}  spend={d['spend_fraction']:.3f}  recall={d['high_recall']:.3f}{marker}"
    )
