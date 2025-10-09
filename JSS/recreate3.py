import pandas as pd
import re
from pathlib import Path

# === change this to your file ===
csv_path = Path("ray_results/PPO_ta72/progress.csv")
df = pd.read_csv(csv_path)
print(df.columns.tolist())

# Heuristics: columns that might contain makespan logged by RLlib custom metrics.
candidate_cols = [c for c in df.columns 
                  if re.search(r"makespan", c, re.IGNORECASE)]

if not candidate_cols:
    # Fallback: some runs log reward as -makespan (so larger reward = better).
    # If that’s your case, we’ll infer makespan from reward.
    reward_cols = [c for c in df.columns if "episode_reward" in c]
    if not reward_cols:
        raise ValueError(
            "Couldn’t find a makespan or reward column. "
            "Open the CSV and tell me the column headers."
        )
    # Try best-episode reward
    # If reward == -makespan, min makespan = -max reward
    reward_col = "episode_reward_mean" if "episode_reward_mean" in df.columns else reward_cols[0]
    df["__makespan_inferred__"] = df[reward_col]
    candidate_cols = ["__makespan_inferred__"]

# Pick the “tightest” makespan column (min across them, just in case several exist)
best_series = pd.concat([df[c] for c in candidate_cols], axis=1).min(axis=1)

best_seen_makespan = best_series.min()
final_window_makespan = best_series.tail(10).mean()  # average of last 10 iters (stable view)

# BKS for ta72 from the literature:
BKS_TA72 = 5181

def gap(val, bks=BKS_TA72):
    return (val - bks) / bks * 100

report = {
    "file": str(csv_path),
    "makespan_column_candidates": candidate_cols,
    "best_seen_makespan": float(best_seen_makespan),
    "best_seen_gap_%": float(gap(best_seen_makespan)),
    "last10it_avg_makespan": float(final_window_makespan),
    "last10it_avg_gap_%": float(gap(final_window_makespan)),
    "BKS_ta72": BKS_TA72
}

print(report)
