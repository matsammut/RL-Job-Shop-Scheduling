import os
import pandas as pd

best_known = {
    "ta42": 642,
    "ta52": 1231,
    "ta62": 3559,
    "ta72": 4616,
}

results_dir = "ray_results"
instances = ["ta42", "ta52", "ta62", "ta72"]

summary = []

for inst in instances:
    # find progress.csv inside PPO run folder
    folder = [f for f in os.listdir(results_dir) if inst in f][0]
    progress_path = os.path.join(results_dir, folder, "progress.csv")
    df = pd.read_csv(progress_path)

    # best makespan
    best_reward = df['episode_reward_mean'].dropna().max()
    best_makespan = -best_reward

    # optimality gap
    gap = ((best_makespan - best_known[inst]) / best_known[inst]) * 100

    summary.append({
        "Instance": inst,
        "Best_Makespan": best_makespan,
        "Best_Known": best_known[inst],
        "Optimality_Gap (%)": gap
    })
print(df['episode_reward_mean'])
summary_df = pd.DataFrame(summary)
print(summary_df)
