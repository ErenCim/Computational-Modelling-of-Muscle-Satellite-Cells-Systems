from pathlib import Path
import pandas as pd

# Update this path if your project is in a different location
DATA_DIR = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis")
RUNS_ROOT = DATA_DIR / "gd_sweep_runs"

def analyze_sweep_iterations(runs_dir: Path, max_iters: int = 1500):
    maxed_out = []
    early_stopped = []
    crashed = []

    # Get all seed directories and sort them numerically
    seed_dirs = sorted(runs_dir.glob("seed_*"), key=lambda x: int(x.name.split('_')[1]))

    for run_dir in seed_dirs:
        seed_num = int(run_dir.name.split('_')[1])
        loss_csv = run_dir / "loss_history.csv"

        if not loss_csv.exists():
            crashed.append(seed_num)
            continue

        try:
            df = pd.read_csv(loss_csv)
            if df.empty:
                crashed.append(seed_num)
                continue
                
            last_iter = df["iteration"].max()

            if last_iter >= max_iters:
                maxed_out.append(seed_num)
            else:
                early_stopped.append((seed_num, last_iter))
        except Exception:
            crashed.append(seed_num)

    # Print Results
    print(f"--- Sweep Analysis (Target: {max_iters} iters) ---")
    
    print(f"\n✅ Maxed Out ({len(maxed_out)} seeds):")
    print(maxed_out if maxed_out else "None")

    '''
    print(f"\n⏹ Early Stopped ({len(early_stopped)} seeds):")
    if early_stopped:
        for s, it in early_stopped:
            print(f"  Seed {s}: stopped at {it}")
    else:
        print("None")

    print(f"\nCrashed/No Data ({len(crashed)} seeds):")
    print(crashed if crashed else "None")
    '''

if __name__ == "__main__":
    if RUNS_ROOT.exists():
        analyze_sweep_iterations(RUNS_ROOT)
    else:
        print(f"Directory not found: {RUNS_ROOT}")