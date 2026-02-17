import os
import subprocess
import sys


def run_command(cmd, cwd=None):
    """Run a command and wait for completion."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HRL Taxi-v3 Pipeline")
    parser.add_argument("--grid", type=int, default=5, help="Grid size for Taxi environment (default: 5)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    hrl_src = os.path.join(base_dir, "hrl_taxi", "src")
    python_exe = sys.executable

    os.environ["TAXI_GRID_SIZE"] = str(args.grid)
    
    print("\n" + "="*60)
    print(f"HRL Taxi-v3 Project Pipeline (Grid: {args.grid}x{args.grid})")
    print("="*60)

    # Step 1: Human play
    print("\n[1/6] Starting Human Play...")
    print("Play a game of Taxi yourself. Arrow keys to move, P=pickup, D=dropoff")
    if not run_command([python_exe, "taxi_ui_2d.py", "human"], cwd=hrl_src):
        print("Human play skipped or failed.")
        return

    # Step 2: Train Flat Q-learning
    print("\n[2/6] Training Flat Q-learning...")
    if not run_command([python_exe, "train_flat.py"], cwd=hrl_src):
        print("Flat training failed.")
        return

    # Step 2b: Display Flat Agent Playing
    print("\n[2b/6] Displaying Trained Flat Agent...")
    print("Watch the flat Q-learning agent play.")
    if not run_command([python_exe, "taxi_ui_2d.py", "flat"], cwd=hrl_src):
        print("Flat agent display failed.")
        return

    # Step 3: Train Options (HRL)
    print("\n[3/5] Training Options (Hierarchical RL)...")
    if not run_command([python_exe, "train_options.py"], cwd=hrl_src):
        print("Options training failed.")
        return

    # Step 3b: Display Options Agent Playing
    print("\n[3b/5] Displaying Trained Options Agent...")
    print("Watch the hierarchical RL agent play.")
    if not run_command([python_exe, "taxi_ui_2d.py", "options"], cwd=hrl_src):
        print("Options agent display failed.")
        return

    # Step 3c: Show Project Summary
    print("\n[3c/6] Project Summary & Details...")
    if not run_command([python_exe, "project_summary_gui.py"], cwd=hrl_src):
        print("Project summary failed.")
        return

    # Step 4: Plot results
    print("\n[5/6] Plotting Training Metrics...")
    if not run_command([python_exe, "plot_results.py"], cwd=hrl_src):
        print("Plotting failed.")
        return

    print("\n" + "="*60)
    print("Pipeline completed! All results displayed.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
