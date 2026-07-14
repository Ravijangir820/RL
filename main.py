import os
import subprocess
import sys
from pathlib import Path


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


def open_file_default(path):
    """Open a file with the system default application."""
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return True
    except Exception as exc:
        print(f"Warning: could not open {path}: {exc}")
        return False


def open_demo_gifs(base_dir):
    """Open generated comparison GIFs in a side-by-side dashboard."""
    demo_dir = Path(base_dir) / "hrl_taxi" / "reports" / "figures" / "demo_episode_comparison"
    budgets = (1, 100, 3000)

    rows = []
    for budget in budgets:
        flat_path = demo_dir / f"flat_budget_{budget}.gif"
        options_path = demo_dir / f"options_budget_{budget}.gif"
        if flat_path.exists() and options_path.exists():
            rows.append((budget, flat_path.as_uri(), options_path.as_uri()))

    if not rows:
        print("No demo GIFs found to open. Run demo comparison first to generate them.")
        return

    dashboard_path = demo_dir / "gif_dashboard.html"
    html_rows = []
    for budget, flat_uri, options_uri in rows:
        html_rows.append(
            f"""
            <div class=\"card\">
                <div class=\"title\">Flat - Budget {budget}</div>
                <img src=\"{flat_uri}\" alt=\"Flat budget {budget}\" />
            </div>
            <div class=\"card\">
                <div class=\"title\">Options - Budget {budget}</div>
                <img src=\"{options_uri}\" alt=\"Options budget {budget}\" />
            </div>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>Flat vs Options GIF Comparison</title>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 16px; background: #f6f8fb; }}
        h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
        p {{ margin: 0 0 14px 0; color: #444; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        .card {{ background: #fff; border: 1px solid #d8dee9; border-radius: 10px; padding: 10px; }}
        .title {{ font-weight: 600; margin-bottom: 8px; }}
        img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
    </style>
</head>
<body>
    <h1>Trajectory Comparison Dashboard</h1>
    <p>Flat on left column, Options on right column (budgets: 1, 100, 3000).</p>
    <div class=\"grid\">
        {''.join(html_rows)}
    </div>
</body>
</html>
"""

    dashboard_path.write_text(html, encoding="utf-8")
    print(f"\nOpening side-by-side GIF dashboard: {dashboard_path}")
    open_file_default(str(dashboard_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HRL Taxi-v3 Pipeline")
    parser.add_argument("--grid", type=int, default=5, help="Grid size for Taxi environment (default: 5)")
    parser.add_argument(
        "--demo-comparison",
        action="store_true",
        help="Run episode-budget demo comparison (1, 10, 100, 1000) with metrics and GIFs.",
    )
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

    # Optional: Episode-budget demo exports
    if args.demo_comparison:
        print("\n[6/6] Running Episode-Budget Demo Comparison...")
        if not run_command([python_exe, "demo_episode_comparison.py"], cwd=hrl_src):
            print("Demo comparison failed.")
            return

    # Always open available GIFs at end for presentation convenience.
    open_demo_gifs(base_dir)

    print("\n" + "="*60)
    print("Pipeline completed! All results displayed.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
