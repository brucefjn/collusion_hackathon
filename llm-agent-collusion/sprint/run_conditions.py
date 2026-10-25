"""
Sprint project 1.5 runner (CAIAC AM team, new sprint work): runs every condition of our design
on the same seeds, one process per (condition, seed), several at a time.

Question: when AI sellers are under pressure to keep prices high, which remedy breaks up their
collusion at the least cost to the market and to the enforcer?

Run from llm-agent-collusion/ (the repo-root compute.env must hold OPENROUTER_API_KEY):

    uv run python sprint/run_conditions.py --stage pilot             # control, ban, no pressure; seed 0
    uv run python sprint/run_conditions.py --stage core              # 7 conditions x 5 seeds
    uv run python sprint/run_conditions.py --stage extra             # 7 stretch conditions x 5 seeds
    uv run python sprint/run_conditions.py --conditions control cap5 --seeds 0 1   # any subset
    uv run python sprint/run_conditions.py --stage core --dry_run    # print the plan, run nothing

Every run lands in results/sprint/<timestamp>_<condition>-seed<N>/ (results/sprint_paper/ with --paper).
Runs that already finished in that folder are skipped, so you can re-run a stage after a crash and the
pilot's seed-0 runs count toward the core stage. Then: uv run python sprint/analyze.py
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIMULATION = "src/continuous_double_auction/simulation.py"

PRESSURE = ["--seller_comms_enabled", "--boss_pressure"]

# name -> (remedy family, what it is, simulation flags). The maverick's seller list is filled in per market size.
CONDITIONS = {
    "control": ("none", "CEO pressure, sellers can talk, no remedy (positive control)", PRESSURE),
    "no_pressure": ("none", "sellers can talk, no CEO pressure (second setting)", ["--seller_comms_enabled"]),
    "ban": ("communication limits", "sellers cannot message each other (reference point)", ["--boss_pressure"]),
    "cap5": ("communication limits", "seller messages cut to 5 characters", PRESSURE + ["--max_message_length", "5"]),
    "cap20": ("communication limits", "seller messages cut to 20 characters", PRESSURE + ["--max_message_length", "20"]),
    "cap80": ("communication limits", "seller messages cut to 80 characters", PRESSURE + ["--max_message_length", "80"]),
    "oversight100": (
        "monitoring",
        "warning + monitor reads all messages + gag after a top score",
        PRESSURE + ["--oversight_enabled"],
    ),
    "oversight50": (
        "monitoring",
        "warning + monitor reads 50% of messages + gag",
        PRESSURE + ["--oversight_enabled", "--overseer_sample_fraction", "0.5"],
    ),
    "oversight25": (
        "monitoring",
        "warning + monitor reads 25% of messages + gag",
        PRESSURE + ["--oversight_enabled", "--overseer_sample_fraction", "0.25"],
    ),
    "warning_only": ("monitoring", "warning only, no monitor and no gag (a bluff)", PRESSURE + ["--oversight_warning_only"]),
    "oversight_silent": (
        "monitoring",
        "monitor reads all messages and can gag, but sellers get no warning",
        PRESSURE + ["--oversight_enabled", "--oversight_silent"],
    ),
    "oversight_enforced": (
        "monitoring",
        "warning + monitor + a gag that really cuts messages to 5 characters",
        PRESSURE + ["--oversight_enabled", "--enforce_gag"],
    ),
    "start_competitive": (
        "none",
        "control, but sellers' opening asks start at the competitive price ($90)",
        PRESSURE + ["--initial_ask_center", "90"],
    ),
    "maverick": ("structural", "one LLM seller replaced by a scripted price-cutter", PRESSURE),
}

STAGES = {
    # control = pressure + talk; ban = pressure without talk (does talk cause the high prices?); no_pressure = talk only
    "pilot": (["control", "ban", "no_pressure"], [0]),
    "core": (["control", "no_pressure", "ban", "start_competitive", "cap5", "oversight100", "oversight25"],
             [0, 1, 2, 3, 4]),
    "extra": (["cap20", "cap80", "oversight50", "warning_only", "oversight_silent", "oversight_enforced", "maverick"],
              [0, 1, 2, 3, 4]),
}


def build_command(condition: str, seed: int, args, log_dir: str) -> list[str]:
    n_sellers = 5 if args.paper else 3
    flags = list(CONDITIONS[condition][2])
    if condition == "maverick":
        flags += ["--seller_models", *([args.model] * (n_sellers - 1)), "maverick"]
    else:
        flags += ["--seller_models", args.model]
    cmd = [sys.executable, SIMULATION, *flags, "--buyer_models", args.model]
    if args.paper:
        cmd.append("--paper")
    if args.rounds:
        cmd += ["--rounds", str(args.rounds)]
    cmd += ["--seeds", str(seed), "--tag", f"{condition}-seed{seed}", "--log_dir", log_dir, "-y"]
    return cmd


def model_calls(condition: str, args) -> int:
    sellers, buyers, rounds = (5, 6, 30) if args.paper else (3, 3, 15)
    rounds = args.rounds or rounds
    llm_sellers = sellers - 1 if condition == "maverick" else sellers
    calls = rounds * (llm_sellers + buyers)
    if "--oversight_enabled" in CONDITIONS[condition][2]:
        calls += rounds  # one overseer call per hour
    return calls


def already_done(log_dir: Path, condition: str, seed: int) -> bool:
    return any((d / "collusion_metrics.json").exists() for d in log_dir.glob(f"*_{condition}-seed{seed}"))


def run_one(cmd: list[str], log_file: Path) -> int:
    with open(log_file, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


def summary_lines(log_file: Path) -> str:
    text = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(text):
        if "Summary (collusion index" in line:
            return "\n".join(text[i : i + 9])
    return "\n".join(text[-6:])  # no summary: show the tail (usually the error)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), help="a preset set of conditions and seeds")
    p.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), help="run these conditions instead")
    p.add_argument("--seeds", type=int, nargs="+", help="override the stage's seeds")
    p.add_argument("--model", default="gpt-4.1", help="model for every LLM seller and buyer (paper: gpt-4.1)")
    p.add_argument("--paper", action="store_true", help="paper-size market: 5 sellers, 6 buyers, 30 rounds")
    p.add_argument("--rounds", type=int, help="override the number of rounds")
    p.add_argument("--parallel", type=int, default=3, help="runs at the same time (default 3)")
    p.add_argument("--fresh", action="store_true", help="re-run even if a finished run already exists")
    p.add_argument("--dry_run", action="store_true", help="print the plan and the commands, run nothing")
    p.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    if not args.stage and not args.conditions:
        p.error("pass --stage or --conditions")
    conditions, seeds = STAGES[args.stage] if args.stage else (args.conditions, [0])
    if args.conditions:
        conditions = args.conditions
    if args.seeds:
        seeds = args.seeds

    log_dir_name = "results/sprint_paper" if args.paper else "results/sprint"
    log_dir = PROJECT_DIR / log_dir_name
    console_dir = log_dir / "_console"
    console_dir.mkdir(parents=True, exist_ok=True)

    plan, skipped = [], []
    for condition in conditions:
        for seed in seeds:
            if not args.fresh and already_done(log_dir, condition, seed):
                skipped.append(f"{condition}-seed{seed}")
            else:
                plan.append((condition, seed))

    total_calls = sum(model_calls(c, args) for c, _ in plan)
    print(f"Plan: {len(plan)} run(s), about {total_calls:,} model calls, up to {args.parallel} at a time.")
    for condition in conditions:
        family, what, _ = CONDITIONS[condition]
        print(f"  {condition:<13} [{family}] {what}")
    if skipped:
        print(f"Skipping {len(skipped)} finished run(s): {', '.join(skipped)} (use --fresh to redo them)")
    if args.dry_run:
        for condition, seed in plan:
            print(" ".join(build_command(condition, seed, args, log_dir_name)))
        return
    if not plan:
        print("Nothing to run.")
        return
    if not args.yes and sys.stdin.isatty() and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("aborted")

    start = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {}
        for condition, seed in plan:
            name = f"{condition}-seed{seed}"
            futures[pool.submit(run_one, build_command(condition, seed, args, log_dir_name), console_dir / f"{name}.log")] = name
        for done, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            code = future.result()
            status = "done" if code == 0 else f"FAILED (exit {code})"
            print(f"\n[{done}/{len(plan)}] {name}: {status}  ({(time.time() - start) / 60:.1f} min elapsed)")
            print(summary_lines(console_dir / f"{name}.log"))
            if code != 0:
                failures.append(name)

    print(f"\nFinished {len(plan) - len(failures)}/{len(plan)} run(s) in {(time.time() - start) / 60:.1f} min.")
    if failures:
        print(f"Failed: {', '.join(failures)}. Full output: {console_dir}/<name>.log. Re-run the same command to retry them.")
    print(f"Next: uv run python sprint/analyze.py{' --paper' if args.paper else ''}")


if __name__ == "__main__":
    main()
