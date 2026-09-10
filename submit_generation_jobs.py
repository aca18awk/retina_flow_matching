"""
Submit one SLURM job per (experiment, seed, N) combination.

Edit the SLURM header variables at the top, then run:
    python submit_generation_jobs.py [--dry-run]
"""

import argparse
import itertools
import os
import subprocess

# ── Job combinations ──────────────────────────────────────j────────────────────
# EXPERIMENTS     = ["all", "dilated", "nondilated"]
# SEEDS           = ["seed_A", "seed_B", "seed_C"]
# N_VALUES        = [20, 50, 100, 200, 300, 500]
EXPERIMENTS     = ["all", "dilated", "nondilated"]
SEEDS           = ["seed_A", "seed_B", "seed_C"]
N_VALUES        = [300]
GUIDANCE_SCALE  = 1.5
K_NEIGHBORS     = 2

# ── SLURM settings ────────────────────────────────────────────────────────────
PARTITION   = "gpus48"
GRES        = "gpu:1"
NODES       = ["loki"]   # jobs round-robin across these nodes
CONDA_SH    = "/vol/biomedic3/awk24/miniconda3/bin/activate"
CONDA_ENV   = "flow-env"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(SCRIPT_DIR, "run_logs")
# ─────────────────────────────────────────────────────────────────────────────


def make_job_script(exp: str, seed: str, N: int, node: str) -> str:
    job_name = f"gen_{exp}_{seed}_N{N}"
    log_file = f"run_logs/16_june_n_abl_{seed}_{exp}.%N.%j.log"

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {PARTITION}
#SBATCH --gres {GRES}
#SBATCH --nodelist {node}
#SBATCH --output={log_file}

source {CONDA_SH}
conda activate {CONDA_ENV}

python Eyepacs_generation.py \\
    --experiment {exp} \\
    --seed {seed} \\
    --N {N} \\
    --guidance_scale {GUIDANCE_SCALE} \\
    --k_neighbors {K_NEIGHBORS}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job scripts without submitting")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    combos = list(itertools.product(EXPERIMENTS, SEEDS, N_VALUES))
    print(f"Submitting {len(combos)} jobs across {NODES}...")

    for i, (exp, seed, N) in enumerate(combos):
        node = NODES[i % len(NODES)]
        script = make_job_script(exp, seed, N, node)
        if args.dry_run:
            print(f"\n{'─'*60}")
            print(script)
        else:
            result = subprocess.run(
                ["sbatch"], input=script, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  [{node}] Submitted {exp}/{seed}/N{N}: {result.stdout.strip()}")
            else:
                print(f"  [{node}] FAILED {exp}/{seed}/N{N}: {result.stderr.strip()}")

    if args.dry_run:
        print(f"\n{len(combos)} jobs would be submitted (--dry-run, nothing submitted)")
    else:
        print(f"\nAll {len(combos)} jobs submitted. Logs → {LOG_DIR}")
