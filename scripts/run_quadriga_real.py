"""Generate real QuaDRiGa multi-cell channel data via MATLAB.

Produces 10 shards x 500 UEs = 5000 samples with realistic
path loss, positions, and multi-cell interference.

Usage:
    python scripts/run_quadriga_real.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATLAB_DIR = _PROJECT_ROOT / "matlab"
OUTPUT_DIR = _PROJECT_ROOT / "artifacts" / "quadriga_real_mat"

NUM_SHARDS = 20
UES_PER_SHARD = 50

CONFIG_TEMPLATE = {
    "num_cells": 7,
    "num_ues": UES_PER_SHARD,
    "num_snapshots": 14,
    "carrier_freq_hz": 3.5e9,
    "scenario": "3GPP_38.901_UMa_NLOS",
    "isd_m": 500,
    "tx_height_m": 25,
    "rx_height_m": 1.5,
    "cell_radius_m": 250,
    "ue_speed_kmh": 3,
    "bs_ant_v": 4,
    "bs_ant_h": 8,
    "ue_ant_v": 1,
    "ue_ant_h": 2,
    "n_rb": 273,
    "sc_inter": 30000,
    "tx_power_dbm": 43.0,
    "output_dir": str(OUTPUT_DIR),
}

MATLAB_EXE = os.environ.get("MSG_MATLAB_EXE", r"D:\matlab\bin\matlab.exe")


def run_shard(shard_id: int) -> bool:
    cfg = dict(CONFIG_TEMPLATE)
    cfg["seed"] = 42 + shard_id
    cfg["shard_id"] = shard_id

    config_path = MATLAB_DIR / f"_config_shard{shard_id:04d}.json"
    config_path.write_text(json.dumps(cfg, indent=2))

    out_file = OUTPUT_DIR / f"QDRG_real_shard{shard_id:04d}.mat"
    if out_file.exists():
        print(f"  Shard {shard_id}: already exists, skipping")
        return True

    cmd = [
        str(MATLAB_EXE),
        "-nosplash",
        "-nodesktop",
        "-batch",
        f"main_multi('{config_path.as_posix()}')",
    ]

    print(f"  Shard {shard_id}: starting MATLAB ({UES_PER_SHARD} UEs, 7 cells)...")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(MATLAB_DIR),
        capture_output=True,
        text=True,
        timeout=7200,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Shard {shard_id}: FAILED ({elapsed:.0f}s)")
        print(f"  STDERR: {result.stderr[:500]}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        return False

    progress_lines = [line for line in result.stdout.split("\n") if "UE " in line or "Saved" in line]
    for line in progress_lines[-3:]:
        print(f"    {line.strip()}")

    print(f"  Shard {shard_id}: OK ({elapsed:.0f}s, {UES_PER_SHARD / elapsed:.1f} UE/s)")

    config_path.unlink(missing_ok=True)
    return True


def main() -> None:
    print("=" * 60)
    print("QuaDRiGa Real Multi-Cell Channel Generation")
    print(f"  {NUM_SHARDS} shards x {UES_PER_SHARD} UEs = {NUM_SHARDS * UES_PER_SHARD} total")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    success = 0
    for i in range(NUM_SHARDS):
        ok = run_shard(i)
        if ok:
            success += 1
        print(f"  Progress: {success}/{i + 1} shards done")

    elapsed = time.time() - t_start
    print(f"\nCompleted: {success}/{NUM_SHARDS} shards in {elapsed / 60:.1f} min")

    if success == NUM_SHARDS:
        print(f"\nAll {NUM_SHARDS * UES_PER_SHARD} samples generated!")
        print(f"Output directory: {OUTPUT_DIR}")
    else:
        print(f"\nWARNING: {NUM_SHARDS - success} shards failed")


if __name__ == "__main__":
    main()
