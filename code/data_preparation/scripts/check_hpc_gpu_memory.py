from __future__ import annotations

import argparse
import json
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-gib", type=float, default=12.0)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    command = ["nvidia-smi"]
    if visible and visible not in {"NoDevFiles", "void"}:
        command.extend(["--id", visible])
    command.extend(
        [
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    output = subprocess.check_output(command, text=True).strip().splitlines()
    if not output:
        raise SystemExit("No GPU was visible to the SLURM job.")

    fields = [item.strip() for item in output[0].split(",")]
    if len(fields) != 4:
        raise SystemExit(f"Unexpected nvidia-smi output: {output[0]}")
    free_mib = float(fields[2])
    total_mib = float(fields[3])
    report = {
        "cuda_visible_devices": visible or None,
        "gpu_index": fields[0],
        "gpu_name": fields[1],
        "free_gib": free_mib / 1024.0,
        "total_gib": total_mib / 1024.0,
        "minimum_free_gib": args.minimum_free_gib,
        "passed": free_mib >= args.minimum_free_gib * 1024.0,
    }
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(
            f"Allocated GPU has only {report['free_gib']:.2f} GiB free; "
            f"at least {args.minimum_free_gib:.2f} GiB is required."
        )


if __name__ == "__main__":
    main()
