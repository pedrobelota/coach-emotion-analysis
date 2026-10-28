"""Explicit local model test. Writes diagnostics only to an ignored output folder."""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Local audio/video; never copied to the repository"
    )
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--offline", action="store_true", help="Require models already in the local cache"
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    output = Path(args.output_dir)
    if output.exists():
        parser.error("Output directory exists; select a new directory")
    output.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "coach_emotion",
        "analyze",
        "--input",
        args.input,
        "--output",
        str(output / "predictions.tsv"),
        "--details",
        str(output / "details.tsv"),
        "--windows",
        str(output / "windows.json"),
        "--device",
        args.device,
        "--model",
        args.model,
    ]
    env = dict(os.environ, PYTHONUTF8="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    log_path = output / "console.log"
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            run = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=args.timeout_seconds,
            )
            exit_code = run.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
    elapsed = time.perf_counter() - started
    console = log_path.read_text(encoding="utf-8")
    versions = {}
    for package in (
        "coach-emotion-analysis",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "transformers",
        "faster-whisper",
        "ctranslate2",
        "librosa",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    summary = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 2),
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "versions": versions,
        "requested_device": args.device,
        "requested_model": args.model,
        "offline": args.offline,
        "cpu_fallback_observed": "Tentando fallback do faster-whisper em CPU" in console,
    }
    try:
        import torch

        summary["cuda_available"] = torch.cuda.is_available()
        summary["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        summary["cuda_available"] = False
    if exit_code == 0:
        with (output / "predictions.tsv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        windows = json.loads((output / "windows.json").read_text(encoding="utf-8"))
        assert rows and windows, "No outputs generated"
        assert all(0 <= int(row["codigo_emocao"]) <= 6 for row in rows)
        summary.update(
            intervals=len(rows),
            windows=len(windows),
            duration_seconds=float(rows[-1]["fim_s"]),
            codes=[int(row["codigo_emocao"]) for row in rows],
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if exit_code:
        print(f"See local diagnostic log: {output / 'console.log'}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
