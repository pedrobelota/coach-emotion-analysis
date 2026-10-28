"""Public CLI. Help and evaluation do not import inference libraries."""

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from coach_emotion import __version__
from coach_emotion.runtime import RuntimeConfig, configuration


def _check_paths(inputs, outputs):
    resolved_inputs = {Path(p).expanduser().resolve() for p in inputs if p}
    resolved_outputs = [Path(p).expanduser().resolve() for p in outputs if p]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Output paths must be distinct")
    for path in resolved_outputs:
        if path in resolved_inputs:
            raise ValueError("An output cannot overwrite an input")
        if path.exists():
            raise ValueError(f"Output already exists: {path}. Choose a new filename.")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Research speech emotion analysis; not a validated measure of mental state"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze audio/video without human labels")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True, help="CSV or TSV output")
    analyze.add_argument("--details", help="Optional detailed CSV/TSV")
    analyze.add_argument("--windows", help="Optional JSON with transcripts: treat as private")
    analyze.add_argument("--config", help="JSON runtime configuration")
    analyze.add_argument("--device", choices=("auto", "cpu", "cuda"))
    analyze.add_argument("--backend", choices=("auto", "faster-whisper", "transformers"))
    analyze.add_argument(
        "--model", choices=("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
    )
    analyze.add_argument("--window-ms", type=int, default=4000)
    analyze.add_argument("--hop-ms", type=int, default=2000)
    analyze.add_argument("--interval-seconds", type=float, default=5.0)
    analyze.add_argument("--start-seconds", type=float)
    analyze.add_argument("--end-seconds", type=float)
    analyze.add_argument("--verbose", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Compare final predictions with human annotations")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--ground-truth", required=True)
    evaluate.add_argument("--output", help="Optional JSON metrics")
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            _check_paths([args.predictions, args.ground_truth], [args.output])
            from coach_emotion.evaluate import avaliar_dataset_final

            result = asdict(avaliar_dataset_final(args.ground_truth, args.predictions))
            text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                destination = Path(args.output).expanduser()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text + "\n", encoding="utf-8")
            print(text)
            return 0

        _check_paths([args.input, args.config], [args.output, args.details, args.windows])
        if args.window_ms <= 0 or args.hop_ms <= 0 or args.hop_ms > args.window_ms:
            raise ValueError("Window and hop must be positive; hop cannot exceed window")
        config = RuntimeConfig.from_file(args.config) if args.config else RuntimeConfig()
        changes = {key: getattr(args, key) for key in ("device", "backend") if getattr(args, key)}
        if args.model:
            changes["stt_model"] = f"openai/whisper-{args.model}"
        config = replace(config, **changes)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
        from coach_emotion.analyze import main_cli

        forwarded = [
            "--entrada",
            args.input,
            "--saida",
            args.output,
            "--jan-ms",
            str(args.window_ms),
            "--hop-ms",
            str(args.hop_ms),
            "--intervalo-s",
            str(args.interval_seconds),
        ]
        for key, flag in (
            ("details", "--saida-detalhes"),
            ("windows", "--saida-janelas"),
            ("start_seconds", "--inicio-s"),
            ("end_seconds", "--fim-s"),
        ):
            value = getattr(args, key)
            if value is not None:
                forwarded.extend([flag, str(value)])
        with configuration(config):
            return main_cli(forwarded)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
