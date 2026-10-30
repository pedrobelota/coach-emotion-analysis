"""Prepare an allowlisted Spaces source folder. Never authenticates or publishes."""

import argparse
import shutil
from pathlib import Path


def prepare(destination: Path) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Destination exists; choose a new directory")
    if root == destination or root.is_relative_to(destination):
        raise ValueError("Destination cannot replace the project or an ancestor")
    files = {Path("pyproject.toml"): root / "pyproject.toml"}
    for name in ("README.md", "Dockerfile"):
        files[Path(name)] = root / "space" / name
    for source in (root / "src" / "coach_emotion").glob("*.py"):
        files[source.relative_to(root)] = source
    for source in (root / "src" / "coach_emotion" / "static").glob("*"):
        if source.suffix in {".html", ".css", ".js"}:
            files[source.relative_to(root)] = source
    for relative, source in files.items():
        if source.is_symlink():
            raise ValueError("Symlink source is not allowed")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return [name.as_posix() for name in sorted(files)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected = prepare(args.output)
    print(f"Prepared {len(selected)} source files. No upload performed.")
