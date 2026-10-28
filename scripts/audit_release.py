"""Check tracked and candidate files for common accidental publication mistakes.

This is a narrow repository hygiene check, not a complete secret detector.
"""

import re
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
    ).strip()
    if Path(git_root).resolve() != root:
        raise SystemExit("Refusing to audit a parent Git repository")
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    files = sorted(set(raw.decode("utf-8").split("\0")) - {""})
    forbidden = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".log",
        ".pdf",
        ".docx",
        ".zip",
    }
    patterns = {
        "personal absolute path": re.compile(
            r"(?i)(?:[a-z]:[/\\]users[/\\]|/(?:home|users)/)[\w .-]+[/\\]"
        ),
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    }
    problems = []
    total = 0
    for name in files:
        path = root / name
        if not path.exists():
            continue
        size = path.stat().st_size
        total += size
        if path.suffix.lower() in forbidden or path.name.startswith(".env"):
            problems.append(f"Excluded artifact type: {name}")
        if size > 1_000_000:
            problems.append(f"Unexpected large file: {name}")
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            problems.append(f"Unexpected binary file: {name}")
            continue
        for label, pattern in patterns.items():
            if pattern.search(content):
                problems.append(f"{label}: {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        f"PASS: {len(files)} publishable text files, {total:,} bytes; no excluded artifacts or matched secrets."
    )


if __name__ == "__main__":
    main()
