import subprocess
import sys
from pathlib import Path


def test_space_export_excludes_private_data_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / "space"
    command = [sys.executable, str(root / "scripts/prepare_space.py"), "--output", str(target)]
    subprocess.run(command, check=True, capture_output=True)
    names = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    assert {
        "Dockerfile",
        "README.md",
        "pyproject.toml",
        "src/coach_emotion/static/index.html",
    } <= names
    assert all(
        name in {"Dockerfile", "README.md", "pyproject.toml"}
        or name.startswith("src/coach_emotion/")
        for name in names
    )
    assert all(
        Path(name).suffix in {".py", ".toml", ".md", ".html", ".css", ".js", ""} for name in names
    )
    assert "sdk: docker" in (target / "README.md").read_text(encoding="utf-8")
    retry = subprocess.run(command, capture_output=True)
    assert retry.returncode != 0
