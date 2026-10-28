"""Generate a silent WAV to check file handling. It does not validate emotions."""

import argparse
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/synthetic-silence.wav")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 16000 * 2 * 10)
    print(f"Created {output}: 10 seconds of silence, no human recordings.")


if __name__ == "__main__":
    main()
