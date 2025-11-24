import sys
from pathlib import Path

try:
    from gtts import gTTS
except ImportError as exc:
    print("Missing dependency: gTTS. Install with: pip install gTTS", file=sys.stderr)
    raise


def read_text_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Input text file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input text file is empty.")
    return text


def synthesize_to_mp3(text: str, output_path: Path, language_code: str = "en") -> None:
    tts = gTTS(text=text, lang=language_code, slow=False)
    tts.save(str(output_path))


def main() -> None:
    input_path = Path("/home/pope/Desktop/tools/tts/audio.txt")
    output_path = Path("/home/pope/Desktop/tools/tts/audio.mp3")

    try:
        text = read_text_file(input_path)
        synthesize_to_mp3(text, output_path)
    except Exception as error:  # Keep simple CLI error surfacing
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"MP3 saved to: {output_path}")


if __name__ == "__main__":
    main()

