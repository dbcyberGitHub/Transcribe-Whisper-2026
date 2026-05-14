import os
import time
from pathlib import Path
from faster_whisper import WhisperModel

DESKTOP       = Path(os.path.expanduser("~")) / "Desktop"
BASE_FOLDER   = DESKTOP / "Transcribe Whisper 2026"
INPUT_FOLDER  = BASE_FOLDER / "Input"
OUTPUT_FOLDER = BASE_FOLDER / "Output"

MODEL_SIZE   = "medium"
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".mp4", ".ogg", ".flac"}
POLL_INTERVAL    = 2

def setup_folders():
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[INPUT]  {INPUT_FOLDER}")
    print(f"[OUTPUT] {OUTPUT_FOLDER}")

def load_model():
    print(f"\nLoading Whisper model ({MODEL_SIZE} / {DEVICE})...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model ready.\n")
    return model

def already_transcribed(audio_file: Path) -> bool:
    return (OUTPUT_FOLDER / (audio_file.stem + ".txt")).exists()

def transcribe_file(model: WhisperModel, audio_file: Path):
    output_txt = OUTPUT_FOLDER / (audio_file.stem + ".txt")
    print(f"[TRANSCRIBING] {audio_file.name} ...")
    try:
        segments, info = model.transcribe(
            str(audio_file),
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {audio_file.name}\n")
            f.write(f"Language: {info.language}  |  Confidence: {info.language_probability:.2%}\n")
            f.write("=" * 60 + "\n\n")
            parts = []
            for segment in segments:
                text = segment.text.strip()
                parts.append(text)
                print(text, end=" ", flush=True)
            f.write(" ".join(parts) + "\n")
            print()
        print(f"[DONE] Saved -> {output_txt.name}\n")
    except Exception as e:
        print(f"[ERROR] {audio_file.name}: {e}\n")

def watch(model: WhisperModel):
    print("Watching for new audio files... (Ctrl+C to stop)\n")
    seen = set()
    while True:
        try:
            for file in sorted(INPUT_FOLDER.iterdir()):
                if file.suffix.lower() in AUDIO_EXTENSIONS:
                    if file not in seen and not already_transcribed(file):
                        seen.add(file)
                        transcribe_file(model, file)
        except Exception as e:
            print(f"[WATCHER ERROR] {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    setup_folders()
    model = load_model()
    try:
        watch(model)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")