"""Download LiftLens's optional local Qwen3-VL video-review runtime."""

from __future__ import annotations

import pathlib
import sys
import zipfile

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "work" / "local_video_downloads"
RUNTIME = ROOT / ".runtime"
MODELS = ROOT / "models"

ASSETS = (
    ("llama-video.zip", "https://github.com/ggml-org/llama.cpp/releases/download/b11249/llama-b11249-bin-win-cuda-13.4-x64.zip", 153_544_476, "zip-llama"),
    ("llama-cuda.zip", "https://github.com/ggml-org/llama.cpp/releases/download/b11249/cudart-llama-bin-win-cuda-13.4-x64.zip", 423_535_356, "zip-llama"),
    ("ffmpeg-essentials.zip", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", 114_768_076, "zip-ffmpeg"),
    ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1_107_409_952, "model"),
    ("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", 445_053_216, "model"),
)


def download(name: str, url: str, expected_size: int) -> pathlib.Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / name
    if target.is_file() and target.stat().st_size == expected_size:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {name} ({expected_size // (1024 * 1024)} MB)", flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        size = 0
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
                    size += len(chunk)
        if size != expected_size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: expected {expected_size} bytes, received {size}")
    partial.replace(target)
    return target


def safe_extract_zip(archive: pathlib.Path, destination: pathlib.Path, wanted=None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        entries = []
        for info in bundle.infolist():
            name = pathlib.PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"Unsafe path in downloaded archive: {info.filename}")
            if wanted is None or wanted(info.filename):
                entries.append(info)
        for info in entries:
            if wanted and info.filename.startswith("ffmpeg-"):
                relative = pathlib.PurePosixPath(info.filename).relative_to(
                    next(part for part in pathlib.PurePosixPath(info.filename).parts if part.startswith("ffmpeg-"))
                )
            else:
                relative = pathlib.PurePosixPath(info.filename)
            target = (destination / pathlib.Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"Unsafe extraction target: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def main() -> int:
    for name, url, size, kind in ASSETS:
        if kind == "model":
            target = MODELS / name
            if target.is_file() and target.stat().st_size == size:
                continue
        elif kind == "zip-llama" and (RUNTIME / "llama" / "llama-mtmd-cli.exe").is_file():
            if (name == "llama-video.zip" or (RUNTIME / "llama" / "cublas64_13.dll").is_file()):
                continue
        elif kind == "zip-ffmpeg" and (RUNTIME / "ffmpeg" / "bin" / "ffmpeg.exe").is_file() and (RUNTIME / "ffmpeg" / "bin" / "ffprobe.exe").is_file():
            continue
        archive = download(name, url, size)
        if kind == "zip-llama":
            safe_extract_zip(archive, RUNTIME / "llama")
        elif kind == "zip-ffmpeg":
            safe_extract_zip(
                archive,
                RUNTIME / "ffmpeg" / "bin",
                wanted=lambda item: item.endswith(("/bin/ffmpeg.exe", "/bin/ffprobe.exe")),
            )
        elif kind == "model":
            MODELS.mkdir(parents=True, exist_ok=True)
            target = MODELS / name
            archive.replace(target)
    print("Local Qwen3-VL video-review runtime is ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
