# archive/media.py
import subprocess
from pathlib import Path

def extract_audio(video_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn",
           "-ac", "1", "-ar", "16000", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

def extract_keyframes(video_path: Path, out_dir: Path, threshold: float = 0.3) -> list[Path]:
    """장면 전환 감지로 keyframe 추출. out_dir/NNNN.jpg 로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vf", f"select='gt(scene,{threshold})',showinfo",
           "-vsync", "vfr", str(out_dir / "%04d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(out_dir.glob("*.jpg"))
