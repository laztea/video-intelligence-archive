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
    """장면 전환 감지로 keyframe 추출. out_dir/NNNN.jpg 로 저장.

    첫 프레임(eq(n,0))을 항상 포함해, 장면 전환이 임계값을 넘지 않는
    정적인 영상에서도 최소 1프레임이 추출되도록 한다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vf", f"select='eq(n,0)+gt(scene,{threshold})'",
           "-fps_mode", "vfr", str(out_dir / "%04d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(out_dir.glob("*.jpg"))
