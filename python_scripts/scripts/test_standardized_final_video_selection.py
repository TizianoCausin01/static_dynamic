import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


@dataclass
class Cfg:
    input_dir: Path = Path("/Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes")
    standardized_dir: Path | None = None
    padded_dir: Path | None = None
    fps: int = 30
    duration: float = 2.5
    padded_duration: float = 3.0
    width: int = 500
    height: int = 500
    duration_tolerance: float = 0.002


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    codec_name: str
    profile: str
    codec_tag_string: str
    pix_fmt: str
    fps: float
    frame_count: int
    duration: float


"""
parse_args
Parse command line arguments into a config object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Check standardized videos have the expected temporal and spatial metadata."
    )
    parser.add_argument("--input_dir", type=Path, default=Cfg.input_dir)
    parser.add_argument("--standardized_dir", type=Path, default=None)
    parser.add_argument("--padded_dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=Cfg.fps)
    parser.add_argument("--duration", type=float, default=Cfg.duration)
    parser.add_argument("--padded_duration", type=float, default=Cfg.padded_duration)
    parser.add_argument("--width", type=int, default=Cfg.width)
    parser.add_argument("--height", type=int, default=Cfg.height)
    parser.add_argument("--duration_tolerance", type=float, default=Cfg.duration_tolerance)
    args = parser.parse_args()

    return Cfg(
        input_dir=args.input_dir,
        standardized_dir=args.standardized_dir,
        padded_dir=args.padded_dir,
        fps=args.fps,
        duration=args.duration,
        padded_duration=args.padded_duration,
        width=args.width,
        height=args.height,
        duration_tolerance=args.duration_tolerance,
    )


def require_tool(tool_name):
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"{tool_name} is required, but it was not found on PATH.")


def iter_video_paths(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def rate_to_float(rate):
    return float(Fraction(rate))


def default_standardized_dir(input_dir, fps):
    if fps == 30:
        return input_dir / "standardized"
    return input_dir / f"standardized_{fps}fps"


def default_padded_dir(standardized_dir, fps):
    if fps == 30:
        return standardized_dir / "padded"
    return standardized_dir / f"padded_{fps}fps"


"""
probe_video
Read spatial and temporal metadata for one video with ffprobe.

INPUT:
    - video_path: Path -> video to inspect

OUTPUT:
    - video_info: VideoInfo -> measured video metadata
"""
def probe_video(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,profile,codec_tag_string,pix_fmt,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    stream = data["streams"][0]
    return VideoInfo(
        path=video_path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec_name=stream["codec_name"],
        profile=stream.get("profile", ""),
        codec_tag_string=stream.get("codec_tag_string", ""),
        pix_fmt=stream.get("pix_fmt", ""),
        fps=rate_to_float(stream["avg_frame_rate"]),
        frame_count=int(stream["nb_read_frames"]),
        duration=float(stream["duration"]),
    )


"""
check_video
Check one video against expected metadata values.

INPUT:
    - video_info: VideoInfo -> measured video metadata
    - cfg: Cfg -> expected values
    - target_frames: int -> expected frame count

OUTPUT:
    - errors: list[str] -> failed checks for this video
"""
def check_video(video_info, cfg, target_frames):
    errors = []
    expected_duration = target_frames / cfg.fps

    if video_info.width != cfg.width or video_info.height != cfg.height:
        errors.append(f"size={video_info.width}x{video_info.height}")

    if abs(video_info.fps - cfg.fps) > 1e-6:
        errors.append(f"fps={video_info.fps:g}")

    if video_info.frame_count != target_frames:
        errors.append(f"frames={video_info.frame_count}")

    if abs(video_info.duration - expected_duration) > cfg.duration_tolerance:
        errors.append(f"duration={video_info.duration:.6f}")

    return errors


def check_quicktime_codec(video_info, label):
    errors = []

    if label == "standardized":
        if video_info.codec_name != "h264":
            errors.append(f"codec={video_info.codec_name}")
        if video_info.profile in {"High 4:4:4 Predictive", "High 4:4:4 Intra"}:
            errors.append(f"profile={video_info.profile}")
        if video_info.pix_fmt != "yuv420p":
            errors.append(f"pix_fmt={video_info.pix_fmt}")
        if video_info.codec_tag_string != "avc1":
            errors.append(f"tag={video_info.codec_tag_string}")

    if label == "padded":
        if video_info.codec_name != "prores":
            errors.append(f"codec={video_info.codec_name}")
        if video_info.profile != "HQ":
            errors.append(f"profile={video_info.profile}")
        if video_info.pix_fmt != "yuv422p10le":
            errors.append(f"pix_fmt={video_info.pix_fmt}")

    return errors


"""
last_frame_hashes
Compute decoded frame hashes for a frame interval.

INPUT:
    - video_path: Path -> video to inspect
    - start_frame: int -> first frame to hash
    - end_frame: int -> last frame to hash, inclusive

OUTPUT:
    - hashes: list[str] -> decoded framemd5 hashes
"""
def last_frame_hashes(video_path, start_frame, end_frame):
    vf = f"select=between(n\\,{start_frame}\\,{end_frame}),format=rgb24"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "framemd5",
        "-",
    ]
    result = subprocess.check_output(cmd, text=True)
    hashes = []
    for line in result.splitlines():
        if not line or line.startswith("#"):
            continue
        hashes.append(line.split(",")[-1].strip())
    return hashes


def check_last_frame_hold(video_path, hold_start_frame, hold_frames):
    hashes = last_frame_hashes(video_path, hold_start_frame, hold_start_frame + hold_frames - 1)
    if len(hashes) != hold_frames:
        return f"hold_frames_decoded={len(hashes)}"
    if len(set(hashes)) != 1:
        return "last_frame_hold_changes"
    return None


def check_folder(folder, expected_names, cfg, target_frames, label):
    paths = list(iter_video_paths(folder))
    names = {path.name for path in paths}
    errors = []

    for name in sorted(expected_names - names):
        errors.append(f"{label}/{name}: missing output")

    for name in sorted(names - expected_names):
        errors.append(f"{label}/{name}: unexpected output")

    for video_path in paths:
        video_info = probe_video(video_path)
        video_errors = check_video(video_info, cfg, target_frames)
        video_errors.extend(check_quicktime_codec(video_info, label))
        if video_errors:
            errors.append(f"{label}/{video_path.name}: {', '.join(video_errors)}")

    return errors, len(paths)


def main():
    cfg = parse_args()
    cfg.input_dir = cfg.input_dir.expanduser().resolve()
    cfg.standardized_dir = (
        cfg.standardized_dir or default_standardized_dir(cfg.input_dir, cfg.fps)
    ).expanduser().resolve()
    cfg.padded_dir = (
        cfg.padded_dir or default_padded_dir(cfg.standardized_dir, cfg.fps)
    ).expanduser().resolve()
    target_frames = int(round(cfg.fps * cfg.duration)) + 1
    padded_frames = int(round(cfg.fps * cfg.padded_duration))
    hold_start_frame = target_frames - 1
    hold_frames = padded_frames - hold_start_frame

    require_tool("ffprobe")
    require_tool("ffmpeg")

    source_paths = list(iter_video_paths(cfg.input_dir))
    expected_names = {path.name for path in source_paths}
    padded_expected_names = {path.with_suffix(".mov").name for path in source_paths}

    errors = []
    standardized_errors, standardized_count = check_folder(
        cfg.standardized_dir,
        expected_names,
        cfg,
        target_frames,
        "standardized",
    )
    padded_errors, padded_count = check_folder(
        cfg.padded_dir,
        padded_expected_names,
        cfg,
        padded_frames,
        "padded",
    )
    errors.extend(standardized_errors)
    errors.extend(padded_errors)

    for video_path in iter_video_paths(cfg.padded_dir):
        hold_error = check_last_frame_hold(video_path, hold_start_frame, hold_frames)
        if hold_error:
            errors.append(f"padded/{video_path.name}: {hold_error}")

    if errors:
        print("FAILED")
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(
        f"PASSED: {standardized_count} standardized videos are "
        f"{cfg.width}x{cfg.height}, {cfg.fps} fps, {target_frames} frames; "
        f"{padded_count} padded videos are {padded_frames} frames with "
        f"{hold_frames / cfg.fps:.3f} s final-frame hold"
    )


if __name__ == "__main__":
    main()


# EOF
