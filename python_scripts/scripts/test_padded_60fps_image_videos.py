import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass
class Cfg:
    input_dir: Path = Path(
        "/Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes/"
        "standardized_60fps/padded_60fps/images"
    )
    expected_count: int = 100
    fps: int = 60
    frames: int = 30
    duration: float = 0.5
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
        description="Check static image-videos have fixed metadata and identical decoded frames."
    )
    parser.add_argument("--input_dir", type=Path, default=Cfg.input_dir)
    parser.add_argument("--expected_count", type=int, default=Cfg.expected_count)
    parser.add_argument("--fps", type=int, default=Cfg.fps)
    parser.add_argument("--frames", type=int, default=Cfg.frames)
    parser.add_argument("--duration", type=float, default=Cfg.duration)
    parser.add_argument("--width", type=int, default=Cfg.width)
    parser.add_argument("--height", type=int, default=Cfg.height)
    parser.add_argument("--duration_tolerance", type=float, default=Cfg.duration_tolerance)
    args = parser.parse_args()

    return Cfg(
        input_dir=args.input_dir,
        expected_count=args.expected_count,
        fps=args.fps,
        frames=args.frames,
        duration=args.duration,
        width=args.width,
        height=args.height,
        duration_tolerance=args.duration_tolerance,
    )


def require_tool(tool_name):
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"{tool_name} is required, but it was not found on PATH.")


def rate_to_float(rate):
    return float(Fraction(rate))


def iter_video_paths(input_dir):
    for path in sorted(input_dir.glob("*.mov")):
        if path.name.startswith(".") or path.is_dir():
            continue
        yield path


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
        "stream=width,height,codec_name,profile,pix_fmt,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    stream = json.loads(subprocess.check_output(cmd, text=True))["streams"][0]
    return VideoInfo(
        path=video_path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec_name=stream["codec_name"],
        profile=stream.get("profile", ""),
        pix_fmt=stream.get("pix_fmt", ""),
        fps=rate_to_float(stream["avg_frame_rate"]),
        frame_count=int(stream["nb_read_frames"]),
        duration=float(stream["duration"]),
    )


def decoded_frame_hashes(video_path):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "format=rgb24",
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


def check_video(video_info, cfg):
    errors = []

    if video_info.width != cfg.width or video_info.height != cfg.height:
        errors.append(f"size={video_info.width}x{video_info.height}")

    if abs(video_info.fps - cfg.fps) > 1e-6:
        errors.append(f"fps={video_info.fps:g}")

    if video_info.frame_count != cfg.frames:
        errors.append(f"frames={video_info.frame_count}")

    if abs(video_info.duration - cfg.duration) > cfg.duration_tolerance:
        errors.append(f"duration={video_info.duration:.6f}")

    if video_info.codec_name != "prores" or video_info.profile != "HQ":
        errors.append(f"codec={video_info.codec_name}/{video_info.profile}")

    if video_info.pix_fmt != "yuv422p10le":
        errors.append(f"pix_fmt={video_info.pix_fmt}")

    hashes = decoded_frame_hashes(video_info.path)
    if len(hashes) != cfg.frames:
        errors.append(f"decoded_frames={len(hashes)}")
    elif len(set(hashes)) != 1:
        errors.append("frames_change")

    return errors


def main():
    cfg = parse_args()
    cfg.input_dir = cfg.input_dir.expanduser().resolve()

    require_tool("ffprobe")
    require_tool("ffmpeg")

    paths = list(iter_video_paths(cfg.input_dir))
    errors = []

    if len(paths) != cfg.expected_count:
        errors.append(f"count={len(paths)}")

    for path in paths:
        video_info = probe_video(path)
        video_errors = check_video(video_info, cfg)
        if video_errors:
            errors.append(f"{path.name}: {', '.join(video_errors)}")

    if errors:
        print("FAILED")
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(
        f"PASSED: {len(paths)} static videos are {cfg.width}x{cfg.height}, "
        f"{cfg.fps} fps, {cfg.frames} frames, {cfg.duration:.3f} s, "
        "ProRes HQ, and frame-constant"
    )


if __name__ == "__main__":
    main()


# EOF
