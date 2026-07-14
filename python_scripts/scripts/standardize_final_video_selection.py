import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
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
    output_dir: Path | None = None
    padded_dir: Path | None = None
    fps: int = 30
    duration: float = 2.5
    padded_duration: float = 3.0
    crf: int = 8
    preset: str = "slow"
    overwrite: bool = True


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int


"""
parse_args
Parse command line arguments into a config object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize selected videos to a fixed fps, duration, and frame count."
    )
    parser.add_argument("--input_dir", type=Path, default=Cfg.input_dir)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--padded_dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=Cfg.fps)
    parser.add_argument("--duration", type=float, default=Cfg.duration)
    parser.add_argument("--padded_duration", type=float, default=Cfg.padded_duration)
    parser.add_argument("--crf", type=int, default=Cfg.crf)
    parser.add_argument("--preset", default=Cfg.preset)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    return Cfg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        padded_dir=args.padded_dir,
        fps=args.fps,
        duration=args.duration,
        padded_duration=args.padded_duration,
        crf=args.crf,
        preset=args.preset,
        overwrite=not args.no_overwrite,
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


def remove_stale_padded_mp4s(padded_dir):
    for path in padded_dir.glob("*.mp4"):
        path.unlink()


def default_output_dir(input_dir, fps):
    if fps == 30:
        return input_dir / "standardized"
    return input_dir / f"standardized_{fps}fps"


def default_padded_dir(output_dir, fps):
    if fps == 30:
        return output_dir / "padded"
    return output_dir / f"padded_{fps}fps"


"""
probe_video
Read spatial metadata for one video with ffprobe.

INPUT:
    - video_path: Path -> video to inspect

OUTPUT:
    - video_info: VideoInfo -> path and frame dimensions
"""
def probe_video(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
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
    )


"""
standardize_video
Write one constant-frame-rate video with exactly target_frames frames.

INPUT:
    - video_info: VideoInfo -> source video metadata
    - output_path: Path -> destination mp4 path
    - cfg: Cfg -> conversion parameters
    - target_frames: int -> exact output frame count
"""
def standardize_video(video_info, output_path, cfg, target_frames):
    overwrite_flag = "-y" if cfg.overwrite else "-n"
    vf = (
        f"fps={cfg.fps},"
        "tpad=stop_mode=clone:stop_duration=1,"
        f"trim=start_frame=0:end_frame={target_frames},"
        f"setpts=N/({cfg.fps}*TB),"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        str(video_info.path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-fps_mode",
        "cfr",
        "-r",
        str(cfg.fps),
        "-frames:v",
        str(target_frames),
        "-c:v",
        "libx264",
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-profile:v",
        "high",
        "-level:v",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


"""
pad_video
Pad one standardized video by freezing its last frame.

INPUT:
    - input_path: Path -> standardized source video
    - output_path: Path -> padded destination mp4 path
    - cfg: Cfg -> conversion parameters
    - padded_frames: int -> exact padded output frame count
"""
def pad_video(input_path, output_path, cfg, padded_frames):
    overwrite_flag = "-y" if cfg.overwrite else "-n"
    hold_start_frame = int(round(cfg.fps * cfg.duration))
    source_frames = hold_start_frame + 1
    hold_frames = padded_frames - hold_start_frame
    vf = (
        f"trim=start_frame=0:end_frame={source_frames},"
        f"loop=loop={hold_frames}:size=1:start={hold_start_frame},"
        f"setpts=N/({cfg.fps}*TB),"
        f"trim=start_frame=0:end_frame={padded_frames},"
        "format=yuv422p10le"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-fps_mode",
        "cfr",
        "-r",
        str(cfg.fps),
        "-frames:v",
        str(padded_frames),
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-pix_fmt",
        "yuv422p10le",
        "-vendor",
        "apl0",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    cfg = parse_args()
    cfg.input_dir = cfg.input_dir.expanduser().resolve()
    cfg.output_dir = (cfg.output_dir or default_output_dir(cfg.input_dir, cfg.fps)).expanduser().resolve()
    cfg.padded_dir = (cfg.padded_dir or default_padded_dir(cfg.output_dir, cfg.fps)).expanduser().resolve()
    target_frames = int(round(cfg.fps * cfg.duration)) + 1
    padded_frames = int(round(cfg.fps * cfg.padded_duration))

    require_tool("ffmpeg")
    require_tool("ffprobe")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.padded_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_padded_mp4s(cfg.padded_dir)

    video_paths = list(iter_video_paths(cfg.input_dir))
    if not video_paths:
        raise RuntimeError(f"No videos found in {cfg.input_dir}")

    print(f"Writing {len(video_paths)} standardized videos to {cfg.output_dir}")
    print(
        f"Standardized target: {cfg.fps} fps, {target_frames} frames, "
        f"including a frame at {cfg.duration:.3f} s"
    )
    print(f"Padded target: {cfg.fps} fps, {padded_frames} frames, {cfg.padded_duration:.3f} s")

    for video_path in video_paths:
        video_info = probe_video(video_path)
        output_path = cfg.output_dir / video_path.name
        padded_path = cfg.padded_dir / video_path.with_suffix(".mov").name
        print(f"{video_path.name} -> {output_path.name} ({video_info.width}x{video_info.height})")
        standardize_video(video_info, output_path, cfg, target_frames)
        pad_video(output_path, padded_path, cfg, padded_frames)

    print("Done.")


if __name__ == "__main__":
    main()


# EOF
