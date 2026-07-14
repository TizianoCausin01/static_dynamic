import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cfg:
    input_dir: Path = Path(
        "/Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes/"
        "standardized_60fps/padded_60fps"
    )
    output_dir: Path | None = None
    fps: int = 60
    duration: float = 0.5
    source_start_seconds: float = 2.5
    overwrite: bool = True


"""
parse_args
Parse command line arguments into a config object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create static video clips from the final held 500 ms of padded videos."
    )
    parser.add_argument("--input_dir", type=Path, default=Cfg.input_dir)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=Cfg.fps)
    parser.add_argument("--duration", type=float, default=Cfg.duration)
    parser.add_argument("--source_start_seconds", type=float, default=Cfg.source_start_seconds)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    return Cfg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        duration=args.duration,
        source_start_seconds=args.source_start_seconds,
        overwrite=not args.no_overwrite,
    )


def require_tool(tool_name):
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"{tool_name} is required, but it was not found on PATH.")


def iter_padded_paths(input_dir):
    for path in sorted(input_dir.glob("*.mov")):
        if path.name.startswith(".") or path.is_dir():
            continue
        yield path


"""
write_static_video
Write a video made from one repeated frame sampled from the held final interval.

INPUT:
    - input_path: Path -> padded source video
    - output_path: Path -> static output video path
    - cfg: Cfg -> conversion parameters
    - output_frames: int -> exact output frame count
"""
def write_static_video(input_path, output_path, cfg, output_frames):
    overwrite_flag = "-y" if cfg.overwrite else "-n"
    source_frame = int(round(cfg.source_start_seconds * cfg.fps))
    vf = (
        f"trim=start_frame={source_frame}:end_frame={source_frame + 1},"
        f"loop=loop={output_frames - 1}:size=1:start=0,"
        f"setpts=N/({cfg.fps}*TB),"
        f"trim=start_frame=0:end_frame={output_frames},"
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
        str(output_frames),
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
    cfg.output_dir = (cfg.output_dir or cfg.input_dir / "images").expanduser().resolve()
    output_frames = int(round(cfg.fps * cfg.duration))

    require_tool("ffmpeg")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = list(iter_padded_paths(cfg.input_dir))
    if not input_paths:
        raise RuntimeError(f"No padded .mov videos found in {cfg.input_dir}")

    print(f"Writing {len(input_paths)} static videos to {cfg.output_dir}")
    print(f"Target: {cfg.fps} fps, {output_frames} frames, {cfg.duration:.3f} s")

    for input_path in input_paths:
        output_path = cfg.output_dir / input_path.name
        print(f"{input_path.name} -> {output_path.name}")
        write_static_video(input_path, output_path, cfg, output_frames)

    print("Done.")


if __name__ == "__main__":
    main()


# EOF
