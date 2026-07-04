import os, yaml, sys
import subprocess
import argparse
from pathlib import Path

ENV = os.getenv("MY_ENV", "dev")
REPO_ROOT = Path(__file__).resolve().parents[2]
with open(REPO_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from useful_stuff.general_utils import print_wise

# e.g. to call it:
# run_download_clip.py --video_url "https://www.youtube.com/watch?v=XXXXXXXXXXX" --video_filename cats --starting_point 54:45.93 01:02:10 01:15:00 --duration 3


parser = argparse.ArgumentParser(
    description="Download one or more clips from a YouTube video without downloading the full video."
)
parser.add_argument("--video_url", required=True, help="YouTube URL.")
parser.add_argument("--video_filename", default="clip", help="Filename prefix used for saved clips.")
parser.add_argument("--starting_point", nargs="+", required=True, help="One or more timestamps (e.g. 54:45.93 01:15:00).")
parser.add_argument("--duration", required=True, type=float, help="Duration of each clip in seconds.")

cfg = parser.parse_args()
output_dir = Path(paths["data_path"]) / "possible_vids"


def timestamp_to_seconds(timestamp):
    fields = timestamp.split(":")
    if len(fields) > 3:
        raise ValueError(f"Invalid timestamp: {timestamp}")

    seconds = 0
    for power, field in enumerate(reversed(fields)):
        seconds += float(field) * 60**power
    return seconds


def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    seconds_string = f"{seconds:06.3f}".rstrip("0").rstrip(".")
    return f"{hours:02d}:{minutes:02d}:{seconds_string}"


def seconds_to_filename_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    seconds_string = f"{seconds:06.3f}".rstrip("0").rstrip(".")
    return f"{hours:02d}.{minutes:02d}.{seconds_string}"


def clean_url(video_url):
    return (
        video_url
        .replace("\\?", "?")
        .replace("\\=", "=")
        .replace("\\&", "&")
    )


def clean_filename(filename):
    filename = Path(filename).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in filename)


def run_command(cmd):
    print_wise(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_link_entry(video_name, video_url):
    links_file = output_dir / "links.txt"
    entry = f"{video_name}: {clean_url(video_url)}"

    existing_lines = []
    if links_file.exists():
        existing_lines = links_file.read_text().splitlines()

    filtered_lines = [
        line for line in existing_lines
        if not line.startswith(f"{video_name}: ")
    ]
    filtered_lines.append(entry)

    links_file.write_text("\n".join(filtered_lines) + "\n")
    print_wise(f"Recorded source link in {links_file}")


def build_yt_dlp_command(video_url, section, output_prefix, extra_options=None):
    extra_options = extra_options or []
    return [
        "yt-dlp",
        "--force-ipv4",
        "--socket-timeout",
        "30",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--no-continue",
        "--force-overwrites",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        *extra_options,
        "--merge-output-format",
        "mp4",
        "-o",
        f"{output_prefix}.%(ext)s",
        clean_url(video_url),
    ]


def download_clip(video_url, start_time, duration, output_prefix):
    end_time = seconds_to_timestamp(timestamp_to_seconds(start_time) + duration)
    section = f"*{start_time}-{end_time}"
    attempts = [
        [
            "-f",
            "bv*+ba/b",
        ],
        [
            "--extractor-args",
            "youtube:player_client=web",
            "--add-headers",
            "Referer:https://www.youtube.com/",
            "--add-headers",
            "User-Agent:Mozilla/5.0",
            "-f",
            "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/b",
        ],
        [
            "--extractor-args",
            "youtube:player_client=default",
            "--add-headers",
            "Referer:https://www.youtube.com/",
            "--add-headers",
            "User-Agent:Mozilla/5.0",
            "-f",
            "b[ext=mp4]/bv*[height<=1080]+ba/b",
        ],
    ]

    last_error = None
    for attempt_number, extra_options in enumerate(attempts, start=1):
        if attempt_number > 1:
            print_wise(f"Retrying clip download with fallback strategy {attempt_number}")

        cmd = build_yt_dlp_command(video_url, section, output_prefix, extra_options)
        try:
            run_command(cmd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print_wise(f"Download attempt {attempt_number} failed with exit code {exc.returncode}")

    raise last_error


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    print_wise(f"Saving clips in {output_dir}")

    video_filename = clean_filename(cfg.video_filename)
    starting_points = sorted(cfg.starting_point, key=timestamp_to_seconds)

    for start in starting_points:
        start_seconds = timestamp_to_seconds(start)
        outfile = output_dir / f"{video_filename}_{seconds_to_filename_timestamp(start_seconds)}"
        print_wise(f"Downloading clip starting at {start}")
        download_clip(
            cfg.video_url,
            start,
            cfg.duration,
            str(outfile),
        )

    write_link_entry(video_filename, cfg.video_url)
    print_wise("Done.")


if __name__ == "__main__":
    main()
