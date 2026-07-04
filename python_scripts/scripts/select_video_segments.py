import argparse
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration: float


def clean_filename(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")


def seconds_to_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    seconds_string = f"{seconds_remainder:06.3f}".rstrip("0").rstrip(".")
    return f"{hours:02d}:{minutes:02d}:{seconds_string}"


def seconds_to_filename_timestamp(seconds):
    return seconds_to_timestamp(seconds).replace(":", ".")


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not read FPS/frame count from {video_path}")

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps,
    )


def iter_video_paths(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def existing_segment_prefix(video_path):
    return f"{clean_filename(video_path.stem)}__"


def segment_exists_for(video_path, segments_dir):
    prefix = existing_segment_prefix(video_path)
    return any(path.is_file() and path.name.startswith(prefix) for path in segments_dir.glob("*"))


def build_output_path(video_path, segments_dir, start_seconds, duration_seconds, end_seconds):
    source_name = clean_filename(video_path.stem)
    start_label = seconds_to_filename_timestamp(start_seconds)
    end_label = seconds_to_filename_timestamp(end_seconds)
    duration_label = seconds_to_filename_timestamp(duration_seconds)
    output_path = (
        segments_dir
        / f"{source_name}__start_{start_label}__end_{end_label}__dur_{duration_label}.mp4"
    )

    if not output_path.exists():
        return output_path

    for counter in range(2, 1000):
        candidate = output_path.with_name(f"{output_path.stem}__v{counter}{output_path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find a free output filename for {video_path.name}")


def write_segment(video_path, output_path, start_seconds, duration_seconds):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write segments, but it was not found on PATH.")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def move_to_discarded(video_path, discarded_dir):
    discarded_dir.mkdir(parents=True, exist_ok=True)
    destination = discarded_dir / video_path.name
    if not destination.exists():
        shutil.move(str(video_path), str(destination))
        return destination

    for counter in range(2, 1000):
        candidate = discarded_dir / f"{video_path.stem}__discarded_{counter}{video_path.suffix}"
        if not candidate.exists():
            shutil.move(str(video_path), str(candidate))
            return candidate

    raise RuntimeError(f"Could not find a free discarded filename for {video_path.name}")


class EndingFrameSelector:
    def __init__(self, video_info, segment_duration, display_width):
        self.video_info = video_info
        self.segment_duration = segment_duration
        self.display_width = display_width
        self.window_name = f"Select ending frame - {video_info.path.name}"
        self.result = None
        self.min_frame_index = min(
            video_info.frame_count - 1,
            max(0, math.ceil(segment_duration * video_info.fps) - 1),
        )
        self.current_frame_index = self.min_frame_index
        self.pending_frame_index = self.current_frame_index
        self.setting_trackbar = False
        self.frame_cache = {}

        self.cap = cv2.VideoCapture(str(video_info.path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open {video_info.path}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "ending frame",
            self.window_name,
            self.current_frame_index,
            self.video_info.frame_count - 1,
            self._on_trackbar,
        )
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self._show_frame(self.current_frame_index)

        while self.result is None:
            if self.pending_frame_index != self.current_frame_index:
                self._show_frame(self.pending_frame_index)

            key = cv2.waitKeyEx(15)
            if key == -1:
                continue

            self._handle_key(key)

        self.cap.release()
        cv2.destroyWindow(self.window_name)
        return self.result, self.current_frame_index

    def _on_trackbar(self, frame_index):
        if not self.setting_trackbar:
            self.pending_frame_index = frame_index

    def _on_mouse(self, event, _x, _y, flags, _param):
        if event != cv2.EVENT_MOUSEWHEEL:
            return

        delta = cv2.getMouseWheelDelta(flags)
        step = 10 if flags & cv2.EVENT_FLAG_SHIFTKEY else 1
        self._queue_frame(self.current_frame_index + (step if delta > 0 else -step))

    def _handle_key(self, key):
        ascii_key = key & 0xFF

        if ascii_key in (ord("s"), 13, 10, 32):
            review_decision = self._review_selected_frames()
            if review_decision == "accept":
                self.result = "save"
            elif review_decision == "quit":
                self.result = "quit"
            else:
                self._show_frame(self.current_frame_index)
        elif ascii_key == ord("n"):
            self.result = "skip"
        elif ascii_key == ord("x"):
            self.result = "discard"
        elif ascii_key == ord("p"):
            subprocess.run(["open", str(self.video_info.path)], check=False)
        elif ascii_key == ord("q") or key == 27:
            self.result = "quit"
        elif key in (81, 2424832) or ascii_key in (ord("h"), ord(",")):
            self._queue_frame(self.current_frame_index - 1)
        elif key in (83, 2555904) or ascii_key in (ord("l"), ord(".")):
            self._queue_frame(self.current_frame_index + 1)
        elif key in (82, 2490368) or ascii_key == ord("k"):
            self._queue_frame(self.current_frame_index + round(self.video_info.fps))
        elif key in (84, 2621440) or ascii_key == ord("j"):
            self._queue_frame(self.current_frame_index - round(self.video_info.fps))
        elif key in (2162688, 85):
            self._queue_frame(self.current_frame_index + round(5 * self.video_info.fps))
        elif key in (2228224, 86):
            self._queue_frame(self.current_frame_index - round(5 * self.video_info.fps))

    def _queue_frame(self, frame_index):
        self.pending_frame_index = min(
            max(self.min_frame_index, int(frame_index)),
            self.video_info.frame_count - 1,
        )

    def _show_frame(self, frame_index):
        frame_index = min(max(self.min_frame_index, int(frame_index)), self.video_info.frame_count - 1)
        frame = self._read_frame(frame_index)

        if frame is None:
            return

        self.current_frame_index = frame_index
        self.pending_frame_index = frame_index
        self._sync_trackbar(frame_index)
        cv2.imshow(self.window_name, self._render_frame())

    def _review_selected_frames(self):
        review_window = f"Review selected frames - {self.video_info.path.name}"
        start_frame_index = self.selected_start_frame_index
        end_frame_index = self.current_frame_index

        selected_frames = self._load_frame_range(start_frame_index, end_frame_index)
        if not selected_frames:
            return "reject"

        cv2.namedWindow(review_window, cv2.WINDOW_NORMAL)
        while True:
            decision = self._play_frames_at_video_fps(review_window, selected_frames)
            if decision is not None and decision != "replay":
                cv2.destroyWindow(review_window)
                return decision

            final_frame_index, final_frame = selected_frames[-1]
            while True:
                cv2.imshow(review_window, self._render_review_frame(final_frame, final_frame_index, done=True))
                decision = self._review_key_to_decision(cv2.waitKeyEx(30))
                if decision == "replay":
                    break
                if decision is not None:
                    cv2.destroyWindow(review_window)
                    return decision

    def _load_frame_range(self, start_frame_index, end_frame_index):
        cap = cv2.VideoCapture(str(self.video_info.path))
        if not cap.isOpened():
            return []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        frames = []
        for frame_index in range(start_frame_index, end_frame_index + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append((frame_index, frame))

        cap.release()
        return frames

    def _play_frames_at_video_fps(self, window_name, frames):
        frame_period = 1 / self.video_info.fps
        playback_start = time.perf_counter()

        for offset, (frame_index, frame) in enumerate(frames):
            cv2.imshow(window_name, self._render_review_frame(frame, frame_index))

            next_frame_time = playback_start + (offset + 1) * frame_period
            while True:
                remaining_ms = int((next_frame_time - time.perf_counter()) * 1000)
                if remaining_ms <= 0:
                    break

                key = cv2.waitKeyEx(min(remaining_ms, 10))
                decision = self._review_key_to_decision(key)
                if decision is not None:
                    return decision

        return None

    def _review_key_to_decision(self, key):
        if key == -1:
            return None

        ascii_key = key & 0xFF
        if ascii_key in (ord("a"), ord("s"), 13, 10):
            return "accept"
        if ascii_key in (ord("r"), ord("n")) or key == 27:
            return "reject"
        if ascii_key in (ord(" "), ord("p")):
            return "replay"
        if ascii_key == ord("q"):
            return "quit"

        return None

    def _render_review_frame(self, frame, frame_index, done=False):
        frame = self._resize_for_display(frame, self.display_width)
        height, width = frame.shape[:2]
        top_bar_height = 82
        bottom_bar_height = 42
        rendered = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        rendered[top_bar_height:top_bar_height + height, :, :] = frame

        status = "review done" if done else "review playing"
        lines = [
            f"{status}: {self.video_info.path.name}",
            (
                f"frame {frame_index + 1}/{self.video_info.frame_count} "
                f"at {seconds_to_timestamp((frame_index + 1) / self.video_info.fps)}"
            ),
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                rendered,
                line,
                (14, 24 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        controls = "a/Enter/s: accept and save | r/n/Esc: reject and keep selecting | Space/p: replay | q: quit"
        cv2.putText(
            rendered,
            controls,
            (14, top_bar_height + height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return rendered

    def _sync_trackbar(self, frame_index):
        self.setting_trackbar = True
        cv2.setTrackbarPos("ending frame", self.window_name, frame_index)
        self.setting_trackbar = False

    def _read_frame(self, frame_index):
        frame_index = min(max(0, int(frame_index)), self.video_info.frame_count - 1)
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            return cached_frame.copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        if len(self.frame_cache) > 12:
            self.frame_cache.clear()
        self.frame_cache[frame_index] = frame.copy()
        return frame

    def _render_frame(self):
        end_seconds = self.selected_end_seconds
        start_seconds = self.selected_start_seconds
        start_frame_index = self.selected_start_frame_index
        start_frame = self._read_frame(start_frame_index)
        end_frame = self._read_frame(self.current_frame_index)
        if start_frame is None or end_frame is None:
            return np.zeros((480, 960, 3), dtype=np.uint8)

        video_frame = self._build_side_by_side_frame(start_frame, end_frame)
        height, width = video_frame.shape[:2]
        top_bar_height = 124
        bottom_bar_height = 42
        frame = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        frame[top_bar_height:top_bar_height + height, :, :] = video_frame

        lines = [
            f"{self.video_info.path.name}",
            (
                f"left first frame {start_frame_index + 1}/{self.video_info.frame_count} "
                f"at {seconds_to_timestamp(start_seconds)}"
            ),
            (
                f"right last frame {self.current_frame_index + 1}/{self.video_info.frame_count} "
                f"at {seconds_to_timestamp(end_seconds)}"
            ),
            (
                f"clip: {seconds_to_timestamp(start_seconds)} -> "
                f"{seconds_to_timestamp(end_seconds)} "
                f"({seconds_to_timestamp(self.segment_duration)})"
            ),
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (14, 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        controls = "left/right or mouse wheel: 1 frame | up/down: 1 sec | PgUp/PgDn: 5 sec | p: play full | s/Enter/Space: review frames | n: skip | x: discard | q: quit"
        cv2.putText(
            frame,
            controls,
            (14, top_bar_height + height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _build_side_by_side_frame(self, start_frame, end_frame):
        panel_width = self.display_width // 2 if self.display_width > 0 else 0
        start_frame = self._resize_for_display(start_frame, panel_width)
        end_frame = self._resize_for_display(end_frame, panel_width)
        target_height = max(start_frame.shape[0], end_frame.shape[0])
        start_frame = self._pad_to_height(start_frame, target_height)
        end_frame = self._pad_to_height(end_frame, target_height)
        divider = np.full((target_height, 6, 3), 35, dtype=np.uint8)
        return np.hstack([start_frame, divider, end_frame])

    def _pad_to_height(self, frame, target_height):
        height, width = frame.shape[:2]
        if height == target_height:
            return frame

        top = (target_height - height) // 2
        bottom = target_height - height - top
        return cv2.copyMakeBorder(
            frame,
            top,
            bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _resize_for_display(self, frame, max_width):
        if max_width <= 0 or frame.shape[1] <= max_width:
            return frame.copy()

        scale = max_width / frame.shape[1]
        new_size = (max_width, int(frame.shape[0] * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    @property
    def selected_end_seconds(self):
        return min(
            self.video_info.duration,
            (self.current_frame_index + 1) / self.video_info.fps,
        )

    @property
    def selected_start_seconds(self):
        return max(0.0, self.selected_end_seconds - self.segment_duration)

    @property
    def selected_start_frame_index(self):
        return min(
            self.current_frame_index,
            max(0, int(round(self.selected_start_seconds * self.video_info.fps))),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively select an ending frame from videos in possible_vids, "
            "then save the previous fixed-duration segment."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/Users/tizianocausin/sd_local/possible_vids"),
        help="Folder containing downloaded videos.",
    )
    parser.add_argument(
        "--segments_dir",
        type=Path,
        default=None,
        help="Folder where selected segments are saved. Defaults to INPUT_DIR/segments.",
    )
    parser.add_argument(
        "--discarded_dir",
        type=Path,
        default=None,
        help="Folder where discarded videos are moved. Defaults to INPUT_DIR/discarded_vids.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.5,
        help="Number of seconds to keep before the selected ending frame.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=1280,
        help="Maximum total display width for the review window. Use 0 for original size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Show videos even if a segment with this source video prefix already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    segments_dir = (
        args.segments_dir.expanduser().resolve()
        if args.segments_dir is not None
        else input_dir / "segments"
    )
    discarded_dir = (
        args.discarded_dir.expanduser().resolve()
        if args.discarded_dir is not None
        else input_dir / "discarded_vids"
    )

    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    segments_dir.mkdir(parents=True, exist_ok=True)
    discarded_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_video_paths(input_dir))
    pending_videos = [
        video for video in videos
        if args.force or not segment_exists_for(video, segments_dir)
    ]

    print(f"Found {len(videos)} videos in {input_dir}")
    print(f"{len(pending_videos)} videos need review")
    print(
        "Controls: left/right or mouse wheel = 1 frame, up/down = 1 second, "
        "PgUp/PgDn = 5 seconds, p = play in default media player, "
        "s/Enter/Space = review selected frames, n = skip, x = discard, q = quit"
    )

    for index, video_path in enumerate(pending_videos, start=1):
        print(f"\n[{index}/{len(pending_videos)}] {video_path.name}")

        try:
            video_info = get_video_info(video_path)
        except ValueError as exc:
            print(f"Skipping unreadable video: {exc}")
            continue

        if video_info.duration < args.duration:
            print(
                "Skipping because video is shorter than requested segment duration: "
                f"{seconds_to_timestamp(video_info.duration)} < {seconds_to_timestamp(args.duration)}"
            )
            continue

        selector = EndingFrameSelector(
            video_info=video_info,
            segment_duration=args.duration,
            display_width=args.display_width,
        )
        decision, ending_frame_index = selector.run()
        end_seconds = min(video_info.duration, (ending_frame_index + 1) / video_info.fps)
        start_seconds = max(0.0, end_seconds - args.duration)

        if decision == "quit":
            print("Stopped by user.")
            break
        if decision == "discard":
            destination = move_to_discarded(video_path, discarded_dir)
            print(f"Discarded to {destination}")
            continue
        if decision != "save":
            print("Skipped.")
            continue

        output_path = build_output_path(
            video_path=video_path,
            segments_dir=segments_dir,
            start_seconds=start_seconds,
            duration_seconds=args.duration,
            end_seconds=end_seconds,
        )
        print(
            "Saving "
            f"{output_path.name} from {seconds_to_timestamp(start_seconds)} "
            f"to {seconds_to_timestamp(end_seconds)}"
        )
        write_segment(video_path, output_path, start_seconds, args.duration)

    print("\nDone.")


if __name__ == "__main__":
    main()
