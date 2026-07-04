import argparse
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
    width: int
    height: int


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not read metadata from {video_path}")

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps,
        width=width,
        height=height,
    )


def seconds_to_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    seconds_string = f"{seconds_remainder:06.3f}".rstrip("0").rstrip(".")
    return f"{hours:02d}:{minutes:02d}:{seconds_string}"


def iter_video_paths(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def output_path_for(video_path, output_dir, source_square_size, output_size):
    return output_dir / f"{video_path.stem}__crop_{source_square_size}px_to_{output_size}x{output_size}.mp4"


def clamp_axis(center, frame_size, crop_size):
    if frame_size <= crop_size:
        return frame_size / 2

    half_crop = crop_size / 2
    return min(max(float(center), half_crop), frame_size - half_crop)


def clamp_center(center, video_info, crop_width, crop_height):
    x, y = center
    return (
        clamp_axis(x, video_info.width, crop_width),
        clamp_axis(y, video_info.height, crop_height),
    )


def center_for_video(video_info):
    return (video_info.width / 2, video_info.height / 2)


def crop_frame(frame, center, crop_width, crop_height):
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = center
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    right = left + crop_width
    bottom = top + crop_height

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(frame_width, right)
    source_bottom = min(frame_height, bottom)

    destination_left = source_left - left
    destination_top = source_top - top

    cropped = np.zeros((crop_height, crop_width, 3), dtype=frame.dtype)
    cropped[
        destination_top:destination_top + (source_bottom - source_top),
        destination_left:destination_left + (source_right - source_left),
    ] = frame[source_top:source_bottom, source_left:source_right]
    return cropped


class CropPathSelector:
    def __init__(self, video_info, source_square_size, output_size, display_width):
        self.video_info = video_info
        self.source_square_size = source_square_size
        self.output_size = output_size
        self.display_width = display_width
        self.window_name = f"Select crop path - {video_info.path.name}"
        self.result = None
        self.current_frame_index = 0
        self.pending_frame_index = 0
        self.setting_trackbar = False
        self.frame_cache = {}
        self.last_render_scale = 1.0
        self.last_render_top_bar = 0
        self.last_render_video_width = video_info.width
        self.last_render_video_height = video_info.height

        center = center_for_video(video_info)
        self.keyframes = {
            0: center,
            video_info.frame_count - 1: center,
        }

        self.cap = cv2.VideoCapture(str(video_info.path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open {video_info.path}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "frame",
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
        return self.result, self.keyframes.copy()

    def _on_trackbar(self, frame_index):
        if not self.setting_trackbar:
            self.pending_frame_index = frame_index

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            step = 10 if flags & cv2.EVENT_FLAG_SHIFTKEY else 1
            self._queue_frame(self.current_frame_index + (step if delta > 0 else -step))
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        video_y = y - self.last_render_top_bar
        if x < 0 or video_y < 0:
            return
        if x >= self.last_render_video_width or video_y >= self.last_render_video_height:
            return

        original_x = x / self.last_render_scale
        original_y = video_y / self.last_render_scale
        self._set_keyframe(self.current_frame_index, (original_x, original_y))

    def _handle_key(self, key):
        ascii_key = key & 0xFF

        if ascii_key == ord("s"):
            review_decision = self._review_crop_path()
            if review_decision == "accept":
                self.result = "save"
            elif review_decision == "quit":
                self.result = "quit"
            else:
                self._show_frame(self.current_frame_index)
        elif ascii_key == ord("c"):
            previous_keyframes = self.keyframes.copy()
            center = center_for_video(self.video_info)
            self.keyframes = {
                0: center,
                self.video_info.frame_count - 1: center,
            }
            review_decision = self._review_crop_path()
            if review_decision == "accept":
                self.result = "save"
            elif review_decision == "quit":
                self.result = "quit"
            else:
                self.keyframes = previous_keyframes
                self._show_frame(self.current_frame_index)
        elif ascii_key in (ord("="), ord("+")):
            self._adjust_square_size(25)
        elif ascii_key in (ord("-"), ord("_")):
            self._adjust_square_size(-25)
        elif ascii_key == ord("]"):
            self._adjust_square_size(5)
        elif ascii_key == ord("["):
            self._adjust_square_size(-5)
        elif ascii_key == ord("i"):
            self._set_keyframe(self.current_frame_index, self.current_center)
        elif ascii_key in (ord("d"), 8):
            self._delete_current_keyframe()
        elif ascii_key == ord("n"):
            self.result = "skip"
        elif ascii_key == ord("p"):
            subprocess.run(["open", str(self.video_info.path)], check=False)
        elif ascii_key == ord("q") or key == 27:
            self.result = "quit"
        elif ascii_key == ord("f"):
            self._queue_frame(0)
        elif ascii_key == ord("g"):
            self._queue_frame(self.video_info.frame_count - 1)
        elif key in (81, 2424832) or ascii_key in (ord("h"), ord(",")):
            self._queue_frame(self.current_frame_index - 1)
        elif key in (83, 2555904) or ascii_key in (ord("l"), ord(".")):
            self._queue_frame(self.current_frame_index + 1)
        elif key in (82, 2490368) or ascii_key == ord("k"):
            self._queue_frame(self.current_frame_index + round(self.video_info.fps))
        elif key in (84, 2621440) or ascii_key == ord("j"):
            self._queue_frame(self.current_frame_index - round(self.video_info.fps))

    def _queue_frame(self, frame_index):
        self.pending_frame_index = min(max(0, int(frame_index)), self.video_info.frame_count - 1)

    def _show_frame(self, frame_index):
        frame_index = min(max(0, int(frame_index)), self.video_info.frame_count - 1)
        frame = self._read_frame(frame_index)
        if frame is None:
            return

        self.current_frame_index = frame_index
        self.pending_frame_index = frame_index
        self._sync_trackbar(frame_index)
        cv2.imshow(self.window_name, self._render_selection_frame(frame))

    def _sync_trackbar(self, frame_index):
        self.setting_trackbar = True
        cv2.setTrackbarPos("frame", self.window_name, frame_index)
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

        if len(self.frame_cache) > 16:
            self.frame_cache.clear()
        self.frame_cache[frame_index] = frame.copy()
        return frame

    def _set_keyframe(self, frame_index, center):
        self.keyframes[int(frame_index)] = clamp_center(
            center,
            self.video_info,
            self.source_square_size,
            self.source_square_size,
        )
        self._show_frame(self.current_frame_index)

    def _adjust_square_size(self, delta):
        max_square_size = max(1, min(self.video_info.width, self.video_info.height))
        self.source_square_size = min(
            max(1, self.source_square_size + delta),
            max_square_size,
        )
        self.keyframes = {
            frame_index: clamp_center(
                center,
                self.video_info,
                self.source_square_size,
                self.source_square_size,
            )
            for frame_index, center in self.keyframes.items()
        }
        self._show_frame(self.current_frame_index)

    def _delete_current_keyframe(self):
        if self.current_frame_index in (0, self.video_info.frame_count - 1):
            return
        self.keyframes.pop(self.current_frame_index, None)
        self._show_frame(self.current_frame_index)

    @property
    def current_center(self):
        return interpolate_center(self.keyframes, self.current_frame_index)

    def _render_selection_frame(self, frame):
        display_frame, scale = resize_for_display(frame, self.display_width)
        self.last_render_scale = scale
        self.last_render_top_bar = 126
        self.last_render_video_width = display_frame.shape[1]
        self.last_render_video_height = display_frame.shape[0]

        height, width = display_frame.shape[:2]
        top_bar_height = self.last_render_top_bar
        bottom_bar_height = 58
        rendered = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        rendered[top_bar_height:top_bar_height + height, :, :] = display_frame

        center_x, center_y = self.current_center
        left = int(round((center_x - self.source_square_size / 2) * scale))
        top = int(round((center_y - self.source_square_size / 2) * scale)) + top_bar_height
        right = int(round((center_x + self.source_square_size / 2) * scale))
        bottom = int(round((center_y + self.source_square_size / 2) * scale)) + top_bar_height
        cv2.rectangle(rendered, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.drawMarker(
            rendered,
            (int(round(center_x * scale)), int(round(center_y * scale)) + top_bar_height),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

        keyframe_status = "keyframe" if self.current_frame_index in self.keyframes else "interpolated"
        lines = [
            f"{self.video_info.path.name}",
            (
                f"frame {self.current_frame_index + 1}/{self.video_info.frame_count} "
                f"at {seconds_to_timestamp((self.current_frame_index + 1) / self.video_info.fps)}"
            ),
            (
                f"source square {self.source_square_size}x{self.source_square_size} -> "
                f"{self.output_size}x{self.output_size}, center ({center_x:.1f}, {center_y:.1f}), "
                f"{keyframe_status}, {len(self.keyframes)} keyframes"
            ),
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                rendered,
                line,
                (14, 24 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        controls = (
            "click: set crop center | i: add keyframe | d: delete keyframe | "
            "+/-: square size 25px | [/]: 5px | f/g: first/last | arrows/wheel: move | "
            "c: center crop | s: review | n: skip | p: play | q: quit"
        )
        cv2.putText(
            rendered,
            controls,
            (14, top_bar_height + height + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return rendered

    def _review_crop_path(self):
        review_window = f"Review crop - {self.video_info.path.name}"
        frames = self._build_cropped_frames()
        if not frames:
            return "reject"

        cv2.namedWindow(review_window, cv2.WINDOW_NORMAL)
        while True:
            decision = play_frames_at_video_fps(
                review_window,
                frames,
                self.video_info.fps,
                self._render_review_frame,
                review_key_to_decision,
            )
            if decision is not None and decision != "replay":
                cv2.destroyWindow(review_window)
                return decision

            final_index, final_frame = frames[-1]
            while True:
                cv2.imshow(review_window, self._render_review_frame(final_frame, final_index, done=True))
                decision = review_key_to_decision(cv2.waitKeyEx(30))
                if decision == "replay":
                    break
                if decision is not None:
                    cv2.destroyWindow(review_window)
                    return decision

    def _build_cropped_frames(self):
        cap = cv2.VideoCapture(str(self.video_info.path))
        if not cap.isOpened():
            return []

        frames = []
        for frame_index in range(self.video_info.frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            center = interpolate_center(self.keyframes, frame_index)
            cropped = crop_frame(frame, center, self.source_square_size, self.source_square_size)
            frames.append((frame_index, resize_frame(cropped, self.output_size, self.output_size)))

        cap.release()
        return frames

    def _render_review_frame(self, frame, frame_index, done=False):
        display_frame, _scale = resize_for_display(frame, self.display_width)
        height, width = display_frame.shape[:2]
        top_bar_height = 82
        bottom_bar_height = 44
        rendered = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        rendered[top_bar_height:top_bar_height + height, :, :] = display_frame

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

        controls = "a/Enter/s: accept and save | r/n/Esc: reject and edit | Space/p: replay | q: quit"
        cv2.putText(
            rendered,
            controls,
            (14, top_bar_height + height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return rendered


def resize_frame(frame, width, height):
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def resize_for_display(frame, max_width):
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame.copy(), 1.0

    scale = max_width / frame.shape[1]
    new_size = (max_width, int(frame.shape[0] * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def interpolate_center(keyframes, frame_index):
    sorted_items = sorted(keyframes.items())
    if frame_index <= sorted_items[0][0]:
        return sorted_items[0][1]
    if frame_index >= sorted_items[-1][0]:
        return sorted_items[-1][1]

    for (left_frame, left_center), (right_frame, right_center) in zip(sorted_items, sorted_items[1:]):
        if left_frame <= frame_index <= right_frame:
            if left_frame == right_frame:
                return left_center
            fraction = (frame_index - left_frame) / (right_frame - left_frame)
            return (
                left_center[0] + fraction * (right_center[0] - left_center[0]),
                left_center[1] + fraction * (right_center[1] - left_center[1]),
            )

    return sorted_items[-1][1]


def play_frames_at_video_fps(window_name, frames, fps, render_frame, key_to_decision):
    frame_period = 1 / fps
    playback_start = time.perf_counter()

    for offset, (frame_index, frame) in enumerate(frames):
        cv2.imshow(window_name, render_frame(frame, frame_index))
        next_frame_time = playback_start + (offset + 1) * frame_period
        while True:
            remaining_ms = int((next_frame_time - time.perf_counter()) * 1000)
            if remaining_ms <= 0:
                break
            decision = key_to_decision(cv2.waitKeyEx(min(remaining_ms, 10)))
            if decision is not None:
                return decision

    return None


def review_key_to_decision(key):
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


def write_cropped_video(video_path, output_path, video_info, keyframes, source_square_size, output_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_info.fps, (output_size, output_size))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for frame_index in range(video_info.frame_count):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        center = interpolate_center(keyframes, frame_index)
        cropped = crop_frame(frame, center, source_square_size, source_square_size)
        writer.write(resize_frame(cropped, output_size, output_size))

    writer.release()
    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively crop segment videos to a fixed size using keyframed crop centers."
        )
    )
    parser.add_argument(
        "--segments_dir",
        type=Path,
        default=Path("/Users/tizianocausin/sd_local/possible_vids/segments"),
        help="Folder containing segment videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Folder for cropped videos. Defaults to SEGMENTS_DIR/cropped_segments.",
    )
    parser.add_argument(
        "--source_square_size",
        type=int,
        default=500,
        help="Initial source square size in pixels. This can be adjusted per video.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=500,
        help="Final square output size in pixels.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=1280,
        help="Maximum display width for selection/review windows. Use 0 for original size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Review videos even if the cropped output already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    segments_dir = args.segments_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else segments_dir / "cropped_segments"
    )

    if args.source_square_size <= 0 or args.output_size <= 0:
        raise ValueError("--source_square_size and --output_size must be positive.")
    if not segments_dir.exists():
        raise FileNotFoundError(segments_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = list(iter_video_paths(segments_dir))
    pending_videos = [
        video for video in videos
        if args.force or not any(output_dir.glob(f"{video.stem}__crop_*_to_{args.output_size}x{args.output_size}.mp4"))
    ]

    print(f"Found {len(videos)} videos in {segments_dir}")
    print(f"{len(pending_videos)} videos need cropping")
    print(
        "Controls: click = set crop center at current frame, i = add intermediate keyframe, "
        "d = delete current keyframe, f/g = first/last frame, arrows or mouse wheel = move, "
        "+/- = adjust square by 25px, [/] = adjust square by 5px, "
        "c = review centered crop, s = review current crop path, n = skip, p = play source, q = quit"
    )

    for index, video_path in enumerate(pending_videos, start=1):
        print(f"\n[{index}/{len(pending_videos)}] {video_path.name}")
        try:
            video_info = get_video_info(video_path)
        except ValueError as exc:
            print(f"Skipping unreadable video: {exc}")
            continue

        selector = CropPathSelector(
            video_info=video_info,
            source_square_size=min(
                args.source_square_size,
                video_info.width,
                video_info.height,
            ),
            output_size=args.output_size,
            display_width=args.display_width,
        )
        decision, keyframes = selector.run()

        if decision == "quit":
            print("Stopped by user.")
            break
        if decision != "save":
            print("Skipped.")
            continue

        output_path = output_path_for(
            video_path,
            output_dir,
            selector.source_square_size,
            args.output_size,
        )
        print(f"Saving {output_path.name}")
        write_cropped_video(
            video_path=video_path,
            output_path=output_path,
            video_info=video_info,
            keyframes=keyframes,
            source_square_size=selector.source_square_size,
            output_size=args.output_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
