"""Warp a stationary-camera video of the track to a flat top-down rectangle.

Stages:
  1. Pick: click 4 corners (TL, TR, BR, BL) of the track on the first frame.
  2. Transform: warp every frame and encode via piped ffmpeg.
  3. Crop (default): trim the transformed video to a start/end range, chosen
     interactively on a scrubbable timeline (or via --start/--end).

By default the script transforms and then crops. Use --save_intermediate to
keep the full transformed video before cropping, or --no-crop to skip cropping.

Default aspect ratio 2:3 matches the "full" RCP track (2.4 m x 3.6 m).
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
INSTRUCTIONS = (
    "Click 4 corners in order: TL -> TR -> BR -> BL "
    "(of the desired output orientation). "
    "u=undo  r=reset  Enter=confirm  q=quit"
)
CROP_HELP = (
    "scrub=trackbar  ,/.=frame -/+  a/d=second -/+  [=set in  ]=set out  "
    "Enter=confirm  q=quit"
)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_aspect(s):
    w, h = s.split(":")
    return float(w) / float(h)


def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"failed to read first frame from {path}")
    return frame


def fmt_time(frame_idx, fps):
    t = frame_idx / fps
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


def pick_corners(frame):
    points = []
    win = "pick corners"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def render():
        canvas = frame.copy()
        cv2.putText(
            canvas,
            INSTRUCTIONS,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"{i}:{CORNER_LABELS[i]}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(points) == 4:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2
            )
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            render()

    cv2.setMouseCallback(win, on_mouse)
    render()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            sys.exit("aborted")
        if key == ord("r"):
            points.clear()
            render()
        elif key == ord("u") and points:
            points.pop()
            render()
        elif key in (13, 10) and len(points) == 4:  # Enter
            cv2.destroyWindow(win)
            return points


def compute_output_size(corners, aspect):
    c = np.array(corners, dtype=np.float64)
    top = np.linalg.norm(c[1] - c[0])
    bottom = np.linalg.norm(c[2] - c[3])
    left = np.linalg.norm(c[3] - c[0])
    right = np.linalg.norm(c[2] - c[1])
    cand_w = max(top, bottom)
    cand_h = max(left, right)
    # Grow the smaller side so neither edge is undersampled.
    out_w = max(cand_w, cand_h * aspect)
    out_h = out_w / aspect
    # libx264 needs even dims.
    return (int(round(out_w / 2)) * 2, int(round(out_h / 2)) * 2)


def transform(input_path, output_path, corners, aspect):
    out_w, out_h = compute_output_size(corners, aspect)
    src = np.array(corners, dtype=np.float32)
    dst = np.array([(0, 0), (out_w, 0), (out_w, out_h), (0, out_h)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"warping {n_frames} frames @ {fps:.3f} fps -> {out_w}x{out_h} "
        f"(aspect {out_w / out_h:.4f}, target {aspect:.4f})"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{out_w}x{out_h}",
        "-framerate",
        f"{fps}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            warped = cv2.warpPerspective(frame, M, (out_w, out_h))
            proc.stdin.write(warped.tobytes())
            i += 1
            if i % 30 == 0 or i == n_frames:
                print(f"  {i}/{n_frames}", end="\r", flush=True)
        print()
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"ffmpeg exited with code {proc.returncode}")
    print(f"wrote {output_path}")


def pick_crop_range(video_path):
    """Interactively choose a [start, end] frame range on a scrubbable timeline.

    Returns (start_frame, end_frame, fps). end_frame is inclusive.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 1:
        cap.release()
        sys.exit(f"could not determine frame count for crop UI: {video_path}")
    last = n_frames - 1

    win = "crop range"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    st = {"pos": 0, "start": 0, "end": last, "cache_idx": -1, "cache": None}

    def get_frame(idx):
        if idx != st["cache_idx"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                st["cache"] = fr
            st["cache_idx"] = idx
        return st["cache"]

    def render():
        frame = get_frame(st["pos"])
        if frame is None:
            return
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        bar_y = h - 30
        bar_x0, bar_x1 = 20, w - 20
        bar_w = bar_x1 - bar_x0

        def fx(idx):
            return int(bar_x0 + bar_w * idx / last)

        # full timeline, then the selected [in, out] span on top.
        cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (180, 180, 180), 2)
        cv2.line(
            canvas,
            (fx(st["start"]), bar_y),
            (fx(st["end"]), bar_y),
            (0, 255, 0),
            4,
        )
        # in / out / cursor ticks.
        cv2.line(
            canvas,
            (fx(st["start"]), bar_y - 10),
            (fx(st["start"]), bar_y + 10),
            (0, 255, 0),
            2,
        )
        cv2.line(
            canvas,
            (fx(st["end"]), bar_y - 10),
            (fx(st["end"]), bar_y + 10),
            (0, 0, 255),
            2,
        )
        cv2.line(
            canvas,
            (fx(st["pos"]), bar_y - 16),
            (fx(st["pos"]), bar_y + 16),
            (0, 255, 255),
            2,
        )

        info = (
            f"pos {st['pos']} {fmt_time(st['pos'], fps)}  |  "
            f"in {st['start']} {fmt_time(st['start'], fps)}  |  "
            f"out {st['end']} {fmt_time(st['end'], fps)}"
        )
        cv2.putText(canvas, info, (20, 30), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas, CROP_HELP, (20, 55), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow(win, canvas)

    def on_track(v):
        st["pos"] = v
        render()

    cv2.createTrackbar("frame", win, 0, last, on_track)

    def set_pos(idx):
        # Drive position through the trackbar so the slider stays in sync;
        # the callback updates st["pos"] and re-renders.
        cv2.setTrackbarPos("frame", win, max(0, min(last, idx)))

    render()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            cap.release()
            cv2.destroyWindow(win)
            sys.exit("aborted")
        elif key == ord(","):
            set_pos(st["pos"] - 1)
        elif key == ord("."):
            set_pos(st["pos"] + 1)
        elif key == ord("a"):
            set_pos(st["pos"] - int(round(fps)))
        elif key == ord("d"):
            set_pos(st["pos"] + int(round(fps)))
        elif key == ord("["):
            st["start"] = st["pos"]
            render()
        elif key == ord("]"):
            st["end"] = st["pos"]
            render()
        elif key in (13, 10):  # Enter
            if st["start"] >= st["end"]:
                print("  in must be before out; adjust and confirm again")
                continue
            cap.release()
            cv2.destroyWindow(win)
            return st["start"], st["end"], fps


def crop(input_path, output_path, start_frame, end_frame, fps):
    start_t = start_frame / fps
    end_t = (end_frame + 1) / fps  # +1 so end_frame is included
    print(
        f"cropping frames {start_frame}..{end_frame} "
        f"({fmt_time(start_frame, fps)} -> {fmt_time(end_frame, fps)})"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ss",
        f"{start_t:.6f}",
        "-to",
        f"{end_t:.6f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        output_path,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg exited with code {proc.returncode}")
    print(f"wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument(
        "--corners", help="transform corners JSON path; default <input>.corners.json"
    )
    p.add_argument("--crop", help="crop-times JSON path; default <input>.crop.json")
    p.add_argument("--aspect", default="2:3", help="W:H (default 2:3)")
    p.add_argument("--pick", action="store_true", help="force re-picking")
    p.add_argument("--pick-only", action="store_true")
    p.add_argument(
        "--no-crop", action="store_true", help="skip the duration crop stage"
    )
    p.add_argument(
        "--save-intermediate",
        action="store_true",
        help="keep the transformed video before cropping",
    )
    p.add_argument(
        "--start",
        type=float,
        help="crop start time in seconds (skips crop UI; needs --end)",
    )
    p.add_argument(
        "--end",
        type=float,
        help="crop end time in seconds (skips crop UI; needs --start)",
    )
    args = p.parse_args()

    corners_path = args.corners or args.input + ".corners.json"
    aspect = parse_aspect(args.aspect)

    # Pick phase (try to read from disk)
    if args.pick or not os.path.exists(corners_path):
        frame = read_first_frame(args.input)
        corners = pick_corners(frame)
        with open(corners_path, "w") as f:
            json.dump(corners, f)
        print(f"saved corners to {corners_path}: {corners}")
    else:
        with open(corners_path) as f:
            corners = json.load(f)
        print(f"loaded corners from {corners_path}: {corners}")

    if args.pick_only:
        return

    # Transform phase
    if not args.output:
        sys.exit("--output is required unless --pick-only")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stem, ext = os.path.splitext(args.output)
    transform_output = (
        args.output
        if args.no_crop
        else (
            f"{stem}.transformed{ext}"
            if args.save_intermediate
            else f"{stem}.transformed.tmp{ext}"
        )
    )
    transform(args.input, transform_output, corners, aspect)

    if args.no_crop:
        return

    # Crop phase. Range precedence: --start/--end > saved JSON > interactive UI.
    crop_path = args.crop or args.input + ".crop.json"
    picked_crop = False
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            sys.exit("--start and --end must be given together")
        fps = cv2.VideoCapture(transform_output).get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(round(args.start * fps))
        end_frame = int(round(args.end * fps))
    elif os.path.exists(crop_path):
        with open(crop_path) as f:
            saved = json.load(f)
        fps = cv2.VideoCapture(transform_output).get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(round(saved["start"] * fps))
        end_frame = int(round(saved["end"] * fps))
        print(f"loaded crop times from {crop_path}: {saved}")
    else:
        start_frame, end_frame, fps = pick_crop_range(transform_output)
        picked_crop = True

    crop(transform_output, args.output, start_frame, end_frame, fps)

    # Only persist crop times we picked interactively. Don't clobber a
    # caller-provided (--start/--end) or previously saved range: the
    # frame-rounded round-trip differs from the input, which would dirty the
    # file (and any DVC stage that depends on it) on every run.
    if picked_crop:
        crop_times = {"start": start_frame / fps, "end": end_frame / fps}
        with open(crop_path, "w") as f:
            json.dump(crop_times, f)
        print(f"saved crop times to {crop_path}: {crop_times}")

    if not args.save_intermediate:
        os.remove(transform_output)
        print(f"removed intermediate {transform_output}")


if __name__ == "__main__":
    main()
