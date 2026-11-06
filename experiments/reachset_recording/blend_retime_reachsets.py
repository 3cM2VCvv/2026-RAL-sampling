"""Composite a controller's visualization over a transformed recording.

The controller logs one visualization frame per control step, but those steps
are not evenly spaced in wall-clock time. This script resamples the
visualization to a fixed frame rate using the per-step timestamps recorded in
the controller data, then overlays the result onto a (top-down, transformed)
recording of the run.

Pipeline:
  1. Read per-step timestamps from the controller data pickle.
  2. Resample the visualization frames to --fps, using --blend-mode to fill
     output times that fall between two control steps:
       - "blend"   linearly cross-fades the two neighboring frames (default)
       - "nearest" snaps to the closer neighbor
  3. Pipe the resampled RGBA frames into ffmpeg, which scales them to the
     recording's resolution and overlays them (using their alpha, scaled by
     --alpha) onto the transformed recording.
"""

import argparse
import pickle
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image


def unit_float(s):
    v = float(s)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be between 0 and 1")
    return v


def load_timestamps(pickle_path):
    """Return the per-control-step timestamps from a controller-data pickle."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return np.array([float(e[0][0]) for e in data], dtype=np.float64)


def probe_video_size(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or h == 0:
        sys.exit(f"failed to read dimensions from {path}")
    return w, h


def load_frame(im, i):
    im.seek(i)
    return np.asarray(im.convert("RGBA"), dtype=np.float32)


def resample(im, ts, t0, n_out, fps, blend_mode, alpha, out_stream):
    """Resample visualization frames to fps and write RGBA bytes to out_stream.

    Walks two neighboring source frames (f1, f2) forward in lockstep with the
    output clock so each frame is decoded at most once.
    """
    N = len(ts)
    i = 0
    f1 = load_frame(im, 0)
    f2 = load_frame(im, 1)

    for k in range(n_out):
        target_t = t0 + k / fps
        while i < N - 2 and target_t >= ts[i + 1]:
            i += 1
            f1 = f2
            f2 = load_frame(im, i + 1)

        if target_t >= ts[-1]:
            out = f2
        elif target_t <= ts[0]:
            out = f1
        else:
            frac = (target_t - ts[i]) / (ts[i + 1] - ts[i])
            if blend_mode == "nearest":
                out = f1 if frac < 0.5 else f2
            else:  # blend
                out = (1.0 - frac) * f1 + frac * f2

        if alpha != 1.0:
            out = out.copy()  # may alias a cached frame; don't mutate in place
            out[..., 3] *= alpha

        try:
            out_stream.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
        except BrokenPipeError:
            # overlay=shortest=1: ffmpeg stops once the base recording ends,
            # which can be a few frames before the resampled stream. Benign.
            print(f"  base recording ended at frame {k}/{n_out}; stopping")
            return
        if k % 300 == 0:
            print(f"  frame {k}/{n_out} (step {i})", flush=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--transformed-recording",
        required=True,
        help="base video: top-down warped recording of the run",
    )
    p.add_argument(
        "--controller-visualization",
        required=True,
        help="animated RGBA frames (.webp), one per control step",
    )
    p.add_argument(
        "--controller-data",
        required=True,
        help="controller-data pickle; provides per-step timestamps",
    )
    p.add_argument("--output", required=True, help="output video path")
    p.add_argument("--fps", type=int, default=60, help="output frame rate (default 60)")
    p.add_argument(
        "--blend-mode",
        choices=["blend", "nearest"],
        default="blend",
        help="how to fill output times between control steps (default blend)",
    )
    p.add_argument(
        "--alpha",
        type=unit_float,
        default=1.0,
        help="overlay opacity in [0, 1] (default 1.0)",
    )
    args = p.parse_args()

    ts = load_timestamps(args.controller_data)
    N = len(ts)

    im = Image.open(args.controller_visualization)
    n_frames = im.n_frames
    W, H = im.size
    print(f"visualization: {n_frames} frames {W}x{H} {im.mode}; control steps: {N}")
    assert n_frames == N, "expected one visualization frame per control step"

    base_w, base_h = probe_video_size(args.transformed_recording)

    t0 = ts[0]
    duration = ts[-1] - t0
    n_out = int(np.floor(duration * args.fps)) + 1
    print(
        f"duration: {duration:.6f} s, output frames: {n_out} @ {args.fps} fps; "
        f"base {base_w}x{base_h}, blend={args.blend_mode}, alpha={args.alpha}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        args.transformed_recording,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{W}x{H}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-filter_complex",
        f"[1:v]scale={base_w}:{base_h}[ov];[0:v][ov]overlay=shortest=1[out]",
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        args.output,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        resample(im, ts, t0, n_out, args.fps, args.blend_mode, args.alpha, proc.stdin)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"ffmpeg exited with code {proc.returncode}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
