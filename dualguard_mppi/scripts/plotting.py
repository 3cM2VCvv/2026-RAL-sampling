"""Figure and GIF output: one plot per file, so each is an addressable DVC output.

The GIF replaces the notebooks' ``ipywidgets`` slider and plays back at real speed:
the delay written into each frame is exactly the simulated time it spans,
``stride * dt``.  GIF stores that delay in centiseconds, so :func:`real_time_stride`
only picks strides whose delay is a whole number of them -- otherwise every viewer
would round it somewhere different and playback would be real speed on none of them.
"""

import json
import os
from io import BytesIO

import matplotlib

# Set before pyplot is imported.  This venv has PyQt5, so matplotlib would otherwise
# default to qtagg, which needs a display these scripts do not have.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def save_fig(fig, out_dir, name, dpi=130):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def save_json(obj, out_dir, name="summary.json"):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"  wrote {path}")
    return path


def trace(out_dir, name, t, series, ylabel="", title="", hlines=(), ylim=None,
          shade=None, shade_label="", figsize=(9, 3)):
    """One time series figure.  ``series`` is a list of (values, label)."""
    fig, ax = plt.subplots(figsize=figsize)
    for values, label in series:
        ax.plot(t, values, lw=0.9, label=label)
    for y, style in hlines:
        ax.axhline(y, **style)
    if shade is not None and shade.any():
        ax.fill_between(t, 0, 1, where=shade, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.15, label=shade_label)
    if ylim:
        ax.set_ylim(*ylim)
    if any(label for _, label in series) or shade_label:
        ax.legend(loc="best", fontsize=8)
    ax.set(xlabel="t [s]", ylabel=ylabel, title=title)
    return save_fig(fig, out_dir, name)


def standard_figures(run, args, dt, static, dynamic, caption, traces, stride,
                     trajectory_title="", figsize=(6, 8), gif_figsize=(5, 7)):
    """The figure set both experiments produce.

    The track panel is the only experiment-specific piece and comes in two halves:
    ``static(ax)`` draws what never changes (walls, the failure-set contours) and
    ``dynamic(ax, k, rollouts)`` draws frame ``k`` and returns the artists it added.
    Splitting them is what makes the GIF affordable -- the static contours are over
    grids up to 608 x 408 and cost more to evaluate than everything else in a frame
    put together, so they are computed once and every frame reuses them.

    ``traces`` is a list of :func:`trace` keyword dicts for the experiment's own time
    series; ``caption(k)`` titles each GIF frame.
    """
    out, log = args.out_dir, run.log

    fig, ax = plt.subplots(figsize=figsize)
    static(ax)
    dynamic(ax, len(run) - 1, rollouts=False)
    ax.scatter(*run.states[0, :2], marker="o", s=70, facecolors="none",
               edgecolors="k", zorder=6)
    ax.set(xlabel="x [m]", ylabel="y [m]", title=trajectory_title)
    save_fig(fig, out, "trajectory.png")

    for spec in traces:
        trace(out, t=run.t, **spec)

    trace(out, "lap_progress.png", run.t, [(log["progress"], "")], "arc length [m]",
          f"lap progress -- {log['laps'][-1]} of {run.laps.target_laps} laps",
          hlines=[(lap * run.laps.proj.track_len, dict(color="0.7", ls=":"))
                  for lap in range(1, int(log["laps"][-1]) + 1)])

    if args.skip_gif:
        print("  skipping animation.gif (--skip-gif)")
        return
    write_gif(out, "animation.gif", static, dynamic, caption,
              range(0, len(run), stride), stride * dt, figsize=gif_figsize)


def real_time_stride(dt, target_fps=25.0):
    """Frames to skip so the GIF runs near ``target_fps`` and still at real speed.

    The frame delay is ``stride * dt``, which GIF can only represent to 10 ms.
    Strides that would need a rounded delay are rejected and the stride grown: a
    slightly lower frame rate is still real speed, a rounded delay is not.
    """
    ideal = max(1, int(round(1.0 / (target_fps * dt))))
    for stride in range(ideal, 4 * ideal + 1):
        ms = stride * dt * 1000.0
        if abs(ms - round(ms)) < 1e-6 and round(ms) % 10 == 0:
            return stride
    raise ValueError(f"no stride near {ideal} gives a whole-centisecond frame delay "
                     f"at dt = {dt}; playback could not be held to real speed")


def write_gif(out_dir, name, static, dynamic, caption, indices, seconds_per_frame,
              figsize=(5, 7), dpi=80):
    """Animate one figure: ``static(ax)`` once, then ``dynamic(ax, k)`` per frame.

    ``dynamic`` returns the artists it added; they are removed again after the frame
    is grabbed, so the figure -- and the expensive static contours on it -- is built
    exactly once for the whole animation.
    """
    delay_ms = seconds_per_frame * 1000.0
    if abs(delay_ms - round(delay_ms)) > 1e-6 or round(delay_ms) % 10:
        raise ValueError(f"frame delay {delay_ms} ms is not a whole number of "
                         "centiseconds, so playback would not be real speed")

    fig, ax = plt.subplots(figsize=figsize)
    static(ax)
    fig.tight_layout()

    frames = []
    for k in indices:
        artists = dynamic(ax, k)
        ax.set_title(caption(k), fontsize=9)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        for artist in artists:
            artist.remove()
    plt.close(fig)

    path = os.path.join(out_dir, name)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(round(delay_ms)), loop=0)
    print(f"  wrote {path}  ({len(frames)} frames, {delay_ms:.0f} ms each, "
          f"{len(frames) * seconds_per_frame:.1f} s real time, "
          f"{os.path.getsize(path) / 2**20:.1f} MiB)")
    return path
