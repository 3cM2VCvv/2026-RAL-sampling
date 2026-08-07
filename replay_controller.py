"""Controller that replays logged controls and visualizes reachable sets."""

import os
import pickle
import shutil
import subprocess
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from immrax import i2ut, icentpert, interval, natemb, ut2i
from PIL import Image, ImageDraw

from buzzracer.common import BASEDIR
from buzzracer.controllers.car_controller import CarController
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian


def _get_control_index(t, planning_dt, planning_horizon):
    """Compute the control index for time t."""
    return (
        jnp.floor((t + 0.5 * planning_dt) / planning_dt).astype(int) % planning_horizon
    )


# Colors (RGBA, 0-1)
COLOR_SAMPLE_REACHSET = (0.5, 0.5, 0.5, 0.15)
COLOR_BEST_REACHSET = (0.3, 0.5, 1.0, 0.8)
COLOR_BEST_CENTERLINE = (1.0, 0.0, 0.0, 1.0)

# Output framerate for the reachsets-only WebP (matches Gifsaver's 30ms/frame).
REACHSET_GIF_FPS = 33


class ReplayController(CarController):
    def __init__(self, car, config):
        # Defaults (overridden by XML config attributes via ConfigObject)
        self.log_dir = "log/stanley_immrax"
        self.log_num = 1
        # When truthy, also render a transparent reachsets-only animation (WebP).
        self.reachset_gif = False

        super().__init__(car, config)
        self.print_debug_enable()

        # XML attributes arrive as strings; normalize the boolean flag.
        if isinstance(self.reachset_gif, str):
            self.reachset_gif = self.reachset_gif.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        self.debug_dict = {}
        self.step = 0

        # Load log files
        log_dir_abs = os.path.join(BASEDIR, self.log_dir)
        full_state_path = os.path.join(log_dir_abs, f"full_state{self.log_num}.p")
        debug_dict_path = os.path.join(log_dir_abs, f"debug_dict{self.log_num}.p")

        self.print_info(f"Loading logs from {log_dir_abs}")
        with open(full_state_path, "rb") as f:
            self.full_state_log = pickle.load(f)
        with open(debug_dict_path, "rb") as f:
            self.debug_dict_log = pickle.load(f)
        self.total_steps = len(self.debug_dict_log)
        self.print_info(f"Loaded {self.total_steps} timesteps")

        # Reachability setup (mirrors immrax_controller.py:97-108)
        self.planning_dt = 0.02
        self.planning_horizon = 30

        self.predictor = DynamicBicycleCartesian(car)
        self.reach_predictor = natemb(self.predictor)

        self._t_start = jax.device_put(np.array(0.0))
        self._t_end = jax.device_put(np.array(self.planning_horizon * self.planning_dt))
        self._planning_dt_jax = jax.device_put(np.array(self.planning_dt))

        self._zero_disturbance_interval = icentpert(
            jnp.zeros(2), 
            jnp.array([1e-1, 2e-1])
            # jax.device_put(np.array([0.0, 0.0])), # FIXME: I used this disturbance size for most visuals, but it doesn't match the controller disturbance
            # jax.device_put(2e-1 * np.ones(2)),
        )
        self._disturbance_fn = lambda t, x: self._zero_disturbance_interval

        # Reachsets-only WebP rendering state (lazily initialized on first frame)
        self._reachset_px = None  # (W, H, x_limit, y_limit) cached transform info
        self._reachset_frame_dir = None
        self._reachset_webp_path = None
        self._reachset_frame_idx = 0
        self._reachset_done = False
        if self.reachset_gif:
            gifs_dir = os.path.join(BASEDIR, "gifs", "replays", os.path.basename(self.log_dir))
            self.print_info(f"Setting up reachsets-only WebP output in {gifs_dir}")
            os.makedirs(gifs_dir, exist_ok=True)
            self._reachset_webp_path = os.path.join(
                gifs_dir, f"reachsets_{self.log_num}.webp"
            )
            self._reachset_frame_dir = os.path.join(
                gifs_dir, f"reachset_frames_{self.log_num}"
            )
            if os.path.isdir(self._reachset_frame_dir):
                shutil.rmtree(self._reachset_frame_dir)
            os.makedirs(self._reachset_frame_dir)
            self.print_info(
                f"Reachsets-only WebP enabled -> {self._reachset_webp_path}"
            )

        # JIT warmup with batched rollout (21 = 20 samples + 1 best)
        self.print_info("Warming up JIT compilation...")
        dummy_state = jax.device_put(np.zeros(6))
        dummy_controls = jax.device_put(np.zeros((21, self.planning_horizon, 2)))
        _trajs = self.rollout_reachsets_batch(
            i2ut(icentpert(dummy_state, jnp.zeros(6))), dummy_controls
        )
        jax.block_until_ready(_trajs)
        jax.block_until_ready(self._extract_reachset_arrays(_trajs.ys))
        self.print_info("JIT warmup complete")

    def _rollout_single(self, x0_2n, planned_controls):
        """Single reachset rollout (mirrors immrax_controller.py:339-360)."""

        def ff_control(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return interval(planned_controls[idx])

        return self.reach_predictor.compute_trajectory(
            self._t_start,
            self._t_end,
            x0_2n,
            (ff_control, self._disturbance_fn),
            dt=self._planning_dt_jax,
            solver="euler",
        )

    @partial(jax.jit, static_argnums=0)
    def rollout_reachsets_batch(self, x0_2n, all_controls):
        """Batch reachset rollout over multiple control sequences using vmap.

        Args:
            x0_2n: Initial state interval (upper-triangle form)
            all_controls: (K, planning_horizon, 2) stacked control sequences

        Returns:
            Batched trajectory with .ys of shape (K, planning_horizon+1, 2*xlen)
        """
        return jax.vmap(self._rollout_single, in_axes=(None, 0))(x0_2n, all_controls)

    @partial(jax.jit, static_argnums=0)
    def _extract_reachset_arrays(self, all_ys):
        """Batched, JIT'd extraction of reachset geometry for all rollouts.

        Replaces ~3,900 per-step host<->device syncs (one float() per scalar)
        with a single device->host transfer of the returned arrays.

        Args:
            all_ys: (K, T, 2n) upper-triangle reachset states for K rollouts.

        Returns:
            bounds: (K, T, 4) float -- (x_lo, y_lo, x_hi, y_hi) per timestep.
            centers: (K, T, 2) float -- (x, y) interval centers.
            valid_mask: (K, T) bool -- False once a timestep overflows
                (NaN/Inf width, or width > 1e6 in any state dimension).
        """
        ints = jax.vmap(jax.vmap(ut2i))(all_ys)  # Interval (K, T, n)
        lower = ints.lower
        upper = ints.upper
        width = ints.width
        center = ints.center

        bounds = jnp.stack(
            [lower[..., 0], lower[..., 1], upper[..., 0], upper[..., 1]], axis=-1
        )
        centers = center[..., :2]
        bad = jnp.isnan(width) | jnp.isinf(width) | (width > 1e6)
        valid_mask = ~jnp.any(bad, axis=-1)
        return bounds, centers, valid_mask

    @staticmethod
    def _valid_count(mask_row):
        """Number of leading valid timesteps (stop at first overflow)."""
        if mask_row.all():
            return int(mask_row.shape[0])
        return int(np.argmin(mask_row))

    def control(self):
        if self.step >= self.total_steps:
            if self.reachset_gif and not self._reachset_done:
                self._assemble_reachset_webp()
            self.print_info("Replay complete")
            self.car.main.exit_request.set()
            self.car.throttle = 0.0
            self.car.steering = 0.0
            return

        # Extract logged data
        dd = self.debug_dict_log[self.step]["cars"][0]
        logged_state = dd["state"]
        sample_controls = dd["sample_controls"]  # (20, 30, 2)
        best_controls = dd["best_controls"]  # (30, 2)

        # Override car state from full_state_log for exact replay
        fs = self.full_state_log[self.step][0]
        # fs = [time, x, y, theta, v_forward, v_sideway, omega, steering, throttle]
        self.car.state = (fs[1], fs[2], fs[3], fs[4], fs[5], fs[6])
        self.car.steering = fs[7]
        self.car.throttle = fs[8]

        # Compute all reachable sets in one batched call
        state_jax = jax.device_put(np.asarray(logged_state))
        x0_2n = i2ut(icentpert(state_jax, jnp.zeros(6)))

        # Stack all controls: samples (20, H, 2) + best (1, H, 2) -> (21, H, 2)
        all_controls = np.concatenate(
            [sample_controls, best_controls[np.newaxis]], axis=0
        )
        all_controls_jax = jax.device_put(all_controls)
        all_trajs = self.rollout_reachsets_batch(x0_2n, all_controls_jax)

        # Single device->host transfer for all 21 reachsets (see Idea 1).
        bounds, centers, valid_mask = jax.device_get(
            self._extract_reachset_arrays(all_trajs.ys)
        )
        num_samples = sample_controls.shape[0]
        valid_counts = [
            self._valid_count(valid_mask[i]) for i in range(bounds.shape[0])
        ]

        self._draw_reachsets(bounds, centers, valid_counts, num_samples)
        if self.reachset_gif:
            self._render_reachset_frame(bounds, centers, valid_counts, num_samples)

        self.step += 1

    def _draw_reachsets(self, bounds, centers, valid_counts, num_samples):
        """Draw reachsets into the live Buzzracer visualization (if present)."""
        viz = getattr(self.car.main, "visualization", None)
        if viz is None:
            return

        # Sampled control reachsets (gray, transparent, no centerline)
        for i in range(num_samples):
            for t in range(valid_counts[i]):
                x_lo, y_lo, x_hi, y_hi = bounds[i, t]
                viz.draw_quad(
                    float(x_lo),
                    float(y_lo),
                    float(x_hi),
                    float(y_hi),
                    COLOR_SAMPLE_REACHSET,
                )

        # Best control reachset (blue) + red centerline
        best = bounds.shape[0] - 1
        n_best = valid_counts[best]
        for t in range(n_best):
            x_lo, y_lo, x_hi, y_hi = bounds[best, t]
            viz.draw_quad(
                float(x_lo),
                float(y_lo),
                float(x_hi),
                float(y_hi),
                COLOR_BEST_REACHSET,
            )
        if n_best > 1:
            best_centers = [tuple(c) for c in centers[best, :n_best]]
            viz.draw_polyline(best_centers, COLOR_BEST_CENTERLINE, width=0.01)

    # ============================================================
    # Reachsets-only transparent WebP rendering
    # ============================================================

    def _ensure_reachset_canvas_info(self):
        """Resolve track->pixel transform once (mirrors the GL renderer)."""
        if self._reachset_px is not None:
            return self._reachset_px
        track = self.car.main.track
        viz = getattr(self.car.main, "visualization", None)
        if viz is not None and getattr(viz, "img_track", None) is not None:
            rows, cols = viz.img_track.shape[:2]
            W, H = int(cols), int(rows)
        else:
            # Fall back to the track's own meter->pixel resolution.
            W = int(track.x_limit * track.resolution)
            H = int(track.y_limit * track.resolution)
        self._reachset_px = (W, H, float(track.x_limit), float(track.y_limit))
        return self._reachset_px

    def _render_reachset_frame(self, bounds, centers, valid_counts, num_samples):
        """Render one transparent RGBA frame of reachsets and write it as PNG."""
        W, H, x_limit, y_limit = self._ensure_reachset_canvas_info()
        # Straight-alpha RGBA canvas, fully transparent background.
        canvas = np.zeros((H, W, 4), dtype=np.float32)

        def to_px(x, y):
            # Match VisualizationGL: x grows right, y grows up; image row 0 = top.
            col = x / x_limit * W
            row = H - y / y_limit * H
            return col, row

        def blend_quad(x_lo, y_lo, x_hi, y_hi, color):
            c0, r1 = to_px(x_lo, y_lo)
            c1, r0 = to_px(x_hi, y_hi)
            c0i, c1i = sorted((int(round(c0)), int(round(c1))))
            r0i, r1i = sorted((int(round(r0)), int(round(r1))))
            c0i, c1i = max(0, c0i), min(W, c1i)
            r0i, r1i = max(0, r0i), min(H, r1i)
            if c1i <= c0i or r1i <= r0i:
                return
            region = canvas[r0i:r1i, c0i:c1i]
            src_rgb = np.asarray(color[:3], dtype=np.float32)
            src_a = float(color[3])
            dst_a = region[..., 3]
            out_a = src_a + dst_a * (1.0 - src_a)
            safe_a = np.maximum(out_a, 1e-6)
            blended = (
                src_rgb * src_a + region[..., :3] * dst_a[..., None] * (1.0 - src_a)
            ) / safe_a[..., None]
            region[..., :3] = np.where(out_a[..., None] > 0, blended, 0.0)
            region[..., 3] = out_a

        # Sampled reachsets (gray, translucent), then best reachset (blue).
        for i in range(num_samples):
            for t in range(valid_counts[i]):
                blend_quad(*bounds[i, t], COLOR_SAMPLE_REACHSET)
        best = bounds.shape[0] - 1
        n_best = valid_counts[best]
        for t in range(n_best):
            blend_quad(*bounds[best, t], COLOR_BEST_REACHSET)

        img = Image.fromarray(
            np.clip(canvas * 255.0, 0, 255).astype(np.uint8), mode="RGBA"
        )

        # Best centerline (opaque red polyline) drawn on top.
        if n_best > 1:
            pts = [to_px(float(x), float(y)) for x, y in centers[best, :n_best]]
            line_w = max(2, int(round(0.01 / x_limit * W)))
            ImageDraw.Draw(img).line(
                pts,
                fill=tuple(int(c * 255) for c in COLOR_BEST_CENTERLINE),
                width=line_w,
            )

        frame_path = os.path.join(
            self._reachset_frame_dir, f"frame_{self._reachset_frame_idx:06d}.png"
        )
        img.save(frame_path)
        self._reachset_frame_idx += 1

    def _assemble_reachset_webp(self):
        """Encode the streamed PNG frames into one animated, transparent WebP."""
        self._reachset_done = True
        if self._reachset_frame_idx == 0:
            self.print_warning("No reachset frames rendered; skipping WebP export")
            return
        self.print_info(
            f"Encoding {self._reachset_frame_idx} frames -> {self._reachset_webp_path}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(REACHSET_GIF_FPS),
            "-i",
            os.path.join(self._reachset_frame_dir, "frame_%06d.png"),
            "-loop",
            "0",
            "-lossless",
            "1",
            self._reachset_webp_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            self.print_warning(
                "ffmpeg not found; reachset PNG frames left in "
                f"{self._reachset_frame_dir}"
            )
            return
        except subprocess.CalledProcessError as e:
            self.print_warning(f"ffmpeg failed: {e.stderr}")
            return
        self.print_info(f"Reachsets-only WebP saved at {self._reachset_webp_path}")
        shutil.rmtree(self._reachset_frame_dir, ignore_errors=True)
