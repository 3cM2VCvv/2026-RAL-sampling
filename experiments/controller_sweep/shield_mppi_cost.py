"""PyTorch rollout cost matching ImmraxController's state cost.

The terms reproduce ``reachset_cost.RolloutCostEvaluator._compute_cost_components``
verbatim (same signs, weights and constants) on a single trajectory per sample,
minus the hard collision penalty.  It deliberately contains no control or input
regularization: that is supplied by the MPPI solver's free-energy term.  The
DCBF penalty is supplied separately by the vendored solver's
``compute_cbf_cost``.

Trajectories hold the ``N`` post-step states of shape ``(K, N, 6)``.
"""

import math

import numpy as np
import torch


def cbf_half_width(track):
    """Tightest one-sided clearance from the raceline to a track boundary."""
    return float(
        min(
            np.min(track.raceline_left_boundary),
            np.min(track.raceline_right_boundary),
        )
    )


class ImmraxRolloutCost:
    """Total rollout cost for a batch of sampled trajectories."""

    def __init__(
        self,
        track,
        horizon,
        planning_dt,
        centering_weight_factor=1.0,
        velocity_scale_factor=1.0,
        projection_chunk=256,
        device="cpu",
    ):
        self.horizon = int(horizon)
        self.planning_dt = float(planning_dt)
        self.centering_weight_factor = float(centering_weight_factor)
        self.velocity_scale_factor = float(velocity_scale_factor)
        self.projection_chunk = int(projection_chunk)
        self.device = torch.device(device)

        def put(values):
            return torch.as_tensor(
                np.asarray(values), dtype=torch.float32, device=self.device
            )

        self._raceline_xy = torch.stack(
            [put(track.raceline_points[0]), put(track.raceline_points[1])], dim=-1
        )
        self._raceline_headings = put(track.raceline_headings)
        self._raceline_velocity = put(track.raceline_velocity)
        self._raceline_left = put(track.raceline_left_boundary)
        self._raceline_right = put(track.raceline_right_boundary)
        self._progress_grid = put(track.ss)
        self.track_len = float(track.raceline_len_m)

    # ------------------------------------------------------------------
    # Frenet projection
    # ------------------------------------------------------------------
    def _project(self, xy):
        """Nearest raceline point for each xy.

        ``xy`` is ``(..., 2)``; returns ``progress``, signed ``lateral_err`` and
        ``closest_idx`` with the leading shape.
        """
        lead = xy.shape[:-1]
        flat = xy.reshape(-1, 2)
        indices = []
        for chunk in flat.split(self.projection_chunk):
            dists = torch.cdist(chunk, self._raceline_xy)
            indices.append(torch.argmin(dists, dim=1))
        idx = torch.cat(indices, dim=0)

        ref_xy = self._raceline_xy[idx]
        heading = self._raceline_headings[idx]
        dx = flat[:, 0] - ref_xy[:, 0]
        dy = flat[:, 1] - ref_xy[:, 1]
        lateral = -dx * torch.sin(heading) + dy * torch.cos(heading)

        return (
            self._progress_grid[idx].reshape(lead),
            lateral.reshape(lead),
            idx.reshape(lead),
        )

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def components(self, x0, traj):
        """Return the individual cost terms, each shape (K,)."""
        batch = traj.shape[0]
        full = torch.cat([x0.reshape(1, 1, -1).expand(batch, -1, -1), traj], dim=1)
        xy = full[..., :2]
        vel = full[..., 3]

        progress, lateral_err, closest_idx = self._project(xy)
        abs_lateral = torch.abs(lateral_err)

        ref_vel = self.velocity_scale_factor * self._raceline_velocity[closest_idx]
        step_cost = (
            self.centering_weight_factor * 5.0 * abs_lateral
            + 2.5 * (vel - ref_vel) ** 2
            - 10.0 * torch.clamp(vel, max=0.0)
        ).sum(dim=1)

        left = self._raceline_left[closest_idx]
        right = self._raceline_right[closest_idx]
        bound = torch.where(lateral_err > 0, left, right)
        boundary_arg = -100.0 * (bound - (abs_lateral + 0.05))
        boundary_cost = (
            self.centering_weight_factor
            * 10.0
            * torch.clamp(
                (2.0 / math.pi) * torch.atan(boundary_arg) + 1.0, min=0.0
            ).sum(dim=1)
        )

        progress_made = progress[:, -1] - progress[:, 0]
        track_len = self.track_len
        progress_made = torch.where(
            progress_made < -track_len / 2,
            progress_made + track_len,
            torch.where(
                progress_made > track_len / 2,
                progress_made - track_len,
                progress_made,
            ),
        )
        terminal_cost = -2.0 * progress_made

        return {
            "step_cost": step_cost,
            "boundary_cost": boundary_cost,
            "terminal_cost": terminal_cost,
        }

    def __call__(self, x0, traj):
        return sum(self.components(x0, traj).values())
