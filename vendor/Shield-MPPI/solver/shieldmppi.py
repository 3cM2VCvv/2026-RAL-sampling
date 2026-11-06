#!/usr/bin/env python3
"""Shield-MPPI sampling utilities for the Buzzracer adapter.

The adapter owns the car dynamics, the rollouts and the local repair.  This
class provides the MPPI weight/constraint utilities and the DCBF cost.

The DCBF is the paper's ``C * sum(clamp(-h(x_{k+1}) + alpha h(x_k), 0))``
with ``h(x) = (w_T / 2)^2 - ||x - x_ref||^2``.  ``alpha = 1 - beta``.
"""

import torch


class MPPIController:
    def __init__(
        self,
        delta_t: float = 0.05,
        time_step: int = 30,
        batch_size: int = 1000,
        sigma: torch.Tensor = torch.tensor([[0.7, 0.0], [0.0, 0.5]]),
        ref_track_path: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 1.0]),
        param_lambda: float = 50.0,
        param_alpha: float = 1.0,
        cbf_cost_weight: float = 1000.0,
        beta: float = 0.5,
        w_T: float = 3.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.dim_x = 6
        self.dim_u = 2
        self.time_step = time_step
        self.batch_size = batch_size
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)

        self.sigma = sigma.to(self.device)
        self.sigma_diag = self.sigma.diag()
        self.sigma_inv = torch.linalg.inv(self.sigma)
        self.sigma_inv_diag = self.sigma_inv.diag()
        self.ref_track_path = ref_track_path.to(self.device)

        self.dt = delta_t
        self.min_v = torch.tensor([-1.0, -1.0], device=self.device)
        self.max_v = torch.tensor([1.0, 1.0], device=self.device)
        self.state = torch.zeros(self.dim_x, device=self.device)
        self.u_prev = torch.zeros((self.time_step, self.dim_u), device=self.device)

        self.beta = beta
        self.alpha = 1.0 - self.beta
        self.w_T = w_T
        self.cbf_cost_weight = cbf_cost_weight

    def h(self, x, ref_path):
        """DCBF value at ``x`` relative to the nearest reference point."""
        if ref_path.ndim == 3:
            ref_xy = ref_path[:, :, :2]
        else:
            ref_xy = ref_path[:, :2]
        half = self.w_T / 2.0
        diff = ref_xy - x
        return half * half - torch.sum(torch.square(diff), dim=-1)

    def compute_cbf_cost(self, trajectory, ref_path):
        """Paper's DCBF penalty over the sampled trajectory."""
        batch = trajectory.shape[0]
        x0 = self.state[:2].view(1, 1, 2)
        x_t_p1 = trajectory[:, :, :2]
        x_t = torch.cat(
            [x0.expand(batch, 1, 2), x_t_p1[:, :-1]],
            dim=1,
        )
        h_x_t_p1 = self.h(x_t_p1, ref_path)
        h_x_t = self.h(x_t, ref_path)
        cbf_cost = self.cbf_cost_weight * torch.clamp(
            -h_x_t_p1 + self.alpha * h_x_t, min=0.0
        )

        # Paper's z_0 = (x_0, x_0); zero unless the initial state is unsafe.
        h_x0 = self.h(x0, ref_path[:, :1, :])
        zero_term = self.cbf_cost_weight * torch.clamp(
            -(1.0 - self.alpha) * h_x0, min=0.0
        )
        return torch.cat([zero_term, cbf_cost], dim=1).sum(dim=1)

    def get_nearest_waypoints(self, trajectory, SEARCH_IDX_LEN=200):
        """Nearest reference point for each trajectory state.

        ``trajectory`` is ``(B, T, state_dim)``; only x/y are used.  The search
        window starts at the reference point nearest the current state.
        """
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
        batch, time_step, _ = trajectory.shape
        N = self.ref_track_path.shape[0]

        ref_xy_all = self.ref_track_path[:, :2]
        current_pos = self.state[:2]
        dists = torch.linalg.norm(ref_xy_all - current_pos[None, :], dim=1)
        start_idx = int(torch.argmin(dists).item())

        end_idx = start_idx + SEARCH_IDX_LEN
        if end_idx <= N:
            search_slice = self.ref_track_path[start_idx:end_idx]
        else:
            tail = self.ref_track_path[start_idx:]
            head = self.ref_track_path[: end_idx % N]
            search_slice = torch.cat([tail, head], dim=0)

        ref_xy = search_slice[:, :2]
        query_xy = trajectory[:, :, :2]
        diff = query_xy[:, :, None, :] - ref_xy[None, None, :, :]
        dists_squared = torch.sum(diff**2, dim=-1)

        nearest_local_idx = torch.argmin(dists_squared, dim=-1)
        nearest_global_idx = (start_idx + nearest_local_idx) % N

        flat_idx = nearest_global_idx.view(-1)
        ref_selected = self.ref_track_path[flat_idx]
        return ref_selected.view(batch, time_step, 4)

    def compute_weights(self, costs):
        return torch.softmax(-costs / self.param_lambda, dim=0)

    def apply_constraint(self, u):
        return torch.clamp(u, min=self.min_v, max=self.max_v)

    def set_state(self, state):
        self.state = state
