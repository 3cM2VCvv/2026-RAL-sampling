"""Batched PyTorch port of the 6-state dynamic bicycle used by ImmraxController.

Mirrors ``immrax_dynamic_bicycle_cartesian.DynamicBicycleCartesian.f`` with a
zero disturbance (same constants, same low-speed branch) so the MPPI rollouts
and the plant/reachability model agree.  States are
``[x, y, heading, v_forward, v_sideway, omega]`` and controls are
``[steering, throttle]``.
"""

import torch


class DynamicBicycleTorch:
    """Continuous dynamics ``f(x, u)`` with an explicit-Euler rollout."""

    TIRE_C = 1.6
    TIRE_B = 2.3
    TIRE_D = 1.1
    MOTOR_GAIN = 6.17
    MOTOR_DRAG = 15.2
    MOTOR_OFFSET = 0.333
    REAR_FORCE_SCALE = 1.15
    GRAVITY = 9.8
    LOW_SPEED = 0.1

    def __init__(self, car):
        car.init_param()
        params = car.params
        self.lf = float(params.lf)
        self.lr = float(params.lr)
        self.L = self.lf + self.lr
        self.Iz = float(params.Iz)
        self.m = float(params.m)

    def _tire_curve(self, slip):
        return self.TIRE_D * torch.sin(self.TIRE_C * torch.atan(self.TIRE_B * slip))

    def _motor_force(self, throttle, v_forward):
        return (
            self.MOTOR_GAIN
            * (throttle - v_forward / self.MOTOR_DRAG - self.MOTOR_OFFSET)
            * torch.where(
                v_forward < 0.0, torch.zeros_like(v_forward), torch.ones_like(v_forward)
            )
        )

    def derivative(self, x, u):
        """Return ``f(x, u)``; leading dimensions are broadcast freely."""
        _, _, heading, v_forward, v_sideway, omega = x.unbind(-1)
        steering, throttle = u.unbind(-1)

        # Kinematic branch (``v_forward < LOW_SPEED``).
        beta = torch.atan(self.lr / self.L * torch.tan(steering))
        omega_ref = v_forward * torch.cos(beta) / self.L * torch.tan(steering)
        kinematic = torch.stack(
            [
                v_forward * torch.cos(heading + beta),
                v_forward * torch.sin(heading + beta),
                omega_ref,
                self._motor_force(throttle, v_forward),
                -v_sideway,
                -(omega - omega_ref),
            ],
            dim=-1,
        )

        # Dynamic branch.  Only evaluated where unselected, so clamp the
        # denominator instead of dividing by ~0.
        v_safe = torch.clamp(v_forward, min=self.LOW_SPEED)
        slip_f = -torch.atan((omega * self.lf + v_sideway) / v_safe) + steering
        slip_r = torch.atan((omega * self.lr - v_sideway) / v_safe)
        mass_ratio = self.m * self.GRAVITY / self.L
        f_fy = self._tire_curve(slip_f) * mass_ratio * self.lr
        f_ry = self.REAR_FORCE_SCALE * self._tire_curve(slip_r) * mass_ratio * self.lf

        a_forward = self._motor_force(throttle, v_forward)
        a_sideway = (f_ry + f_fy - self.m * v_forward * omega) / self.m
        a_heading = (f_fy * self.lf - f_ry * self.lr) / self.Iz

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        dynamic = torch.stack(
            [
                v_forward * cos_h - v_sideway * sin_h,
                v_forward * sin_h + v_sideway * cos_h,
                omega,
                a_forward,
                a_sideway,
                a_heading,
            ],
            dim=-1,
        )

        low_speed = (v_forward < self.LOW_SPEED).unsqueeze(-1)
        return torch.where(low_speed, kinematic, dynamic)

    def step(self, x, u, dt):
        """One explicit-Euler step."""
        return x + dt * self.derivative(x, u)

    def rollout(self, x0, controls, dt):
        """Roll ``x0`` through ``controls`` of shape ``(K, N, 2)``.

        Returns the post-step states, shape ``(K, N, 6)``.
        """
        batch, horizon, _ = controls.shape
        x = x0.unsqueeze(0).expand(batch, -1).clone()
        states = []
        for k in range(horizon):
            x = self.step(x, controls[:, k, :], dt)
            states.append(x)
        if not states:
            return x.new_zeros((batch, 0, x.shape[-1]))
        return torch.stack(states, dim=1)
