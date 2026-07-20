"""AC power-flow safety projection for normalised Volt-VAR actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp
from scipy.optimize import Bounds, LinearConstraint, minimize

import Env


@dataclass(frozen=True)
class ProjectionResult:
    action: np.ndarray
    intervened: bool
    success: bool
    iterations: int
    power_flow_previews: int
    raw_vmin: float
    raw_vmax: float
    projected_vmin: float
    projected_vmax: float
    correction_norm: float


class ACPowerFlowSafetyProjector:
    """Project an action with local AC sensitivities and nonlinear verification."""

    def __init__(
        self,
        lower_voltage: float = 0.9505,
        upper_voltage: float = 1.0495,
        finite_difference_step: float = 0.02,
        max_iterations: int = 3,
        optimiser_tolerance: float = 1e-9,
    ):
        if not lower_voltage < upper_voltage:
            raise ValueError("lower_voltage must be smaller than upper_voltage")
        if finite_difference_step <= 0.0 or max_iterations <= 0:
            raise ValueError("finite_difference_step and max_iterations must be positive")
        self.lower_voltage = float(lower_voltage)
        self.upper_voltage = float(upper_voltage)
        self.finite_difference_step = float(finite_difference_step)
        self.max_iterations = int(max_iterations)
        self.optimiser_tolerance = float(optimiser_tolerance)

    def _preview(self, env: Env.grid_case, action: np.ndarray) -> np.ndarray:
        q_mvar = env.action_clip(np.asarray(action, dtype=np.float64))
        env.model.sgen.loc[env.model.sgen.index, "q_mvar"] = q_mvar
        pp.runpp(env.model, algorithm="bfsw", numba=False)
        return np.asarray(env.model.res_bus.vm_pu, dtype=np.float64).copy()

    def _is_safe(self, voltage: np.ndarray) -> bool:
        return bool(
            np.all(voltage >= self.lower_voltage)
            and np.all(voltage <= self.upper_voltage)
        )

    def _violation(self, voltage: np.ndarray) -> float:
        under = max(0.0, self.lower_voltage - float(np.min(voltage)))
        over = max(0.0, float(np.max(voltage)) - self.upper_voltage)
        return under + over

    def _jacobian(
        self, env: Env.grid_case, action: np.ndarray, base_voltage: np.ndarray
    ) -> tuple[np.ndarray, int]:
        jacobian = np.zeros((base_voltage.size, action.size), dtype=np.float64)
        calls = 0
        for index in range(action.size):
            direction = 1.0 if action[index] <= 1.0 - self.finite_difference_step else -1.0
            perturbed = action.copy()
            perturbed[index] = np.clip(
                perturbed[index] + direction * self.finite_difference_step, -1.0, 1.0
            )
            displacement = perturbed[index] - action[index]
            if abs(displacement) < 1e-12:
                continue
            voltage = self._preview(env, perturbed)
            calls += 1
            jacobian[:, index] = (voltage - base_voltage) / displacement
        return jacobian, calls

    def project(self, env: Env.grid_case, policy_action: np.ndarray) -> ProjectionResult:
        raw_action = np.clip(np.asarray(policy_action, dtype=np.float64), -1.0, 1.0)
        current_action = raw_action.copy()
        current_voltage = self._preview(env, current_action)
        calls = 1
        raw_vmin = float(np.min(current_voltage))
        raw_vmax = float(np.max(current_voltage))
        if self._is_safe(current_voltage):
            return ProjectionResult(
                action=raw_action.astype(np.float32),
                intervened=False,
                success=True,
                iterations=0,
                power_flow_previews=calls,
                raw_vmin=raw_vmin,
                raw_vmax=raw_vmax,
                projected_vmin=raw_vmin,
                projected_vmax=raw_vmax,
                correction_norm=0.0,
            )

        best_action = current_action.copy()
        best_voltage = current_voltage.copy()
        best_violation = self._violation(current_voltage)
        completed_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration
            jacobian, jacobian_calls = self._jacobian(env, current_action, current_voltage)
            calls += jacobian_calls
            offset = jacobian @ current_action
            lower = self.lower_voltage - current_voltage + offset
            upper = self.upper_voltage - current_voltage + offset
            constraint = LinearConstraint(jacobian, lower, upper)

            result = minimize(
                lambda candidate: 0.5 * float(np.square(candidate - raw_action).sum()),
                current_action,
                jac=lambda candidate: candidate - raw_action,
                method="SLSQP",
                bounds=Bounds(-np.ones_like(raw_action), np.ones_like(raw_action)),
                constraints=[constraint],
                options={"ftol": self.optimiser_tolerance, "maxiter": 100, "disp": False},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                break
            candidate = np.clip(np.asarray(result.x, dtype=np.float64), -1.0, 1.0)
            voltage = self._preview(env, candidate)
            calls += 1
            violation = self._violation(voltage)
            if violation < best_violation:
                best_action, best_voltage, best_violation = candidate.copy(), voltage.copy(), violation
            if self._is_safe(voltage):
                best_action, best_voltage = candidate.copy(), voltage.copy()
                break
            current_action, current_voltage = candidate, voltage

        success = self._is_safe(best_voltage)
        return ProjectionResult(
            action=best_action.astype(np.float32),
            intervened=True,
            success=success,
            iterations=completed_iterations,
            power_flow_previews=calls,
            raw_vmin=raw_vmin,
            raw_vmax=raw_vmax,
            projected_vmin=float(np.min(best_voltage)),
            projected_vmax=float(np.max(best_voltage)),
            correction_norm=float(np.linalg.norm(best_action - raw_action)),
        )
