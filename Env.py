'''
@Author: Qiong Liu, Ye Guo, Lirong Deng, Haotian Liu, Dongyu Li, Hongbin Sun, and Wenqi Huang
@Email: liuqiong_yl@outlook.com
@Description:
# code for paper
# @article{liu2022reducing,
#   title={Reducing Learning Difficulties: One-Step Two-Critic Deep Reinforcement Learning for Inverter-based Volt-Var Control},
#   author={Liu, Qiong and Guo, Ye and Deng, Lirong and Liu, Haotian and Li, Dongyu and Sun, Hongbin and Huang, Wenqi},
#   journal={arXiv preprint arXiv:2203.16289},
#   year={2022}
# }
'''

import copy
import random
from typing import Dict, List, Tuple
import pickle
from time import perf_counter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import pandapower as pp
from pandapower.pypower.idx_bus import VA, VM
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeYbus import makeYbus
try:
    # pandapower>=3.x
    from pandapower.converter.matpower import from_mpc
except Exception:  # pragma: no cover - compatibility fallback
    try:
        # older pandapower versions
        from pandapower.converter import from_mpc
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Cannot import Matpower converter 'from_mpc'. "
            "Please check your pandapower installation."
        ) from e
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd

try:
    import numba as _numba  # noqa: F401
    _PP_RUNPP_NUMBA = True
except Exception:
    # Avoid repeated pandapower warning spam when numba is not available.
    _PP_RUNPP_NUMBA = False

# Silence a noisy pandapower->pandas FutureWarning during Matpower conversion.
# This does NOT affect correctness for current pandas versions; it only prevents console spam.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Setting an item of incompatible dtype is deprecated.*",
    module=r"pandapower\.converter\.pypower\.from_ppc",
)


def Relu(x: np.ndarray):
    return np.maximum(0, x)


def ac_power_balance_residual(net) -> tuple[float, float]:
    """Return maximum nodal active/reactive AC mismatch in MW/MVAr."""
    ppc = net._ppc
    bus = ppc["bus"]
    ybus, _, _ = makeYbus(ppc["baseMVA"], bus, ppc["branch"])
    voltage = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
    specified = makeSbus(ppc["baseMVA"], bus, ppc["gen"])
    mismatch = voltage * np.conj(ybus.dot(voltage)) - specified
    return (
        float(np.max(np.abs(mismatch.real)) * ppc["baseMVA"]),
        float(np.max(np.abs(mismatch.imag)) * ppc["baseMVA"]),
    )


class grid_case:

    def __init__(self,
                 env_name,
                 load_pu: np.ndarray,
                 gene_pu: np.ndarray,
                 id_iber,
                 id_svc,
                 enable_pq_curve: bool = False,
                 pv_s_scale: float = 1.0,
                 pv_generation_scale: float = 1.0,
                 svc_q_scale: float = 1.0,
                 svc_absorption_ratio: float = 0.0,
                 cap_buses: List[int] | None = None,
                 cap_q_mvar: List[float] | None = None,
                 action_parameterization: str = "relative",
                 reference_pv_s_scale: float = 1.0,
                 reference_svc_q_scale: float = 1.0,
                 audit_power_balance: bool = False,
                 audit_physical_trace: bool = False):
        """Initializate."""
        self.id_iber = id_iber
        self.id_svc = id_svc
        self.env_name = env_name
        self.enable_pq_curve = bool(enable_pq_curve)
        self.pv_s_scale = float(pv_s_scale)
        self.pv_generation_scale = float(pv_generation_scale)
        self.svc_q_scale = float(svc_q_scale)
        self.svc_absorption_ratio = float(svc_absorption_ratio)
        if not 0.0 <= self.svc_absorption_ratio <= 1.0:
            raise ValueError("svc_absorption_ratio must lie in [0, 1]")
        if action_parameterization not in {"relative", "reference_mvar"}:
            raise ValueError("action_parameterization must be relative or reference_mvar")
        self.action_parameterization = action_parameterization
        self.reference_pv_s_scale = float(reference_pv_s_scale)
        self.reference_svc_q_scale = float(reference_svc_q_scale)
        self.audit_power_balance = bool(audit_power_balance)
        self.audit_physical_trace = bool(audit_physical_trace)
        self.last_p_balance_residual_mw = float("nan")
        self.last_q_balance_residual_mvar = float("nan")
        self.last_balance_audit_seconds = 0.0
        self.last_executed_trace: Dict[str, np.ndarray | float] = {}
        if self.env_name == 33:
            self.model = from_mpc(str(_resolve_case_file('case33_bw.mat')), f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 69:
            self.model = from_mpc(str(_resolve_case_file('case69.mat')), f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 118:
            self.model = from_mpc(str(_resolve_case_file('case1180zh.mat')), f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.action_dim = len(self.id_iber) + len(self.id_svc)
        self.step_n = 0
        self.done = False
        self.n_bus = len(self.model.bus)

        # 1.32
        self._idx_iber_sgen: List[int] = []
        self._idx_svc_sgen: List[int] = []
        for i in self.id_iber:
            idx = pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                                 max_p_mw=4, min_p_mw=0, max_q_mvar=2, min_q_mvar=-2, controllable=True)
            self._idx_iber_sgen.append(int(idx))
        for i in self.id_svc:
            idx = pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='SVC', scaling=1.0, in_service=True,
                                 max_p_mw=0, min_p_mw=0, max_q_mvar=2,
                                 min_q_mvar=-2 * self.svc_absorption_ratio, controllable=True)
            self._idx_svc_sgen.append(int(idx))
        self._svc_base_min_q = np.asarray(
            self.model.sgen.loc[self._idx_svc_sgen, "min_q_mvar"], dtype=float
        )
        self._svc_base_max_q = np.asarray(
            self.model.sgen.loc[self._idx_svc_sgen, "max_q_mvar"], dtype=float
        )

        # Optional: scale centralized device (SVC/SVG) reactive capacity
        if self.svc_q_scale != 1.0 and len(self._idx_svc_sgen) > 0:
            self.model.sgen.loc[self._idx_svc_sgen, "max_q_mvar"] = self.model.sgen.loc[self._idx_svc_sgen, "max_q_mvar"] * self.svc_q_scale
            self.model.sgen.loc[self._idx_svc_sgen, "min_q_mvar"] = self.model.sgen.loc[self._idx_svc_sgen, "min_q_mvar"] * self.svc_q_scale

        # Optional: distributed capacitors modeled as shunts (do NOT change action_dim)
        self._cap_idx: List[int] = []
        self._cap_q_mvar: List[float] = []
        if cap_buses is not None and cap_q_mvar is not None:
            if len(cap_buses) != len(cap_q_mvar):
                raise ValueError("cap_buses and cap_q_mvar must have the same length")
            for b, q in zip(cap_buses, cap_q_mvar):
                # Capacitor injects reactive power => negative shunt q_mvar in pandapower convention.
                idx = pp.create_shunt(self.model, bus=int(b), p_mw=0.0, q_mvar=-abs(float(q)), name="CAP")
                self._cap_idx.append(int(idx))
                self._cap_q_mvar.append(-abs(float(q)))

        # Precompute PV inverter apparent power ratings for P-Q capability curve
        self._pv_s_base_rated: np.ndarray | None = None
        self._pv_s_rated: np.ndarray | None = None
        if len(self._idx_iber_sgen) > 0:
            p_r = np.asarray(self.model.sgen.loc[self._idx_iber_sgen, "max_p_mw"], dtype=float)
            q_r = np.asarray(self.model.sgen.loc[self._idx_iber_sgen, "max_q_mvar"], dtype=float)
            self._pv_s_base_rated = np.sqrt(np.maximum(0.0, p_r * p_r + q_r * q_r))
            if self.enable_pq_curve:
                self._pv_s_rated = self._pv_s_base_rated * max(self.pv_s_scale, 0.0)

        pp.runpp(self.model, algorithm='bfsw', numba=_PP_RUNPP_NUMBA)
        self.observation_space = copy.deepcopy(np.hstack(
            (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar),
             np.zeros(self.action_dim))))

        # self.id_Gp = self.id_iber
        # self.id_Gp = [x-1 for x in self.id_Gp]
        # self.id_Gq = self.id_iber + self.id_svc
        # self.id_Gq = [x+self.n_bus-1-1 for x in self.id_Gq]

        self.id_bus_load = self.model.load.bus.values

        # self.observation_space = np.hstack((np.ones(33), np.zeros(33*2+4)))
        # np.zeros(self.net.mu.shape[1] + self.net.label_mu.shape[1] + len(self.id_Gq))
        # self.injection_pq = np.zeros(self.net.mu.shape[1])
        # self.injection_pq = np.hstack((self.model.res_bus.p_mw.values[1:], self.model.res_bus.q_mvar.values[1:]))
        self.action_space = np.zeros(self.action_dim)

        self.init_load_p_mw = copy.deepcopy(self.model.load.p_mw)
        self.init_load_q_mvar = copy.deepcopy(self.model.load.q_mvar)
        self.init_max_q_mavr = copy.deepcopy(self.model.sgen.max_q_mvar)
        self.init_min_q_mavr = copy.deepcopy(self.model.sgen.min_q_mvar)

        self.init_line_r_ohm_per_km = copy.deepcopy(self.model.line.r_ohm_per_km)
        self.init_line_x_ohm_per_km = copy.deepcopy(self.model.line.x_ohm_per_km)

        supplied_load = np.asarray(self.load_pu)
        supplied_generation = np.asarray(self.gene_pu)
        if supplied_load.ndim == 2 or supplied_generation.ndim == 2:
            if supplied_load.ndim != 2 or supplied_generation.ndim != 2:
                raise ValueError("load and generation profiles must both be one- or two-dimensional")
            if supplied_load.shape[0] != supplied_generation.shape[0]:
                raise ValueError("load and generation profiles must have equal time length")
            if supplied_load.shape[1] != len(self.init_load_p_mw):
                raise ValueError("pre-expanded load profile has the wrong number of load columns")
            if supplied_generation.shape[1] != len(self._idx_iber_sgen):
                raise ValueError("pre-expanded generation profile has the wrong inverter columns")
            self.load_pu = supplied_load.astype(float, copy=False)
            self.gene_pu = supplied_generation.astype(float, copy=False)
            return

        # The legacy training scripts expect time-series files:
        #   - two{n_bus}load.npy : (T, n_loads)
        #   - two{n_bus}gen.npy  : (T, n_inverters)
        # If they are missing, generate them from the provided 1-D profiles.
        load_path = Path(f"two{self.n_bus}load.npy")
        gen_path = Path(f"two{self.n_bus}gen.npy")

        if (not load_path.exists()) or (not gen_path.exists()):
            base_load = np.asarray(self.load_pu).reshape(-1)
            base_gen = np.asarray(self.gene_pu).reshape(-1)

            # Broadcast to per-load / per-inverter matrices
            load_mat = base_load[:, np.newaxis] * np.ones_like(self.init_load_p_mw)[np.newaxis, :]
            gen_mat = base_gen[:, np.newaxis] * np.ones((base_gen.shape[0], len(self.id_iber)), dtype=float)

            # Optional disturbance/noise (fixed RNG for reproducibility across seeds)
            rng = np.random.default_rng(0)
            t0 = min(load_mat.shape[0], 370 * 96)
            if t0 > 0:
                load_mat[:t0, :] = load_mat[:t0, :] * rng.uniform(low=0.8, high=1.2, size=load_mat[:t0, :].shape)
                gen_mat[:t0, :] = gen_mat[:t0, :] * rng.uniform(low=0.8, high=1.2, size=gen_mat[:t0, :].shape)

            np.save(str(load_path), load_mat)
            np.save(str(gen_path), gen_mat)

        self.load_pu = np.load(str(load_path))
        self.gene_pu = np.load(str(gen_path))

    def _update_pv_q_limits_from_p(self, p_mw: np.ndarray) -> None:
        """
        Update PV inverters (IBVR) min/max q_mvar using capability curve:
            |Q| <= sqrt(S^2 - P^2)
        Only applied when enable_pq_curve=True.
        """
        if (not self.enable_pq_curve) or (self._pv_s_rated is None) or (len(self._idx_iber_sgen) == 0):
            return
        p = np.asarray(p_mw, dtype=float).reshape(-1)
        s = np.asarray(self._pv_s_rated, dtype=float).reshape(-1)
        if p.shape[0] != s.shape[0]:
            # best-effort: broadcast if scalar
            if p.shape[0] == 1:
                p = np.ones_like(s) * float(p[0])
            else:
                return
        qmax = np.sqrt(np.maximum(0.0, s * s - p * p))
        self.model.sgen.loc[self._idx_iber_sgen, "max_q_mvar"] = qmax
        self.model.sgen.loc[self._idx_iber_sgen, "min_q_mvar"] = -qmax

    def _pv_generation_at(self, t: int) -> np.ndarray:
        """Return the PV injection profile at a time index under a penetration multiplier."""
        return np.asarray(self.gene_pu[int(t)], dtype=float) * self.pv_generation_scale

    def reset_at_step(self, step_n: int) -> np.ndarray:
        """Reset the electrical state to a specific time index without advancing the simulation."""
        max_step = min(len(self.load_pu), len(self.gene_pu)) - 1
        self.step_n = int(np.clip(int(step_n), 0, max_step))
        self.done = False
        t = self.step_n
        self.model.load.p_mw = self.load_pu[t] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu[t] * self.init_load_q_mvar
        pv_p = self._pv_generation_at(t)
        self.model.sgen.loc[self._idx_iber_sgen, "p_mw"] = pv_p
        if self.enable_pq_curve:
            self._update_pv_q_limits_from_p(pv_p)
        self.model.sgen.loc[self.model.sgen.index, "q_mvar"] = 0.0
        pp.runpp(self.model, algorithm='bfsw', numba=_PP_RUNPP_NUMBA)
        return np.hstack((
            np.array(self.model.res_bus.vm_pu),
            np.array(self.model.res_bus.p_mw),
            np.array(self.model.res_bus.q_mvar),
            np.zeros(self.action_dim),
        ))

    def set_capacitor_steps(self, cap_q_mvar: List[float]) -> None:
        """
        Set capacitor shunt injections (negative for capacitive). Length must match created caps.
        """
        if len(self._cap_idx) == 0:
            return
        if len(cap_q_mvar) != len(self._cap_idx):
            raise ValueError("cap_q_mvar length must match number of caps")
        qv = [-abs(float(q)) for q in cap_q_mvar]
        self.model.shunt.loc[self._cap_idx, "q_mvar"] = qv
        self._cap_q_mvar = qv


    def _action_mapping_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the physical-Q bounds used to interpret actor outputs."""
        low = np.array(self.model.sgen.min_q_mvar)
        high = np.array(self.model.sgen.max_q_mvar)
        mapping_low = low.copy()
        mapping_high = high.copy()
        if self.action_parameterization == "reference_mvar":
            if self.enable_pq_curve and self._pv_s_base_rated is not None:
                p = np.asarray(
                    self.model.sgen.loc[self._idx_iber_sgen, "p_mw"], dtype=float
                )
                reference_s = self._pv_s_base_rated * max(self.reference_pv_s_scale, 0.0)
                reference_q = np.sqrt(np.maximum(0.0, reference_s * reference_s - p * p))
                mapping_low[self._idx_iber_sgen] = -reference_q
                mapping_high[self._idx_iber_sgen] = reference_q
            if self._idx_svc_sgen:
                mapping_low[self._idx_svc_sgen] = (
                    self._svc_base_min_q * max(self.reference_svc_q_scale, 0.0)
                )
                mapping_high[self._idx_svc_sgen] = (
                    self._svc_base_max_q * max(self.reference_svc_q_scale, 0.0)
                )
        return mapping_low, mapping_high

    def physical_q_to_action(self, q_mvar: np.ndarray) -> np.ndarray:
        """Invert the configured action map for a representable physical-Q target."""
        mapping_low, mapping_high = self._action_mapping_bounds()
        scale = (mapping_high - mapping_low) / 2.0
        centre = mapping_low + scale
        q = np.asarray(q_mvar, dtype=float)
        if q.shape != scale.shape:
            raise ValueError("q_mvar must match the environment action dimension")
        action = np.zeros_like(q)
        movable = np.abs(scale) > 1e-12
        action[movable] = (q[movable] - centre[movable]) / scale[movable]
        return np.clip(action, -1.0, 1.0)

    def action_clip(self, action: np.ndarray) -> np.ndarray:
        """Map actor outputs in [-1, 1] to physical reactive-power commands."""
        low = np.array(self.model.sgen.min_q_mvar)
        high = np.array(self.model.sgen.max_q_mvar)
        mapping_low, mapping_high = self._action_mapping_bounds()

        scale_factor = (mapping_high - mapping_low) / 2
        reloc_factor = mapping_high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def step_model(self,
             action: np.ndarray,
             ):
        self.step_n = self.step_n + 1
        # Prepare injections at time t = step_n-1
        t = self.step_n - 1
        self.model.load.p_mw = self.load_pu[t] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu[t] * self.init_load_q_mvar
        pv_p = self._pv_generation_at(t)
        self.model.sgen.loc[self._idx_iber_sgen, "p_mw"] = pv_p
        if self.enable_pq_curve:
            self._update_pv_q_limits_from_p(pv_p)

        action = self.action_clip(action)
        # Use .loc to avoid pandas chained-assignment warnings (future pandas behavior change).
        self.model.sgen.loc[self.model.sgen.index, "q_mvar"] = action
        #
        # self.model.line.r_ohm_per_km = self.init_line_r_ohm_per_km*(1+0.2*(np.random.rand(1)-0.5))
        # self.model.line.x_ohm_per_km = self.init_line_x_ohm_per_km*(1+0.2*(np.random.rand(1)-0.5))

        pp.runpp(self.model, algorithm='bfsw', numba=_PP_RUNPP_NUMBA)
        if self.audit_power_balance:
            audit_start = perf_counter()
            (
                self.last_p_balance_residual_mw,
                self.last_q_balance_residual_mvar,
            ) = ac_power_balance_residual(self.model)
            self.last_balance_audit_seconds = perf_counter() - audit_start
        if self.audit_physical_trace:
            self.last_executed_trace = {
                "sgen_q_mvar": self.model.res_sgen.q_mvar.to_numpy(dtype=float).copy(),
                "ext_grid_q_mvar": float(self.model.res_ext_grid.q_mvar.sum()),
                "line_q_from_mvar": self.model.res_line.q_from_mvar.to_numpy(dtype=float).copy(),
                "line_q_to_mvar": self.model.res_line.q_to_mvar.to_numpy(dtype=float).copy(),
                "line_i_ka": self.model.res_line.i_ka.to_numpy(dtype=float).copy(),
            }
        violation_M = Relu(self.model.res_bus.vm_pu - 1.05).sum()
        violation_N = Relu(0.95-self.model.res_bus.vm_pu).sum()
        grid_loss = -self.model.res_line.pl_mw.sum()
        # grid_loss1 = self.model.res_bus.p_mw.sum()

        reward_p = grid_loss
        reward_v = -  violation_M - violation_N
        reward = np.array((reward_p, reward_v))

        violation = 0
        if (1 * (self.model.res_bus.vm_pu > 1.05).sum() + 1 * (self.model.res_bus.vm_pu < 0.95).sum()) > 0:
            violation = 1

        next_state = np.hstack( (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))
        voltage_M = self.model.res_bus.vm_pu.max()
        voltage_N =  self.model.res_bus.vm_pu.min()

        # new disturbance at time t+1 (keeps same q command)
        t2 = min(self.step_n, min(len(self.load_pu), len(self.gene_pu)) - 1)
        self.model.load.p_mw = self.load_pu[t2] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu[t2] * self.init_load_q_mvar
        pv_p2 = self._pv_generation_at(t2)
        self.model.sgen.loc[self._idx_iber_sgen, "p_mw"] = pv_p2
        if self.enable_pq_curve:
            self._update_pv_q_limits_from_p(pv_p2)
            # clamp q to new capability
            q_now = np.asarray(self.model.sgen.q_mvar, dtype=float)
            q_low = np.asarray(self.model.sgen.min_q_mvar, dtype=float)
            q_high = np.asarray(self.model.sgen.max_q_mvar, dtype=float)
            self.model.sgen.loc[self.model.sgen.index, "q_mvar"] = np.clip(q_now, q_low, q_high)
        else:
            self.model.sgen.loc[self.model.sgen.index, "q_mvar"] = action

        pp.runpp(self.model, algorithm='bfsw', numba=_PP_RUNPP_NUMBA)

        new_state = np.hstack((np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))

        return next_state, reward, self.done, violation, violation_M, violation_N,voltage_M,voltage_N, grid_loss, new_state



    def reset(self):
        self.done = False
        return self.observation_space
def _resolve_case_file(name: str) -> Path:
    """Resolve benchmark cases in both the development tree and public release."""
    module_dir = Path(__file__).resolve().parent
    candidates = (
        Path.cwd() / name,
        module_dir / name,
        module_dir.parent / "data" / "inputs" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Benchmark case not found: {name}")
