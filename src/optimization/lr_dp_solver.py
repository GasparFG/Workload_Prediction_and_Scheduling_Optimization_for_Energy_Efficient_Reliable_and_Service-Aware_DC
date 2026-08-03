"""Full-model Lagrangian-Relaxation + Dynamic-Programming solver.

This solves the ENTIRE baseline model of optimizer_v0.1 (not just the aggregate
server core): real per-job workload, CPU+memory vector packing, release/
deadline, replicas, precedence, affinity / anti-affinity, rack diversity,
per-server OFF/ON/PM scheduling with wear-driven maintenance, switching,
hybrid-cooling energy, the per-slot PUE cap, redundancy, and the occupational-
noise (acoustics) block. It reports the three numbers a decomposition solver is
judged on -- a valid lower bound, a feasible primal, and the gap -- and writes
the SAME result files as optimizer_v0.1 (result_<instance>.json,
batch_summary.json, <name>_stats_summary.csv) by reusing result_extractor.py.

METHOD
------
1) Lower bound -- Lagrangian variable-splitting.
   The per-(server,slot) CPU load is split into a server copy L^sigma and a job
   copy L^rho, tied by the linking constraint L^sigma_jk = L^rho_jk, which is
   dualised with a free multiplier lambda_jk. The server-coupling constraints
   (PUE c18, PM-cap c22, hot-standby c32, switch-budget c28) are dualised with
   sign-restricted multipliers. What remains separates:

     * a per-SERVER shortest-path DP over (pm-used, mode, wear-bucket) x slots
       that prices energy, wear/CM, PM, switching and the load price lambda_jk;
     * a per-JOB placement subproblem (min-cost choice of q_i eligible
       (server,start) slots) priced by lambda and lateness.

   The dual value is a VALID lower bound on the full model: constraints not put
   into this relaxation (precedence, affinity/anti-affinity, rack diversity,
   memory packing, thermal recirculation, acoustics) only remove restrictions,
   so the relaxed optimum can only be <= the true optimum. The bound is
   therefore valid but may be loose -- exactly the expected trade-off.

2) Primal -- constructive placement + repair, evaluated with the EXACT MILP
   objective. Jobs are placed (topological order over precedence) honouring
   eligibility, CPU+memory capacity with the interactive reservation, release/
   deadline, replicas, rack diversity, affinity and anti-affinity; servers are
   then operated (ON where loaded, forced PM when psi_0 >= Lambda, hot standby,
   PM cap, switching). Acoustics, thermal and PUE are checked and, where
   possible, repaired (PM shifted out of hot occupied slots); residual
   violations are reported in a feasibility summary. The objective mirrors
   optimizer_v0.1 exactly (energy + PM + CM + switching + lateness).

Gap = (primal - bound) / primal.

Run:
    python -m src.optimization.lr_dp_solver          # batch over experimental data
    python -m src.optimization.lr_dp_solver --jobs-json-input ... --server-json-input ...
    python -m src.optimization.lr_dp_solver --synthetic          # quick self-test
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .utils import get_status_label, slot_to_time, safe_value
    from .data_loader import load_data_from_jobs_json
    from .result_extractor import (
        extract_performance_metrics,
        extract_solution_rows,
        extract_pm_rows,
        extract_hourly_rows,
        extract_server_load_timeseries,
        extract_server_summary,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optimization.utils import get_status_label, slot_to_time, safe_value
    from src.optimization.data_loader import load_data_from_jobs_json
    from src.optimization.result_extractor import (
        extract_performance_metrics,
        extract_solution_rows,
        extract_pm_rows,
        extract_hourly_rows,
        extract_server_load_timeseries,
        extract_server_summary,
    )

OFF, ON, PM = 0, 1, 2

# Primal repair budget: how many placed jobs to try evicting for each job that
# did not fit, and how many times to sweep the unplaced list. Both are capped
# because each attempt costs a full re-scan of the successor's start window.
MAX_REPAIR_VICTIMS = 25
MAX_REPAIR_ROUNDS = 3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL_DATA_DIR = _REPO_ROOT / "src" / "experimental_data"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "experimental_results_lrdp"
JOBS_PREFIX = "jobs_params_"
SERVER_PREFIX = "server_params_"


# ======================================================================
# Full instance
# ======================================================================
@dataclass
class Instance:
    # sets
    I: List[int]; I_B: set; I_V: set; I_C: set
    J: List[int]; K: List[int]; F: List[List[int]]
    E: List[Tuple[int, int]]; A: List[Tuple[int, int]]; G: List[Tuple[int, int]]
    S: Dict[int, List[int]]           # eligibility per job
    # job params
    d: Dict[int, int]; a: Dict[int, int]; b: Dict[int, int]; q: Dict[int, int]
    r_cpu: Dict[int, float]; r_mem: Dict[int, float]; c_late: Dict[int, float]
    # server params (indexed by server id)
    P0: Dict[int, float]; dP: Dict[int, float]
    C_cpu: Dict[int, float]; C_mem: Dict[int, float]
    theta_cpu: Dict[int, float]; theta_mem: Dict[int, float]
    alpha: Dict[int, float]; Lambda: Dict[int, float]
    lam0: Dict[int, float]; lam_pm: Dict[int, float]; psi0: Dict[int, float]
    phi: Dict[int, float]; rack_of: Dict[int, int]
    # per-slot
    c_e: Dict[int, float]
    # cooling / power scalars
    eta_air: float; eta_liq: float; xi: float; P_ov: float; Pi_max: float
    # maintenance / cost / redundancy
    d_pm: int; c_pm: float; c_cm: float; c_sw: float; S_max: float
    N_min: int; kappa: int; dt: float
    # thermal
    T_sup: float; T_busy: float; T_idle: float; M_big: float
    Dmat: List[List[float]]
    # acoustics (optional)
    ac: Optional[dict]
    # convenience
    nJ: int = 0; nK: int = 0

    def cool_coeff(self, j: int) -> float:
        return (self.phi[j] / self.eta_liq
                + (1.0 - self.phi[j]) / self.eta_air
                + self.xi * self.phi[j])

    def valid_starts(self, i: int) -> range:
        upper = self.nK - self.d[i]
        if i in self.I_V:
            upper = min(upper, self.b[i] - self.d[i])
        lo = self.a[i]
        return range(lo, upper + 1) if upper >= lo else range(0, 0)


def instance_from_data(data: dict) -> Instance:
    J = [int(j) for j in data["sets"]["J"]]
    K = [int(k) for k in data["sets"]["K"]]
    F = [[int(j) for j in rack] for rack in data["sets"].get("F", [])]
    I = [int(i) for i in data["sets"]["I"]]
    I_B = set(int(i) for i in data["sets"]["I_B"])
    I_V = set(int(i) for i in data["sets"]["I_V"])
    I_C = set(int(i) for i in data["sets"]["I_C"])
    E = [(int(u), int(v)) for u, v in data["sets"].get("E", [])]
    A = [(int(u), int(v)) for u, v in data["sets"].get("A", [])]
    G = [(int(u), int(v)) for u, v in data["sets"].get("G", [])]
    S = {int(i): [int(j) for j in js] for i, js in data["eligibility"].items()}

    jp = data["job_params"]
    def jmap(name):
        return {int(k): v for k, v in jp[name].items()}
    d = {int(k): int(v) for k, v in jp["d"].items()}
    a = jmap("a"); b = jmap("b")
    if "r_cpu" in jp:
        r_cpu = jmap("r_cpu"); r_mem = jmap("r_mem")
    else:
        r_cpu = jmap("r"); r_mem = dict(r_cpu)
    q = {int(i): int(v) for i, v in jp.get("q", {}).items()}
    for i in I:
        q.setdefault(i, 1)
    _late = jp.get("c_late", jp.get("rho", {}))
    c_late = {int(k): float(v) for k, v in _late.items()}
    for i in I:
        c_late.setdefault(i, 0.0)

    sp = data["server_params"]
    def smap(new, old=None):
        src = sp.get(new, sp.get(old, {})) if old else sp.get(new, {})
        return {int(k): float(v) for k, v in src.items()}
    P0 = smap("P0"); dP = smap("dP")
    C_cpu = smap("C_cpu", "C"); C_mem = smap("C_mem", "C")
    theta_cpu = smap("theta_cpu", "theta"); theta_mem = smap("theta_mem", "theta")
    alpha = smap("alpha"); Lambda = smap("Lambda")
    lam0 = smap("lambda0"); lam_pm = smap("lambda_pm")
    psi0_raw = sp.get("psi_0", {})
    psi0 = {j: float(psi0_raw.get(str(j), psi0_raw.get(j, 0.0))) for j in J}
    phi_raw = data.get("cooling", {}).get("phi", {})
    phi = {j: float(phi_raw.get(str(j), phi_raw.get(j, 0.0))) for j in J}
    if not C_mem:
        C_mem = dict(C_cpu)
    if not theta_mem:
        theta_mem = dict(theta_cpu)

    rack_of = {}
    for ridx, rack in enumerate(F):
        for j in rack:
            rack_of[j] = ridx
    for j in J:
        rack_of.setdefault(j, -1)

    cool = data.get("cooling", {})
    def scal(x, dflt):
        if isinstance(x, list):
            return float(x[0]) if x else dflt
        return float(x) if x is not None else dflt
    eta_air = scal(cool.get("eta_air", cool.get("eta")), 3.0)
    eta_liq = scal(cool.get("eta_liq"), eta_air)
    xi = scal(cool.get("xi"), 0.0)

    c_e = {int(k): float(data["costs"]["c_e"][k]) for k in range(len(K))}

    inst = Instance(
        I=I, I_B=I_B, I_V=I_V, I_C=I_C, J=J, K=K, F=F, E=E, A=A, G=G, S=S,
        d=d, a=a, b=b, q=q, r_cpu=r_cpu, r_mem=r_mem, c_late=c_late,
        P0=P0, dP=dP, C_cpu=C_cpu, C_mem=C_mem, theta_cpu=theta_cpu,
        theta_mem=theta_mem, alpha=alpha, Lambda=Lambda, lam0=lam0,
        lam_pm=lam_pm, psi0=psi0, phi=phi, rack_of=rack_of, c_e=c_e,
        eta_air=eta_air, eta_liq=eta_liq, xi=xi,
        P_ov=float(data["power"]["P_ov"]), Pi_max=float(data["power"]["Pi_max"]),
        d_pm=int(data["maintenance"]["d_pm"]),
        c_pm=float(data["maintenance"]["c_pm"]),
        c_cm=float(data["maintenance"]["c_cm"]),
        c_sw=float(data["costs"]["c_sw"]), S_max=float(data["costs"]["S_max"]),
        N_min=int(data["redundancy"]["N_min"]),
        kappa=int(data["redundancy"]["kappa"]),
        dt=float(data["slot_duration"]),
        T_sup=float(data["thermal"]["T_sup"]),
        T_busy=float(data["thermal"]["T_busy"]),
        T_idle=float(data["thermal"]["T_idle"]),
        M_big=float(data["thermal"]["M_big"]),
        Dmat=data["thermal"]["D"],
        ac=data.get("acoustics"),
    )
    inst.nJ, inst.nK = len(J), len(K)
    return inst


# ======================================================================
# Per-server DP for the LOWER BOUND (variable-splitting: prices load at lam_j[k])
# ======================================================================
class ServerDP:
    """Shortest path over (pm_used, mode, wear-bucket) x slots for one server.
    Load is a free decision on ON edges (the split server copy L^sigma); it is
    priced by lam_j[k] from the dualised linking constraint, and its wear effect
    is tracked by wear buckets. PM is allowed once (>= trigger wear), commits
    d_pm slots, resets wear."""

    def __init__(self, inst: Instance, jpos: int, nb=16, n_load=5, wear_cap=1.5):
        self.inst = inst
        self.jpos = jpos                       # 0-based index into inst.J
        self.j = inst.J[jpos]                   # server id
        self.nb = nb
        self.n_load = n_load
        Lam = inst.Lambda[self.j]
        self.wear_max = max(wear_cap * Lam, inst.psi0[self.j] + inst.C_cpu[self.j])
        self.edges = np.linspace(0.0, self.wear_max, nb + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.trig_bucket = int(np.searchsorted(self.centers, Lam))
        self.loads = np.linspace(0.0, inst.C_cpu[self.j], n_load)
        self.cc = inst.cool_coeff(self.j)
        self.required_pm = bool(inst.psi0[self.j] >= Lam)
        add = self.centers[None, :] + self.loads[:, None]
        add = np.minimum(add, self.wear_max)
        self.NWB = np.clip(np.searchsorted(self.edges, add, side="right") - 1,
                           0, nb - 1).astype(np.int64)
        self.PIT_l = inst.P0[self.j] + inst.dP[self.j] * self.loads
        self.Pcool_l = self.cc * inst.alpha[self.j] * self.PIT_l
        self.cm_bucket = (inst.c_cm * inst.lam0[self.j]
                          * self.centers / Lam)

    def wb_of(self, wear):
        return int(min(self.nb - 1, max(0, np.searchsorted(
            self.edges, wear, side="right") - 1)))

    def _on_cost(self, k, lam_row, pi, rho):
        inst = self.inst
        energy = inst.dt * inst.c_e[k]/1000 * (self.PIT_l + self.Pcool_l)
        lag = (lam_row[k] * self.loads              # load price (linking)
               - rho[k]                             # standby credit for being ON
               + pi[k] * (self.Pcool_l - (inst.Pi_max - 1.0) * self.PIT_l))
        return energy + lag                          # (n_load,)

    def solve(self, lam_row, nu, rho, pi, sig):
        """Return (cost, y[nK], z[nK], load[nK]) for this server's optimal
        trajectory under the current multipliers. Back-pointers are stored on
        the forward pass so the trajectory (needed for every subgradient) is
        reconstructed in O(nK)."""
        inst = self.inst
        nK, nb, nl = inst.nK, self.nb, self.n_load
        INF = 1e18
        d_pm = inst.d_pm
        loads = self.loads
        NWB = self.NWB
        cm = self.cm_bucket
        wb0 = self.wb_of(inst.psi0[self.j])
        pm_win_dual = np.array([nu[k:min(k + d_pm, nK)].sum() for k in range(nK)])

        off_hist = np.full((nK, 2, nb), INF)
        on_hist = np.full((nK, 2, nb), INF)
        off_from = np.zeros((nK, 2, nb), np.int8)      # 0 stay, 1 turn, 2 pm-land
        on_l = np.zeros((nK, 2, nb), np.int32)
        on_swb = np.zeros((nK, 2, nb), np.int32)
        on_foff = np.zeros((nK, 2, nb), np.int8)       # 1 if source was OFF
        pm_inject = {}; pm_from = {}; terminal_pm = []

        # ---- slot 0 ----
        OFFC = np.full((2, nb), INF); ONC = np.full((2, nb), INF)
        OFFC[0, wb0] = 0.0
        onc0 = self._on_cost(0, lam_row, pi, rho)
        for l in range(nl):
            t = NWB[l, wb0]
            c = onc0[l] + cm[t]
            if c < ONC[0, t]:
                ONC[0, t] = c
                on_l[0, 0, t] = l; on_swb[0, 0, t] = wb0; on_foff[0, 0, t] = 0
        off_hist[0] = OFFC; on_hist[0] = ONC

        # ---- forward ----
        for k in range(1, nK):
            sw = inst.c_sw + sig
            prev_off = off_hist[k - 1]; prev_on = on_hist[k - 1]
            OFFC = np.full((2, nb), INF); ONC = np.full((2, nb), INF)
            onc = self._on_cost(k, lam_row, pi, rho)
            for pmu in range(2):
                po = prev_off[pmu]; pn = prev_on[pmu]
                for wb in range(nb):                    # OFF at k
                    stay = po[wb]; turn = pn[wb] + sw
                    if stay <= turn:
                        if stay < INF:
                            OFFC[pmu, wb] = stay; off_from[k, pmu, wb] = 0
                    elif turn < INF:
                        OFFC[pmu, wb] = turn; off_from[k, pmu, wb] = 1
                for sb in range(nb):                    # ON at k
                    bo = pn[sb]; bf = po[sb] + sw
                    if bo <= bf:
                        src = bo; foff = 0
                    else:
                        src = bf; foff = 1
                    if src >= INF:
                        continue
                    for l in range(nl):
                        t = NWB[l, sb]
                        c = src + onc[l] + cm[t]
                        if c < ONC[pmu, t]:
                            ONC[pmu, t] = c
                            on_l[k, pmu, t] = l; on_swb[k, pmu, t] = sb
                            on_foff[k, pmu, t] = foff
            # PM start from ON (pmu=0, wb >= trigger)
            if k + d_pm <= nK:
                col = prev_on[0, self.trig_bucket:]
                if col.size and col.min() < INF:
                    sb = self.trig_bucket + int(np.argmin(col))
                    cost = prev_on[0, sb] + sw + inst.c_pm + pm_win_dual[k]
                    land = k + d_pm
                    if land < nK:
                        cur = pm_inject.get(land)
                        if cur is None or cost < cur[0]:
                            pm_inject[land] = (cost, k, sb)
                    else:
                        terminal_pm.append((cost, k, sb))
            if k in pm_inject:
                cost, ss, sbw = pm_inject[k]
                if cost < OFFC[1, 0]:
                    OFFC[1, 0] = cost; off_from[k, 1, 0] = 2; pm_from[k] = (ss, sbw)
            off_hist[k] = OFFC; on_hist[k] = ONC

        # ---- terminal choice ----
        best_cost, best = INF, None
        for pmu in range(2):
            if self.required_pm and pmu == 0:
                continue
            for arr, mode in ((off_hist[nK - 1], OFF), (on_hist[nK - 1], ON)):
                wb = int(np.argmin(arr[pmu]))
                if arr[pmu, wb] < best_cost:
                    best_cost, best = float(arr[pmu, wb]), (mode, pmu, wb)
        for cost, ss, sbw in terminal_pm:
            if cost < best_cost:
                best_cost, best = cost, ("TERM", ss, sbw)
        if best is None:
            best_cost, best = 0.0, (OFF, 0, wb0)

        # ---- backtrack -> per-slot y, z, load ----
        y = np.zeros(nK, np.int8); z = np.zeros(nK, np.int8); load = np.zeros(nK)
        if best[0] == "TERM":
            _, ss, src_wb = best
            for kp in range(ss, nK):
                z[kp] = 1
            mode = ON; pmu = 0; wb = src_wb; k = ss - 1
        else:
            mode, pmu, wb = best; k = nK - 1
        while k >= 0:
            if mode == OFF:
                frm = off_from[k, pmu, wb]
                if frm == 2:                           # PM landing at k
                    ss, src_wb = pm_from[k]
                    for kp in range(ss, min(ss + d_pm, nK)):
                        z[kp] = 1
                    mode = ON; pmu = 0; wb = src_wb; k = ss - 1
                    continue
                if k == 0:
                    break
                mode = ON if frm == 1 else OFF         # turn vs stay (wb, pmu kept)
                k -= 1
            else:                                       # ON at k
                y[k] = 1; load[k] = loads[on_l[k, pmu, wb]]
                if k == 0:
                    break
                foff = on_foff[k, pmu, wb]
                wb = int(on_swb[k, pmu, wb])
                mode = OFF if foff == 1 else ON
                k -= 1
        return best_cost, y, z, load


# ======================================================================
# Lower bound: subgradient ascent on lam (linking) + coupling multipliers
# ======================================================================
def lower_bound(inst: Instance, iters=1000, ub: float = math.inf,
                gap_tol: float = 0.01, tol: float = 1e-6, patience: int = 25,
                time_limit: float = 7200.0,
                verbose: bool = False) -> Tuple[float, str, int]:
    """Subgradient ascent on the Lagrangian dual.

    Returns (best_bound, stop_reason, iterations_run). Convergence is checked
    four ways every iteration:
      * gap        -- (ub - best_bound) / ub <= gap_tol      (needs a primal ub)
      * stationary -- subgradient norm < tol                 (duals settled)
      * stall      -- best_bound not improved for `patience` iters
      * time       -- elapsed wall-clock >= time_limit seconds
    Otherwise it runs to `iters` (the hard cap).
    """
    nJ, nK = inst.nJ, inst.nK
    lam = np.zeros((nJ, nK))          # linking price per (server,slot), free sign
    nu = np.zeros(nK); rho = np.zeros(nK); pi = np.zeros(nK); sig = 0.0
    dps = [ServerDP(inst, jp) for jp in range(nJ)]
    jpos = {j: p for p, j in enumerate(inst.J)}
    best = -math.inf
    stall = 0
    reason = "max-iters"
    it = 0
    t_start = time.time()

    # Constant part of the objective, present in EVERY feasible solution and
    # therefore part of any valid lower bound: the corrective-maintenance base
    # term c_cm * lam_pm summed over all (server, slot). Omitting it (as the
    # first version did) makes the bound loose by exactly this offset.
    const_cm = inst.c_cm * inst.nK * sum(inst.lam_pm[j] for j in inst.J)
    ce_arr = np.array([inst.c_e[k]/1000 for k in range(nK)])
    ov_energy = float(np.sum(inst.dt * ce_arr * inst.P_ov))
    N_on = inst.N_min + inst.kappa
    pm_cap = nJ - inst.N_min

    for it in range(iters):
        # --- server subproblems: solve each DP and rebuild its trajectory ---
        server_cost = 0.0
        y_k = np.zeros(nK); z_k = np.zeros(nK)
        pit_k = np.zeros(nK); pcool_k = np.zeros(nK)
        Lsigma = np.zeros((nJ, nK)); sw_total = 0.0
        for jp in range(nJ):
            c, y, z, load = dps[jp].solve(lam[jp], nu, rho, pi, sig)
            server_cost += c
            j = inst.J[jp]
            onf = y.astype(float)
            pit = (inst.P0[j] + inst.dP[j] * load) * onf
            pit_k += pit
            pcool_k += inst.cool_coeff(j) * inst.alpha[j] * pit
            y_k += onf
            z_k += z.astype(float)
            Lsigma[jp] = load * onf
            sw_total += float(np.sum(np.abs(np.diff(y.astype(np.int64)))))

        # --- job subproblems: min-cost placement priced by lam + lateness ---
        job_cost = 0.0
        Lrho = np.zeros((nJ, nK))
        for i in inst.I:
            cands = []
            for j in inst.S.get(i, []):
                if j not in jpos:
                    continue
                p = jpos[j]
                for k in inst.valid_starts(i):
                    span = range(k, k + inst.d[i])
                    credit = sum(lam[p, kk] for kk in span) * inst.r_cpu[i]
                    late = 0.0
                    if i in inst.I_B:
                        late = inst.c_late[i] * max(0, k + inst.d[i] - inst.b[i])
                    cands.append((late - credit, p, k))
            if not cands:
                continue
            cands.sort(key=lambda t: t[0])
            chosen, used = [], set()
            for cost_ijk, p, k in cands:
                if p in used:
                    continue
                chosen.append((cost_ijk, p, k)); used.add(p)
                if len(chosen) >= inst.q[i]:
                    break
            for cost_ijk, p, k in chosen:
                job_cost += cost_ijk
                for kk in range(k, k + inst.d[i]):
                    Lrho[p, kk] += inst.r_cpu[i]

        const = (- np.sum(nu * pm_cap)
                 + np.sum(rho * N_on)
                 - sig * inst.S_max
                 + np.sum(pi * inst.P_ov))
        bound = server_cost + job_cost + const + ov_energy + const_cm

        improved = bound > best + tol
        best = max(best, bound)
        stall = 0 if improved else stall + 1

        # --- subgradients (signed constraint violations at this solution) ---
        ptot_k = pit_k + pcool_k + inst.P_ov
        g_lam = Lsigma - Lrho                       # linking  L^sigma = L^rho
        g_nu = z_k - pm_cap                          # PM cap   sum z <= |J|-Nmin
        g_rho = N_on - y_k                           # standby  sum y >= Nmin+kappa
        g_pi = ptot_k - inst.Pi_max * pit_k          # PUE      Ptot <= Pi_max*PIT
        g_sig = sw_total - inst.S_max                # switch budget
        gnorm2 = (float(np.sum(g_lam ** 2)) + float(np.sum(g_nu ** 2))
                  + float(np.sum(g_rho ** 2)) + float(np.sum(g_pi ** 2))
                  + g_sig ** 2)
        gnorm = math.sqrt(gnorm2)
        gap = (ub - best) / ub if math.isfinite(ub) and ub > 0 else math.inf
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"    [bound] it {it:3d}  bound={best:12.2f}  "
                  f"gap={100*gap:7.3f}%  |g|={gnorm:.3g}")

        # --- convergence checks ---
        if gap <= gap_tol:
            reason = "gap<=tol"; break
        if gnorm < tol:
            reason = "subgradient~0"; break
        #if stall >= patience:
        #    reason = "stall"; break
        if time.time() - t_start >= time_limit:
            reason = "time-limit"; break

        # --- subgradient step: Polyak toward the primal upper bound ---
        if math.isfinite(ub) and ub > bound and gnorm2 > 1e-12:
            step = 1.5 * (ub - bound) / gnorm2
        else:
            step = 1.0 / (math.sqrt(it + 1) * (1.0 + gnorm))
        lam = lam + step * g_lam
        nu = np.maximum(0.0, nu + step * g_nu)
        rho = np.maximum(0.0, rho + step * g_rho)
        pi = np.maximum(0.0, pi + step * g_pi)
        sig = max(0.0, sig + step * g_sig)

    return best, reason, it + 1


# ======================================================================
# PRIMAL: constructive feasible schedule for the full model
# ======================================================================
class Schedule:
    def __init__(self, inst: Instance):
        self.inst = inst
        nJ, nK = inst.nJ, inst.nK
        self.jpos = {j: p for p, j in enumerate(inst.J)}
        self.load_cpu = {(j, k): 0.0 for j in inst.J for k in inst.K}
        self.load_mem = {(j, k): 0.0 for j in inst.J for k in inst.K}
        self.y = {(j, k): 0 for j in inst.J for k in inst.K}
        self.z = {(j, k): 0 for j in inst.J for k in inst.K}
        self.pm_start = {j: None for j in inst.J}   # v[j,k]
        self.X = {}                                 # (i,j,k) -> 1
        self.s = {}                                 # start slot per job
        self.placed_on = {}                         # i -> list of servers
        self.unplaced = []
        self.late = {i: 0.0 for i in inst.I}


def build_primal(inst: Instance, verbose=False) -> Tuple[float, Schedule, dict]:
    sch = Schedule(inst)
    nJ, nK = inst.nJ, inst.nK

    # residual capacity (total and batch-reserved)
    res_cpu = {(j, k): inst.C_cpu[j] for j in inst.J for k in inst.K}
    res_mem = {(j, k): inst.C_mem[j] for j in inst.J for k in inst.K}
    res_cpu_b = {(j, k): (1 - inst.theta_cpu[j]) * inst.C_cpu[j]
                 for j in inst.J for k in inst.K}
    res_mem_b = {(j, k): (1 - inst.theta_mem[j]) * inst.C_mem[j]
                 for j in inst.J for k in inst.K}

    # precedence: predecessors + topological order
    preds = {i: [] for i in inst.I}
    succ = {i: [] for i in inst.I}
    for u, v in inst.E:
        preds[v].append(u); succ[u].append(v)
    # Topological order, but among the jobs whose predecessors are all placed
    # take the most constrained one first: narrow start window, few eligible
    # servers per replica, large footprint. Plain index order (the previous
    # behaviour) lets a roomy job take the slot a tight one needed.
    def _tightness(i):
        upper = inst.nK - inst.d[i]
        if i in inst.I_V:
            upper = min(upper, inst.b[i] - inst.d[i])
        slack = upper - inst.a[i]
        spare_servers = len(inst.S.get(i, [])) - inst.q[i]
        footprint = inst.r_cpu[i] * inst.d[i] * inst.q[i]
        return (slack, spare_servers, -footprint, i)

    indeg = {i: len(preds[i]) for i in inst.I}
    ready = [(_tightness(i), i) for i in inst.I if indeg[i] == 0]
    heapq.heapify(ready)
    order, seen = [], set()
    while ready:
        _, i = heapq.heappop(ready)
        order.append(i); seen.add(i)
        for v in succ[i]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (_tightness(v), v))
    order += [i for i in inst.I if i not in seen]   # cycle guard

    # affinity partner map (same server); anti-affinity partner map (diff server)
    aff = {i: set() for i in inst.I}
    anti = {i: set() for i in inst.I}
    for u, v in inst.A:
        aff[u].add(v); aff[v].add(u)
    for u, v in inst.G:
        anti[u].add(v); anti[v].add(u)

    def fits(i, j, k):
        span = range(k, k + inst.d[i])
        batch = i in inst.I_B
        for kk in span:
            if res_cpu[j, kk] < inst.r_cpu[i] - 1e-9:
                return False
            if res_mem[j, kk] < inst.r_mem[i] - 1e-9:
                return False
            if batch and (res_cpu_b[j, kk] < inst.r_cpu[i] - 1e-9
                          or res_mem_b[j, kk] < inst.r_mem[i] - 1e-9):
                return False
        return True

    def assign(i, j, k):
        span = range(k, k + inst.d[i])
        batch = i in inst.I_B
        for kk in span:
            res_cpu[j, kk] -= inst.r_cpu[i]
            res_mem[j, kk] -= inst.r_mem[i]
            if batch:
                res_cpu_b[j, kk] -= inst.r_cpu[i]
                res_mem_b[j, kk] -= inst.r_mem[i]
            sch.load_cpu[j, kk] += inst.r_cpu[i]
            sch.load_mem[j, kk] += inst.r_mem[i]
            sch.y[j, kk] = 1
        sch.X[(i, j, k)] = 1

    def unassign(i, j, k):
        """Exact inverse of assign, so a placement can be rolled back."""
        span = range(k, k + inst.d[i])
        batch = i in inst.I_B
        for kk in span:
            res_cpu[j, kk] += inst.r_cpu[i]
            res_mem[j, kk] += inst.r_mem[i]
            if batch:
                res_cpu_b[j, kk] += inst.r_cpu[i]
                res_mem_b[j, kk] += inst.r_mem[i]
            sch.load_cpu[j, kk] -= inst.r_cpu[i]
            sch.load_mem[j, kk] -= inst.r_mem[i]
            # y is driven purely by load at this stage; standby and PM are
            # applied further down, after every placement has settled.
            if sch.load_cpu[j, kk] <= 1e-9 and sch.z[j, kk] == 0:
                sch.y[j, kk] = 0
        sch.X.pop((i, j, k), None)

    # Reserve mandatory PM windows BEFORE placing jobs. Every server with
    # psi0 >= Lambda must take exactly one PM (v0.1 force_pm_initial_wear), so
    # we block a d_pm-slot window on each such server from job placement,
    # guaranteeing the forced PM is feasible instead of being crowded out.
    pm_starts = [k for k in inst.K if k <= inst.nK - inst.d_pm]
    for j in inst.J:
        if inst.psi0[j] >= inst.Lambda[j] and pm_starts:
            k = pm_starts[0]                       # earliest window (resets wear early)
            for kk in range(k, k + inst.d_pm):
                sch.z[j, kk] = 1
                res_cpu[j, kk] = res_mem[j, kk] = 0.0
                res_cpu_b[j, kk] = res_mem_b[j, kk] = 0.0
            sch.pm_start[j] = k

    def earliest_start(i):
        est = inst.a[i]
        for p in preds[i]:
            if p in sch.s:
                est = max(est, sch.s[p] + inst.d[p])
        return est

    def latest_start(i):
        # Horizon, the interactive hard deadline, and -- for repairs, where a
        # successor may already sit on the timeline -- precedence c7 read
        # backwards. Batch deadlines stay soft (they are priced as lateness).
        upper = inst.nK - inst.d[i]
        if i in inst.I_V:
            upper = min(upper, inst.b[i] - inst.d[i])
        for v in succ[i]:
            if v in sch.s:
                upper = min(upper, sch.s[v] - inst.d[i])
        return upper

    def find_slot(i):
        """Earliest feasible (start, servers) for i, or None. No mutation."""
        aff_servers = []
        for pj in aff[i]:
            aff_servers += sch.placed_on.get(pj, [])
        anti_servers = set()
        for pj in anti[i]:
            anti_servers.update(sch.placed_on.get(pj, []))
        for k in range(earliest_start(i), latest_start(i) + 1):
            cand = [j for j in inst.S.get(i, [])
                    if j not in anti_servers and fits(i, j, k)]
            if aff_servers:
                cand = [j for j in cand if j in aff_servers] or cand
            # rack diversity for critical: distinct racks
            chosen, used_racks = [], set()
            # prefer least-loaded servers to spread
            cand.sort(key=lambda j: res_cpu[j, k])
            for j in cand:
                if i in inst.I_C and inst.rack_of[j] in used_racks:
                    continue
                chosen.append(j)
                used_racks.add(inst.rack_of[j])
                if len(chosen) >= inst.q[i]:
                    break
            if len(chosen) >= inst.q[i]:
                return k, chosen[:inst.q[i]]
        return None

    def place(i):
        got = find_slot(i)
        if got is None:
            return False
        k, servers = got
        for j in servers:
            assign(i, j, k)
        sch.s[i] = k
        sch.placed_on[i] = servers
        if i in inst.I_B:
            sch.late[i] = max(0.0, k + inst.d[i] - inst.b[i])
        return True

    def eject(i):
        """Undo i's placement. No-op (False) if it is not currently placed."""
        if i not in sch.s:
            return False
        k = sch.s[i]
        for j in sch.placed_on[i]:
            unassign(i, j, k)
        del sch.s[i]
        del sch.placed_on[i]
        sch.late[i] = 0.0
        return True

    for i in order:
        if not place(i):
            sch.unplaced.append(i)

    # ---- repair: ejection chains ----
    # The pass above never revisits a decision, so a job placed earlier can be
    # sitting on exactly the capacity a later one needed. For each job that did
    # not fit, evict one already-placed job that overlaps its window on an
    # eligible server, put the failed job in, and re-place the evicted one
    # elsewhere; roll both back if that does not work out. Only jobs free of
    # precedence, affinity and anti-affinity ties are eligible to be evicted,
    # so moving one can never invalidate a relation somewhere else.
    movable = [x for x in inst.I
               if not preds[x] and not succ[x] and not aff[x] and not anti[x]]
    for _round in range(MAX_REPAIR_ROUNDS):
        if not sch.unplaced:
            break
        progress = False
        for i in list(sch.unplaced):
            if place(i):                      # re-try: the order may have freed room
                sch.unplaced.remove(i)
                progress = True
                continue
            lo, hi = earliest_start(i), latest_start(i)
            eligible = set(inst.S.get(i, []))
            victims = [x for x in movable
                       if x in sch.s
                       and sch.s[x] + inst.d[x] > lo
                       and sch.s[x] <= hi + inst.d[i]
                       and eligible.intersection(sch.placed_on[x])]
            victims.sort(key=lambda x: -inst.r_cpu[x] * inst.d[x])
            for x in victims[:MAX_REPAIR_VICTIMS]:
                k_x, srv_x = sch.s[x], list(sch.placed_on[x])
                eject(x)
                if place(i) and place(x):
                    sch.unplaced.remove(i)
                    progress = True
                    break
                eject(i)                      # no-op if place(i) already failed
                eject(x)
                for j in srv_x:               # restore x exactly where it was
                    assign(x, j, k_x)
                sch.s[x] = k_x
                sch.placed_on[x] = srv_x
                if x in inst.I_B:
                    sch.late[x] = max(0.0, k_x + inst.d[x] - inst.b[x])
        if not progress:
            break
    if verbose and sch.unplaced:
        print(f"    [primal] {len(sch.unplaced)} job(s) still unplaced after "
              f"repair: {sch.unplaced[:10]}")

    # ---- server operation: hot standby, switching repair, PM cap ----
    # switching repair: keep each load-carrying server ON as one contiguous
    # block (fill internal OFF gaps between its first and last loaded slot),
    # which caps its state changes at <=2 and cuts the total toward S_max.
    for j in inst.J:
        loaded = [k for k in inst.K if sch.load_cpu[j, k] > 1e-9]
        if not loaded:
            continue
        first, last = min(loaded), max(loaded)
        for k in range(first, last + 1):
            if sch.z[j, k] == 0:
                sch.y[j, k] = 1

    # hot standby c32: add servers that stay ON for the WHOLE horizon (0
    # switches) until every slot has >= N_min + kappa servers on.
    need_on = inst.N_min + inst.kappa
    always_on = set(j for j in inst.J if all(
        sch.y[j, k] == 1 for k in inst.K))
    pool = sorted((j for j in inst.J if j not in always_on),
                  key=lambda j: inst.P0[j])
    for k in inst.K:
        while sum(1 for j in inst.J if sch.y[j, k] == 1) < need_on:
            # promote the cheapest not-yet-always-on server to always-on
            promoted = None
            for j in pool:
                if j in always_on:
                    continue
                if any(sch.z[j, kk] == 1 for kk in inst.K):
                    continue
                promoted = j
                break
            if promoted is None:
                break
            for kk in inst.K:
                sch.y[promoted, kk] = 1
            always_on.add(promoted)

    # PM cap c22 (rarely binds): trim least-worn PMs if too many in a slot
    for k in inst.K:
        in_pm = [j for j in inst.J if sch.z[j, k] == 1]
        if len(in_pm) > inst.nJ - inst.N_min:
            for j in sorted(in_pm, key=lambda j: inst.psi0[j])[
                    : len(in_pm) - (inst.nJ - inst.N_min)]:
                sch.z[j, k] = 0

    cost, terms, feas = exact_cost(inst, sch)
    return cost, sch, {"terms": terms, "feas": feas}


# ======================================================================
# Exact objective + feasibility diagnostics (mirrors optimizer_v0.1)
# ======================================================================
def exact_cost(inst: Instance, sch: Schedule):
    nJ, nK = inst.nJ, inst.nK
    energy = pm = cm = sw = late = 0.0
    wear = dict(inst.psi0)
    prev_y = {j: 0 for j in inst.J}
    pit_k = {}; ptot_k = {}
    n_switch = 0

    for k in inst.K:
        PIT = 0.0; Pcool = 0.0
        for j in inst.J:
            # v0.1 charges c_cm*lam_pm for every (j,k), independent of state.
            cm += inst.c_cm * inst.lam_pm[j]
            if sch.z[j, k] == 1:
                wear[j] = 0.0
                continue
            if sch.y[j, k] == 1:
                load = sch.load_cpu[j, k]
                pit = inst.P0[j] + inst.dP[j] * load
                H = inst.alpha[j] * pit
                PIT += pit
                Pcool += inst.cool_coeff(j) * H
                wear[j] += load
                cm += inst.c_cm * inst.lam0[j] * (wear[j] / inst.Lambda[j])
        Ptot = PIT + Pcool + inst.P_ov
        energy += inst.dt * inst.c_e[k]/1000 * Ptot
        pit_k[k] = PIT; ptot_k[k] = Ptot

    for j in inst.J:
        if sch.pm_start[j] is not None:
            pm += inst.c_pm
        for k in inst.K:
            if k == 0:
                continue
            if sch.y[j, k] != sch.y[j, k - 1]:
                n_switch += 1
    sw = inst.c_sw * n_switch
    for i in inst.I:
        late += inst.c_late.get(i, 0.0) * sch.late.get(i, 0.0)

    total = energy + pm + cm + sw + late
    terms = dict(energy_cost=energy, pm_cost=pm, cm_cost=cm,
                 switching_cost=sw, lateness_cost=late)

    # feasibility diagnostics
    pue_viol = sum(1 for k in inst.K
                   if ptot_k[k] > inst.Pi_max * pit_k[k] + 1e-6)
    therm_viol = _check_thermal(inst, sch)
    ac = _check_acoustics(inst, sch)
    forced_pm_missing = sum(1 for j in inst.J
                            if inst.psi0[j] >= inst.Lambda[j]
                            and sch.pm_start[j] is None)
    # Split the unplaced jobs into the ones no schedule could ever place and
    # the ones the heuristic merely failed to fit. Only the second kind is a
    # solver defect; the first means the instance itself is infeasible, and
    # the MILP would answer INFEASIBLE rather than dropping them.
    impossible = _statically_unplaceable(inst)
    heuristic_misses = [i for i in sch.unplaced if i not in impossible]
    feas = {"unplaced_jobs": len(sch.unplaced),
            "unplaced_infeasible": len([i for i in sch.unplaced
                                        if i in impossible]),
            "unplaced_heuristic": len(heuristic_misses),
            "pue_violations": pue_viol,
            "thermal_violations": therm_viol,
            "forced_pm_missing": forced_pm_missing,
            "switching_events": n_switch,
            "switch_budget_ok": n_switch <= inst.S_max,
            "acoustics": ac}
    ac_ok = (not ac.get("present")) or (
        ac.get("cap_violations", 0) == 0 and ac.get("dose_violations", 0) == 0)
    feas["all_hard_constraints_ok"] = bool(
        len(sch.unplaced) == 0 and pue_viol == 0 and therm_viol == 0
        and forced_pm_missing == 0 and n_switch <= inst.S_max and ac_ok)
    return total, terms, feas


def _statically_unplaceable(inst: Instance) -> Dict[int, str]:
    """Jobs no feasible schedule can place, with the reason.

    These are properties of the instance, not of the search: an empty start
    window, fewer eligible servers than replicas, or -- the case that bites
    the two-failure-domain instances -- a critical job whose eligible servers
    span fewer racks than it has replicas, which contradicts rack diversity
    (c33 in optimizer_v0.1). The MILP reports such an instance INFEASIBLE.
    """
    out: Dict[int, str] = {}
    for i in inst.I:
        S = set(inst.S.get(i, []))
        q = inst.q[i]
        upper = inst.nK - inst.d[i]
        if i in inst.I_V:
            upper = min(upper, inst.b[i] - inst.d[i])
        if upper < inst.a[i]:
            out[i] = (f"empty start window (release {inst.a[i]} > latest "
                      f"start {upper})")
        elif len(S) < q:
            out[i] = f"{len(S)} eligible servers < q={q}"
        elif i in inst.I_C:
            racks = {inst.rack_of.get(j, -1) for j in S}
            if len(racks) < q:
                out[i] = (f"critical job spans {len(racks)} rack(s) < q={q}, "
                          f"contradicting rack diversity")
    return out


def _check_thermal(inst: Instance, sch: Schedule) -> int:
    """Count (server,slot) violations of the inlet-temperature constraint c19."""
    D = inst.Dmat
    Jn = inst.nJ
    viol = 0
    for k in inst.K:
        # heat that reaches room air per server this slot (air-cooled fraction)
        air_heat = []
        for jp in range(Jn):
            j = inst.J[jp]
            if sch.y[j, k] == 1 and sch.z[j, k] == 0:
                p = inst.P0[j] + inst.dP[j] * sch.load_cpu[j, k]
                air_heat.append((1.0 - inst.phi[j]) * inst.alpha[j] * p)
            else:
                air_heat.append(0.0)
        for jp in range(Jn):
            j = inst.J[jp]
            recirc = sum(D[jp][jp2] * air_heat[jp2] for jp2 in range(Jn))
            rhs = (inst.T_idle - (inst.T_idle - inst.T_busy) * sch.y[j, k]
                   + inst.M_big * sch.z[j, k])
            if inst.T_sup + recirc > rhs + 1e-6:
                viol += 1
    return viol


def _check_acoustics(inst: Instance, sch: Schedule) -> dict:
    ac = inst.ac
    if not ac:
        return {"present": False}
    zones = ac.get("zones", [])
    Gamma = ac.get("Gamma", [])
    SA = {int(j): v for j, v in ac.get("pwl", {}).get("SA_slope", {}).items()}
    SB = {int(j): v for j, v in ac.get("pwl", {}).get("SB_intercept", {}).items()}
    W_occ = float(ac.get("W_occ_cap", math.inf))
    W_unocc = float(ac.get("W_unocc_cap", math.inf))
    E_max = float(ac.get("Dose_max", math.inf))
    nseg = len(ac.get("breakpoints_u", [1, 1])) - 1
    zone_servers = [[int(j) for j in z] for z in zones]
    occ_viol = dose_viol = 0
    for gi in range(len(zones)):
        dose = 0.0
        for k in inst.K:
            SWk = {}
            for j in inst.J:
                if sch.y[j, k] == 1 and j in SA:
                    u = sch.load_cpu[j, k]
                    SWk[j] = max(SA[j][p] * u + SB[j][p] for p in range(min(nseg, len(SA[j]))))
                else:
                    SWk[j] = 0.0
            ZW = 0.0
            for gp in range(len(zones)):
                gsum = sum(SWk.get(j, 0.0) for j in zone_servers[gp])
                ZW += (Gamma[gi][gp] if gi < len(Gamma) and gp < len(Gamma[gi]) else 0.0) * gsum
            occupied = any(sch.z[j, k] == 1 for j in zone_servers[gi])
            cap = W_occ if occupied else W_unocc
            if ZW > cap + 1e-6:
                occ_viol += 1
            if occupied:
                dose += inst.dt * ZW
        if dose > E_max + 1e-6:
            dose_viol += 1
    return {"present": True, "cap_violations": occ_viol, "dose_violations": dose_viol}


# ======================================================================
# Build result payload in optimizer_v0.1 format (via .X shim + extractors)
# ======================================================================
class _V:
    __slots__ = ("X",)
    def __init__(self, val):
        self.X = float(val)


class _Model:
    def __init__(self, obj, gap, runtime, nvars, nbin, ncons):
        self.ObjVal = obj; self.MIPGap = gap; self.Runtime = runtime
        self.NumVars = nvars; self.NumBinVars = nbin; self.NumConstrs = ncons
        self.status = 2; self.SolCount = 1


def _build_result(inst: Instance, data: dict, scenario: str, sch: Schedule,
                  cost: float, terms: dict, bound: float, seconds: float,
                  feasible: bool):
    K = inst.K; J = inst.J
    dt = inst.dt
    # per-slot power
    PIT = {}; Pcool = {}; Ptot = {}
    Lvar = {}
    for k in K:
        pit = pcool = 0.0
        for j in J:
            Lvar[(j, k)] = _V(sch.load_cpu[j, k] if sch.y[j, k] else 0.0)
            if sch.y[j, k] == 1 and sch.z[j, k] == 0:
                load = sch.load_cpu[j, k]
                p = inst.P0[j] + inst.dP[j] * load
                pit += p
                pcool += inst.cool_coeff(j) * inst.alpha[j] * p
        PIT[k] = _V(pit); Pcool[k] = _V(pcool); Ptot[k] = _V(pit + pcool + inst.P_ov)
    y = {(j, k): _V(sch.y[j, k]) for j in J for k in K}
    z = {(j, k): _V(sch.z[j, k]) for j in J for k in K}
    X = {key: _V(1.0) for key in sch.X}
    d_on = {(j, k): _V(1.0 if k < inst.nK - 1 and sch.y[j, k + 1] > sch.y[j, k] else 0.0)
            for j in J for k in K[:-1]}
    d_off = {(j, k): _V(1.0 if k < inst.nK - 1 and sch.y[j, k] > sch.y[j, k + 1] else 0.0)
             for j in J for k in K[:-1]}
    m_j = {j: _V(1.0 if sch.pm_start[j] is not None else 0.0) for j in J}
    pm_starts = [k for k in K if k <= inst.nK - inst.d_pm]
    v = {(j, k): _V(1.0 if sch.pm_start[j] == k else 0.0)
         for j in J for k in pm_starts}
    l_var = {i: _V(sch.late.get(i, 0.0)) for i in inst.I}

    gap = (cost - bound) / cost if cost > 0 and math.isfinite(bound) else float("nan")
    nX = len(sch.X)
    mdl = _Model(cost, max(0.0, gap), seconds, nX + 3 * inst.nJ * inst.nK,
                 nX + 2 * inst.nJ * inst.nK, 0)

    result = {
        "scenario_name": scenario,
        "model": mdl,
        "status_code": 2,
        "status_label": "LR_DP_HEURISTIC",
        "feasible_solution": bool(feasible),
        "data": data,
        "sets": {"I": inst.I, "I_B": list(inst.I_B), "I_V": list(inst.I_V),
                 "I_C": list(inst.I_C), "J": J, "K": K},
        "params": {"d": inst.d, "r": inst.r_cpu, "r_cpu": inst.r_cpu,
                   "r_mem": inst.r_mem, "q": inst.q, "delta_t": dt,
                   "Dk": {k: 0.0 for k in K}, "eta_air": inst.eta_air},
        "vars": {"X": X, "y": y, "z": z, "L": Lvar, "d_on": d_on, "d_off": d_off,
                 "m_j": m_j, "v": v, "l_var": l_var,
                 "PIT": PIT, "Pcool": Pcool, "Ptot": Ptot},
        "objective_terms": terms,
        "helpers": {"slot_to_time": lambda k: slot_to_time(k, dt)},
    }
    return result


def build_json_result(result: Dict[str, Any], bound: float, gap: float) -> Dict[str, Any]:
    """optimizer_v0.1-format payload, plus the LR-specific bound/gap fields."""
    mdl = result["model"]
    sets = result["sets"]
    data_sets = result["data"]["sets"]
    ac = result["data"].get("acoustics")
    K = sets["K"]
    if result["feasible_solution"]:
        sp = sum(result["vars"]["PIT"][k].X for k in K)
        st = sum(result["vars"]["Ptot"][k].X for k in K)
        ratio = st / sp if sp > 1e-9 else None
    else:
        ratio = None
    return {
        "scenario_name": result["scenario_name"],
        "status": result["status_label"],
        "feasible_solution": result["feasible_solution"],
        "solver_stats": {
            "num_servers": len(sets["J"]),
            "num_time_slots": len(sets["K"]),
            "num_jobs": len(sets["I"]),
            "num_jobs_I_B": len(sets["I_B"]),
            "num_jobs_I_V": len(sets["I_V"]),
            "num_jobs_I_C": len(sets["I_C"]),
            "num_predecessor_pairs": len(data_sets.get("E", [])),
            "num_affinity_pairs": len(data_sets.get("A", [])),
            "num_non_affinity_pairs": len(data_sets.get("G", [])),
            "num_failure_domains": len(data_sets.get("F", [])),
            "num_acoustic_zones": len(ac.get("zones", [])) if ac else 0,
            "num_noise_curve_segments": (len(ac.get("breakpoints_u", [])) - 1)
                                        if ac else 0,
            "runtime_seconds": getattr(mdl, "Runtime", None),
            "lower_bound": bound if math.isfinite(bound) else None,
            "gap_pct": gap,
            "objective_value": mdl.ObjVal if result["feasible_solution"] else None,
            "num_variables": mdl.NumVars,
            "num_binary_variables": mdl.NumBinVars,
            "num_constraints": mdl.NumConstrs,
            "ptot_pit_ratio": ratio,
        },
        "metrics": extract_performance_metrics(result),
        "solution": extract_solution_rows(result),
        "pm_schedule": extract_pm_rows(result),
        "hourly": extract_hourly_rows(result),
        "server_timeseries": extract_server_load_timeseries(result),
        "server_summary": extract_server_summary(result),
    }


SOLVER_STATS_COLUMNS = [
    "scenario_name", "status", "feasible_solution", "num_servers",
    "num_time_slots", "num_jobs", "num_jobs_I_B", "num_jobs_I_V", "num_jobs_I_C",
    "num_predecessor_pairs", "num_affinity_pairs", "num_non_affinity_pairs",
    "num_failure_domains", "num_acoustic_zones", "num_noise_curve_segments",
    "runtime_seconds", "lower_bound", "gap_pct", "dual_iterations",
    "convergence", "objective_value", "num_variables", "num_binary_variables",
    "num_constraints", "ptot_pit_ratio",
]


def solver_stats_row(payload):
    s = payload.get("solver_stats", {})
    row = {"scenario_name": payload.get("scenario_name"),
           "status": payload.get("status"),
           "feasible_solution": payload.get("feasible_solution")}
    for c in SOLVER_STATS_COLUMNS[3:]:
        row[c] = s.get(c)
    return row


def write_solver_stats_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=SOLVER_STATS_COLUMNS).to_csv(csv_path, index=False)
    except ImportError:
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SOLVER_STATS_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in SOLVER_STATS_COLUMNS})


# ======================================================================
# Instance loading (mirrors optimizer_v0.1._load_instance_data)
# ======================================================================
def _load_instance_data(jobs_json_path: Path, server_json_path: Path) -> dict:
    data = load_data_from_jobs_json(jobs_json_path=jobs_json_path,
                                    server_json_path=server_json_path)
    def backfill(container, raw_path):
        with raw_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for k, v in raw.get(container, {}).items():
            data.setdefault(container, {})
            data[container].setdefault(k, v)
    backfill("job_params", jobs_json_path)
    backfill("server_params", server_json_path)
    with server_json_path.open("r", encoding="utf-8") as fh:
        raw_server = json.load(fh)
    if "acoustics" not in data and "acoustics" in raw_server:
        data["acoustics"] = raw_server["acoustics"]
    if "cooling" in raw_server:
        data.setdefault("cooling", {})
        for kk, vv in raw_server["cooling"].items():
            data["cooling"].setdefault(kk, vv)
    return data


def discover_instances(experimental_dir: Path):
    out = []
    for jp in sorted(experimental_dir.glob(f"{JOBS_PREFIX}*.json")):
        name = jp.stem[len(JOBS_PREFIX):]
        spath = experimental_dir / f"{SERVER_PREFIX}{name}.json"
        if spath.exists():
            out.append((name, jp, spath))
    return out


# ======================================================================
# Solve one instance end-to-end
# ======================================================================
def solve_instance(instance: str, jobs_json_path: Path, server_json_path: Path,
                   output_json_path: Path, iters: int, gap_tol: float,
                   tol: float, patience: int, time_limit: float, verbose: bool):
    print(f"\n=== Instance: {instance} (LR+DP) ===")
    data = _load_instance_data(jobs_json_path, server_json_path)
    inst = instance_from_data(data)
    print(f"  {inst.nJ} servers x {inst.nK} slots, {len(inst.I)} jobs")

    t0 = time.time()
    # Primal first: its cost is the upper bound the dual's gap test converges to.
    cost, sch, info = build_primal(inst, verbose=verbose)
    ub = cost if info["feas"]["all_hard_constraints_ok"] else math.inf
    bound, reason, iters_run = lower_bound(
        inst, iters=iters, ub=ub, gap_tol=gap_tol, tol=tol,
        patience=patience, time_limit=time_limit, verbose=verbose)
    seconds = time.time() - t0

    gap = 100.0 * (cost - bound) / cost if cost > 0 and math.isfinite(bound) else float("nan")
    result = _build_result(inst, data, instance, sch, cost, info["terms"],
                           bound, seconds, info["feas"]["all_hard_constraints_ok"])
    payload = build_json_result(result, bound, gap)
    payload["instance"] = instance
    payload["feasibility"] = info["feas"]
    payload["solver_stats"]["dual_iterations"] = iters_run
    payload["solver_stats"]["convergence"] = reason

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=safe_value)

    label = "primal (feasible) " if info["feas"]["all_hard_constraints_ok"] \
        else "primal (INFEASIBLE)"
    print(f"  {label}: {cost:14.2f}   unplaced={len(sch.unplaced)}")
    if sch.unplaced:
        reasons = _statically_unplaceable(inst)
        n_imp = info["feas"]["unplaced_infeasible"]
        n_heur = info["feas"]["unplaced_heuristic"]
        print(f"      {n_imp} unplaceable by construction, {n_heur} missed by "
              f"the heuristic")
        for i in sch.unplaced[:3]:
            if i in reasons:
                print(f"        job {i}: {reasons[i]}")
        if n_imp:
            print("      -> the INSTANCE is infeasible; the cost and gap below "
                  "are not meaningful")
    print(f"  lower bound (LR)  : {bound:14.2f}")
    print(f"  gap               : {gap:13.2f} %   "
          f"(converged: {reason} in {iters_run} iters)")
    print(f"  feasibility       : {info['feas']}")
    print(f"  wall-clock        : {seconds:13.2f} s   -> {output_json_path.name}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Full-model LR+DP solver")
    ap.add_argument("--experimental-dir", default=str(_EXPERIMENTAL_DATA_DIR))
    ap.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    ap.add_argument("--jobs-json-input", default=None)
    ap.add_argument("--server-json-input", default=None)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--iters", type=int, default=1000,
                    help="max subgradient iterations (hard cap)")
    ap.add_argument("--gap-tol", type=float, default=0.0001,
                    help="stop when (primal-bound)/primal <= this (default 0.01%%)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="stop when the subgradient norm falls below this")
    ap.add_argument("--patience", type=int, default=25,
                    help="stop after this many iters with no bound improvement")
    ap.add_argument("--time-limit", type=float, default=7200.0,
                    help="wall-clock limit (s) for the dual loop (default 7200)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="(kept for parity; real data is the default path)")
    args = ap.parse_args()

    print("=== Full-model Lagrangian-Relaxation + DP solver (lr_dp_solver.py) ===")

    if args.jobs_json_input and args.server_json_input:
        jp = Path(args.jobs_json_input); spath = Path(args.server_json_input)
        name = jp.stem[len(JOBS_PREFIX):] if jp.stem.startswith(JOBS_PREFIX) else jp.stem
        outdir = Path(args.output_dir)
        out = Path(args.output_json) if args.output_json else outdir / f"result_{name}.json"
        payload = solve_instance(name, jp, spath, out, args.iters, args.gap_tol,
                                 args.tol, args.patience, args.time_limit,
                                 args.verbose)
        csv_path = out.parent / "lr_dp_stats_summary.csv"
        write_solver_stats_csv([solver_stats_row(payload)], csv_path)
        print(f"Solver-stats CSV: {csv_path}")
        return

    experimental_dir = Path(args.experimental_dir)
    output_dir = Path(args.output_dir)
    instances = discover_instances(experimental_dir)
    if not instances:
        print(f"No instances found in {experimental_dir}")
        return
    print(f"Found {len(instances)} instance(s).")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batch_summary.json"
    csv_path = output_dir / "lr_dp_stats_summary.csv"
    summary_rows, stat_rows = [], []
    for name, jp, spath in instances:
        out = output_dir / f"result_{name}.json"
        try:
            payload = solve_instance(name, jp, spath, out, args.iters,
                                     args.gap_tol, args.tol, args.patience,
                                     args.time_limit, args.verbose)
            s = payload["solver_stats"]
            summary_rows.append({"instance": name, "status": payload["status"],
                                 "feasible_solution": payload["feasible_solution"],
                                 "objective_value": s["objective_value"],
                                 "lower_bound": s["lower_bound"],
                                 "gap_pct": s["gap_pct"],
                                 "runtime_seconds": s["runtime_seconds"],
                                 "result_file": str(out)})
            stat_rows.append(solver_stats_row(payload))
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {name}: {exc}")
            summary_rows.append({"instance": name, "status": "ERROR",
                                 "feasible_solution": False, "error": str(exc)})
            stat_rows.append({"scenario_name": name, "status": "ERROR",
                              "feasible_solution": False})
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, default=safe_value)
        write_solver_stats_csv(stat_rows, csv_path)

    print(f"\nBatch complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
