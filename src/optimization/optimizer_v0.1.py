"""MILP builder and solver for the data-centre scheduling model
This version is the straightforward coding of the model published in the paper
"""

from typing import Any, Dict, List, Tuple
import argparse
import json
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

try:
    from .utils import get_status_label, slot_to_time
    from .data_loader import load_data_from_jobs_json
    from .update_psi_0 import update_psi_0
    from .result_extractor import (
        extract_performance_metrics,
        extract_solution_rows,
        extract_pm_rows,
        extract_hourly_rows,
        extract_server_load_timeseries,
        extract_server_summary,
    )
except ImportError:
    # Executed directly (e.g. `python optimization_model.py`) rather than
    # via `-m src.optimization...`, so relative imports have no parent
    # package to resolve against. Fall back to an absolute import after
    # putting the repo root on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optimization.utils import get_status_label, slot_to_time
    from src.optimization.data_loader import load_data_from_jobs_json
    from src.optimization.update_psi_0 import update_psi_0
    from src.optimization.result_extractor import (
        extract_performance_metrics,
        extract_solution_rows,
        extract_pm_rows,
        extract_hourly_rows,
        extract_server_load_timeseries,
        extract_server_summary,
    )

# This standalone entry point sources its inputs from an experimental
# data folder rather than the main pipeline's data/processed and data/raw
# (used by cli.py), since it runs independently of the forecast pipeline.
# Resolved as absolute paths from this file's location (not the process's
# working directory), since IDE run configurations often launch scripts
# with the working directory set to the script's own folder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL_DATA_DIR = _REPO_ROOT / "src" / "experimental_data"
# Anchored to the repo root (not the process CWD) so results always land in
# the same place regardless of where the script is launched from.
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "experimental_results_v0.1"


def solve_datacenter_model(
    data: Dict[str, Any],
    scenario_name: str,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Build and solve the MILP for one scenario."""

    # -----------------------------
    # 4.1 Load sets and parameters
    # -----------------------------
    I = data["sets"]["I"]
    I_B = data["sets"]["I_B"]
    I_V = data["sets"]["I_V"]
    I_C = data["sets"]["I_C"]
    J = data["sets"]["J"]
    K = data["sets"]["K"]
    F = data["sets"]["F"]
    E = data["sets"]["E"]
    A = data["sets"]["A"]
    G = data["sets"]["G"]
    S = {int(k): v for k, v in data["eligibility"].items()}

    jp = data["job_params"]
    d = {int(k): v for k, v in jp["d"].items()}
    # Resource demands are split into CPU and memory. A file may provide the
    # new r_cpu/r_mem fields, or only the legacy scalar r (used for CPU, with
    # memory copied from it so old inputs remain valid).
    if "r_cpu" in jp:
        r_cpu = {int(k): v for k, v in jp["r_cpu"].items()}
        r_mem = {int(k): v for k, v in jp["r_mem"].items()}
    else:
        r_cpu = {int(k): v for k, v in jp["r"].items()}
        r_mem = dict(r_cpu)
    # Resources iterated by the vector-packing constraints. CPU is listed first
    # because it is also the load that feeds power, heat, wear (and noise).
    RES = ("cpu", "mem")
    r_res = {"cpu": r_cpu, "mem": r_mem}
    a = {int(k): v for k, v in jp["a"].items()}
    b = {int(k): v for k, v in jp["b"].items()}
    q = {int(k): v for k, v in jp["q"].items()}
    # Per-job batch lateness cost coefficient. Named c_late to match the other
    # cost coefficients (c_pm, c_cm, c_sw); the data field may be "c_late" or the
    # legacy "rho" (same meaning: dollars per unit of tardiness for batch job i).
    _late_src = jp.get("c_late", jp.get("rho", {}))
    c_late = {int(k): v for k, v in _late_src.items()}

    # Non-critical jobs need one replica by definition.
    for i in I:
        if i not in q:
            q[i] = 1

    sp = data["server_params"]
    # Per-resource server capacities and interactive reservations, with a
    # fallback to the legacy scalar fields for backward compatibility.
    if "C_cpu" in sp:
        C_cpu = {int(k): v for k, v in sp["C_cpu"].items()}
        C_mem = {int(k): v for k, v in sp["C_mem"].items()}
    else:
        C_cpu = {int(k): v for k, v in sp["C"].items()}
        C_mem = dict(C_cpu)
    if "theta_cpu" in sp:
        theta_cpu = {int(k): v for k, v in sp["theta_cpu"].items()}
        theta_mem = {int(k): v for k, v in sp["theta_mem"].items()}
    else:
        theta_cpu = {int(k): v for k, v in sp["theta"].items()}
        theta_mem = dict(theta_cpu)
    C_res = {"cpu": C_cpu, "mem": C_mem}
    theta_res = {"cpu": theta_cpu, "mem": theta_mem}
    # CPU capacity/load is the physical driver of power, heat and wear.
    C = C_cpu
    P0 = {int(k): v for k, v in sp["P0"].items()}
    dP = {int(k): v for k, v in sp["dP"].items()}
    alpha = {int(k): v for k, v in sp["alpha"].items()}
    lambda0 = {int(k): v for k, v in sp["lambda0"].items()}
    lambda_pm = {int(k): v for k, v in sp["lambda_pm"].items()}
    Lambda = {int(k): v for k, v in sp["Lambda"].items()}

    # psi_0[j]: accumulated load since last PM at the start of this horizon (k=0).
    # Loaded from server_params if present; defaults to 0.0 (fresh or recently maintained).
    # This parameter carries wear history across daily optimization cycles.
    psi_0_raw = sp.get("psi_0", {})
    psi_0 = {j: float(psi_0_raw.get(str(j), psi_0_raw.get(j, 0.0))) for j in J}

    th = data["thermal"]
    T_sup = th["T_sup"]
    T_busy = th["T_busy"]
    T_idle = th["T_idle"]
    M_big = th["M_big"]
    D = th["D"]

    # Hybrid cooling parameters. eta_air is the air-path COP (the legacy "eta");
    # eta_liq is the much higher liquid-path COP; phi[j] is the fraction of
    # server j's heat captured by the liquid loop; xi is the pump power per unit
    # of liquid-carried heat. Absent fields fall back to a pure air-cooled model.
    cool = data["cooling"]
    _eta_legacy = cool.get("eta")
    if isinstance(_eta_legacy, list):
        _eta_legacy = _eta_legacy[0]
    eta_air = cool.get("eta_air", _eta_legacy)
    if isinstance(eta_air, list):
        eta_air = eta_air[0]
    eta_liq = cool.get("eta_liq", eta_air)   # if absent, liquid == air (no gain)
    if isinstance(eta_liq, list):
        eta_liq = eta_liq[0]
    xi_pump = cool.get("xi", 0.0)            # pump coefficient; 0 => no pump term
    phi_raw = cool.get("phi", {})           # per-server liquid capture fraction
    phi = {j: float(phi_raw.get(str(j), phi_raw.get(j, 0.0))) for j in J}
    hybrid_cooling = ("eta_liq" in cool) or ("phi" in cool)
    P_ov = data["power"]["P_ov"]
    Pi_max = data["power"]["Pi_max"]
    d_pm = data["maintenance"]["d_pm"]
    c_pm = data["maintenance"]["c_pm"]
    c_cm = data["maintenance"]["c_cm"]
    c_e = {k: data["costs"]["c_e"][k] for k in K}
    c_sw = data["costs"]["c_sw"]
    S_max = data["costs"]["S_max"]
    Dk = {k: data["demand"]["D"][k] for k in K}
    N_min = data["redundancy"]["N_min"]
    kappa = data["redundancy"]["kappa"]
    red = data["redundancy"]
    if "Q_max_cpu" in red:
        Q_max_res = {"cpu": red["Q_max_cpu"], "mem": red["Q_max_mem"]}
    else:
        Q_max_res = {"cpu": red["Q_max"], "mem": red["Q_max"]}
    delta_t = data["slot_duration"]

    # --- Acoustics / occupational noise (optional block) --------------------
    # Present in the *_noise_split.json instances. When absent, every acoustic
    # variable, constraint and objective term below is skipped and the model
    # reduces exactly to the noise-free formulation.
    ac = data.get("acoustics")
    if ac is not None:
        Zn = list(range(len(ac["zones"])))                 # acoustic zones g
        zone_servers = {g: [int(j) for j in ac["zones"][g]] for g in Zn}
        Gamma = ac["Gamma"]                                # Gamma_{g,g'}
        n_seg = len(ac["breakpoints_u"]) - 1               # PWL segments p
        SA = {int(j): v for j, v in ac["pwl"]["SA_slope"].items()}
        SB = {int(j): v for j, v in ac["pwl"]["SB_intercept"].items()}
        W_occ = ac["W_occ_cap"]                            # W^occ
        W_unocc = ac["W_unocc_cap"]                        # W^unocc
        E_max = ac["Dose_max"]                             # E^max
        # c_ex is no longer used: occupational noise is enforced by the hard
        # constraints (40)-(42), not priced in the objective. Read defensively.
        c_ex = ac.get("c_ex", 0.0)                         # retained, unused
        # M_W retained for backward-compatible data; the dose linearization
        # (41a)-(41c) uses W_unocc as its big-M, so M_W is no longer required.
        M_W = ac.get("M_W", 0.0)
    nK = len(K)

    def local_slot_to_time(k: int) -> str:
        return slot_to_time(k, delta_t)

    def valid_starts(job: int) -> List[int]:
        """Valid start slots: release time, horizon end, and hard deadline for interactive jobs."""
        upper = nK - d[job]
        if job in I_V:
            upper = min(upper, b[job] - d[job])
        return [k for k in K if a[job] <= k <= upper]

    # X is referenced by running_at, so the helper is defined after X exists.
    X: Dict[Tuple[int, int, int], gp.Var] = {}

    def running_at(job: int, server: int, slot: int) -> List[int]:
        """Return start slots where job is running on server during slot."""
        return [
            kp for kp in valid_starts(job)
            if (job, server, kp) in X and kp <= slot < kp + d[job]
        ]

    # -----------------------------
    # 4.2 Build model
    # -----------------------------
    params = {
        "WLSACCESSID": "fc17fa3a-ef7f-41d2-b95c-20c3b221a483",
        "WLSSECRET": "6bee54d1-5c9f-4f12-9d64-0c7b16e0dd52",
        "LICENSEID": 2804943
    }

    env = gp.Env(empty=True)
    for key, value in params.items():
        env.setParam(key, value)
    env.start()
    mdl = gp.Model(f"datacenter_1day_{scenario_name}", env=env)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", mip_gap)
    mdl.setParam("Presolve", 2)       # aggressive presolve
    mdl.setParam("MIPFocus", 1)       # focus on finding feasible solutions fast
    mdl.setParam("Cuts", 2)           # aggressive cuts
    mdl.setParam("Heuristics", 0.3)   # more time on heuristics early
    mdl.setParam("Threads", 0)        # use all available cores
    if not verbose:
        mdl.setParam("OutputFlag", 0)

    # -----------------------------
    # 4.3 Decision variables
    # -----------------------------
    # R_{ijk} remains eliminated. It is fully determined by X.
    X.update({
        (i, j, k): mdl.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
        for i in I for j in S[i] for k in valid_starts(i)
    })

    y = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
         for j in J for k in K}
    d_on = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"don_{j}_{k}")
            for j in J for k in K[:-1]}
    d_off = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"doff_{j}_{k}")
             for j in J for k in K[:-1]}
    m_j = {j: mdl.addVar(vtype=GRB.BINARY, name=f"m_{j}") for j in J}

    # All possible starts for preventive maintenance
    pm_starts = [k for k in K if k <= nK - d_pm]
    v = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"v_{j}_{k}")
         for j in J for k in pm_starts}
    z = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")
         for j in J for k in K}


    l_var = {i: mdl.addVar(lb=0.0, name=f"l_{i}") for i in I_B}
    # Per-resource aggregate load. L_res["cpu"][j,k] and L_res["mem"][j,k].
    L_res = {
        res: {
            (j, k): mdl.addVar(lb=0.0, ub=GRB.INFINITY, name=f"L{res}_{j}_{k}")
            for j in J for k in K
        }
        for res in RES
    }
    # CPU load alias: power, heat, wear and thermal recirculation all key off
    # the CPU dimension, so `L` continues to refer to CPU load throughout.
    L = L_res["cpu"]
    H = {(j, k): mdl.addVar(lb=0.0, name=f"H_{j}_{k}") for j in J for k in K}
    PIT = {k: mdl.addVar(lb=0.0, name=f"PIT_{k}") for k in K}
    Pcool = {k: mdl.addVar(lb=0.0, name=f"Pcool_{k}") for k in K}
    # P_pump[k]: electrical power drawn by the coolant distribution pumps.
    Ppump = {k: mdl.addVar(lb=0.0, name=f"Ppump_{k}") for k in K}
    Ptot = {k: mdl.addVar(lb=0.0, name=f"Ptot_{k}") for k in K}
    s = {i: mdl.addVar(lb=0.0, ub=nK - 1, name=f"s_{i}") for i in I}

    # M_psi: tight upper bound for cumulative wear psi[j,k] and auxiliary w[j,k].
    # Worst case: server runs at full capacity every slot from maximum inherited load.
    M_psi = max(psi_0.values(), default=0.0) + nK * max(C_cpu.values())
    psi = {(j, k): mdl.addVar(lb=0.0, ub=M_psi, name=f"psi_{j}_{k}")
           for j in J for k in K}
    # w[j,k]: pre-reset accumulation = psi[j,k-1] + L[j,k].
    # Separating w from psi allows a clean linearisation of the conditional
    # reset without creating contradictory equality/inequality pairs on psi.
    w = {(j, k): mdl.addVar(lb=0.0, ub=M_psi, name=f"w_{j}_{k}")
         for j in J for k in K}

    # --- Acoustic variables (only when the acoustics block is present) ------
    if ac is not None:
        # SW[j,k]: sound power emitted by server j in slot k (linear units)
        SW = {(j, k): mdl.addVar(lb=0.0, name=f"SW_{j}_{k}")
              for j in J for k in K}
        # ZW[g,k]: aggregate sound power perceived in zone g in slot k
        ZW = {(g, k): mdl.addVar(lb=0.0, name=f"ZW_{g}_{k}")
              for g in Zn for k in K}
        # o[g,k]: 1 if a technician is present in zone g (PM active there)
        o = {(g, k): mdl.addVar(vtype=GRB.BINARY, name=f"o_{g}_{k}")
             for g in Zn for k in K}
        # Dz[g,k]: occupied-noise contribution of zone g in slot k. Equals the
        # zone noise ZW[g,k] when a technician is present and 0 otherwise; used
        # by the dose constraint (41a)-(41c). Linearized exactly by (41a)-(41c) below.
        Dz = {(g, k): mdl.addVar(lb=0.0, name=f"Dz_{g}_{k}")
             for g in Zn for k in K}
    else:
        SW, ZW, o, Dz = {}, {}, {}, {}


    mdl.update()

    # -----------------------------
    # 4.4 Objective (#4)
    # -----------------------------
    energy_cost = delta_t * gp.quicksum(c_e[k]/1000 * Ptot[k] for k in K)
    pm_cost = gp.quicksum(c_pm * m_j[j] for j in J)
    # CM cost: scales with normalized accumulated wear psi[j,k] / Lambda[j].
    # A server at full wear (psi = Lambda) has failure rate lambda0[j];
    # a freshly maintained server has near-zero risk. This creates a genuine
    # economic incentive to schedule PM when wear is high across cycles.
    cm_cost = c_cm * gp.quicksum(
        lambda_pm[j] + lambda0[j] * (w[j, k] / Lambda[j]) * y[j, k]
        for j in J for k in K
    )
    sw_cost = c_sw * gp.quicksum(d_on[j, k] + d_off[j, k]
                                 for j in J for k in K[:-1])
    late_cost = gp.quicksum(c_late[i] * l_var[i] for i in I_B)
    # Occupational noise is NOT priced in the objective. It is enforced entirely
    # by the hard safety constraints (40) instantaneous and (42) cumulative dose,
    # so worker protection is a feasibility requirement rather than a cost traded
    # off against energy or maintenance. The objective is a pure monetary cost.
    mdl.setObjective(energy_cost + pm_cost + cm_cost +
                     sw_cost + late_cost, GRB.MINIMIZE)

    # -----------------------------
    # 4.5 Constraints
    # -----------------------------

    # --- (4) Job assignment (exact replica count) ---
    for i in I:
        mdl.addConstr(
            gp.quicksum(X[i, j, k] for j in S[i]
                        for k in valid_starts(i)) == q[i],
            name=f"c4_{i}",
        )

    # --- (5)/(6) Release time and interactive hard deadlines are enforced in valid_starts(). ---

    # --- (7) Precedence ---
    for i_pred, i_succ in E:
        mdl.addConstr(s[i_succ] >= s[i_pred] + d[i_pred],
                      name=f"c7_{i_pred}_{i_succ}")

    # --- (3) Start-time definition ---
    # to capture the latest start among all replicas since staggering is allowed in the model
    # not all replicas start at the same time
    for i in I:
        for j in S[i]:
            for k in valid_starts(i):
                mdl.addConstr(s[i] >= k * X[i, j, k], name=f"c3_{i}_{j}_{k}")

    # --- (8) Batch-only capacity (interactive reservation), per resource ---
    # Neither CPU nor memory may be monopolised by batch work: the reservation
    # is enforced independently on each resource.
    for res in RES:
        r_r = r_res[res]
        C_r = C_res[res]
        th_r = theta_res[res]
        for j in J:
            for k in K:
                batch_load = gp.quicksum(
                    r_r[i] * X[i, j, kp]
                    for i in I_B if j in S[i]
                    for kp in running_at(i, j, k)
                )
                mdl.addConstr(
                    batch_load <= (1 - th_r[j]) * C_r[j] * y[j, k],
                    name=f"c8_{res}_{j}_{k}",
                )

    # --- (9) Batch lateness ---
    for i in I_B:
        if i not in I_C:
            mdl.addConstr(l_var[i] >= s[i] + d[i] - b[i], name=f"c9nc_{i}")

    M_lat = nK
    for i in I_B:
        if i in I_C:
            for j in S[i]:
                for k in valid_starts(i):
                    mdl.addConstr(
                        l_var[i] >= k + d[i] - b[i] - M_lat * (1 - X[i, j, k]),
                        name=f"c9cr_{i}_{j}_{k}",
                    )

    # --- (10)/(11) Load definition and server capacity, per resource ---
    # Each resource load is the sum of that resource's demand over running
    # jobs, and is capped by the server's capacity in that resource. Enforcing
    # CPU and memory independently is what makes this genuine vector bin
    # packing: a CPU-heavy job can share a server with a memory-heavy one.
    for res in RES:
        r_r = r_res[res]
        C_r = C_res[res]
        Lr = L_res[res]
        for j in J:
            for k in K:
                load_expr = gp.quicksum(
                    r_r[i] * X[i, j, kp]
                    for i in I if j in S[i]
                    for kp in running_at(i, j, k)
                )
                mdl.addConstr(Lr[j, k] == load_expr, name=f"c10_{res}_{j}_{k}")
                mdl.addConstr(Lr[j, k] <= C_r[j] * y[j, k],
                              name=f"c11_{res}_{j}_{k}")

    # --- (12) Execution only on a powered-on server (R_ijk <= y_jk) ---
    # Not implied by (11): a job whose CPU and memory demands are both zero
    # adds nothing to either load, so the capacity constraints alone would
    # permit it to "run" on an off server. This gates execution itself.
    for i in I:
        for j in S[i]:
            for k in K:
                run = running_at(i, j, k)
                if run:
                    mdl.addConstr(
                        gp.quicksum(X[i, j, kp] for kp in run) <= y[j, k],
                        name=f"c12_{i}_{j}_{k}",
                    )

    # --- (13) Server cannot be active and under preventive maintenance simultaneously ---
    for j in J:
        for k in K:
            mdl.addConstr(y[j, k] + z[j, k] <= 1, name=f"c13_{j}_{k}")

    # --- (14) Total IT power ---
    for k in K:
        mdl.addConstr(
            PIT[k] == gp.quicksum(P0[j] * y[j, k] + dP[j]
                                  * L[j, k] for j in J),
            name=f"c14_{k}",
        )

    # --- (15) Heat per server ---
    for j in J:
        for k in K:
            mdl.addConstr(
                H[j, k] == alpha[j] * (P0[j] * y[j, k] + dP[j] * L[j, k]),
                name=f"c15_{j}_{k}",
            )

    # --- (16) Cooling power (hybrid air + liquid) ---
    # Each server's heat H[j,k] splits into a liquid-captured fraction phi[j],
    # removed at the high liquid-path COP eta_liq, and an air-cooled remainder
    # (1 - phi[j]), removed at the air-path COP eta_air; the pump term (16b) is
    # added on top. With phi=0 and eta_liq=eta_air this reduces exactly to the
    # single-COP air-cooling model.
    for k in K:
        mdl.addConstr(
            Pcool[k]
            == (1.0 / eta_liq) * gp.quicksum(phi[j] * H[j, k] for j in J)
            + (1.0 / eta_air) * gp.quicksum((1.0 - phi[j]) * H[j, k] for j in J)
            + Ppump[k],
            name=f"c16_{k}",
        )

    # --- (16b) Coolant-pump power ---
    # Pump electrical power is proportional to the heat carried by the liquid
    # loop. When xi_pump = 0 (no hybrid data) this simply fixes Ppump = 0.
    for k in K:
        mdl.addConstr(
            Ppump[k] == xi_pump * gp.quicksum(phi[j] * H[j, k] for j in J),
            name=f"c16b_{k}",
        )

    # --- (17) Total facility power ---
    for k in K:
        mdl.addConstr(Ptot[k] == PIT[k] + Pcool[k] + P_ov, name=f"c17_{k}")

    # --- (18) PUE cap ---
    for k in K:
        mdl.addConstr(Ptot[k] <= Pi_max * PIT[k], name=f"c18_{k}")

    # --- (19) Thermal: server inlet temperature ---
    for j in J:
        for k in K:
            # Only the air-cooled fraction of a neighbour's heat reaches the
            # room air and recirculates; cold plates remove the rest at source.
            recirc = gp.quicksum(
                (1.0 - phi[jp2]) * D[j][jp2] * alpha[jp2] *
                (P0[jp2] * y[jp2, k] + dP[jp2] * L[jp2, k])
                for jp2 in J
            )
            mdl.addConstr(
                T_sup + recirc <= T_idle -
                (T_idle - T_busy) * y[j, k] + M_big * z[j, k],
                name=f"c19_{j}_{k}",
            )

    # --- (20) PM count per server ---
    for j in J:
        mdl.addConstr(gp.quicksum(v[j, k]
                      for k in pm_starts) == m_j[j], name=f"c20_{j}")

    # --- (21) PM active window ---
    for j in J:
        for k in K:
            win = [kp for kp in pm_starts if max(0, k - d_pm + 1) <= kp <= k]
            mdl.addConstr(z[j, k] == gp.quicksum(v[j, kp]
                          for kp in win), name=f"c21_{j}_{k}")

    # --- (22) Max servers under PM per slot ---
    for k in K:
        mdl.addConstr(gp.quicksum(z[j, k] for j in J)
                      <= len(J) - N_min, name=f"c22_{k}")

    # --- (23)/(24) Cumulative wear with psi_0 and linearised conditional reset ---
    #
    # Goal: psi[j,k] = (psi[j,k-1] + L[j,k]) * (1 - z[j,k])
    #   i.e. accumulate load every slot, but reset to 0 whenever the server
    #   is under PM (z[j,k]=1).  The product is non-linear, so we introduce
    #   auxiliary variable w[j,k] = pre-reset accumulation and linearize:
    #
    #   w[j,k]   = psi_0[j] + L[j,0]            if k == 0      (c23w_init)
    #   w[j,k]   = psi[j,k-1] + L[j,k]          if k  > 0      (c23w)
    #   psi[j,k] <= w[j,k]                                       (c24a)
    #   psi[j,k] <= M_psi * (1 - z[j,k])                        (c24b)
    #   psi[j,k] >= w[j,k] - M_psi * z[j,k]                    (c24c)
    #
    # When z[j,k]=0: c24b is inactive, c24a+c24c force psi[j,k] = w[j,k].
    # When z[j,k]=1: c24b forces psi[j,k] = 0, c24c is inactive (RHS <= 0).
    # There is no contradictory equality: w always equals the raw accumulation
    # and psi is the reset-gated version.
    for j in J:
        for k in K:
            if k == 0:
                mdl.addConstr(
                    w[j, k] == psi_0[j] + L[j, k],
                    name=f"c23w_init_{j}",
                )
            else:
                mdl.addConstr(
                    w[j, k] == psi[j, k - 1] + L[j, k],
                    name=f"c23w_{j}_{k}",
                )
            mdl.addConstr(psi[j, k] <= w[j, k],
                          name=f"c24a_{j}_{k}")
            mdl.addConstr(psi[j, k] <= M_psi * (1 - z[j, k]),
                          name=f"c24b_{j}_{k}")
            mdl.addConstr(psi[j, k] >= w[j, k] - M_psi * z[j, k],
                          name=f"c24c_{j}_{k}")

    # --- (25) PM trigger: only allowed once wear reaches threshold Lambda[j] ---
    # v[j,k]=1 is only feasible when psi[j,k] >= Lambda[j].
    # Because psi is already 0 during PM slots (c24b), this constraint is
    # evaluated on the slot immediately before the PM window starts, which
    # is the last slot where psi still reflects real accumulated wear.

    for j in J:
        for k in pm_starts:
            if k == 0:
                mdl.addConstr(
                    psi_0[j] >= Lambda[j] * v[j, k],
                    name=f"c25_init_{j}_{k}",
                )
            else:
                mdl.addConstr(
                    w[j, k-1] >= Lambda[j] * v[j, k],
                    name=f"c25_{j}_{k}",
                    )

    # --- Force PM if server starts the horizon at or above threshold ---
    for j in J:
        if psi_0[j] >= Lambda[j]:
            mdl.addConstr(
                gp.quicksum(v[j, k] for k in pm_starts) == 1,
                name = f"force_pm_initial_wear_{j}",
            )


    # --- (26)/(27) Server state-change tracking ---
    for j in J:
        for k in K[:-1]:
            mdl.addConstr(y[j, k + 1] - y[j, k] <=
                          d_on[j, k], name=f"c26_{j}_{k}")
            mdl.addConstr(y[j, k] - y[j, k + 1] <=
                          d_off[j, k], name=f"c27_{j}_{k}")

    # --- (28) Total switching budget ---
    mdl.addConstr(gp.quicksum(d_on[j, k] + d_off[j, k]
                  for j in J for k in K[:-1]) <= S_max, name="c28")

    # --- Forecast demand representation (no numbered constraint in the model) ---
    # The synthetic forecast is represented directly as aggregate workload jobs.
    # Therefore, the old hard aggregate demand constraint is not added here.
    # Adding it would double-count the same forecast demand and can make the
    # model infeasible.

    # --- (29) Anti-affinity isolation for critical job pairs ---
    for i1, i2 in G:
        if i1 in I_C and i2 in I_C:
            for j in J:
                for k in K:
                    r1 = gp.quicksum(X[i1, j, kp]
                                     for kp in running_at(i1, j, k))
                    r2 = gp.quicksum(X[i2, j, kp]
                                     for kp in running_at(i2, j, k))
                    mdl.addConstr(r1 + r2 <= 1, name=f"c29_{i1}_{i2}_{j}_{k}")

    # --- (30) Affinity: same server ---
    for i1, i2 in A:
        for j in J:
            s1 = gp.quicksum(X[i1, j, k]
                             for k in valid_starts(i1) if (i1, j, k) in X)
            s2 = gp.quicksum(X[i2, j, k]
                             for k in valid_starts(i2) if (i2, j, k) in X)
            mdl.addConstr(s1 == s2, name=f"c30_{i1}_{i2}_{j}")

    # --- (31) Affinity critical pairs: same replica count data check ---
    for i1, i2 in A:
        if i1 in I_C:
            assert q[i1] == q[i2], f"Affinity pair ({i1},{i2}): replica counts must match"

    # --- (32) Hot standby buffer ---
    for k in K:
        mdl.addConstr(gp.quicksum(y[j, k]
                      for j in J) >= N_min + kappa, name=f"c32_{k}")

    # --- (33) Rack diversity for critical jobs ---
    for i in I_C:
        for f_idx, Ff in enumerate(F):
            mdl.addConstr(
                gp.quicksum(
                    X[i, j, k]
                    for j in Ff if j in S[i]
                    for k in valid_starts(i) if (i, j, k) in X
                ) <= 1,
                name=f"c33_{i}_{f_idx}",
            )

    # --- (34) Replication overhead budget, per resource ---
    for res in RES:
        r_r = r_res[res]
        mdl.addConstr(
            gp.quicksum((q[i] - 1) * r_r[i] for i in I_C) <= Q_max_res[res],
            name=f"c34_{res}",
        )


    # --- Non-critical anti-affinity: jobs cannot share a server ---
    for i1, i2 in G:
        if not (i1 in I_C and i2 in I_C):
            for j in J:
                lhs = gp.quicksum(X[i1, j, k]
                                  for k in valid_starts(i1) if (i1, j, k) in X)
                rhs = gp.quicksum(X[i2, j, k]
                                  for k in valid_starts(i2) if (i2, j, k) in X)
                mdl.addConstr(lhs + rhs <= 1, name=f"c_nca_{i1}_{i2}_{j}")

    # -----------------------------
    # 4.5b Occupational noise constraints (36)-(42)
    # -----------------------------
    if ac is not None:
        # --- (36) Fan-law sound power, convex secant PWL (exact epigraph) ---
        #   SW[j,k] >= SA[j][p] * L_cpu[j,k] + SB[j][p] * y[j,k]   for all p
        # W(u) = W_full * n(u)^5 is convex in utilisation, so the secant
        # interpolation is the maximum of its affine pieces and needs no extra
        # binaries, no SOS2 sets and no big-M. Multiplying the intercept by
        # y[j,k] silences a server that is off or under PM, while an active
        # idle server is held at its idle sound power by the flat first piece.
        # Noise is driven by CPU load, consistent with (14)-(15) and (23).
        L_cpu = L_res["cpu"]
        for j in J:
            for k in K:
                for p in range(n_seg):
                    mdl.addConstr(
                        SW[j, k] >= SA[j][p] * L_cpu[j, k] + SB[j][p] * y[j, k],
                        name=f"c36_{j}_{k}_{p}",
                    )

        # --- (37) Zone acoustic aggregation (linear sound-power sums) ---
        #   ZW[g,k] = sum_g' Gamma[g][g'] * sum_{j in Z_g'} SW[j,k]
        # Valid because sound powers are in linear units; structurally the
        # acoustic analogue of the heat-recirculation term in (19).
        for g in Zn:
            for k in K:
                mdl.addConstr(
                    ZW[g, k] == gp.quicksum(
                        Gamma[g][g2] * gp.quicksum(
                            SW[j, k] for j in zone_servers[g2])
                        for g2 in Zn),
                    name=f"c37_{g}_{k}",
                )

        # --- (38)/(39) Occupancy linking: technician present <=> PM in zone ---
        # (38) forces occupancy whenever any server of the zone is under PM,
        #      which is the safety-binding direction.
        # (39) forbids phantom occupancy so the dose accounting in (41)-(42)
        #      stays honest.
        for g in Zn:
            for k in K:
                for j in zone_servers[g]:
                    mdl.addConstr(o[g, k] >= z[j, k], name=f"c38_{g}_{j}_{k}")
                mdl.addConstr(
                    o[g, k] <= gp.quicksum(
                        z[j, k] for j in zone_servers[g]),
                    name=f"c39_{g}_{k}",
                )

        # --- (40) Occupancy-activated acoustic cap ---
        #   ZW[g,k] <= W_unocc - (W_unocc - W_occ) * o[g,k]
        # Unoccupied zones face only the loose engineering limit; while a
        # technician is present the limit tightens to the occupational action
        # level. This is the central worker-safety coupling of the model.
        for g in Zn:
            for k in K:
                mdl.addConstr(
                    ZW[g, k] <= W_unocc - (W_unocc - W_occ) * o[g, k],
                    name=f"c40_{g}_{k}",
                )

        # --- (41a)-(41c) Occupied-noise variable D (exact linearization) ---
        # D[g,k] is the product ZW[g,k] * o[g,k] linearized: it equals the zone
        # noise when the zone is occupied and 0 when it is not. Because the noise
        # term was removed from the objective, D must be pinned from both sides
        # (a one-sided bound would float free), so all three inequalities are
        # required. W_unocc is a valid upper bound on ZW and serves as the big-M.
        for g in Zn:
            for k in K:
                mdl.addConstr(
                    Dz[g, k] <= W_unocc * o[g, k], name=f"c41a_{g}_{k}",
                )
                mdl.addConstr(
                    Dz[g, k] <= ZW[g, k], name=f"c41b_{g}_{k}",
                )
                mdl.addConstr(
                    Dz[g, k] >= ZW[g, k] - W_unocc * (1 - o[g, k]),
                    name=f"c41c_{g}_{k}",
                )

        # --- (42) Equal-energy noise-dose (TWA) budget per zone ---
        #   delta_t * sum_k Dz[g,k] <= E_max
        # The daily dose counts only occupied slots (Dz is 0 when unoccupied).
        # Repeated maintenance windows in one zone draw on a shared allowance,
        # which the per-slot cap (40) alone would not enforce. This is a hard
        # safety constraint; there is no accompanying objective penalty.
        for g in Zn:
            mdl.addConstr(
                delta_t * gp.quicksum(Dz[g, k] for k in K) <= E_max,
                name=f"c42_{g}",
            )

    # -----------------------------
    # 4.6 Solve
    # -----------------------------

    mdl.optimize()
    if mdl.status == GRB.INFEASIBLE:
        print("\nModel is infeasible. Computing IIS...")
        mdl.computeIIS()

        iis_path = f"infeasible_{scenario_name}.ilp"
        mdl.write(iis_path)

        print(f"\nIIS written to: {iis_path}")

        print("\nConstraints included in IIS:")
        for constr in mdl.getConstrs():
            if constr.IISConstr:
                print(constr.ConstrName)

        print("\nVariable bounds included in IIS:")
        for var in mdl.getVars():
            if var.IISLB or var.IISUB:
                print(
                    var.VarName,
                    "LB" if var.IISLB else "",
                    "UB" if var.IISUB else "",
                )
    feasible_solution = mdl.status in (
        GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and mdl.SolCount > 0

    result: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "model": mdl,
        "status_code": mdl.status,
        "status_label": get_status_label(mdl.status),
        "feasible_solution": feasible_solution,
        "data": data,
        "sets": {"I": I, "I_B": I_B, "I_V": I_V, "I_C": I_C, "J": J, "K": K},
        "params": {
            "d": d,
            "r": r_cpu,            # CPU demand (back-compat: `r` == CPU load)
            "r_cpu": r_cpu,
            "r_mem": r_mem,
            "q": q,
            "delta_t": delta_t,
            "Dk": Dk,
            "eta_air": eta_air,
            "eta_liq": eta_liq,
            "xi_pump": xi_pump,
            "phi": phi,
            "hybrid_cooling": hybrid_cooling,
        },
        "vars": {
            "X": X,
            "y": y,
            "d_on": d_on,
            "d_off": d_off,
            "m_j": m_j,
            "v": v,
            "z": z,
            "l_var": l_var,
            "L": L,                # CPU load (alias of L_res["cpu"])
            "L_cpu": L_res["cpu"],
            "L_mem": L_res["mem"],
            "H": H,
            "PIT": PIT,
            "Pcool": Pcool,
            "Ppump": Ppump,
            "Ptot": Ptot,
            "s": s,
            "psi": psi,
            "w": w,
            "SW": SW,
            "ZW": ZW,
            "o": o,
            "Dz": Dz,
        },
        "objective_terms": {
            "energy_cost": energy_cost,
            "pm_cost": pm_cost,
            "cm_cost": cm_cost,
            "switching_cost": sw_cost,
            "lateness_cost": late_cost,
        },
        "helpers": {"slot_to_time": local_slot_to_time},
    }

    return result


# -----------------------------
# Standalone entry point
# -----------------------------

def build_json_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a solve_datacenter_model() result into a JSON-serialisable dict.

    Drops the live Gurobi Model/Var objects and keeps only plain
    numbers/strings, reusing the same row extractors as the CSV pipeline
    (result_extractor.py) so the two output formats stay consistent.
    """
    mdl = result["model"]
    sets = result["sets"]
    # Precedence / affinity / anti-affinity relations are edge lists of job
    # pairs. Report the number of pairs (relations) in each: E holds
    # (predecessor, successor) precedence pairs, A holds affinity pairs, and G
    # holds non-affinity (anti-affinity) pairs.
    data_sets = result["data"]["sets"]
    num_predecessor_pairs = len(data_sets.get("E", []))
    num_affinity_pairs = len(data_sets.get("A", []))
    num_non_affinity_pairs = len(data_sets.get("G", []))
    # Structural scale dimensions. F is the set of failure domains (racks). The
    # acoustics block (present only in *_noise_split instances) defines the
    # acoustic zones g and a piecewise-linear noise curve whose segment count is
    # (breakpoints - 1); both are 0 when acoustics is absent.
    num_failure_domains = len(data_sets.get("F", []))
    ac = result["data"].get("acoustics")
    if ac is not None:
        num_acoustic_zones = len(ac.get("zones", []))
        num_noise_curve_segments = max(len(ac.get("breakpoints_u", [])) - 1, 0)
    else:
        num_acoustic_zones = 0
        num_noise_curve_segments = 0
    # Horizon-average PUE: total facility power over total IT power across all
    # slots (sum Ptot / sum PIT). Only defined when a feasible solution exists.
    ptot_pit_ratio = None
    if result["feasible_solution"]:
        K = result["sets"]["K"]
        vars_ = result["vars"]
        sum_pit = sum(vars_["PIT"][k].X for k in K)
        sum_ptot = sum(vars_["Ptot"][k].X for k in K)
        if sum_pit > 1e-9:
            ptot_pit_ratio = sum_ptot / sum_pit
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
            "num_predecessor_pairs": num_predecessor_pairs,
            "num_affinity_pairs": num_affinity_pairs,
            "num_non_affinity_pairs": num_non_affinity_pairs,
            "num_failure_domains": num_failure_domains,
            "num_acoustic_zones": num_acoustic_zones,
            "num_noise_curve_segments": num_noise_curve_segments,
            "runtime_seconds": getattr(mdl, "Runtime", None),
            "mip_gap": getattr(mdl, "MIPGap", None) if result["feasible_solution"] else None,
            "objective_value": mdl.ObjVal if result["feasible_solution"] else None,
            "num_variables": mdl.NumVars,
            "num_binary_variables": mdl.NumBinVars,
            "num_constraints": mdl.NumConstrs,
            "ptot_pit_ratio": ptot_pit_ratio,
        },
        "metrics": extract_performance_metrics(result),
        "solution": extract_solution_rows(result),
        "pm_schedule": extract_pm_rows(result),
        "hourly": extract_hourly_rows(result),
        "server_timeseries": extract_server_load_timeseries(result),
        "server_summary": extract_server_summary(result),
    }


# -----------------------------
# Solver-stats summary (CSV)
# -----------------------------

# Column order for the solver-statistics summary table. Mirrors the
# "solver_stats" block of each result payload, prefixed by the scenario
# identity columns so the CSV is self-describing.
SOLVER_STATS_COLUMNS = [
    "scenario_name",
    "status",
    "feasible_solution",
    "num_servers",
    "num_time_slots",
    "num_jobs",
    "num_jobs_I_B",
    "num_jobs_I_V",
    "num_jobs_I_C",
    "num_predecessor_pairs",
    "num_affinity_pairs",
    "num_non_affinity_pairs",
    "num_failure_domains",
    "num_acoustic_zones",
    "num_noise_curve_segments",
    "runtime_seconds",
    "mip_gap",
    "objective_value",
    "num_variables",
    "num_binary_variables",
    "num_constraints",
    "ptot_pit_ratio",
]


def solver_stats_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the flat solver-statistics row for one scenario's payload.

    Pulls the same fields shown in each result JSON's ``solver_stats`` block
    (plus the scenario name, status and feasibility flag) into a single flat
    dict suitable for a DataFrame row / CSV line.
    """
    stats = payload.get("solver_stats", {}) or {}
    return {
        "scenario_name": payload.get("scenario_name", payload.get("instance")),
        "status": payload.get("status"),
        "feasible_solution": payload.get("feasible_solution"),
        "num_servers": stats.get("num_servers"),
        "num_time_slots": stats.get("num_time_slots"),
        "num_jobs": stats.get("num_jobs"),
        "num_jobs_I_B": stats.get("num_jobs_I_B"),
        "num_jobs_I_V": stats.get("num_jobs_I_V"),
        "num_jobs_I_C": stats.get("num_jobs_I_C"),
        "num_predecessor_pairs": stats.get("num_predecessor_pairs"),
        "num_affinity_pairs": stats.get("num_affinity_pairs"),
        "num_non_affinity_pairs": stats.get("num_non_affinity_pairs"),
        "num_failure_domains": stats.get("num_failure_domains"),
        "num_acoustic_zones": stats.get("num_acoustic_zones"),
        "num_noise_curve_segments": stats.get("num_noise_curve_segments"),
        "runtime_seconds": stats.get("runtime_seconds"),
        "mip_gap": stats.get("mip_gap"),
        "objective_value": stats.get("objective_value"),
        "num_variables": stats.get("num_variables"),
        "num_binary_variables": stats.get("num_binary_variables"),
        "num_constraints": stats.get("num_constraints"),
        "ptot_pit_ratio": stats.get("ptot_pit_ratio"),
    }


def append_solver_stats_row(row: Dict[str, Any], csv_path: Path) -> None:
    """Append (or upsert) a single solver-stats row into a shared CSV.

    Reads any existing rows, replaces the row for the same scenario_name if it
    is already present (so re-solving an instance updates rather than
    duplicates it), then rewrites the file. Uses pandas when available and the
    csv module otherwise.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if csv_path.exists():
        try:
            import pandas as pd
            existing = pd.read_csv(csv_path).to_dict("records")
        except ImportError:
            import csv
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                existing = list(csv.DictReader(fh))
        except Exception:
            existing = []
    # Drop any prior row for this scenario, then append the fresh one.
    name = row.get("scenario_name")
    existing = [r for r in existing if str(r.get("scenario_name")) != str(name)]
    existing.append(row)
    write_solver_stats_csv(existing, csv_path)


def write_solver_stats_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Write the collected solver-statistics rows to a CSV file.

    Builds a pandas DataFrame when pandas is available (so the result is easy
    to load back for analysis) and falls back to the standard-library csv
    module otherwise, so the summary is produced regardless of environment.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # local import: optional dependency
        df = pd.DataFrame(rows, columns=SOLVER_STATS_COLUMNS)
        df.to_csv(csv_path, index=False)
    except ImportError:
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SOLVER_STATS_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in SOLVER_STATS_COLUMNS})


JOBS_PREFIX = "jobs_params_"
SERVER_PREFIX = "server_params_"


def discover_instances(experimental_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Pair up experimental data files into solvable instances.

    Each instance is defined by a ``jobs_params_<instance>.json`` file and a
    matching ``server_params_<instance>.json`` file in the same folder. The
    instance name is whatever follows the ``jobs_params_`` prefix (minus the
    ``.json`` suffix), e.g. ``T1_A_baseline_small``.

    Returns a list of (instance_name, jobs_path, server_path) sorted by name.
    Job files without a matching server file are skipped with a warning.
    """
    instances: List[Tuple[str, Path, Path]] = []
    for jobs_path in sorted(experimental_dir.glob(f"{JOBS_PREFIX}*.json")):
        instance = jobs_path.stem[len(JOBS_PREFIX):]
        server_path = experimental_dir / f"{SERVER_PREFIX}{instance}.json"
        if not server_path.exists():
            print(f"  [skip] No matching server file for instance "
                  f"'{instance}' (expected {server_path.name})")
            continue
        instances.append((instance, jobs_path, server_path))
    return instances


def _load_instance_data(jobs_json_path: Path, server_json_path: Path) -> Dict[str, Any]:
    """Load and assemble the optimization input dict for one instance.

    Wraps load_data_from_jobs_json and backfills the newer CPU/memory split
    fields straight from the raw JSON when the data_loader did not forward
    them, so the split always reaches solve_datacenter_model. Older scalar-only
    files are left untouched and handled by the in-solver fallback.
    """
    data = load_data_from_jobs_json(
        jobs_json_path=jobs_json_path,
        server_json_path=server_json_path,
    )

    def _backfill(container_key, raw_path):
        with raw_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for k, v in raw.get(container_key, {}).items():
            data.setdefault(container_key, {})
            if k not in data[container_key]:
                data[container_key][k] = v

    _backfill("job_params", jobs_json_path)
    _backfill("server_params", server_json_path)
    with server_json_path.open("r", encoding="utf-8") as fh:
        raw_server = json.load(fh)
    # The acoustics block is newer than some data_loader versions; read it
    # directly from the server JSON when the loader has not forwarded it.
    if "acoustics" not in data and "acoustics" in raw_server:
        data["acoustics"] = raw_server["acoustics"]
    # Forward hybrid-cooling fields (eta_air/eta_liq/phi/xi) if the loader
    # dropped them; older single-COP files are left untouched.
    if "cooling" in raw_server:
        data.setdefault("cooling", {})
        for kk, vv in raw_server["cooling"].items():
            data["cooling"].setdefault(kk, vv)
    for k in ("Q_max_cpu", "Q_max_mem"):
        if k in raw_server.get("redundancy", {}):
            data.setdefault("redundancy", {})
            data["redundancy"].setdefault(k, raw_server["redundancy"][k])
    return data


def solve_instance(
    instance: str,
    jobs_json_path: Path,
    server_json_path: Path,
    output_json_path: Path,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
    update_psi0: bool,
) -> Dict[str, Any]:
    """Solve a single instance and write its results to output_json_path.

    Returns the JSON-serialisable payload (also used to build the combined
    batch summary).
    """
    print(f"\n=== Instance: {instance} ===")
    print(f"  jobs   : {jobs_json_path}")
    print(f"  server : {server_json_path}")

    data = _load_instance_data(jobs_json_path, server_json_path)

    result = solve_datacenter_model(
        data=data,
        scenario_name=instance,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )

    if update_psi0:
        update_psi_0(result, server_json_path)

    payload = build_json_result(result)
    payload["instance"] = instance

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  Status: {result['status_label']}")
    if result["feasible_solution"]:
        print(f"  Objective value: {result['model'].ObjVal:.2f}")
    print(f"  Results written to: {output_json_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the data-centre scheduling MILP (hazard-included). By "
            "default, batch-solves every instance found in the experimental "
            "data folder and writes one JSON result file per instance. Pass "
            "both --jobs-json-input and --server-json-input to solve a single "
            "instance instead."
        )
    )
    parser.add_argument(
        "--experimental-dir",
        default=str(_EXPERIMENTAL_DATA_DIR),
        help="Folder of jobs_params_*/server_params_* files to batch-solve.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Folder to write per-instance result JSON files into.",
    )
    parser.add_argument(
        "--jobs-json-input",
        default=None,
        help="Single-instance mode: path to a jobs_params_*.json input file.",
    )
    parser.add_argument(
        "--server-json-input",
        default=None,
        help="Single-instance mode: path to a server_params_*.json input file.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Single-instance mode: path to write the result JSON to.",
    )
    parser.add_argument("--time-limit", type=int, default=7200,
                        help="Gurobi time limit in seconds (per instance).")
    parser.add_argument("--mip-gap", type=float,
                        default=0.0001, help="Target MIP gap (0.01 %%).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full Gurobi solver log.")
    parser.add_argument(
        "--update-psi0",
        action="store_true",
        help=(
            "After solving, write end-of-horizon wear back to "
            "server_params.psi_0 in the instance's server JSON so it carries "
            "forward into the next day's run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Single-instance mode: both explicit input files given.
    if args.jobs_json_input and args.server_json_input:
        jobs_json_path = Path(args.jobs_json_input)
        server_json_path = Path(args.server_json_input)
        instance = jobs_json_path.stem[len(JOBS_PREFIX):] \
            if jobs_json_path.stem.startswith(JOBS_PREFIX) else jobs_json_path.stem
        output_json_path = Path(args.output_json) if args.output_json else \
            Path(args.output_dir) / f"result_{instance}.json"
        payload = solve_instance(
            instance=instance,
            jobs_json_path=jobs_json_path,
            server_json_path=server_json_path,
            output_json_path=output_json_path,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            verbose=args.verbose,
            update_psi0=args.update_psi0,
        )
        # Append this scenario's solver statistics to a shared CSV in the same
        # folder, so solving instances one at a time still accumulates a single
        # combined table (existing rows for other scenarios are preserved).
        solver_stats_csv_path = output_json_path.parent / "optimizer_v0.1_stats_summary.csv"
        append_solver_stats_row(solver_stats_row(payload), solver_stats_csv_path)
        print(f"  Solver-stats CSV: {solver_stats_csv_path}")
        return

    # Batch mode: solve every instance in the experimental data folder.
    experimental_dir = Path(args.experimental_dir)
    output_dir = Path(args.output_dir)
    instances = discover_instances(experimental_dir)

    if not instances:
        print(f"No instances found in {experimental_dir} "
              f"(expected {JOBS_PREFIX}*.json / {SERVER_PREFIX}*.json pairs).")
        return

    print(f"Found {len(instances)} instance(s) in {experimental_dir}:")
    for instance, _, _ in instances:
        print(f"  - {instance}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batch_summary.json"
    solver_stats_csv_path = output_dir / "optimizer_v0.1_stats_summary.csv"

    summary_rows = []
    solver_stats_rows = []
    for instance, jobs_json_path, server_json_path in instances:
        output_json_path = output_dir / f"result_{instance}.json"
        try:
            payload = solve_instance(
                instance=instance,
                jobs_json_path=jobs_json_path,
                server_json_path=server_json_path,
                output_json_path=output_json_path,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                verbose=args.verbose,
                update_psi0=args.update_psi0,
            )
            summary_rows.append({
                "instance": instance,
                "status": payload["status"],
                "feasible_solution": payload["feasible_solution"],
                "objective_value": payload["solver_stats"]["objective_value"],
                "runtime_seconds": payload["solver_stats"]["runtime_seconds"],
                "mip_gap": payload["solver_stats"]["mip_gap"],
                "result_file": str(output_json_path),
            })
            solver_stats_rows.append(solver_stats_row(payload))
        except Exception as exc:  # noqa: BLE001 - keep batch going on failure
            print(f"  [error] Instance '{instance}' failed: {exc}")
            summary_rows.append({
                "instance": instance,
                "status": "ERROR",
                "feasible_solution": False,
                "objective_value": None,
                "runtime_seconds": None,
                "mip_gap": None,
                "error": str(exc),
                "result_file": None,
            })
            solver_stats_rows.append({
                "scenario_name": instance,
                "status": "ERROR",
                "feasible_solution": False,
                "runtime_seconds": None,
                "mip_gap": None,
                "objective_value": None,
                "num_variables": None,
                "num_binary_variables": None,
                "num_constraints": None,
            })

        # Rewrite the combined summary after every instance so progress is
        # persisted incrementally and survives an interrupted batch.
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)
        # Persist the solver-statistics table as CSV after every instance so a
        # partial run still yields a usable summary if the batch is interrupted.
        write_solver_stats_csv(solver_stats_rows, solver_stats_csv_path)

    solved = sum(1 for row in summary_rows if row["feasible_solution"])
    print(f"\nBatch finished: {solved}/{len(summary_rows)} instance(s) "
          f"produced a feasible solution.")
    print(f"Per-instance results: {output_dir}")
    print(f"Combined summary: {summary_path}")
    print(f"Solver-stats CSV: {solver_stats_csv_path}")


if __name__ == "__main__":
    main()