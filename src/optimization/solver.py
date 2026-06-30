"""MILP builder and solver for the data-centre scheduling model."""

from typing import Any, Dict, List, Tuple
import math
import os

import gurobipy as gp
from gurobipy import GRB

from .utils import get_status_label, slot_to_time


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
    r = {int(k): v for k, v in jp["r"].items()}
    a = {int(k): v for k, v in jp["a"].items()}
    b = {int(k): v for k, v in jp["b"].items()}
    q = {int(k): v for k, v in jp["q"].items()}
    rho = {int(k): v for k, v in jp["rho"].items()}

    # Non-critical jobs need one replica by definition.
    for i in I:
        if i not in q:
            q[i] = 1

    sp = data["server_params"]
    C = {int(k): v for k, v in sp["C"].items()}
    theta = {int(k): v for k, v in sp["theta"].items()}
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

    eta = data["cooling"]["eta"]
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
    Q_max = data["redundancy"]["Q_max"]
    delta_t = data["slot_duration"]
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
    L = {(j, k): mdl.addVar(lb=0.0, ub=GRB.INFINITY, name=f"L_{j}_{k}")
         for j in J for k in K}
    H = {(j, k): mdl.addVar(lb=0.0, name=f"H_{j}_{k}") for j in J for k in K}
    PIT = {k: mdl.addVar(lb=0.0, name=f"PIT_{k}") for k in K}
    Pcool = {k: mdl.addVar(lb=0.0, name=f"Pcool_{k}") for k in K}
    Ptot = {k: mdl.addVar(lb=0.0, name=f"Ptot_{k}") for k in K}
    s = {i: mdl.addVar(lb=0.0, ub=nK - 1, name=f"s_{i}") for i in I}

    # M_psi: tight upper bound for cumulative wear psi[j,k] and auxiliary w[j,k].
    # Worst case: server runs at full capacity every slot from maximum inherited load.
    M_psi = max(psi_0.values(), default=0.0) + nK * max(C.values())
    psi = {(j, k): mdl.addVar(lb=0.0, ub=M_psi, name=f"psi_{j}_{k}")
           for j in J for k in K}
    # w[j,k]: pre-reset accumulation = psi[j,k-1] + L[j,k].
    # Separating w from psi allows a clean linearisation of the conditional
    # reset without creating contradictory equality/inequality pairs on psi.
    w = {(j, k): mdl.addVar(lb=0.0, ub=M_psi, name=f"w_{j}_{k}")
         for j in J for k in K}


    mdl.update()

    # -----------------------------
    # 4.4 Objective (#4)
    # -----------------------------
    energy_cost = delta_t * gp.quicksum(c_e[k] * Ptot[k] for k in K)
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
    late_cost = gp.quicksum(rho[i] * l_var[i] for i in I_B)

    mdl.setObjective(energy_cost + pm_cost + cm_cost +
                     sw_cost + late_cost, GRB.MINIMIZE)

    # -----------------------------
    # 4.5 Constraints
    # -----------------------------

    # --- #5 Job assignment (exact replica count) ---
    for i in I:
        mdl.addConstr(
            gp.quicksum(X[i, j, k] for j in S[i]
                        for k in valid_starts(i)) == q[i],
            name=f"c5_{i}",
        )

    # --- #6/#7 Release time and interactive hard deadlines are enforced in valid_starts(). ---

    # --- #8 Precedence ---
    for i_pred, i_succ in E:
        mdl.addConstr(s[i_succ] >= s[i_pred] + d[i_pred],
                      name=f"c8_{i_pred}_{i_succ}")

    # --- Start-time definition ---
    # to capture the latest start among all replicas since staggering is allowed in the model
    # not all replicas start at the same time
    for i in I:
        for j in S[i]:
            for k in valid_starts(i):
                mdl.addConstr(s[i] >= k * X[i, j, k], name=f"cs_{i}_{j}_{k}")

    # --- #9 Batch-only capacity (interactive reservation) ---
    for j in J:
        for k in K:
            batch_load = gp.quicksum(
                r[i] * X[i, j, kp]
                for i in I_B if j in S[i]
                for kp in running_at(i, j, k)
            )
            mdl.addConstr(batch_load <= (
                1 - theta[j]) * C[j] * y[j, k], name=f"c9_{j}_{k}")

    # --- #10 Batch lateness ---
    for i in I_B:
        if i not in I_C:
            mdl.addConstr(l_var[i] >= s[i] + d[i] - b[i], name=f"c10nc_{i}")

    M_lat = nK
    for i in I_B:
        if i in I_C:
            for j in S[i]:
                for k in valid_starts(i):
                    mdl.addConstr(
                        l_var[i] >= k + d[i] - b[i] - M_lat * (1 - X[i, j, k]),
                        name=f"c10cr_{i}_{j}_{k}",
                    )

    # --- #12/#13 Load definition and server capacity ---
    for j in J:
        for k in K:
            load_expr = gp.quicksum(
                r[i] * X[i, j, kp]
                for i in I if j in S[i]
                for kp in running_at(i, j, k)
            )
            mdl.addConstr(L[j, k] == load_expr, name=f"c13_{j}_{k}")
            mdl.addConstr(L[j, k] <= C[j] * y[j, k], name=f"c12_{j}_{k}")

    # --- #14 Server cannot be active and under preventive maintenance simultaneously ---
    for j in J:
        for k in K:
            mdl.addConstr(y[j, k] + z[j, k] <= 1, name=f"c14_{j}_{k}")

    # --- #15 Total IT power ---
    for k in K:
        mdl.addConstr(
            PIT[k] == gp.quicksum(P0[j] * y[j, k] + dP[j]
                                  * L[j, k] for j in J),
            name=f"c15_{k}",
        )

    # --- #16 Heat per server ---
    for j in J:
        for k in K:
            mdl.addConstr(
                H[j, k] == alpha[j] * (P0[j] * y[j, k] + dP[j] * L[j, k]),
                name=f"c16_{j}_{k}",
            )

    # --- #17 Cooling power ---
    for k in K:
        mdl.addConstr(Pcool[k] == (1.0 / eta[0]) *
                      gp.quicksum(H[j, k] for j in J), name=f"c17_{k}")

    # --- #18 Total facility power ---
    for k in K:
        mdl.addConstr(Ptot[k] == PIT[k] + Pcool[k] + P_ov, name=f"c18_{k}")

    # --- #20 PUE cap ---
    for k in K:
        mdl.addConstr(Ptot[k] <= Pi_max * PIT[k], name=f"c20_{k}")

    # --- #21 Thermal: server inlet temperature ---
    for j in J:
        for k in K:
            recirc = gp.quicksum(
                D[j][jp2] * alpha[jp2] *
                (P0[jp2] * y[jp2, k] + dP[jp2] * L[jp2, k])
                for jp2 in J
            )
            mdl.addConstr(
                T_sup + recirc <= T_idle -
                (T_idle - T_busy) * y[j, k] + M_big * z[j, k],
                name=f"c21_{j}_{k}",
            )

    # --- #22 PM count per server ---
    for j in J:
        mdl.addConstr(gp.quicksum(v[j, k]
                      for k in pm_starts) == m_j[j], name=f"c22_{j}")

    # --- #23 PM active window ---
    for j in J:
        for k in K:
            win = [kp for kp in pm_starts if max(0, k - d_pm + 1) <= kp <= k]
            mdl.addConstr(z[j, k] == gp.quicksum(v[j, kp]
                          for kp in win), name=f"c23_{j}_{k}")

    # --- #24 Max servers under PM per slot ---
    for k in K:
        mdl.addConstr(gp.quicksum(z[j, k] for j in J)
                      <= len(J) - N_min, name=f"c24_{k}")

    # --- #25 Cumulative wear with psi_0 and linearised conditional reset ---
    #
    # Goal: psi[j,k] = (psi[j,k-1] + L[j,k]) * (1 - z[j,k])
    #   i.e. accumulate load every slot, but reset to 0 whenever the server
    #   is under PM (z[j,k]=1).  The product is non-linear, so we introduce
    #   auxiliary variable w[j,k] = pre-reset accumulation and linearise:
    #
    #   w[j,k]   = psi_0[j] + L[j,0]            if k == 0      (c25w_init)
    #   w[j,k]   = psi[j,k-1] + L[j,k]          if k  > 0      (c25w)
    #   psi[j,k] <= w[j,k]                                       (c25a)
    #   psi[j,k] <= M_psi * (1 - z[j,k])                        (c25b)
    #   psi[j,k] >= w[j,k] - M_psi * z[j,k]                    (c25c)
    #
    # When z[j,k]=0: c25b is inactive, c25a+c25c force psi[j,k] = w[j,k].
    # When z[j,k]=1: c25b forces psi[j,k] = 0, c25c is inactive (RHS <= 0).
    # There is no contradictory equality: w always equals the raw accumulation
    # and psi is the reset-gated version.
    for j in J:
        for k in K:
            if k == 0:
                mdl.addConstr(
                    w[j, k] == psi_0[j] + L[j, k],
                    name=f"c25w_init_{j}",
                )
            else:
                mdl.addConstr(
                    w[j, k] == psi[j, k - 1] + L[j, k],
                    name=f"c25w_{j}_{k}",
                )
            mdl.addConstr(psi[j, k] <= w[j, k],
                          name=f"c25a_{j}_{k}")
            mdl.addConstr(psi[j, k] <= M_psi * (1 - z[j, k]),
                          name=f"c25b_{j}_{k}")
            mdl.addConstr(psi[j, k] >= w[j, k] - M_psi * z[j, k],
                          name=f"c25c_{j}_{k}")

    # --- #26 PM trigger: only allowed once wear reaches threshold Lambda[j] ---
    # v[j,k]=1 is only feasible when psi[j,k] >= Lambda[j].
    # Because psi is already 0 during PM slots (c25b), this constraint is
    # evaluated on the slot immediately before the PM window starts, which
    # is the last slot where psi still reflects real accumulated wear.

    for j in J:
        for k in pm_starts:
            if k == 0:
                mdl.addConstr(
                    psi_0[j] >= Lambda[j] * v[j, k],
                    name=f"c26_init_{j}_{k}",
                )
            else:
                mdl.addConstr(
                    w[j, k-1] >= Lambda[j] * v[j, k],
                    name=f"c26_{j}_{k}",
                    )

    # --- Force PM if server starts the horizon at or above threshold ---
    for j in J:
        if psi_0[j] >= Lambda[j]:
            mdl.addConstr(
                gp.quicksum(v[j, k] for k in pm_starts) == 1,
                name = f"force_pm_initial_wear_{j}",
            )


    # --- #27/#28 Server state-change tracking ---
    for j in J:
        for k in K[:-1]:
            mdl.addConstr(y[j, k + 1] - y[j, k] <=
                          d_on[j, k], name=f"c27_{j}_{k}")
            mdl.addConstr(y[j, k] - y[j, k + 1] <=
                          d_off[j, k], name=f"c28_{j}_{k}")

    # --- #29 Total switching budget ---
    mdl.addConstr(gp.quicksum(d_on[j, k] + d_off[j, k]
                  for j in J for k in K[:-1]) <= S_max, name="c29")

    # --- #30 Forecast demand representation ---
    # The synthetic forecast is represented directly as aggregate workload jobs.
    # Therefore, the old hard aggregate demand constraint is not added here.
    # Adding it would double-count the same forecast demand and can make the
    # model infeasible.

    # --- #31 Anti-affinity isolation for critical job pairs ---
    for i1, i2 in G:
        if i1 in I_C and i2 in I_C:
            for j in J:
                for k in K:
                    r1 = gp.quicksum(X[i1, j, kp]
                                     for kp in running_at(i1, j, k))
                    r2 = gp.quicksum(X[i2, j, kp]
                                     for kp in running_at(i2, j, k))
                    mdl.addConstr(r1 + r2 <= 1, name=f"c31_{i1}_{i2}_{j}_{k}")

    # --- #32 Affinity: same server ---
    for i1, i2 in A:
        for j in J:
            s1 = gp.quicksum(X[i1, j, k]
                             for k in valid_starts(i1) if (i1, j, k) in X)
            s2 = gp.quicksum(X[i2, j, k]
                             for k in valid_starts(i2) if (i2, j, k) in X)
            mdl.addConstr(s1 == s2, name=f"c32_{i1}_{i2}_{j}")

    # --- #33 Affinity critical pairs: same replica count data check ---
    for i1, i2 in A:
        if i1 in I_C:
            assert q[i1] == q[i2], f"Affinity pair ({i1},{i2}): replica counts must match"

    # --- #34 Hot standby buffer ---
    for k in K:
        mdl.addConstr(gp.quicksum(y[j, k]
                      for j in J) >= N_min + kappa, name=f"c34_{k}")

    # --- #35 Rack diversity for critical jobs ---
    for i in I_C:
        for f_idx, Ff in enumerate(F):
            mdl.addConstr(
                gp.quicksum(
                    X[i, j, k]
                    for j in Ff if j in S[i]
                    for k in valid_starts(i) if (i, j, k) in X
                ) <= 1,
                name=f"c35_{i}_{f_idx}",
            )

    # --- #36 Replication overhead budget ---
    mdl.addConstr(gp.quicksum((q[i] - 1) * r[i]
                  for i in I_C) <= Q_max, name="c36")


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
        "params": {"d": d, "r": r, "q": q, "delta_t": delta_t, "Dk": Dk, "eta": eta},
        "vars": {
            "X": X,
            "y": y,
            "d_on": d_on,
            "d_off": d_off,
            "m_j": m_j,
            "v": v,
            "z": z,
            "l_var": l_var,
            "L": L,
            "H": H,
            "PIT": PIT,
            "Pcool": Pcool,
            "Ptot": Ptot,
            "s": s,
            "psi": psi,
            "w": w,
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
