"""Extract CSV-ready result tables from solved optimization models."""

from collections import defaultdict
from typing import Any, Dict, List
import math

from .utils import safe_value


def extract_solution_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create job-level optimization_solution rows."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    I_B = result["sets"]["I_B"]
    I_C = result["sets"]["I_C"]
    d = result["params"]["d"]
    delta_t = result["params"]["delta_t"]
    X = result["vars"]["X"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    rows = []
    for (i, j, k), var in sorted(X.items()):
        if var.X > 0.5:
            rows.append({
                "scenario": scenario,
                "job_id": i,
                "job_type": "batch" if i in I_B else "interactive",
                "is_critical": int(i in I_C),
                "server_id": j,
                "start_slot": k,
                "end_slot": k + d[i],
                "start_time": local_slot_to_time(k),
                "end_time": local_slot_to_time(k + d[i]),
                "duration_hours": d[i] * delta_t,
            })
    return rows


def extract_hourly_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create hourly power/load rows for tables and analysis."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    J = result["sets"]["J"]
    K = result["sets"]["K"]
    Dk = result["params"]["Dk"]
    # Cooling COP. The hybrid-cooling refactor renamed the legacy scalar "eta"
    # to the air-path COP "eta_air"; fall back to it (and still support the old
    # per-slot dict form) so the COP column keeps working either way.
    params = result["params"]
    eta = params.get("eta", params.get("eta_air"))
    local_slot_to_time = result["helpers"]["slot_to_time"]

    y = result["vars"]["y"]
    L = result["vars"]["L"]
    PIT = result["vars"]["PIT"]
    Pcool = result["vars"]["Pcool"]
    Ptot = result["vars"]["Ptot"]

    rows = []
    for k in K:
        pit = PIT[k].X
        pcool = Pcool[k].X
        ptot = Ptot[k].X
        total_load = sum(L[j, k].X for j in J)
        rows.append({
            "scenario": scenario,
            "slot": k,
            "time": local_slot_to_time(k),
            "demand": Dk[k],
            "served_load": total_load,
            "active_servers": sum(1 for j in J if y[j, k].X > 0.5),
            "PIT_W": pit,
            "Pcool_W": pcool,
            "Ptot_W": ptot,
            "PUE": ptot / pit if pit > 1e-6 else math.nan,
            "COP": eta[k] if isinstance(eta, dict) else eta,
        })
    return rows


def extract_pm_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create preventive-maintenance schedule rows."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    J = result["sets"]["J"]
    d_pm = result["data"]["maintenance"]["d_pm"]
    K = result["sets"]["K"]
    nK = len(K)
    pm_starts = [k for k in K if k <= nK - d_pm]
    v = result["vars"]["v"]
    m_j = result["vars"]["m_j"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    rows = []
    for j in J:
        scheduled = int(m_j[j].X > 0.5)
        starts = [k for k in pm_starts if v[j, k].X > 0.5]
        if not starts:
            rows.append({
                "scenario": scenario,
                "server_id": j,
                "pm_scheduled": scheduled,
                "pm_start_slot": "",
                "pm_end_slot": "",
                "pm_start_time": "",
                "pm_end_time": "",
            })
        for k in starts:
            rows.append({
                "scenario": scenario,
                "server_id": j,
                "pm_scheduled": scheduled,
                "pm_start_slot": k,
                "pm_end_slot": k + d_pm,
                "pm_start_time": local_slot_to_time(k),
                "pm_end_time": local_slot_to_time(k + d_pm),
            })
    return rows


def extract_performance_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create one scenario-level performance metrics row."""
    scenario = result["scenario_name"]
    mdl = result["model"]

    metrics = {
        "scenario": scenario,
        "status": result["status_label"],
        "has_solution": int(result["feasible_solution"]),
        "objective_value": "",
        "mip_gap_pct": "",
        "runtime_seconds": safe_value(getattr(mdl, "Runtime", 0.0)),
        "num_variables": mdl.NumVars,
        "num_binary_variables": mdl.NumBinVars,
        "num_constraints": mdl.NumConstrs,
        "jobs_in_forecast": "",
        "jobs_placed": "",
        "jobs_on_time": "",
        "jobs_late_count": "",
        "energy_cost": "",
        "pm_cost": "",
        "expected_cm_cost": "",
        "switching_cost": "",
        "lateness_cost": "",
        "total_cost": "",
        "total_facility_energy_kwh": "",
        "average_pue": "",
        "max_pue": "",
        "total_switching_events": "",
        "total_lateness_hours": "",
    }

    if not result["feasible_solution"]:
        return metrics

    I = result["sets"]["I"]
    K = result["sets"]["K"]
    I_B = result["sets"]["I_B"]
    J = result["sets"]["J"]
    delta_t = result["params"]["delta_t"]
    d_on = result["vars"]["d_on"]
    d_off = result["vars"]["d_off"]
    l_var = result["vars"]["l_var"]
    PIT = result["vars"]["PIT"]
    Ptot = result["vars"]["Ptot"]
    X = result["vars"]["X"]

    placed_jobs = set(i for (i, j, k), var in X.items() if var.X > 0.5)
    late_jobs = set(i for i in I_B if l_var[i].X > 1e-6)

    pue_values = [Ptot[k].X / PIT[k].X for k in K if PIT[k].X > 1e-6]
    total_energy_kwh = sum(Ptot[k].X * delta_t / 1000.0 for k in K)
    total_switching = sum(
        d_on[j, k].X + d_off[j, k].X for j in J for k in K[:-1])
    total_lateness_hours = sum(max(0.0, l_var[i].X) * delta_t for i in I_B)

    objective_terms = result["objective_terms"]
    energy_cost_val = safe_value(objective_terms["energy_cost"])
    pm_cost_val = safe_value(objective_terms["pm_cost"])
    cm_cost_val = safe_value(objective_terms["cm_cost"])
    sw_cost_val = safe_value(objective_terms["switching_cost"])
    late_cost_val = safe_value(objective_terms["lateness_cost"])

    metrics.update({
        "objective_value": mdl.ObjVal,
        "mip_gap_pct": 100.0 * mdl.MIPGap,
        "jobs_in_forecast": len(I),
        "jobs_placed": len(placed_jobs),
        "jobs_on_time": len(placed_jobs) - len(late_jobs),
        "jobs_late_count": len(late_jobs),
        "energy_cost": energy_cost_val,
        "pm_cost": pm_cost_val,
        "expected_cm_cost": cm_cost_val,
        "switching_cost": sw_cost_val,
        "lateness_cost": late_cost_val,
        "total_cost": mdl.ObjVal,
        "total_facility_energy_kwh": total_energy_kwh,
        "average_pue": sum(pue_values) / len(pue_values) if pue_values else math.nan,
        "max_pue": max(pue_values) if pue_values else math.nan,
        "total_switching_events": total_switching,
        "total_lateness_hours": total_lateness_hours,
    })
    return metrics


def extract_server_load_timeseries(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Per-server per-slot load timeseries for Power BI line charts and semaphore.

    Columns: scenario, server_id, server_type, slot, time,
             load, batch_load, interactive_load, active, in_maintenance, status

    Powers: line chart of server usage, peak hours, semaphore traffic light.
    """
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    I_B = result["sets"]["I_B"]
    J = result["sets"]["J"]
    K = result["sets"]["K"]
    nK = len(K)
    d = result["params"]["d"]
    r = result["params"]["r"]
    d_pm = result["data"]["maintenance"]["d_pm"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    X = result["vars"]["X"]
    y = result["vars"]["y"]
    L = result["vars"]["L"]
    m_j = result["vars"]["m_j"]
    v = result["vars"]["v"]

    server_type = {j: "GPU" if j < 34 else "CPU" for j in J}
    pm_starts_list = [k for k in K if k <= nK - d_pm]

    # Maintenance slots per server
    pm_slots: Dict[int, set] = {j: set() for j in J}
    for j in J:
        if m_j[j].X > 0.5:
            for k in pm_starts_list:
                if v[j, k].X > 0.5:
                    for kk in range(k, min(k + d_pm, nK)):
                        pm_slots[j].add(kk)

    # Batch / interactive load split per (server, slot)
    batch_jk: Dict[tuple, float] = {(j, k): 0.0 for j in J for k in K}
    inter_jk: Dict[tuple, float] = {(j, k): 0.0 for j in J for k in K}
    for (i, j, k_start), var in X.items():
        if var.X > 0.5:
            for kk in range(k_start, k_start + d[i]):
                if kk in K:
                    if i in I_B:
                        batch_jk[j, kk] += r[i]
                    else:
                        inter_jk[j, kk] += r[i]

    rows = []
    for j in J:
        for k in K:
            is_active = y[j, k].X > 0.5
            in_maint = k in pm_slots[j]
            status = "maintenance" if in_maint else ("active" if is_active else "idle")
            rows.append({
                "scenario": scenario,
                "server_id": j,
                "server_type": server_type[j],
                "slot": k,
                "time": local_slot_to_time(k),
                "load": round(L[j, k].X, 6),
                "batch_load": round(batch_jk[j, k], 6),
                "interactive_load": round(inter_jk[j, k], 6),
                "active": int(is_active),
                "in_maintenance": int(in_maint),
                "status": status,
            })
    return rows


def extract_server_summary(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Per-server aggregate stats for Power BI bar charts and utilization table.

    Columns: scenario, server_id, server_type, slots_active, avg_load,
             max_load, batch_avg_load, interactive_avg_load,
             utilization_rate_pct, pm_scheduled, pm_start_time,
             pm_end_time, final_status

    Powers: highest-utilization server, load by job type, PM semaphore.
    """
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    J = result["sets"]["J"]
    K = result["sets"]["K"]
    nK = len(K)
    d_pm = result["data"]["maintenance"]["d_pm"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    m_j = result["vars"]["m_j"]
    v = result["vars"]["v"]

    pm_starts_list = [k for k in K if k <= nK - d_pm]
    server_type = {j: "GPU" if j < 34 else "CPU" for j in J}

    ts_rows = extract_server_load_timeseries(result)
    by_server: Dict[int, list] = defaultdict(list)
    for row in ts_rows:
        by_server[row["server_id"]].append(row)

    rows = []
    for j in J:
        srows = by_server[j]
        loads = [row["load"] for row in srows]
        batch_loads = [row["batch_load"] for row in srows]
        inter_loads = [row["interactive_load"] for row in srows]
        active_count = sum(1 for row in srows if row["active"])

        pm_sched = int(m_j[j].X > 0.5)
        pm_start_time, pm_end_time = "", ""
        if pm_sched:
            starts = [k for k in pm_starts_list if v[j, k].X > 0.5]
            if starts:
                k_pm = starts[0]
                pm_start_time = local_slot_to_time(k_pm)
                pm_end_time = local_slot_to_time(k_pm + d_pm)

        in_maint_any = any(row["in_maintenance"] for row in srows)
        final_status = "maintenance" if in_maint_any else ("active" if active_count > 0 else "idle")

        rows.append({
            "scenario": scenario,
            "server_id": j,
            "server_type": server_type[j],
            "slots_active": active_count,
            "avg_load": round(sum(loads) / nK, 6) if nK else 0.0,
            "max_load": round(max(loads), 6) if loads else 0.0,
            "batch_avg_load": round(sum(batch_loads) / nK, 6) if nK else 0.0,
            "interactive_avg_load": round(sum(inter_loads) / nK, 6) if nK else 0.0,
            "utilization_rate_pct": round(100.0 * active_count / nK, 2) if nK else 0.0,
            "pm_scheduled": pm_sched,
            "pm_start_time": pm_start_time,
            "pm_end_time": pm_end_time,
            "final_status": final_status,
        })
    return rows
