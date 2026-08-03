"""Load forecast output and convert it into optimization input data."""

from pathlib import Path
from typing import Any, Dict, List
import math
import json


def _normalize_cooling(cooling, K):
    """Ensure cooling["eta"] is always a dict keyed by slot index."""
    eta = cooling.get("eta")
    if isinstance(eta, (int, float)):
        cooling = dict(cooling)
        cooling["eta"] = {k: float(eta) for k in K}
    elif isinstance(eta, list):
        cooling = dict(cooling)
        cooling["eta"] = {k: float(eta[k]) for k in K}
    return cooling


def load_data_from_jobs_json(
    jobs_json_path: Path,
    server_json_path: Path,
) -> Dict[str, Any]:
    """
    Load forecast-translated job parameters and server infrastructure parameters
    from JSON files, then merge them into the complete optimization input
    dictionary expected by solver.py.

    Inputs:
        jobs_json_path:
            data/processed/optimization_jobs_params.json

        server_json_path:
            server_params_42servers_v6.json
    """

    if not jobs_json_path.exists():
        raise FileNotFoundError(
            f"Jobs JSON file not found: {jobs_json_path}. "
            "Run build_jobs_json_from_forecast.py first."
        )

    if not server_json_path.exists():
        raise FileNotFoundError(
            f"Server parameters JSON file not found: {server_json_path}."
        )

    with open(jobs_json_path, "r", encoding="utf-8") as file:
        jobs_data = json.load(file)

    with open(server_json_path, "r", encoding="utf-8") as file:
        server_data = json.load(file)

    required_job_sections = [
        "sets",
        "eligibility",
        "job_params",
    ]

    missing_job_sections = [
        section for section in required_job_sections
        if section not in jobs_data
    ]

    if missing_job_sections:
        raise ValueError(
            f"Missing required sections in {jobs_json_path}: "
            f"{missing_job_sections}. "
            f"Available sections: {list(jobs_data.keys())}"
        )

    required_server_sections = [
        "sets",
        "server_params",
        "thermal",
        "cooling",
        "power",
        "maintenance",
        "costs",
        "demand",
        "redundancy",
        "slot_duration",
    ]

    missing_server_sections = [
        section for section in required_server_sections
        if section not in server_data
    ]

    if missing_server_sections:
        raise ValueError(
            f"Missing required sections in {server_json_path}: "
            f"{missing_server_sections}. "
            f"Available sections: {list(server_data.keys())}"
        )

    job_sets = jobs_data["sets"]
    server_sets = server_data["sets"]

    required_job_sets = [
        "I",
        "I_B",
        "I_V",
        "I_C",
        "E",
        "A",
        "G",
    ]

    missing_job_sets = [
        set_name for set_name in required_job_sets
        if set_name not in job_sets
    ]

    if missing_job_sets:
        raise ValueError(
            f"Missing required job sets in {jobs_json_path}: "
            f"{missing_job_sets}."
        )

    required_server_sets = [
        "J",
        "K",
        "F",
    ]

    missing_server_sets = [
        set_name for set_name in required_server_sets
        if set_name not in server_sets
    ]

    if missing_server_sets:
        raise ValueError(
            f"Missing required server sets in {server_json_path}: "
            f"{missing_server_sets}."
        )

    I = job_sets["I"]
    I_B = job_sets["I_B"]
    I_V = job_sets["I_V"]
    I_C = job_sets["I_C"]

    J = server_sets["J"]
    K = server_sets["K"]
    F = server_sets["F"]

    E = job_sets.get("E", [])
    A = job_sets.get("A", [])
    G = job_sets.get("G", [])

    eligibility = jobs_data["eligibility"]
    job_params = jobs_data["job_params"]

    valid_servers = set(J)

    for job_id, servers in eligibility.items():
        invalid_servers = [
            server for server in servers
            if server not in valid_servers
        ]

        if invalid_servers:
            raise ValueError(
                f"Job {job_id} has invalid eligible servers: "
                f"{invalid_servers}. "
                f"Valid servers are: {min(J)} to {max(J)}."
            )

    required_job_params = [
        "d",
        "a",
        "b",
        "q",
        "c_late",
    ]

    missing_job_params = [
        param for param in required_job_params
        if param not in job_params
    ]

    # Resource demand is either the legacy scalar "r", or the newer
    # "r_cpu"/"r_mem" split (see solve_datacenter_model's CPU/memory
    # vector-packing support). At least one form must be present.
    has_resource_demand = "r" in job_params or (
        "r_cpu" in job_params and "r_mem" in job_params
    )
    if not has_resource_demand:
        missing_job_params.append("r (or r_cpu and r_mem)")

    if missing_job_params:
        raise ValueError(
            f"Missing required job_params in {jobs_json_path}: "
            f"{missing_job_params}."
        )

    data = {
        "pipeline_note": (
            f"Optimization input loaded from two JSON files. "
            f"Jobs source: {jobs_json_path}. "
            f"Server source: {server_json_path}."
        ),

        "sets": {
            "I": I,
            "I_B": I_B,
            "I_V": I_V,
            "I_C": I_C,
            "J": J,
            "K": K,
            "F": F,
            "E": E,
            "A": A,
            "G": G,
        },

        "eligibility": eligibility,

        "job_params": job_params,

        "server_params": server_data["server_params"],

        "thermal": server_data["thermal"],

        "cooling": _normalize_cooling(server_data["cooling"], K),

        "power": server_data["power"],

        "maintenance": server_data["maintenance"],

        "costs": server_data["costs"],

        "demand": server_data["demand"],

        "redundancy": server_data["redundancy"],

        "slot_duration": server_data["slot_duration"],

        "metadata": {
            "jobs_metadata": jobs_data.get("metadata", {}),
            "server_metadata": server_data.get("_comments", {}),
        },
    }

    return data


