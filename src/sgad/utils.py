import math


def sum_delta_g_parallel(delta_g_kcal: list[float], temp_c: float = 37.0) -> float:
    """Combine multiple ΔG values for alternative/competing states (parallel) into an effective ΔG.

    Uses:
        ΔG_eff = -RT * ln(Σ exp(-ΔG_i / (RT)))

    Args:
        delta_g_kcal: ΔG values in kcal/mol for mutually exclusive alternative states/paths.
        temp_c: Temperature in Celsius.

    Returns:
        Effective ΔG in kcal/mol.

    Raises:
        ValueError: If the input is empty, contains non-finite values, or temperature is invalid.
    """
    _R_KCAL_PER_MOL_K = 0.00198720425864083  # kcal/(mol·K)

    if not delta_g_kcal:
        raise ValueError("delta_g_kcal must be non-empty.")

    temp_k = float(temp_c) + 273.15
    if not math.isfinite(temp_k) or temp_k <= 0.0:
        raise ValueError(f"Invalid temperature (C): {temp_c}")

    rt = _R_KCAL_PER_MOL_K * temp_k

    # x_i = -ΔG_i / (RT); logsumexp for numerical stability
    xs: list[float] = []
    for dg in delta_g_kcal:
        if not math.isfinite(dg):
            raise ValueError(f"Non-finite ΔG encountered: {dg}")
        xs.append(-float(dg) / rt)

    x_max = max(xs)
    sum_exp = 0.0
    for x in xs:
        sum_exp += math.exp(x - x_max)

    log_sum = x_max + math.log(sum_exp)
    return -rt * log_sum
