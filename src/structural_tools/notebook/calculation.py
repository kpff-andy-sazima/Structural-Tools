import handcalcs


def set_params_columns(num_cols: int):
    handcalcs.set_option("param_columns", num_cols)


def check_value(value: float, check_value: float = 1, inequality: str = "leq"):
    pass_latex = r"\textbf{\color{OK}OK}\ {\color{OK}\checkmark}"
    fail_latex = r"\textbf{{\color{NG}NG !}}"
    outcome = fail_latex
    operator = "ERROR"
    match inequality:
        case "leq" | "<=":
            if value <= check_value:
                outcome = pass_latex
                operator = "\\leq"
            else:
                operator = "\\gt"
        case "lt" | "<":
            if value < check_value:
                outcome = pass_latex
                operator = "\\lt"
            else:
                operator = "\\geq"
        case "geq" | ">=":
            if value >= check_value:
                outcome = pass_latex
                operator = "\\geq"
            else:
                operator = "\\lt"
        case "gt" | ">":
            if value > check_value:
                outcome = pass_latex
                operator = "\\gt"
            else:
                operator = "\\leq"
        case "eq" | "=":
            if value == check_value:
                outcome = pass_latex
                operator = "="
            else:
                operator = "\\neq"
    return f"{value:.3g} {operator} {check_value:.3g} \\quad {outcome}"


def highlight_insufficient_capacity(row):
    styles = [""] * len(row)

    if row["$0.7 v$ [plf]"] > row["$v_{cap}$ [plf]"]:
        demand_idx = row.index.get_loc("$0.7 v$ [plf]")
        capacity_idx = row.index.get_loc("$v_{cap}$ [plf]")

        styles[demand_idx] = "background-color: red; font-weight: bold"
        styles[capacity_idx] = "background-color: red; font-weight: bold"

    return styles


VALID_RETURN_UNITS_FEET = {"feet", "ft"}
VALID_RETURN_UNITS_INCHES = {"inches", "in", "inch"}
VALID_RETURN_UNITS = VALID_RETURN_UNITS_FEET.union(VALID_RETURN_UNITS_INCHES)


def feet_inches(
    feet: float = 0,
    inches: float = 0,
    fractional_inches: float = 0,
    return_unit: str = "feet",
) -> float:
    if return_unit not in VALID_RETURN_UNITS:
        raise ValueError(f"Invalid return unit {return_unit}. Use one of {VALID_RETURN_UNITS}")

    # Default to calc'ing inches. Then convert to feet if asked to.

    length = (feet * 12) + inches + fractional_inches
    if return_unit in {"feet", "ft"}:
        length = length / 12

    return length
