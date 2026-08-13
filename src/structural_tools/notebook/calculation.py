"""Calculation tools for notebooks"""

from structural_tools.typing import FloatLike


def check_value(value: FloatLike, check_value: FloatLike = 1.0, inequality: str = "leq"):
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


VALID_RETURN_UNITS_FEET = {"feet", "ft"}
VALID_RETURN_UNITS_INCHES = {"inches", "in", "inch"}
VALID_RETURN_UNITS = VALID_RETURN_UNITS_FEET.union(VALID_RETURN_UNITS_INCHES)


def feet_inches(
    feet: float = 0,
    inches: float = 0,
    return_unit: str = "feet",
) -> float:
    if return_unit not in VALID_RETURN_UNITS:
        raise ValueError(f"Invalid return unit {return_unit}. Use one of {VALID_RETURN_UNITS}")

    # Default to calc'ing inches. Then convert to feet if asked to.

    length = (feet * 12) + inches
    if return_unit in {"feet", "ft"}:
        length = length / 12

    return length
