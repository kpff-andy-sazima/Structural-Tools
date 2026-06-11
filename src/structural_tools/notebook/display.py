from collections.abc import Callable, Sequence
import math
import os
from typing import Literal

import pandas as pd
from IPython.display import Latex, display


def sig_figs(x: float, sig_figs: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    if pd.isna(x):
        return ""

    if x == 0:
        return float(0)

    # If the value is not a number, just return it as is (e.g. for strings in the dataframe)
    try:
        x = float(x)
    except ValueError, TypeError:
        return x

    sig_figs = int(sig_figs)

    if 1e-3 <= abs(x) < 1e6:
        decimals = sig_figs - int(math.floor(math.log10(abs(x)))) - 1
        decimals = max(decimals, 0)
        return f"{x:.{decimals}f}"

    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig_figs - 1))


def display_table(
    dataframe: pd.DataFrame,
    levels: str | list[str] | None = None,
    column_names_filter_and_map: dict[str, str | None] | None = None,
    style_functions: Sequence[tuple[Callable, Literal[0, 1]]] | None = None,
    formatter_functions: Sequence[Callable] | None = None,
    hrules: bool = True,
    clines: str = "all;data",
    position: str = "H",
    position_float: str = "centering",
    **kwargs,
) -> Latex | None:
    """Displays a table if run in a Jupyter ipynb, or returns LaTeX code if run by nbconvert when exporting to PDF

    Args:
        dataframe (pd.DataFrame): Your dataframe
        level (str, list[str], optional): Levels you want to display. If None, then display all levels. Defaults to None.
        column_names_filter_and_map (dict[str, str | None], optional): A dictionary of the columns you wish to show, and their mapped display names (give None for the value to leave the display name the same). Every key needs a value or None. If no mapping is given, all columns will be shown with their default names. Defaults to None.
        style_functions (Sequence[tuple[Callable, Literal[0, 1]]], optional): A list of tuples containing the function to apply to the styler for table formatting and the axis (0 or 1) to apply that function to. Note that the columns will use the renamed columns following column_names_filter_and_map. Defaults to None.
        position_float (str, optional): See pandas documentation for Styler.to_latex(). Defaults to "centering".
        hrules (bool, optional): See pandas documentation for Styler.to_latex(). Defaults to True.
        position (str, optional): See pandas documentation for Styler.to_latex(). Defaults to "H".

    Returns:
        Latex: String of latex text that will be picked up by NBConvert if printed to the output of a cell
    """
    if dataframe.empty:
        return

    if column_names_filter_and_map:
        display_columns = list(column_names_filter_and_map.keys())
        df_display = dataframe[display_columns].copy()
        df_display.rename(columns=column_names_filter_and_map, errors="raise", inplace=True)
    else:
        df_display = dataframe.copy()
    if levels:
        df_display = df_display.loc[levels]

    # Need to use the "export_to_*" scripts to set the NBCONVERT environment variable for this to work,
    # otherwise it will just display the dataframe as normal without LaTeX formatting
    if "NBCONVERT" in os.environ:
        styler = df_display.style

        if formatter_functions:
            for function in formatter_functions:
                styler = styler.format(function)
        else:
            styler.format(lambda value: sig_figs(value, sig_figs=3))

        if style_functions:
            for function, axis in style_functions:
                styler = styler.apply(function, axis)

        display(
            Latex(
                styler.to_latex(
                    hrules=hrules,
                    clines=clines,
                    position=position,
                    convert_css=True,
                    position_float=position_float,
                    **kwargs,
                )
            )
        )
    else:
        display(df_display)


def display_text(text: str) -> Latex | None:
    if "NBCONVERT" in os.environ:
        return Latex(text)
    else:
        display(text)
