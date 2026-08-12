import forallpeople as units
from handcalcs.global_config import set_option
from .plotting import set_plot_style
from .calculation import check_value
from .display import (
    display_figure,
    display_markdown,
    display_math,
    display_table,
    display_text,
    set_params_columns,
    sig_figs,
)


def initialize_notebook() -> None:
    units.environment(env_name="structural", top_level=True)
    set_option("custom_symbols", {"star": "^*", "__": ",", "plus": "+", "minus": "-"})
    set_option("greek_exclusions", ["psi"])
    set_option("display_precision", 4)
    set_option("param_columns", 1)
    set_option("latex_block_start", "\\begin{equation*}")
    set_option("latex_block_end", "\\end{equation*}")
    set_plot_style()


__all__ = [
    "check_value",
    "display_figure",
    "display_markdown",
    "display_math",
    "display_table",
    "display_text",
    "set_params_columns",
    "sig_figs",
]
