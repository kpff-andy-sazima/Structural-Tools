import units
import handcalcs
from .plotting import set_plot_style


def initialize_notebook() -> None:
    units.environment(env_name="structural", top_level=True)
    handcalcs.set_option("custom_symbols", {"star": "^*", "__": ",", "plus": "+", "minus": "-"})
    handcalcs.set_option("greek_exclusions", ["psi"])
    handcalcs.set_option("display_precision", 4)
    handcalcs.set_option("param_columns", 1)
    handcalcs.set_option("latex_block_start", "\\begin{equation*}")
    handcalcs.set_option("latex_block_end", "\\end{equation*}")
    set_plot_style()
