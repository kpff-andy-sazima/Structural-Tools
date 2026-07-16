import argparse
import subprocess
import time
from pathlib import Path
import os

# from importlib.resources import as_file, files
import shutil
import nbformat
import yaml
import re


# point nbconvert at your package templates
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def extract_yaml_frontmatter(cell_source: str):
    """
    Extract YAML frontmatter from a markdown cell.
    Returns dict or empty dict if none found.
    """
    match = re.match(r"^---\n(.*?)\n---\s*", cell_source, re.DOTALL)
    if not match:
        return {}

    return yaml.safe_load(match.group(1)) or {}


def check_for_kpff_background():
    template_pdf = TEMPLATE_DIR / "kpff_calc_pad_letter_vertical_no_grid.pdf"
    shutil.copy(template_pdf, ".")


def run_jupytext(python_file_path: Path) -> None:
    print(f"Converting {python_file_path} to a Jupyter Notebook with Jupytext\n")

    if not python_file_path.exists():
        raise FileNotFoundError(f"Notebook not found: {python_file_path}")

    cmd = [
        "jupytext",
        "--to",
        "notebook",
        str(python_file_path),
    ]

    start = time.perf_counter()

    subprocess.run(
        cmd,
        check=True,
    )

    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.1f} sec\n")


def run_nbconvert(output_type: str, notebook_path: Path, template: str) -> None:
    print(f"Exporting {notebook_path} to {output_type} with the {template} template\n")

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    env = os.environ.copy()
    env["NBCONVERT"] = "1"

    cmd = [
        "jupyter",
        "nbconvert",
        "--no-input",
        "--execute",
        "--to",
        output_type,
        f"--template={template}",
        f"--TemplateExporter.extra_template_basedirs={str(TEMPLATE_DIR)}",
        str(notebook_path),
    ]

    start = time.perf_counter()

    subprocess.run(
        cmd,
        env=env,
        check=True,
    )

    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.1f} sec\n")


def main():
    parser = argparse.ArgumentParser(
        prog="export-notebook",
        description="Run jupyter nbconvert on a notebook with structural_tools wrapper.",
    )

    parser.add_argument(
        "output_type",
        choices=["pdf", "latex"],
        help="Output format (pdf, html, markdown, etc.)",
    )

    parser.add_argument(
        "notebooks",
        nargs="+",
        help="Notebook name without .ipynb extension",
    )

    parser.add_argument(
        "--template",
        default="latex-article",
        choices=["latex-article", "latex-report"],
        help="Name of the template to use",
    )

    parser.add_argument(
        "--background",
        default="kpff_calc_pad_letter_vertical_no_grid.pdf",
        help="Path to the pdf background you wish to use",
    )

    args = parser.parse_args()

    check_for_kpff_background()

    notebook_paths: list[Path] = []
    for notebook in args.notebooks:
        notebook = Path(notebook)
        if notebook.suffix == ".py":
            run_jupytext(notebook)

        notebook_paths.append(notebook.with_suffix(".ipynb"))

    for notebook_path in notebook_paths:
        nb = nbformat.read(notebook_path, as_version=4)

        for cell in nb.cells:
            if cell.cell_type in {"raw", "markdown"}:
                meta = extract_yaml_frontmatter(cell.source)
                if meta:
                    # merge into notebook metadata
                    nb.metadata.update(meta)
                    break  # usually only one frontmatter block
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        run_nbconvert(args.output_type, notebook_path, args.template)


if __name__ == "__main__":
    main()
