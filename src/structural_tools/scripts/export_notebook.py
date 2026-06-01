import argparse
import subprocess
import time
from pathlib import Path
import os
from importlib.resources import as_file, files


# point nbconvert at your package templates
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def run_nbconvert(output_type: str, notebooks: list[str], template: str) -> None:
    for notebook in notebooks:
        notebook_path = Path(notebook).with_suffix(".ipynb")
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
        "notebook",
        nargs="+",
        help="Notebook name without .ipynb extension",
    )

    parser.add_argument(
        "--template",
        default="latex-article",
        choices=["latex-article", "latex-report"],
        help="Name of the template to use",
    )

    args = parser.parse_args()

    run_nbconvert(args.output_type, args.notebook, args.template)


if __name__ == "__main__":
    main()
