# structural_tools

A Python package for structural engineering analysis, design, reporting, and workflow automation.

The goal of `structural_tools` is to provide reusable building blocks for common structural engineering tasks. This includes code-based calculations, seismic analysis, steel and wood design, reporting, visualization, and data processing.

---

## Features

### Structural Design

- Light-frame wood shear wall design using flexible, rigid, or envelope procedures
  - Shear wall force distribution
  - Center of rigidity calculations
  - Drift calculations
- AISC 360 steel design utilities

### Seismic Analysis

- Equivalent lateral force procedure
  - Story force distribution
  - Diaphragm story force
  - Automatic querying of USGS earthquake parameters from latitude and longitude input

### Reporting

- Engineering calculation reports from step-by-step calculations
- Automatic typesetting of variable names
- Notebook-to-PDF workflows

### Visualization

- Structural model plotting
- 3D frame visualization
- Deformed shape plotting
- Load and reaction graphing visualization

---

## Installation

### From Source

```bash
git clone https://github.com/kpff-andy-sazima/structural_tools.git
cd structural_tools
pip install -e .
```

### In your Python project

If you are using a `pyproject.toml` file, then be sure that the following is in it:

```toml
[project]
name = "project_name"
version = "0.1.0"
description = "Project description"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "ipython",
    "ipykernel<7",
    "jupyter",
    "jupytext",
    "matplotlib",
    "numpy",
    "pandas",
    "pandas-stubs",
    "PyniteFEA[all]",
    "sectionproperties",
    "scipy",
    "steelpy",
    "sympy",
    # local packages
    "forallpeople",
    "handcalcs",
    "structural_tools",
]

[tool.uv.sources]
structural_tools = { path = "your/path/to/structural_tools", editable = true }
handcalcs = { path = "your/path/to/handcalcs", editable = true }
forallpeople = { path = "your/path/to/forallpeople", editable = true }
```

- Note that "handcalcs" and "forallpeople" will most likely be used with "structural tools."
