# LTB Calculator (browser tool)

An interactive, client-side calculator for **lateral-torsional buckling of
steel beams** per **AISC 360-22 §F2** (compact doubly symmetric I-shapes and
channels, major-axis bending). Pick a shape, enter an unbraced length, and get
the nominal moment capacity with a live M<sub>n</sub>-vs-L<sub>b</sub> curve.

## Files

| File | Source | Commit it? |
|------|--------|------------|
| `ltb_calculator.html` | Hand-written | Yes |
| `shapes.js` | **Generated** from steelpy | Yes (see below) |

## Usage

Just open `ltb_calculator.html` in any browser — double-click it, or:

```bash
# macOS
open web/ltb/ltb_calculator.html
# Linux
xdg-open web/ltb/ltb_calculator.html
# Windows
start web/ltb/ltb_calculator.html
