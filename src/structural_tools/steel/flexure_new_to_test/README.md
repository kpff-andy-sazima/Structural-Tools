# `structural_tools.steel.flexure` — AISC 360-22 Chapter F

Full coverage of Sections F1 through F13. Drop the whole folder in as
`structural_tools/steel/flexure/`, replacing the old single-file
`structural_tools/steel/flexure.py`.

Units are kip-inch throughout (stresses ksi, lengths in).

## Layout

| Module | Contents |
|---|---|
| `common.py` | Enums, `LimitState`, `FlexuralStrength`, F1 (`C_b`, φ<sub>b</sub>/Ω<sub>b</sub>), shared interpolation |
| `properties.py` | `SectionProperties` bag + derivations (r<sub>ts</sub>, c, r<sub>t</sub>, a<sub>w</sub>, k<sub>c</sub>, F<sub>L</sub>, S<sub>xc</sub>/S<sub>xt</sub>, bar Z/S) |
| `classification.py` | Table B4.1b flange/web slenderness |
| `f2.py` … `f12.py` | One module per Specification section; pure equation functions plus a `check_f*` driver |
| `f13.py` | F13.1 tension-flange rupture, F13.2 proportioning limits |
| `dispatch.py` | `calculate_nominal_flexural_strength` — routes per Table User Note F1.1 |
| `beam.py` | `evaluate_beam_flexure`, `flexure_results_to_dataframe` |

## Usage

```python
from structural_tools.steel.flexure import (
    Axis, SectionProperties, ShapeGroup, calculate_nominal_flexural_strength,
)

w18x50 = SectionProperties(
    group=ShapeGroup.I_SHAPE, name="W18X50",
    plastic_modulus_x=101.0, section_modulus_x=88.9, radius_of_gyration_y=1.65,
    torsional_constant=1.24, warping_constant=3040.0, second_moment_y=40.1,
    flange_centroid_distance=17.4, flange_slenderness=6.57, web_slenderness=45.2,
)

result = calculate_nominal_flexural_strength(w18x50, 50.0, 20 * 12, Axis.MAJOR, 1.32)
result.nominal_moment        # 3170.6 kip-in
result.governing.equation    # "F2-3 / F2-4"
result.design_moment         # phi_b * M_n
```

`SectionProperties.from_wsection(section, group=ShapeGroup.I_SHAPE)` adapts an
existing `WSection`. Every limit state is retained in `result.limit_states`,
including the ones that do not apply, each with a `note` giving the reason — so
a calculation table can grey them out instead of silently omitting them.

## Verification

`W18X50, F_y = 50 ksi, L_b = 20 ft, C_b = 1.32` reproduces the published
F2 calculator output exactly: M<sub>p</sub> = 420.8 k-ft, L<sub>p</sub> = 5.83 ft,
L<sub>r</sub> = 16.96 ft, M<sub>n</sub> = 264.2 k-ft, governed by elastic LTB.

## Corrections relative to `aisc_360_22_chapter_f_calculator.html`

The HTML calculator carries several **AISC 360-16** forms. The Python here
follows the 2022 text:

1. **F7-2 / F7-6 (HSS flange and web local buckling)** — 360-22 uses the standard
   λ interpolation between λ<sub>pf</sub>/λ<sub>rf</sub> and
   λ<sub>pw</sub>/λ<sub>rw</sub>. The `3.57√(F_y/E) − 4.0` and
   `0.305√(F_y/E) − 0.738` closed forms are 360-16 and were removed.
2. **F7-5** — box sections use a 0.34 coefficient in b<sub>e</sub>, not the 0.38
   used for HSS (Eq. F7-4). Set `box_section=True` on the properties.
3. **F7 LTB equation numbers** — inelastic is F7-8, elastic is F7-9;
   F7-10 and F7-11 are L<sub>p</sub> and L<sub>r</sub>.
4. **F9-10 (tee / double-angle M_cr)** — 360-22 writes it as
   `1.95(E/L_b)√(I_y J)·[B + √(1+B²)]` with **no C_b and no G**. The HTML uses
   the older `πC_b√(E I_y G J)/L_b` form, which is both a different edition and
   applies a moment-gradient factor F9 does not authorize.
5. **F9 numbering** — yielding is F9-2…F9-5, inelastic LTB is F9-6, L_p/L_r are
   F9-8/F9-9, tee stems in compression are capped by F9-13 (M_n = M_cr ≤ M_y),
   FLB is F9-14/F9-15, stem local buckling is F9-16…F9-19.
6. **F10-6/F10-7/F10-8 (leg local buckling)** — numbering shifted; the HTML
   labels these F10-7/F10-8.
7. **F11-1 (rectangular bars)** — the yielding cap is **1.5** F<sub>y</sub>S<sub>x</sub>;
   1.6 applies only to rounds (F11-2). The HTML uses 1.6 for both.
8. **F10 geometric-axis bending** — M<sub>y</sub> *and* S<sub>c</sub> are taken as
   0.80 of the geometric values only in the unrestrained case; the restrained
   case uses the full geometric S with M<sub>cr</sub> × 1.25.

## Breaking changes from the old single-file module

- `calculate_nominal_flexural_strength` now returns `FlexuralStrength`
  (all limit states) instead of `LateralTorsionalBucklingResult`. The F2-only
  behaviour is still available as `check_f2`.
- `calculate_inelastic_lateral_torsional_buckling_stress` became
  `calculate_inelastic_lateral_torsional_buckling_moment`, returning M<sub>n</sub>
  directly as Eq. F2-2 is written rather than an equivalent extreme-fibre stress.
- `FlexuralSegmentResult.lateral_torsional_buckling` became `.strength`, and
  gains `.governing_equation`, `.governing_limit_state`, `.unbraced_length`.
- `region` strings are replaced by `result.governing.equation` / `.name`.
