# AISC 360-22 Chapter F Calculator — expansion notes

Expands the original §F2-only LTB calculator to the full flexural chapter (F2–F13).

## Files

| File | Purpose |
|---|---|
| `aisc_360_22_chapter_f_calculator.html` | Single-file app. Loads `shapes.js` if present, otherwise falls back to a small verified demo set. |
| `generate_shapes_chapter_f.py` | Replacement for `scripts/generate_shapes.py`. Emits every property the new engine needs, for every family. |

Drop the HTML next to a regenerated `shapes.js` and open it. No build step, no dependencies.

```bash
python generate_shapes_chapter_f.py -o shapes.js
# if steelpy's attribute names differ from the alias table:
python generate_shapes_chapter_f.py --inspect W
```

## What was added

**Dispatcher.** A section is routed to the correct Chapter F clause from its family, bending
axis, and Table B4.1b classification:

| Route | Trigger |
|---|---|
| F2 | I-shape/channel, major axis, compact web + compact flange |
| F3 | Doubly symmetric I, major axis, compact web, noncompact/slender flange |
| F4 | I-shape, major axis, noncompact web **or** singly symmetric |
| F5 | I-shape, major axis, slender web |
| F6 | I-shape/channel, minor axis |
| F7 | Square/rectangular HSS and box |
| F8 | Round HSS and pipe |
| F9 | Tees and double angles |
| F10 | Single angles |
| F11 | Rectangular bars and rounds |
| F12 | Unsymmetrical/general (user-supplied `Fcr`) |
| F13.1 | Tension-flange rupture, layered onto any of the above |

**Every limit state is reported, not just the governing one.** `Mn` is the minimum over
yielding, LTB, FLB, WLB, stem/leg local buckling, tension-flange yielding and rupture. Limit
states that don't apply are greyed out with the reason (`Lb ≤ Lp`, `compact flange`,
`Sxt ≥ Sxc`, …).

**Section classification table** — λ, λ<sub>p</sub>, λ<sub>r</sub> and the compact /
noncompact / slender verdict for the flange and web, with the Table B4.1b case number.

**Custom section properties** — every input is editable, so plate girders, bars, imported
shapes and anything else outside the database work without touching `shapes.js`.
`BAR_RECT` / `BAR_ROUND` derive `Z` and `S` from the entered dimensions.

**Chart** — envelope of all limit states (nominal dashed, design solid) plus a thin dotted
curve per limit state so you can see which one takes over and where.

**Derived in-browser:** `rts` (F2-7), `c` (F2-8), `rt` (F4-11), `Rpc`/`Rpt` (F4-9),
`Rpg` (F5-6), `kc`, `FL` (F4-6), tee `Sxc`/`Sxt` from `Ix` and `ȳ`, HSS `Se` from `be` (F7-3).

## Verification

Checked against published values:

| Case | Computed | Reference |
|---|---|---|
| W18X50, Fy 50, Lb 20 ft, Cb 1.32 | Mn = 264.2 kip-ft, Lp = 5.83 ft, Lr = 16.96 ft | your existing F2 tool / Manual Table 3-2 |
| W21X48, Fy 50, braced (noncompact flange → F3) | φMn = 397.9 kip-ft | Manual Table 3-2: 398 kip-ft |
| HSS8×8×1/2, Fy 46 | φMn = 124.9 kip-ft | Manual Table 3-12: 125 kip-ft |
| HSS12×8×1/4, Fy 46 (noncompact flange) | φMn = 116.3 kip-ft | Manual Table 3-12: 116 kip-ft |
| C15×33.9, Fy 50 | Lp = 3.18 ft | Manual Table 3-8 |
| Plate girder 16×1 fl / 60×5/16 web | Rpg = 0.959, F5 governs | hand check |
| 1"×6" bar, Fy 36, Lb 8 ft | Mn = 23.8 kip-ft (F11-2) | hand check |

## Things you should look at before trusting this

1. **Equation numbers.** I labelled every result with a 360-22 equation number. Cross-check
   them against your printed copy — the numbering shifted in a few clauses between -16 and -22
   and I'd rather you catch a mislabel than inherit one.
2. **F9 is the riskiest module.** 360-22 replaced the -16 single-`Mcr` approach for tees with
   an `Lp`/`Lr` branch when the stem is in tension. I implemented Eq. F9-4 *without* `Cb` in
   the linear branch (spec-literal) and force `Cb = 1.0` with `B` negative when the stem is in
   compression. Some references show `Cb` in F9-4. Verify against §F9.2(a)(ii).
3. **Channels with noncompact flanges fall in a gap.** F2 covers compact channels; F3/F4 are
   written for I-shapes. The tool applies an F3-form FLB check and flags it. If KPFF has a
   house position on this, hard-code it.
4. **F7 slender-web box sections** use `Rpg` per the F5-6 form with `aw = 2ht/bt`. Confirm
   §F7.3 in your copy.
5. **Single angles.** `βw = 0` is assumed for equal legs. For unequal legs you must enter `βw`
   (Manual Table 17-27) — it isn't in the shapes database and can't be derived from the
   tabulated properties.
6. **F13.2 proportioning limits and F13.3/F13.4 are not checked**, nor is Chapter G shear,
   J10 web crippling, or Chapter H interaction.

## Suggestion

You now have Chapter F logic in two places: `structural_tools/steel/` in Python and this
JavaScript engine. That will drift. Consider making the Python package the single source of
truth and either (a) transpiling the limit-state functions, or (b) generating a JSON fixture
of `(shape, Fy, Lb) → Mn` from Python and running it as a regression test against the JS in
CI. Option (b) is cheap and catches divergence immediately.

The engine section of the HTML (`F2()` … `F13_1()`, `classifySection()`, `runChapterF()`) is
deliberately DOM-free and pure, so it lifts out into a `.js` module or ports to Python
one-for-one if you want to go that route.
