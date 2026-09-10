# THGEM_with_mesh — what does a mesh do that a second THGEM does not?

A [Garfield++](https://garfieldpp.web.cern.ch/) simulation of a **THGEM feeding a micromegas-style
mesh**, under a **plane of wires as the drift cathode**. Primary electrons released in the drift gap
are funnelled into the THGEM's holes and multiplied there, transferred across a gap to the mesh,
threaded through its apertures, and multiplied a second time in the amplification gap between the
mesh and the anode.

The mesh is configurable across the whole practically interesting range: from a 400/500 LPI woven
micromesh (~50–65 µm pitch, 18–30 µm wire) up to thick meshes with mm-scale pitch.

The project ships a C++ simulation binary and a PyQt5 desktop GUI that share one JSON configuration
schema.

It is the fourth in a family: [tgc-garfield](https://github.com/lmoleri/tgc-garfield) (a wire
chamber, analytic field), [THGEM-garfield](https://github.com/lmoleri/THGEM-garfield) (one hole
plate, neBEM) and [double-THGEM](https://github.com/lmoleri/double-THGEM) (two plates, wire cathode).
This project takes double-THGEM's stack and **replaces the second THGEM with a mesh**; the gas,
avalanche, signal, track and I/O machinery is shared, and the wire cathode is unchanged.

> **Status:** the field solve and the transport chain are verified end to end — the neBEM solve, the
> five-zone field validation, the mesh-resolution and field-ratio checks, the geometry-keyed caches
> (including the snapped mesh pitch, checked against six deliberate variations), a multiplying
> two-stage avalanche with per-stage gain and a measured mesh transparency, and the GUI's derived
> readouts checked digit-for-digit against the binary's own. The mesh model itself is validated by
> the acceptance test below: measured electron transparency sits between the geometric transparency
> and 1, and rises with the amplification field (0.686 → 0.759 for E_amp 20 → 40 kV/cm on the smoke
> geometry). Which shipped config was run how is noted in [Configuration](#configuration).

---

## Detector model

```
   z                ○   ○   ○   ○   ○      wire cathode  V_wire   (wires ∥ y)
   ↑                      drift gap
   │        ▓▓▓▓░░▓▓▓▓▓▓░░▓▓▓▓            THGEM   V_thgem_top / V_thgem_bot
   │                     transfer gap
   │        ╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═          mesh   V_mesh
   │                  amplification gap
   │        ═══════════════════            anode pad  0 V
            |←── hole pitch p ──→|
```

Electrons drift **−z**, toward the anode. The configuration is given in physics terms and the five
electrode potentials are derived with the anode as the 0 V reference:

```math
\begin{aligned}
V_\text{anode}      &= 0 \\
V_\text{mesh}       &= V_\text{anode}      - E_\text{amp} \cdot d_\text{amp} \\
V_\text{thgem,bot}  &= V_\text{mesh}       - E_\text{transfer} \cdot d_\text{transfer} \\
V_\text{thgem,top}  &= V_\text{thgem,bot}  - \Delta V_\text{THGEM} \\
V_\text{wire}       &= V_\text{thgem,top}  - E_\text{drift} \cdot d_\text{drift}
\end{aligned}
```

**There is no `delta_v_mesh`, and that is the point.** The mesh is a single conductor, so its stage
has no voltage *across* it: its gain comes from `e_amplification_kvcm` over
`amplification_gap_mm` — exactly how a micromegas is specified in practice (40–60 kV/cm over
64–320 µm). The same field also sets how much charge gets through the mesh at all, through the ratio
`E_amp / E_transfer`.

**The anode is mandatory here.** double-THGEM could drop its anode because THGEM 2's bottom copper is
a continuous conductor that legitimately terminates the gas volume. A mesh is 35–65 % open and cannot:
charge passes *through* it. Without an anode there is no amplification gap and no second stage at
all, so `geometry.anode_enabled` and `geometry.induction_gap_mm` are rejected with an explanation
rather than silently ignored.

## The periodic cell, and why the mesh pitch gets snapped

The neBEM solve is one periodic cell, tiled with **translation** periodicity in x and y:

- x-period `cellX = holes_per_wire × hole_pitch` — one wire, `holes_per_wire` holes.
- y-period `cellY = hole_pitch`.

The mesh lattice has to fit that cell a whole number of times too. It looks like two constraints, but
`cellX` is always an integer multiple of `cellY`, so a pitch that divides `cellY` automatically
divides `cellX`. The two collapse into one, and `holes_per_wire > 1` needs no special handling:

```math
k = \max\!\left(1, \left[\frac{p_\text{hole}}{p_\text{requested}}\right]\right), \qquad
p_\text{snapped} = \frac{p_\text{hole}}{k}
```

The snapped pitch is what is simulated, echoed into `run_config.json`, and keyed into the field
cache — never the requested one. A run says so when it snaps:

```
  mesh    : woven, pitch 237.0 um -> 266.7 um (snapped to fit the cell; 3 per hole pitch)
            50.0 um wire, 100.0 um thick, 3 x 3 per cell, optical transparency 0.640
```

At a 1000 µm hole pitch, a 400 LPI mesh (63.5 µm) snaps to 62.5 µm — a 1.6 % shift, physically
nothing. Asking for a mesh *coarser* than the hole pitch is a warning, not a silent clamp: raise
`hole_pitch_um` instead.

## Mesh models: woven vs perforated

This is the non-obvious result of the project, so it is up front.

| | primitives per feature | elements per feature | cell total |
|---|---|---|---|
| **woven** wire (`SolidWire`) | 1 | ≤ `max_elements` | k=8, N=1: 16 wires ≈ **64 elements** |
| **perforated** aperture (`SolidHole`, sectors 4) | ≈ 95 | ≈ 390 | k=4: 16 apertures ≈ **6 200**; k=8: **≈ 25 000** |

For comparison, the whole double-THGEM reference solve is ~2300 elements. neBEM inverts a dense
N × N matrix: the cost is O(N³) and the stored inverse is N² doubles, so N = 25 000 means ~5 GB and
roughly a thousand times the reference solve.

**The woven model is the one that reaches micromegas pitch.** A `SolidWire` becomes a single
line-charge primitive, so a whole woven mesh costs a handful of boundary elements — the same trick
that makes the 50 µm cathode wire affordable next to millimetre plates. The perforated model is for
coarse and thick meshes and does not go below roughly `hole_pitch / 3`; `geometry.max_elements_budget`
(default 8000) refuses such a solve up front rather than after a day, printing the projected element
count and the estimated cost relative to the reference solve.

Every mesh wire (or aperture tile) carries the same `SetLabel("mesh")`, and neBEM groups
same-labelled solids into one readout group — so however many wires the mesh is drawn with, it costs
exactly **one** weighting-field solve.

## Resolving the mesh — the failure mode that is otherwise invisible

`ComponentGrid` has no material map of its own. What stops a drifting electron passing through metal
is an in-solid flag stamped at each grid node when the neBEM field is sampled. If the transport grid
is too coarse for no node to land inside a mesh wire, **nothing absorbs charge there**: every
electron threads the mesh, the measured electron transparency comes out at exactly 1.00, and the run
looks completely healthy while telling you nothing.

The grid is therefore what bounds this geometry, not neBEM — `ComponentGrid::SetMesh()` is
uniform-only, so it has to resolve a 25–50 µm wire in x and y *and* a sub-millimetre amplification
gap in z while still spanning a millimetres-long drift gap. Every run prints both budgets and, when
either is short, the values needed to fix it:

```
    transport grid  dx 28.6 µm, dy 28.6 µm, dz 27.4 µm   nodes across the mesh 3.5, cells across the amp gap 11
```

and the acceptance line, on every transport run:

```
  [mesh] optical transparency 0.562 (geometric) | electron transparency 0.897 (measured, N = 3)
```

| measured ε | verdict |
|---|---|
| ≈ 1.00 | **fail** — the grid is not resolving the mesh; check `nodes across the mesh` |
| ≈ optical | **fail** — the funnelling into the apertures is unresolved; the mesh is acting as a pure geometric stop |
| between the two, **rising with `e_amplification_kvcm`** | correct |

The tempting cheap fix — flagging a node absorbing whenever a wire passes within half a grid cell —
is deliberately *not* done. It over-blocks, and biases the transparency downward by an amount that
depends on the grid rather than on the physics, which is worse than a known-wrong 1.00.

**Practical lower bound:** `mesh.pitch_um ≥ hole_pitch_um / 16`, and independently
`wire_diameter_um ≥ 3 · max(dx, dy, dz)`. A true 400 LPI mesh under a *millimetre*-pitch THGEM needs
~7.8 M grid nodes and is not a default; under a 500 µm THGEM it is reachable, which is what
`config/micromegas_400lpi_woven.json` demonstrates.

The second lever, when the grid alone is not enough, is per-solid discretisation — and it works
backwards from what you would expect. neBEM's `min/max_elements` clamp is **global**, so a per-solid
element size can only make a solid *coarser*, never finer. To spend elements on the mesh and not on
the plate: raise `max_elements`, set a **large** `thgem_element_size_um` (pinning the plate at
`min_elements`) and a **small** `mesh.element_size_um` (driving each wire to `max_elements`).

## Field validation

Five zones, each sampled on an axis that is actually gas there. Unless the mesh lattice happens to
line up with the THGEM hole, the hole axis runs straight through mesh metal — so the amplification
gap and the mesh's own z-band are probed on the mesh's nearest *open cell* instead:

```
  Field validation on the THGEM hole axis (x = 0 cm, y = 0 cm):
    amplification  Ez     5.543 ..     32.85 kV/cm   peak |E|    32.85 kV/cm   dV      -886 V
    mesh aperture  Ez     4.855 ..    23.264 kV/cm   peak |E|    23.26 kV/cm   dV      -248 V
    transfer      Ez      1.81 ..    10.039 kV/cm   peak |E|    10.04 kV/cm   dV    -164.4 V
    THGEM hole    Ez    11.033 ..     22.57 kV/cm   peak |E|    22.57 kV/cm   dV    -847.2 V
    drift         Ez     0.945 ..     9.374 kV/cm   peak |E|     9.37 kV/cm   dV    -128.7 V
    cathode wire surface (estimate)  |E| ~ 2.78 kV/cm
    mesh wire surface (estimate)     |E| ~ 92.39 kV/cm
    mesh optical transparency 0.562 (woven, geometric)
    field ratio E_amp / E_transfer = 40
```

`Ez > 0` in every zone and a monotonically falling potential are the two things that must hold. A
**reversed** zone is almost always too few `periodic_copies` — the tiled wire and anode patches only
approximate infinite structures — and the run says so.

Two checks are specific to the mesh:

- **Mesh wire surface field.** In a micromegas this is the largest field anywhere in the detector; it
  is what sets the sparking limit, and it is the field most likely to run off the end of the Magboltz
  table, so it is folded into the table-ceiling check.
- **Field ratio.** `E_amp / E_transfer` below ~20 warns: at that ratio the mesh collects most of the
  charge the THGEM produced instead of passing it through.

## Cascade diagnostics

```
  [cascade] <G_thgem> = 13.33, <mesh transparency> = 0.897, <G_amp> = 154.78, <total e-> = 390818.3
```

A charge is counted as having entered the amplification gap when it crosses **`z_mesh_bot`**, not
`z_mesh_top`. That matters: a track that lands on a mesh wire ends *inside* the mesh's own z-band and
is correctly not counted, while one that threads an aperture ends below it and is. Testing against
the top surface would count both and report a transparency of 1.

Transparency is measured against **all** the charge that existed above the mesh, not just the THGEM's
own output:

```math
\varepsilon_\text{mesh} = \frac{n_\text{entered amp}}{1 + n_\text{born THGEM} + n_\text{born transfer}}
```

The transfer gap and the mesh funnel are not field-free — the funnel runs at tens of kV/cm — so at a
high transfer field they multiply too. Dividing by the THGEM's gain alone yields a "transparency"
above 1, which is not a quantity. That charge is reported separately as `n_born_transfer`, and the
run says so when it is significant.

## Readout electrodes

| id | what it is |
|---|---|
| `wire_cathode` | the wire plane at the top of the drift gap |
| `thgem_top` | the THGEM's upper copper face |
| `thgem_bottom` | the THGEM's lower copper face |
| `mesh` | the whole mesh — one conductor, one weighting field, however many wires |
| `anode` | the readout pad below the amplification gap (default) |

Each carries its true Shockley–Ramo weighting potential, solved natively by neBEM and sampled onto
the transport grid. Every extra electrode costs one sampling pass and one cached grid.

## GUI

```bash
python3 gui/app.py
```

Tabs: Log, Summary, Plots, Waveforms, Integrals, **Geometry**, 3D Tracks, E-Field, Weighting Field,
Magboltz. The GUI writes a temporary JSON and runs the same binary — it never re-implements physics.
Two exceptions are deliberate: the derived-potential label and the mesh-snap label duplicate the C++
arithmetic in Python, so a disagreement is visible before anything is solved.

### The Geometry tab

The resolved stack on its own, drawn from a run's `run_config.json` — so it needs no transported
events, and **`Load run …` opens any run directory from any earlier session**, which matters when a
cold solve of the default takes an hour.

- **The periodic cell** is drawn solid, heavy and magenta; the tiled display copies are dotted, thin
  and grey, footprints only. Three cues at once, because everything except that one box is a
  convenience of the display rather than something that was solved.
- **The woven mesh's two layers are separated by colour and line style** — upper (wires ∥ x) at
  `z_upper`, lower (wires ∥ y) at `z_lower`. They are one electrode at one potential; the split says
  which layer is which, and the annotation column says so explicitly.
- **The THGEM hole is drawn as its three real z-segments**, so an etched rim is visible instead of
  being averaged into a single barrel.
- **The annotation column** lists every z plane and derived potential, each gap's thickness and its
  field, the mesh lattice and optical transparency, and the two transport-grid budgets — reddened at
  the same thresholds the binary's own validation warns at. The gap fields are recomputed there as
  |ΔV|/Δz rather than read back from `fields`, which makes the column a live cross-check of the
  potential chain.
- **The cut-away** removes geometry on one side of an axis-aligned plane: spans are truncated, round
  cross-sections are restricted to the exact surviving arc, and zero-thickness planes are culled or
  clipped. It is a filter over wireframe outlines — it does **not** cap the section into a solid
  face, and ROOT's 3D painter does not depth-sort, so the far half of what remains still draws over
  the near half.

Drawing is budgeted: a fine mesh at many tiled copies runs to thousands of `TPolyLine3D`, so the tab
projects the cost first and coarsens along a fixed ladder if needed — announcing what it reduced in
the Qt status label, the canvas annotation and the log, never silently.

Note what the E-Field tab's x–z slice does to a woven mesh: it cuts the lower layer (wires ∥ y)
transversely, drawn as ticks, while the upper layer (wires ∥ x) is cut lengthwise and cannot be
resolved in x at all. The upper layer is drawn as a single dashed line marking its plane — it is not
a solid sheet.

## Layout

```
src/thgem_mesh_sim.cc     the simulation: geometry, neBEM solve, transport, ROOT output
gui/app.py                PyQt5 desktop GUI (runs the binary, reads its ROOT/CSV/JSON output)
config/*.json             the shipped working points (see below)
docs/manual.md            developer & physics manual
gas/                      committed Magboltz tables (Ar:CO2 70:30)
third_party/nlohmann/     vendored json.hpp
field_cache/              generated, gitignored — see the note under Run
results/                  generated, gitignored
```

## Build

ROOT's bundled `FindVdt.cmake` does not locate Vdt on this machine, so it is passed explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVDT_INCLUDE_DIR=/Users/luca/miniforge3/include \
  -DVDT_LIBRARY=/Users/luca/miniforge3/lib/libvdt.dylib \
  -DCMAKE_PREFIX_PATH="/Users/luca/Documents/software/Garfield++/local/garfield;/Users/luca/miniforge3"
cmake --build build -j4
```

## Run

```bash
export GARFIELD_INSTALL=/Users/luca/Documents/software/Garfield++/local/garfield
export HEED_DATABASE=$GARFIELD_INSTALL/share/Heed/database
./build/thgem_mesh_sim --config config/default_thgem_mesh.json --out results
```

| flag | meaning |
|---|---|
| `--config <path>` | JSON config (default `config/default_thgem_mesh.json`) |
| `--out <dir>` | output root (default `results`) |
| `--run-name <name>` | override the auto-generated run folder |
| `--distance <mm>` | single source height, overriding the config list |
| `--field-only` | solve, sample, dump the maps and stop before any transport |
| `--help` | usage |

| exit code | meaning |
|---|---|
| `0` | clean |
| `2` | the run completed and wrote everything, but a self-check flagged something |
| `1` | a genuine error (the message says what) |

`n_events: 0` in the config means the same thing as `--field-only`; it is how the GUI asks for it,
and it is the shipped default — the default config is a geometry to look at, not a run to wait for.

```bash
ctest --test-dir build --output-on-failure
```

**A note on `field_cache/`.** Resolving a mesh wire in x and y while still spanning the drift gap in
z needs a much finer transport grid than the THGEM siblings: the shipped default samples ~1.0 M nodes
(~75 MB) and the 400 LPI working point ~1.3 M. A cold solve of the default takes on the order of an
hour, almost all of it in the two sampling passes; every run after that loads the cache in seconds.
The caches are gitignored — far too large to commit.

The cache key covers the geometry, the discretisation **and** the snapped mesh lattice. Two
verified consequences: changing only `target_element_size_um` re-solves (the double-THGEM sibling has
a bug here — its transport cache is keyed without the discretisation and is silently reused), and
offsetting the mesh by exactly one whole mesh pitch, which changes nothing, leaves the cache
filename untouched.

## Configuration

| section | keys |
|---|---|
| `geometry` | `hole_pitch_um`, `wire_diameter_um`, `holes_per_wire`, `wire_between_holes`, `drift_gap_mm`, `transfer_gap_mm`, `amplification_gap_mm`, `thgem{…}`, `mesh{…}`, plus the neBEM and transport-grid controls |
| `geometry.thgem` | `hole_diameter_um`, `plate_thickness_um`, `copper_thickness_um`, `rim_um`, `dielectric_material` |
| `geometry.mesh` | `model` (`woven`\|`perforated`), `pitch_um`, `wire_diameter_um`, `thickness_um` (0 = auto), `aperture_um`, `aperture_sectors`, `offset_x_um`, `offset_y_um`, `element_size_um` |
| `fields` | `e_drift_kvcm`, `delta_v_thgem_V`, `e_transfer_kvcm`, `e_amplification_kvcm` |
| `readout` | `electrodes` (null → `anode`) |
| `source` | `energy_keV`, `source_distances_mm`, `x_positions_cm` |
| `gas` | mixture, T, p, Penning, Magboltz table bounds |
| `simulation` | `n_events` (0 = field only), avalanche cap, time window, ion drift, seed |
| `amplifier` | CIVIDEC C2-TCT transimpedance front end |

Shipped working points:

| config | what it is | nodes | ε_optical | verified |
|---|---|---|---|---|
| `smoke_thgem_mesh.json` | shrunk clone with a fixed seed, used by CTest | 61 k | 0.562 | end to end, incl. transport |
| `default_thgem_mesh.json` | 1 mm THGEM over a 250 µm woven mesh, 45 kV/cm amplification | 1.0 M | 0.640 | field solved + validated |
| `thick_mesh_perforated.json` | the coarse end: a 500 µm perforated sheet, 300 µm apertures | 405 k | 0.270 | field solved + validated |
| `micromegas_400lpi_woven.json` | a real 400 LPI mesh (62.5 µm, 25 µm wire) under a 500 µm THGEM | 1.3 M | 0.360 | solved + validated, not sampled |
| `thgem_mesh_3000V_60kVcm.json` | high-gain hybrid: ΔV 3000 V over a 128 µm gap at 60 kV/cm | 1.4 M | 0.640 | solved + validated, not sampled |

"Solved + validated" means the neBEM solve ran and the five-zone field validation passed on it, but
the multi-hour grid-sampling pass has not been run here — the first run of one of those configs will
take the time noted below. "Field solved + validated" adds the sampling pass and the validation
re-run on the interpolated grid.

`0.0` means "auto" for `mesh.thickness_um` (woven → 2 × wire diameter) and for the two per-solid
element sizes. `null` means "unpinned" for `readout.electrodes`, `source.source_distances_mm` and
`source.x_positions_cm`.

## Transport bounds and known limitations

- **The transport grid, not neBEM, is the binding constraint** — see *Resolving the mesh*. Every run
  prints both budgets; a run that trips them exits 2.
- **The Sensor's drift area is inset by one grid cell** at each end, so a source height at or above
  the cathode plane lands outside it. That is rejected by `LoadConfig` with the key named, rather
  than surfacing as an AvalancheMicroscopic error about the active area.
- **`SolidHole` meshes a polygon, not a circle** — a 4(n−1)-gon of *circumradius* r, so
  `sectors = 2` is a square of half-side r/√2. The in-solid test reproduces that polygon exactly;
  testing a plain circle would be 3 % wrong in area at `sectors = 4` and 29 % wrong at `sectors = 2`.
- **A woven mesh is modelled as two crossed flat layers**, tangent at the default thickness of
  2 × wire diameter, not as a truly interwoven weave with a z-modulated centreline.
- **Conductor singularities.** Field values on a wire axis are non-finite and are zeroed and marked
  absorbing when the grid is sampled; the run reports how many nodes that affected.
- **neBEM console noise** during `Initialise()` (`UpdatePeriodicity: Periodic length is not set`) is
  harmless — the periodicity is set immediately afterwards.
