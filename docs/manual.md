# THGEM_with_mesh — developer & physics manual

The companion to [`README.md`](../README.md). The README says what the device is and how to run it;
this says how the code is put together and which parts will bite you.

Everything lives in one translation unit, `src/thgem_mesh_sim.cc`, in an anonymous namespace, in
pipeline order — the same shape as the sibling projects. References below are by **function / struct
name** rather than line number, so they survive edits.

**Contents**

1. [The device](#1-the-device)
2. [Coordinates, the cell, and the commensurability rule](#2-coordinates-the-cell-and-the-commensurability-rule)
3. [Garfield++ classes used, and why](#3-garfield-classes-used-and-why)
4. [Code map](#4-code-map)
5. [The field: neBEM → ComponentGrid, and the grid budget](#5-the-field-nebem--componentgrid-and-the-grid-budget)
6. [Validation: what every run checks](#6-validation-what-every-run-checks)
7. [The event loop and the cascade counters](#7-the-event-loop-and-the-cascade-counters)
8. [ROOT output schema](#8-root-output-schema)
9. [The GUI](#9-the-gui)
10. [Numerical subtleties](#10-numerical-subtleties)
11. [Extending it](#11-extending-it)

---

## 1. The device

A THGEM over a mesh over an anode, under a wire cathode. Two multiplying stages:

- **Stage 1, the THGEM.** A dielectric foil clad in copper on both faces, drilled with a lattice of
  holes. The dipole field across the foil concentrates in each hole; electrons funnelled in from the
  drift gap multiply there. Gain is set by `delta_v_thgem_V`.
- **Stage 2, the mesh + amplification gap + anode.** This is a micromegas. The mesh is a **single
  conductor**, so — unlike a second THGEM — it has no voltage across it and no gain of its own. The
  gain belongs to the parallel-plate gap below it, and is set by `e_amplification_kvcm` over
  `amplification_gap_mm`.

That difference propagates everywhere: there is no `delta_v_mesh_V`, the mesh is one readout
electrode rather than two, and the anode is mandatory because a 35–65 % open mesh cannot terminate
the gas volume the way a second plate's bottom copper could.

Two mesh models, because the two ends of the pitch range are physically different objects:
`"woven"` (two orthogonal layers of wires) and `"perforated"` (a thin sheet with a lattice of
apertures). They cost wildly different amounts to solve — see §5.

## 2. Coordinates, the cell, and the commensurability rule

Electrodes lie in x–y; the drift axis is z and electrons drift **−z**. The z stack is anchored at the
anode (`ComputeGeom`):

```
zAnode    = 0
zMeshBot  = zAnode   + dAmp
zMeshTop  = zMeshBot + tMesh
zThgemBot = zMeshTop + dTransfer
zThgemTop = zThgemBot + tCu + tDiel + tCu
zWire     = zThgemTop + dDrift
```

For the woven model the two wire layers sit at `zCen ∓ (tMesh − d_wire)/2`, placed so their outer
surfaces land exactly on `zBot` / `zTop`. At the default thickness of `2·d_wire` the two layers are
tangent, which is the physical weave. **Layer convention, because it shows up in every plot:** the
*lower* layer runs ∥ y and is spaced along x; the *upper* layer runs ∥ x and is spaced along y.

The neBEM cell is `cellX = holes_per_wire × hole_pitch` by `cellY = hole_pitch`, tiled with
**translation** periodicity — a staggered mesh lattice or a wire between holes leaves the cell
without a mirror plane, so mirroring would fabricate a geometry neBEM never solved.

**The commensurability rule.** The mesh lattice must fit the cell a whole number of times in both x
and y. That looks like two constraints, but `cellX` is an integer multiple of `cellY`, so a pitch
dividing `cellY` divides `cellX` automatically. One snap satisfies both, and `holes_per_wire > 1`
needs no special handling:

```cpp
k         = max(1, llround(cellY / m_requested));
m_snapped = cellY / k;              // = hole_pitch / k
nY = k;   nX = k * nHolesX;
```

Three things follow, and all three are load-bearing:

1. **The snapped pitch is the only one that exists downstream** — solids, in-solid test, cache key,
   echoed config. `ValidateGeometry()` runs *after* `ComputeGeom` for exactly this reason: the checks
   that compare the pitch against the wire diameter or the aperture cannot live in `LoadConfig`,
   which does not know the cell.
2. **User offsets are wrapped into `(−m/2, m/2]` before anything else**, so a stagger by a whole mesh
   pitch — which changes nothing — does not perturb the lattice phase or invalidate the cache.
3. **The lattice phase is stored** (`MeshGeom::phaseX/YCm = WrapToCell(X(0), pitch)`) because the
   analytic in-solid test has to wrap to the *same* lattice the solids were placed on. Parity is the
   trap here: whether the canonical centred lattice `(i − (n−1)/2)·p` puts a feature at 0 or at ±p/2
   depends on the parity of `n`. The cathode's `latShiftCm` hits the same trap.

## 3. Garfield++ classes used, and why

| class | role | why this one |
|---|---|---|
| `ComponentNeBem3d` | solves the field | Neither a THGEM hole nor a woven mesh has a usable closed form, and no external FEM tool is needed — neBEM is native and handles the periodic cell |
| `SolidHole` | THGEM holes, perforated apertures | A box with a polygonal hole, tiled to cover the cell exactly |
| `SolidWire` | cathode wire, woven mesh wires | **One line-charge primitive each.** This is what makes a woven mesh affordable — see §5 |
| `SolidBox` | anode pad | Zero-thickness patch, tiled by periodicity |
| `MediumPlastic` / `MediumConductor` | dielectric / copper | Boundary conditions, not transport |
| `ComponentGrid` | the transport field | Trilinear interpolation; direct neBEM lookups are ~10³× too slow for an avalanche |
| `MediumMagboltz` | the gas | Cached `.gas` table plus a `_props.csv` sidecar |
| `AvalancheMicroscopic` | electron transport | Collision-by-collision; the only way to get the hole and aperture fields right |
| `AvalancheMC` | ion drift | Ions are ~10³× slower; distance-stepped |
| `Sensor` | electrodes, signals | Shockley–Ramo from each electrode's own weighting field |

**A fact worth internalising:** `neBEM::Initialise()` groups solids **by label** into weighting
readout groups (`ComponentNeBem3d.cc`, the `std::set<std::string> labels` loop). Every mesh wire and
every aperture tile carries `SetLabel(kElecMesh)`, so however many solids the mesh is drawn with, it
gets exactly one weighting-field solve.

## 4. Code map

| region | what it does |
|---|---|
| file header | the stack diagram, the potential chain, and why the anode is mandatory |
| `kElec*`, `kAllElectrodeIds` | the shared electrode vocabulary: Solid labels ≡ Sensor names ≡ ROOT branch prefixes ≡ GUI ids |
| `PlateConfig`, `MeshConfig`, `GeometryConfig`, `FieldConfig`, … | the JSON schema, mirrored into structs with defaults in the member initialisers, so every key is optional |
| `ReadPlate`, `ReadMesh`, `LoadConfig` | parse + validate; rejects the sibling's dead keys with an explanation |
| `MeshModel`, `PlateGeom`, `MeshGeom`, `ThgemMeshGeom` | computed geometry in cm and V |
| `WrapToCell`, `InHolePolygon`, `HolePolygonArea` | lattice wrapping and the exact `SolidHole` polygon |
| `ThgemMeshDetector` | the solids, `InGas`, `InMesh`, `InMeshAperture`, `AddPlate`, `AddMeshWoven`, `AddMeshPerforated`, `ComputeGeom` |
| `ValidateGeometry` | the post-snap lattice checks |
| `EstimateElements` | the pre-solve element projection and budget |
| `ReferenceHoleAxis`, `MeshApertureAxis` | the two sampling axes |
| `DumpFieldMap`, `DumpWeightingMap` | the ROOT maps the GUI draws |
| `WireSurfaceFieldVcm`, `MeshWireSurfaceFieldVcm` | Sauli eq. 2.3 surface-field estimates |
| `ValidateGridResolution`, `ValidateField` | §6 |
| `EstimateTransitTimeNs` | sizes `simulation.time_window_ns` |
| `SampleFieldToFile`, `ScrubNonFinite`, `CountFileLines` | neBEM → grid cache, and its hygiene |
| `GeometryKey`, `DeriveFieldCacheName`, `DeriveWeightingCacheName` | the cache keys |
| `SetupGas`, `ExportGasProps` | Magboltz table caching |
| `RunDistancePoint` | the event loop, the cascade counters, the signals |
| `WriteSummaryGraphs`, `WriteSummaryCsv`, `ConfigToJson` | the outputs |
| `main` | wiring, the preflight, the two validation passes, exit codes |

## 5. The field: neBEM → ComponentGrid, and the grid budget

```
ThgemMeshDetector ctor   solids + SetPeriodicityX/Y + SetPrimAfter(1) + UseLUInversion()
     ├── cache hit ────────────────────► ComponentGrid::LoadElectricField(field_cache/*.txt)
     └── miss → nebem_.Initialise()      (BEM solve, O(n³))
                 ├── ValidateField(neBEM, onGrid = false)
                 ├── SampleFieldToFile          nx·ny·nz nodes → "xyz" text + in-solid flag
                 └── SaveWeightingField per electrode → ScrubNonFinite → Load
                                        ValidateField(grid, onGrid = true)
```

### The element budget, and why woven wins

`NbOfSegments(len, target) = clamp(floor(len / target), min_elements, max_elements)`. The element
count is driven by the **primitive count**, not by `target_element_size_um` — on primitives shorter
than `min_elements × target` the target size does nothing at all.

`SolidWire::SolidPanels()` returns no panels: a wire becomes exactly **one** primitive, discretised
into at most `max_elements` line elements. A `SolidHole` at `sectors = 4` is ≈ 95 primitives ≈ 390
elements. So a woven mesh with `k = 8` costs ~64 elements, while a perforated one at the same pitch
costs ~25 000 — and neBEM inverts a dense N × N matrix. `EstimateElements()` projects the count
before anything is built and `geometry.max_elements_budget` refuses the solve; the calibration
constants come from the double-THGEM reference solve (569 primitives, ~2300 elements).

### The grid budget

`ComponentGrid::SetMesh()` is uniform-only, so one spacing has to serve a 25–50 µm mesh wire in x/y
and a sub-millimetre amplification gap in z across a millimetres-long stack. `ValidateGridResolution`
checks two numbers and prints the values needed to fix either:

- `nodes across the mesh = feature / max(dx, dy, dz)` — warn below 3, hard-warn below 2. The feature
  is the wire diameter (woven) or the metal web between apertures (perforated).
- `cells across the amp gap = dAmp / dz` — warn below 10, because `SetupSensor` insets the drift area
  by one grid cell at each end.

**Why this check exists at all.** `ComponentGrid` has no material map; the in-solid flag stamped by
`SampleFieldToFile` is the only thing that absorbs charge. If no node lands inside a mesh wire, every
electron threads the mesh and the measured transparency is exactly 1.00 — a run that looks healthy
and means nothing. **Do not** "fix" this by dilating the flag to nodes within half a cell of a wire:
it over-blocks and biases transparency downward by a grid-dependent amount, which is worse than a
known-wrong 1.00.

### The per-solid refinement recipe (inverted)

`Solid::SetDiscretisationLevel()` is honoured per solid but is still clamped by the **global**
`min/max_elements`, so it can only make a solid *coarser*. To spend elements on the mesh and not the
plate:

```json
"max_elements": 8,
"thgem_element_size_um": 5000,
"mesh": { "element_size_um": 5 }
```

A large THGEM element size pins the plate at `min_elements`; a small mesh element size drives each
wire to `max_elements`.

## 6. Validation: what every run checks

`ValidateField` runs **twice** — once on the raw neBEM solution, once on the interpolated grid. The
grid pass is the one that decides the exit code, because that is the field the avalanche sees, and it
is the only pass that runs the grid-resolution check (`onGrid`).

Five zones: `amplification`, `mesh aperture`, `transfer`, `THGEM hole`, `drift`. The first two are
sampled on the **mesh aperture axis** (`MeshApertureAxis`) — for the perforated model the nearest
lattice point, for the woven model the window midway between four wires. Probing them on the THGEM
hole axis would, whenever the two lattices differ, land the probe on mesh metal and report a garbage
"reversed" field.

`Ez > 0` in every zone and a monotonically falling potential are the invariants. A reversed zone with
a field applied across it is almost always too few `periodic_copies`; a reversed zone whose own
applied field is ~0 is the configuration, not the solve, and is reported separately.

Beyond the zones: both wire-surface field estimates against the Magboltz table ceiling, the
`E_amp / E_transfer` ratio (warn below 20), the two grid budgets, and — from `EstimateTransitTimeNs`
— the worst-case transit time against `simulation.time_window_ns`.

## 7. The event loop and the cascade counters

One representative electron is transported and the result scaled by `nPrimary = E/W`: the mean
per-electrode charge is exactly linear in the number of primaries.

The counters, from `GetElectronEndpoint`'s birth and end z:

```cpp
if (i > 0) {                                                   // 0 is the primary
  if      (bz <= thgem.zTopCuTop && bz >= thgem.zBotCuBot) ++evtBornThgem;
  else if (bz <  mesh.zBot)                                ++evtBornAmp;
  else if (bz <  thgem.zBotCuBot)                          ++evtBornTransfer;
}
if (bz > mesh.zBot && ez <= mesh.zBot)                     ++evtEnterAmp;
```

Two decisions here are easy to get wrong, and both were:

- **The crossing plane is `mesh.zBot`, not `mesh.zTop`.** A track that lands on a mesh wire ends
  inside the mesh's own z-band, so testing against the top surface counts it as having crossed.
  Against the bottom surface it correctly is not counted, and one that threads an aperture is.
- **`evtBornTransfer` is in the transparency denominator.** The transfer gap and the mesh funnel run
  at tens of kV/cm and multiply in their own right. Dividing by `1 + bornThgem` alone produced a
  measured "transparency" of 1.283 on the first run — not a quantity. The denominator is all the
  charge that existed above the mesh:

```
gainThgem   = 1 + bornThgem
epsilon     = enterAmp / (1 + bornThgem + bornTransfer)
gainAmp     = (enterAmp + bornAmp) / enterAmp
```

`classifyZone` labels where a charge ended: `drift`, `thgem-hole`, `thgem-plate`, `transfer`,
`mesh-aperture`, `mesh-wire`/`mesh-plate`, `amplification`. For the woven model "aperture" is the
square window between four wires, not merely "not in a wire" — a point level with the upper layer but
shadowed by a lower wire is not a clear path.

## 8. ROOT output schema

```
field/     h_field_mag, h_potential, h_field_ez, h_field_ex          x–z slice
           h_field_mag_yz, h_potential_yz                            y–z slice (along a wire)
           g_axis_field, g_axis_ez, g_axis_potential                 THGEM hole axis
           g_axis_field_mesh, g_axis_ez_mesh, g_axis_potential_mesh  mesh aperture axis
           h_wpot_<id>, h_wfield_mag_<id>, g_axis_wpot_<id>          per readout electrode
summary/   g_<id>_charge, g_avalanche_size, g_gain_thgem,
           g_gain_amp, g_mesh_transparency
dist_<h>mm[_x<x>mm]/
           h_<id>_charge, h_avalanche_size, h_n_primary_electrons,
           h_gain_thgem, h_gain_amp, h_mesh_transparency
           p_<id>_signal, p_<id>_electron, p_<id>_ion, p_<id>_amp, p_<id>_amp_int
           t_signals
```

`t_signals` scalar branches: `gain_thgem`, `gain_amp`, `mesh_transparency`, `n_born_thgem`,
`n_born_transfer`, `n_born_amp`, `n_entered_amp`, plus the per-electrode current/charge branches and
the track arrays. `SetBasketSize("*", 24 MB)` is deliberate — uproot mis-parses multi-basket branches.

`summary.csv` carries `mean_gain_thgem`, `mean_gain_amp`, `mean_mesh_transparency`,
`mean_born_thgem`, `mean_born_transfer`, `mean_born_amp` alongside the per-electrode charges.

`run_config.json` echoes the resolved config plus a `derived` block: every potential, every z plane,
the cell, the solved `mesh{}` lattice (snapped and requested pitch, `k`, wire/aperture centres,
optical transparency) and a `grid{}` block with the two budgets. **The GUI reads only this block** —
it never re-derives geometry from the config panel, which may have been edited since the run.

## 9. The GUI

`gui/app.py`, PyQt5, one file. It writes a temp JSON and runs `build/thgem_mesh_sim --config … --out
…`, streaming stdout; exit code 2 is treated as success-with-a-note, not a crash.

Two pieces of arithmetic are duplicated in Python **on purpose**: `_update_derived_voltages` (the
potential chain) and `_update_mesh_derived` (the snap and the optical transparency). A drift between
the GUI and the C++ then shows up as a disagreement between two labels, before anything is solved.

`_root_geometry_lines` draws the x–z overlay. Note what that slice does to a woven mesh: it cuts the
lower layer (∥ y) transversely — ticks — while the upper layer (∥ x) is cut lengthwise and cannot be
resolved in x. The upper layer is a single dashed line marking its plane, **not** a solid sheet.
`_update_track_plot` draws the 3D view; `_draw_wire_y` and `_draw_wire_x` are mirrors, and both take
a radius argument because the cathode and mesh wires differ.

## 10. Numerical subtleties

- **`SolidHole` meshes a polygon, not a circle.** A 4(n−1)-gon of *circumradius* r with its first
  vertex at −45°, inradius `r·cos(π/(4(n−1)))`. `sectors = 2` is therefore a square of half-side
  r/√2, not r. `InHolePolygon` reproduces it exactly; a plain circle of radius r is 3 % wrong in area
  at `sectors = 4` and **29 % wrong** at `sectors = 2`, which the perforated mesh uses.
- **`InGas` is analytic, not a `GeometrySimple::GetMedium` lookup.** The lookup only knows solids at
  their literal positions, so a staggered lattice puts material across the cell edge where no literal
  solid sits. Wrapping to the nearest lattice point is correct for any offset — and far cheaper over
  the millions of nodes a sampling pass visits.
- **Wire primitives are added after neBEM reduces panels to the basic period,** so wire centres are
  not wrapped for you. Every mesh wire centre must stay inside `(−cell/2, +cell/2)`; `MeshGeom::X/Y`
  centre the lattice canonically, which guarantees it.
- **Conductor singularities.** 1/r blows up on a wire axis; `SampleFieldToFile` and `ScrubNonFinite`
  zero non-finite values and mark those nodes absorbing.
- **A truncated cache is silently accepted** by `ComponentGrid::LoadElectricField` — it leaves the
  missing nodes at zero. `CountFileLines` validates the node count before the cache is trusted, and
  sampling writes atomically via `.part` + `rename`.
- **`grid.SetMedium()` must follow `SetMesh()`** — `SetMesh()` calls `Reset()`.

## 11. Extending it

**Add a config knob.** struct → `LoadConfig` → `ConfigToJson` → GUI `to_config_dict` /
`load_from_dict` → **`GeometryKey`**. Forgetting the last one is the classic mistake: the run
silently reuses a cache solved for different geometry. This project's key deliberately includes the
discretisation, which the double-THGEM sibling's does not.

**Add a readout electrode.** `SetLabel` on the solid + `kAllElectrodeIds` + the GUI's
`ELECTRODE_IDS` / `_LABELS` / `_COLOR_OFFSETS`. Solids sharing a label become one electrode.

**Add a third mesh model.** `MeshModel` enum → `ComputeGeom`'s `fillMesh` block (thickness, optical
transparency) → a new `AddMesh*` in the detector ctor → a branch in `InMesh` and `InMeshAperture` →
`EstimateElements` → `GeometryKey` → the GUI's `MeshWidgets.apply_model` and both renderers. The
`InGas` predicate and the element estimate are the two that will be forgotten.

**Make the perforated model reach finer pitches.** The blocker is 390 elements per `SolidHole`. The
honest fix is a coarser `sectors` (an octagon is ~273, a square ~156) plus a raised
`max_elements_budget` — not silently dropping the budget check.
