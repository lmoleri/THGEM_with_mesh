#!/usr/bin/env python3
"""Check the Geometry tab's cut-plane maths and its primitive budget.

Usage:  python3 tools/verify_geometry_view.py [<run dir> ...]
Exit status 0 on pass, 1 on fail.

Three things are checked, in increasing cost:

1. `_GeoView._arc` against brute-force half-plane sampling.  The cut-away
   restricts a circle to the arc that survives a half-plane, in closed form —
   `c0 + r*cos(t - phi) >= p`.  A sign slip there is invisible on screen (you
   get a plausible-looking arc on the wrong side), so it is checked against
   721-point sampling over tens of thousands of random circles and planes.

2. The `derived` block against the run's own `fields` block.  The annotation
   column recomputes each gap's field as |dV| / dz rather than reading it back,
   so this identity is simultaneously the column's content and its test: if the
   two disagree, the solved stack is not the one the config asked for.

3. `estimate_objects` against what a real draw emits.  The reduction ladder
   picks a rung *before* drawing, so an estimator that under-counts would let a
   pathological geometry blow the budget it was supposed to enforce.

Needs PyQt5 and PyROOT; runs headless (QT_QPA_PLATFORM=offscreen, ROOT batch).
"""
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
_QAPP = None
_PANEL = None


def load_app():
    import importlib.util
    spec = importlib.util.spec_from_file_location("guiapp", PROJ / "gui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_arc(GeoView, n_cases=30000):
    random.seed(20260910)
    bad = 0
    t = np.linspace(0.0, 2.0 * math.pi, 721, endpoint=False)
    for _ in range(n_cases):
        c_a, c_b = random.uniform(-1, 1), random.uniform(-1, 1)
        r = random.uniform(1e-3, 2.0)
        ang = random.uniform(0, 2 * math.pi)
        n_a, n_b = math.cos(ang), math.sin(ang)
        p, keep = random.uniform(-2, 2), random.choice((-1, 1))
        rng = GeoView._arc(c_a, c_b, r, n_a, n_b, p, keep)
        a, b = c_a + r * np.cos(t), c_b + r * np.sin(t)
        want = keep * (n_a * a + n_b * b - p) >= 0
        if rng is None:
            if want.any():
                bad += 1
        else:
            t0, t1 = rng
            got = ((t - t0) % (2.0 * math.pi)) <= (t1 - t0) + 1e-12
            if (want != got).sum() > 2:      # one sample either side is fine
                bad += 1
    print(f"  cut arc vs brute force ({n_cases} cases): {bad} mismatches")
    return bad == 0


def check_fields(run_dir):
    import json
    path = Path(run_dir) / "run_config.json"
    cfg = json.loads(path.read_text())
    d, f = cfg["derived"], cfg["fields"]
    p, m = d["thgem"], d["mesh"]

    def field(z_hi, z_lo, v_hi, v_lo):
        return abs(v_hi - v_lo) / (z_hi - z_lo) * 1e-3

    checks = [
        ("e_drift_kvcm", field(d["z_wire_cm"], p["z_top_cu_top_cm"],
                               d["v_wire"], d["v_thgem_top"]), f["e_drift_kvcm"]),
        ("e_transfer_kvcm", field(p["z_bot_cu_bot_cm"], m["z_top_cm"],
                                  d["v_thgem_bot"], d["v_mesh"]), f["e_transfer_kvcm"]),
        ("delta_v_thgem_V", d["v_thgem_bot"] - d["v_thgem_top"],
         f["delta_v_thgem_V"]),
    ]
    # The amplification gap is set by its voltage, so check it as one: comparing
    # the derived field against delta_v_mesh_anode_V / d_amp would just re-divide
    # what the config re-multiplied, which proves nothing.
    if "delta_v_mesh_anode_V" in f:
        checks.append(("delta_v_mesh_anode_V", d["v_anode"] - d["v_mesh"],
                       f["delta_v_mesh_anode_V"]))
    elif "e_amplification_kvcm" in f:
        # A run from before the reparameterisation: check what it recorded.
        checks.append(("e_amplification_kvcm (legacy)",
                       field(m["z_bot_cm"], d["z_anode_cm"],
                             d["v_mesh"], d["v_anode"]),
                       f["e_amplification_kvcm"]))
    ok = True
    for name, got, want in checks:
        good = abs(got - want) <= 1e-6 * max(1.0, abs(want))
        ok &= good
        print(f"    {name:22s} derived {got:10.4f}  config {want:10.4f}"
              f"   {'ok' if good else 'MISMATCH'}")
    return ok


def check_budget(mod, run_dir):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    from PyQt5.QtWidgets import QApplication
    global _QAPP, _PANEL
    if QApplication.instance() is None:
        # Must be kept alive: an unreferenced QApplication is collected before
        # the first QWidget is built, and Qt aborts the process.
        _QAPP = QApplication([])
    if _PANEL is None:
        _PANEL = mod.ResultsPanel()   # one panel: a second would re-create the canvas
    rp = _PANEL
    if not rp.load_geometry(str(run_dir)):
        print(f"    could not load {run_dir}")
        return False
    ok = True
    for cells in (1, 3, 8, 15):
        rp._geo_cut_axis, rp._geo_n_holes = None, cells
        rp._geo_update_cut_pos()
        rp._update_geometry_plot()
        drew = rp._geo_last_n_obj
        _, _, _, est, note = rp._geo_pick_detail(rp._geo_geom)
        err = abs(drew - est) / max(1, est) * 100.0
        # The estimator is exact by construction — it counts the same lines the
        # drawers emit.  Any drift means a drawer changed and estimate_objects
        # did not follow, which is precisely the regression worth catching.
        within = err < 2.0
        under  = drew <= 1.25 * mod._GEO_OBJECT_BUDGET
        ok &= within and under
        print(f"    cells {cells:2d}: drew {drew:5d}  projected {est:5d}"
              f"  ({err:4.1f}% off)  {'ok' if within and under else 'FAIL'}"
              + (f"   [{note}]" if note else ""))
    # A cut removes geometry, but not monotonically in the primitive count:
    # draw_cylinder always draws both arc-end longitudinals so a section reads
    # as a cut rather than as a thinner tube, so a cylinder sliced at a grazing
    # angle costs one line MORE than the whole one did.  The real invariants are
    # that a cut keeping nothing draws nothing, that one keeping everything
    # matches the uncut count, and that the two halves of a mid-cut cover the
    # whole; plus a loose ceiling to catch a runaway.
    rp._geo_n_holes = 3
    rp._geo_cut_axis = None
    rp._geo_update_cut_pos(); rp._update_geometry_plot()
    base = rp._geo_last_n_obj
    worst = 0
    for axis in ("x", "y", "z"):
        for frac in (0.25, 0.5, 0.75):
            for keep in (-1, 1):
                rp._geo_cut_axis, rp._geo_cut_frac, rp._geo_cut_keep = axis, frac, keep
                rp._geo_update_cut_pos(); rp._update_geometry_plot()
                worst = max(worst, rp._geo_last_n_obj)
    within = worst <= 1.15 * base
    ok &= within
    print(f"    18 cut cases: worst {worst} vs uncut {base} "
          f"({100.0 * worst / max(1, base):.0f} %)  {'ok' if within else 'FAIL'}")

    for axis, frac, keep, want, label in (
            ("z", 0.0, -1, 0,    "keeps nothing"),
            ("z", 1.0, -1, base, "keeps everything"),
    ):
        rp._geo_cut_axis, rp._geo_cut_frac, rp._geo_cut_keep = axis, frac, keep
        rp._geo_update_cut_pos(); rp._update_geometry_plot()
        got = rp._geo_last_n_obj
        # frac 1 is not a perfect no-op: the plane lands on the topmost cathode
        # wire's axis, halving its facets.
        good = (got <= 2) if want == 0 else (got >= 0.95 * base)
        ok &= good
        print(f"    cut {axis} frac {frac} {label}: {got} "
              f"(uncut {base})  {'ok' if good else 'FAIL'}")

    lo_hi = []
    for keep in (-1, 1):
        rp._geo_cut_axis, rp._geo_cut_frac, rp._geo_cut_keep = "x", 0.5, keep
        rp._geo_update_cut_pos(); rp._update_geometry_plot()
        lo_hi.append(rp._geo_last_n_obj)
    covers = sum(lo_hi) >= base
    ok &= covers
    print(f"    x mid-cut halves {lo_hi[0]} + {lo_hi[1]} >= uncut {base}: "
          f"{'ok' if covers else 'FAIL'}")
    return ok


def main():
    mod = load_app()
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        runs = sorted(p.parent for p in (PROJ / "results").glob("*/run_config.json"))
    if not runs:
        print("No run directories found; pass one on the command line.")
        return 1

    ok = True
    print("Cut geometry")
    ok &= check_arc(mod._GeoView)
    print("\nPotential chain (derived vs fields)")
    for r in runs:
        print(f"  {r.name}")
        ok &= check_fields(r)
    print("\nPrimitive budget")
    for r in runs:
        print(f"  {r.name}")
        ok &= check_budget(mod, r)

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # ROOT's atexit canvas teardown segfaults on macOS; the app does the same.
    os._exit(code)
