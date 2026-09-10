#!/usr/bin/env python3
"""
THGEM + mesh Simulation GUI

A PyQt5 desktop application for configuring, running and displaying results from
the thgem_mesh_sim Garfield++ binary: a THGEM feeding a micromegas-style mesh
under a wire cathode, read out at the anode pad below the amplification gap.

Launch from anywhere:
    python3 projects/THGEM_mesh/gui/app.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# ROOT colour palettes offered by the E-Field / Weighting Field tabs.  The maps are
# drawn in an interactive ROOT canvas (right-click to zoom), so these are ROOT's own
# palettes (TColor::EColorPalette) rather than matplotlib names.
_ROOT_PALETTE = {
    "Viridis":     112,
    "Bird":         57,
    "Cividis":     113,
    "Thermometer": 105,
    "Dark body":    53,
}

# ---------------------------------------------------------------------------
# Readout electrodes
# ---------------------------------------------------------------------------

# The five electrodes thgem_mesh_sim can read out, in physical stack order
# (top → bottom).  These strings are the C++ Solid labels, the Sensor electrode
# names and the ROOT branch prefixes, so the same vocabulary runs from the config
# panel to the plots.  The mesh is a single conductor and so a single electrode,
# however many wires it is drawn with.
ELECTRODE_IDS = [
    "wire_cathode",
    "thgem_top",
    "thgem_bottom",
    "mesh",
    "anode",
]

ELECTRODE_LABELS = {
    "wire_cathode":  "Wire cathode",
    "thgem_top":     "THGEM top Cu",
    "thgem_bottom":  "THGEM bottom Cu",
    "mesh":          "Mesh",
    "anode":         "Anode pad",
}

# Distinct ROOT colours, one per electrode, so a given electrode keeps its colour
# across the Waveforms, Integrals and Weighting Field tabs.
ELECTRODE_COLOR_OFFSETS = {
    "wire_cathode":  ("kGreen",   2),
    "thgem_top":     ("kOrange",  7),
    "thgem_bottom":  ("kRed",     1),
    "mesh":          ("kTeal",    3),
    "anode":         ("kBlue",    1),
}


def electrode_color(eid):
    """ROOT colour index for one electrode id (lazy: needs PyROOT imported)."""
    import ROOT
    base, off = ELECTRODE_COLOR_OFFSETS.get(eid, ("kBlack", 0))
    return getattr(ROOT, base) + off


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR       = Path(__file__).parent.resolve()                # …/projects/THGEM_mesh/gui/
PROJ_DIR         = (SCRIPT_DIR / "..").resolve()                  # …/projects/THGEM_mesh/
BINARY           = PROJ_DIR / "build" / "thgem_mesh_sim"
GARFIELD_INSTALL = (PROJ_DIR / "../../local/garfield").resolve()  # …/local/garfield/
GAS_DIR          = PROJ_DIR / "gas"                               # gas tables + _props.csv sidecars


# ---------------------------------------------------------------------------
# Gas filename derivation
# ---------------------------------------------------------------------------

def derive_gas_filename(gas: dict) -> str:
    """Return a deterministic .gas filename from gas config parameters."""
    g1  = gas.get("gas1", "ar")
    f1  = round(gas.get("gas1_fraction_pct", 70.0))
    g2  = gas.get("gas2", "co2")
    f2  = 100 - f1
    T   = round(gas.get("temperature_K", 293.15))
    P   = round(gas.get("pressure_Torr", 760.0))
    Ee  = round(gas.get("max_electron_energy_eV", 2000.0))
    Ef    = round(gas.get("e_field_max_vcm", 400000.0) / 1000)
    EfMin = round(gas.get("e_field_min_vcm", 100.0))
    n   = gas.get("n_field_points", 10)
    c   = gas.get("n_magboltz_collisions", 2)
    pen = "pen" if gas.get("enable_penning", True) else "nopen"
    return f"{g1}{f1}_{g2}_{f2}_T{T}_P{P}_Ee{Ee}_Ef{EfMin}v-{Ef}k_n{n}_c{c}_{pen}.gas"


def _file_safe_number(v: float) -> str:
    """Mirror FileSafeNumber() in thgem_mesh_sim.cc: 0.4 -> "0p4", 45.0 -> "45"."""
    txt = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return txt.replace(".", "p") if txt else "0"


def derive_gas_props_filename(gas: dict) -> str:
    """Return the sidecar CSV filename for Magboltz transport properties."""
    return derive_gas_filename(gas).replace(".gas", "_props.csv")


# ---------------------------------------------------------------------------
# Background simulation runner
# ---------------------------------------------------------------------------

class SimRunner(QThread):
    """Runs thgem_mesh_sim in a background thread and emits stdout line-by-line."""

    log_line = pyqtSignal(str)   # one stdout line
    finished = pyqtSignal(str)   # emits the run output directory on success
    failed   = pyqtSignal(str)   # emits an error message on failure
    stopped  = pyqtSignal()      # user asked to stop (not an error)

    # thgem_mesh_sim exit codes: 0 = clean, 2 = the run completed and wrote its output
    # but the field validation flagged something, anything else = a real error.
    EXIT_OK       = 0
    EXIT_WARNINGS = 2

    def __init__(self, config_dict: dict, out_dir: str,
                 run_name: str = "", parent=None):
        super().__init__(parent)
        self._config    = config_dict
        self._out_dir   = out_dir
        self._run_name  = run_name          # passed as --run-name to binary
        self._proc: subprocess.Popen | None = None
        self._stopped   = False             # set when stop() is requested
        self.exit_code  = None              # read by MainWindow after finished

    # ── public ──────────────────────────────────────────────────────────

    def stop(self):
        """Ask the subprocess to terminate (called from the main thread)."""
        self._stopped = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    # ── private ─────────────────────────────────────────────────────────

    def run(self):  # noqa: PLR0912 — runs in the worker thread
        # Write a temporary config file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tf:
                json.dump(self._config, tf, indent=2)
                tmp_cfg = tf.name
        except OSError as exc:
            self.failed.emit(f"Could not write temporary config: {exc}")
            return

        cmd = [str(BINARY), "--config", tmp_cfg, "--out", self._out_dir]
        if self._run_name:
            cmd += ["--run-name", self._run_name]

        env = os.environ.copy()
        env["GARFIELD_INSTALL"] = str(GARFIELD_INSTALL)
        env["HEED_DATABASE"] = str(GARFIELD_INSTALL / "share" / "Heed" / "database")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJ_DIR),   # relative gas-file paths resolve from here
                env=env,
            )
            for line in self._proc.stdout:
                self.log_line.emit(line.rstrip())
            self._proc.wait()
            ret = self._proc.returncode
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        finally:
            try:
                os.unlink(tmp_cfg)
            except OSError:
                pass

        if self._stopped:
            self.stopped.emit()          # user-initiated stop — not an error
            return
        self.exit_code = ret
        if ret not in (self.EXIT_OK, self.EXIT_WARNINGS):
            self.failed.emit(f"Binary exited with code {ret}")
            return
        if ret == self.EXIT_WARNINGS:
            # The run finished and wrote everything; only the field validation
            # flagged something.  Say so and carry on loading the results —
            # treating this as a crash used to throw away a completed run.
            self.log_line.emit(
                "\n[GUI] The run completed and wrote its output, but the field "
                "validation flagged something — see the WARNING above.")

        # Locate the sub-directory the binary created.
        # If we passed --run-name we already know the exact path; otherwise
        # glob for any folder matching the naming pattern (supports both the
        # new date-prefixed names and old-style V<V>V__n<N> directories).
        out_path = Path(self._out_dir)
        if self._run_name:
            run_dir = str(out_path / self._run_name)
        else:
            # Matches both "__n<N>" runs and the "__field" name a field-only
            # run gets; the date prefix is what makes "*__*" specific enough.
            subdirs = sorted((p for p in out_path.glob("*__*") if p.is_dir()),
                             key=lambda p: p.stat().st_mtime)
            run_dir = str(subdirs[-1]) if subdirs else self._out_dir
        self.finished.emit(run_dir)


# ---------------------------------------------------------------------------
# Config panel (left side)
# ---------------------------------------------------------------------------

class PlateWidgets:
    """The five spinboxes/combo describing the THGEM plate."""

    def __init__(self, form, mk_dspin, defaults):
        self.hole_diameter = mk_dspin(50.0, 2000.0, 10.0, 1, defaults["hole"])
        self.hole_diameter.setToolTip("Hole diameter [μm]")
        self.plate_thickness = mk_dspin(50.0, 3000.0, 10.0, 1, defaults["thick"])
        self.plate_thickness.setToolTip("Dielectric foil thickness [μm]")
        self.copper_thickness = mk_dspin(0.0, 200.0, 5.0, 1, defaults["cu"])
        self.copper_thickness.setToolTip("Copper cladding thickness per face [μm]")
        self.rim_um = mk_dspin(0.0, 500.0, 5.0, 1, defaults["rim"])
        self.rim_um.setToolTip(
            "Etched rim [μm]: the copper is etched back from the hole edge, so the copper "
            "openings are wider than the dielectric hole by this amount (0 = straight hole).")
        self.dielectric = QComboBox()
        self.dielectric.addItems(["FR4", "Kapton"])

        form.addRow("Hole diameter [μm]",    self.hole_diameter)
        form.addRow("Plate thickness [μm]",  self.plate_thickness)
        form.addRow("Copper thickness [μm]", self.copper_thickness)
        form.addRow("Rim [μm]",              self.rim_um)
        form.addRow("Dielectric",            self.dielectric)

    def spinboxes(self):
        return [self.hole_diameter, self.plate_thickness, self.copper_thickness, self.rim_um]

    def to_dict(self) -> dict:
        return {
            "hole_diameter_um":    self.hole_diameter.value(),
            "plate_thickness_um":  self.plate_thickness.value(),
            "copper_thickness_um": self.copper_thickness.value(),
            "rim_um":              self.rim_um.value(),
            "dielectric_material": self.dielectric.currentText().lower(),
        }

    def load(self, d: dict, fallback: dict):
        def get(key, default):
            return d.get(key, fallback.get(key, default))
        self.hole_diameter.setValue(   get("hole_diameter_um", 500.0))
        self.plate_thickness.setValue( get("plate_thickness_um", 400.0))
        self.copper_thickness.setValue(get("copper_thickness_um", 35.0))
        self.rim_um.setValue(          get("rim_um", 0.0))
        diel = str(get("dielectric_material", "fr4")).lower()
        self.dielectric.setCurrentText("Kapton" if diel == "kapton" else "FR4")


# Aperture shape combo → SolidHole "sectors".  SolidHole meshes a 4(n-1)-gon of
# *circumradius* r, so "square" is sectors = 2 and its side is aperture/√2, not
# the aperture itself.  The binary is always handed the circumscribed size.
APERTURE_SHAPES = [("Square", 2), ("Octagon", 3), ("Round (12-gon)", 4)]


class MeshWidgets:
    """The widget set describing the mesh electrode, for either model.

    Which widgets matter depends on `model`, so the irrelevant half is disabled
    rather than hidden — switching model back and forth keeps both sets of
    numbers, and a config written for one model still round-trips through the
    other's keys.
    """

    def __init__(self, form, mk_dspin, mk_spin):
        self.model = QComboBox()
        self.model.addItems(["Woven", "Perforated"])
        self.model.setToolTip(
            "Woven: two orthogonal layers of wires — a micromegas micromesh or a wire "
            "grid.\nEach wire is one neBEM line-charge primitive, so this model is cheap "
            "enough to\nreach true micromegas pitch (~50-65 μm).\n\n"
            "Perforated: a thin conductor sheet with a lattice of apertures.  Each "
            "aperture\ncosts ~390 boundary elements, so this model is for coarse and "
            "thick meshes only\n(roughly hole pitch / 3 and above).")

        self.pitch = mk_dspin(20.0, 5000.0, 10.0, 1, 250.0)
        self.pitch.setToolTip(
            "Mesh lattice pitch [μm].  One periodic cell must hold a whole number of "
            "mesh cells,\nso this is snapped to hole_pitch / k — the label below shows "
            "what is actually simulated.")
        self.wire_diameter = mk_dspin(5.0, 500.0, 5.0, 1, 50.0)
        self.wire_diameter.setToolTip("Woven: wire diameter [μm].")
        self.thickness = mk_dspin(0.0, 2000.0, 10.0, 1, 0.0)
        self.thickness.setToolTip(
            "Mesh thickness [μm].  0 = auto: for a woven mesh that is 2 × the wire "
            "diameter,\nwhich puts the two crossed layers exactly tangent (the physical "
            "weave).\nRequired for the perforated model, which has no wire to derive it from.")
        self.aperture = mk_dspin(10.0, 4000.0, 10.0, 1, 300.0)
        self.aperture.setToolTip(
            "Perforated: aperture size across corners [μm] — the circumscribed diameter "
            "of the\npolygon, which is what SolidHole is given.")
        self.aperture_shape = QComboBox()
        self.aperture_shape.addItems([n for n, _ in APERTURE_SHAPES])
        self.aperture_shape.setCurrentText("Round (12-gon)")
        self.aperture_shape.setToolTip(
            "Perforated: aperture polygon.  'Square' is SolidHole sectors = 2, whose side "
            "is\naperture/√2 rather than the aperture itself.")
        self.offset_x = mk_dspin(-5000.0, 5000.0, 10.0, 1, 0.0)
        self.offset_x.setToolTip(
            "Mesh-lattice stagger in x [μm] relative to the THGEM hole lattice.\n"
            "A stagger of one whole mesh pitch is a no-op and is wrapped away.")
        self.offset_y = mk_dspin(-5000.0, 5000.0, 10.0, 1, 0.0)
        self.offset_y.setToolTip("Mesh-lattice stagger in y [μm].")

        self.snap_label = QLabel("—")
        self.snap_label.setWordWrap(True)
        self.snap_label.setStyleSheet("color: #888;")

        form.addRow("Model",              self.model)
        form.addRow("Pitch [μm]",         self.pitch)
        form.addRow("Wire diameter [μm]", self.wire_diameter)
        form.addRow("Thickness [μm]",     self.thickness)
        form.addRow("Aperture [μm]",      self.aperture)
        form.addRow("Aperture shape",     self.aperture_shape)
        form.addRow("Offset x [μm]",      self.offset_x)
        form.addRow("Offset y [μm]",      self.offset_y)
        form.addRow("", self.snap_label)

    def is_woven(self) -> bool:
        return self.model.currentText() == "Woven"

    def apply_model(self):
        """Enable the half of the widget set the chosen model actually uses."""
        woven = self.is_woven()
        self.wire_diameter.setEnabled(woven)
        self.aperture.setEnabled(not woven)
        self.aperture_shape.setEnabled(not woven)
        # A perforated mesh has no wire to derive a thickness from, so 0 is not
        # a legal "auto" there.
        self.thickness.setMinimum(0.0 if woven else 10.0)
        if not woven and self.thickness.value() < 10.0:
            self.thickness.setValue(100.0)

    def spinboxes(self):
        return [self.pitch, self.wire_diameter, self.thickness, self.aperture,
                self.offset_x, self.offset_y]

    def to_dict(self) -> dict:
        sectors = dict(APERTURE_SHAPES)[self.aperture_shape.currentText()]
        return {
            "model":            "woven" if self.is_woven() else "perforated",
            "pitch_um":         self.pitch.value(),
            "wire_diameter_um": self.wire_diameter.value(),
            "thickness_um":     self.thickness.value(),
            "aperture_um":      self.aperture.value(),
            "aperture_sectors": sectors,
            "offset_x_um":      self.offset_x.value(),
            "offset_y_um":      self.offset_y.value(),
        }

    def load(self, d: dict):
        model = str(d.get("model", "woven")).lower()
        self.model.setCurrentText("Perforated" if model == "perforated" else "Woven")
        self.pitch.setValue(         float(d.get("pitch_um", 250.0)))
        self.wire_diameter.setValue( float(d.get("wire_diameter_um", 50.0)))
        self.aperture.setValue(      float(d.get("aperture_um", 300.0)))
        sectors = int(d.get("aperture_sectors", 4))
        for name, n in APERTURE_SHAPES:
            if n == sectors:
                self.aperture_shape.setCurrentText(name)
                break
        self.apply_model()   # sets the thickness minimum before the value
        self.thickness.setValue(float(d.get("thickness_um", 0.0)))
        self.offset_x.setValue(float(d.get("offset_x_um", 0.0)))
        self.offset_y.setValue(float(d.get("offset_y_um", 0.0)))


class ConfigPanel(QScrollArea):
    """Scrollable panel with one QGroupBox per config section."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(360)

        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setSpacing(8)

        # ── Cell & wire cathode ───────────────────────────────────────────
        cell_box  = QGroupBox("Cell && wire cathode")
        cell_form = self._left_form_layout(cell_box)

        self.hole_pitch = self._dspin(100.0, 5000.0, 10.0, 1, 800.0)
        self.hole_pitch.setToolTip(
            "Hole lattice pitch [μm], square, shared by both plates.\n"
            "One periodic cell cannot hold two incommensurate lattices, so the\n"
            "plates differ by an offset rather than by pitch.")
        self.wire_diameter = self._dspin(5.0, 500.0, 5.0, 1, 50.0)
        self.wire_diameter.setToolTip("Cathode wire diameter [μm]")
        self.holes_per_wire = self._spin(1, 8, 1)
        self.holes_per_wire.setToolTip(
            "Holes per wire: the wire pitch is this many hole pitches, and the\n"
            "periodic cell holds one wire and this many holes per plate.\n"
            "The neBEM solve cost grows with it — start at 1.")
        self.wire_between_holes = QCheckBox("Wire sits between holes")
        self.wire_between_holes.setToolTip(
            "Shift the hole lattice by half a pitch so the wires run between hole\n"
            "columns instead of directly above one.")
        self.wire_pitch_label = QLabel("—")
        self.wire_pitch_label.setStyleSheet("font-size: 10px; color: grey;")

        cell_form.addRow("Hole pitch [μm]",     self.hole_pitch)
        cell_form.addRow("Wire diameter [μm]",  self.wire_diameter)
        cell_form.addRow("Holes per wire",      self.holes_per_wire)
        cell_form.addRow("",                    self.wire_between_holes)
        cell_form.addRow("Wire pitch (derived)", self.wire_pitch_label)
        root_layout.addWidget(cell_box)

        # ── THGEM ─────────────────────────────────────────────────────────
        p1_box  = QGroupBox("THGEM")
        p1_form = self._left_form_layout(p1_box)
        self.plate1 = PlateWidgets(
            p1_form, self._dspin,
            {"hole": 500.0, "thick": 400.0, "cu": 35.0, "rim": 100.0})
        root_layout.addWidget(p1_box)

        # ── Mesh ──────────────────────────────────────────────────────────
        mesh_box  = QGroupBox("Mesh")
        mesh_form = self._left_form_layout(mesh_box)
        self.mesh = MeshWidgets(mesh_form, self._dspin, self._spin)
        root_layout.addWidget(mesh_box)
        self.mesh.model.currentIndexChanged.connect(self._on_mesh_model_changed)

        # ── Gaps ──────────────────────────────────────────────────────────
        gap_box  = QGroupBox("Gaps")
        gap_form = self._left_form_layout(gap_box)

        self.drift_gap = self._dspin(0.2, 20.0, 0.5, 2, 2.0)
        self.drift_gap.setToolTip("Wire plane → THGEM top copper [mm]")
        self.transfer_gap = self._dspin(0.1, 20.0, 0.5, 2, 1.0)
        self.transfer_gap.setToolTip("THGEM bottom copper → mesh top surface [mm]")
        self.amplification_gap = self._dspin(0.02, 2.0, 0.01, 3, 0.2)
        self.amplification_gap.setToolTip(
            "Mesh bottom → anode pad [mm].  This is the micromegas amplification gap:\n"
            "the whole second-stage gain happens here, and there is no anode toggle —\n"
            "a mesh is 35-65 % open and cannot terminate the volume by itself.")

        gap_form.addRow("Drift gap [mm]",         self.drift_gap)
        gap_form.addRow("THGEM → mesh gap [mm]",  self.transfer_gap)
        gap_form.addRow("Mesh → anode gap [mm]",  self.amplification_gap)
        root_layout.addWidget(gap_box)

        # ── Fields (physics inputs → derived electrode potentials) ────────
        fld_box  = QGroupBox("Fields")
        fld_form = self._left_form_layout(fld_box)

        self.e_drift = self._dspin(0.0, 10.0, 0.1, 3, 0.5)
        self.e_drift.setToolTip(
            "Drift-gap field [kV/cm].\n"
            "This is the gap AVERAGE (ΔV / gap). Under a wire cathode the local\n"
            "field sags to roughly half of it midway down the gap, which is what\n"
            "sets the transit time.")
        self.delta_v1 = self._dspin(100.0, 3000.0, 25.0, 1, 1200.0)
        self.delta_v1.setToolTip("Voltage across the THGEM, top → bottom copper [V]")
        self.e_transfer = self._dspin(0.0, 200.0, 0.5, 3, 1.0)
        self.e_transfer.setToolTip(
            "THGEM → mesh gap field [kV/cm].\n"
            "Above roughly 20-30 kV/cm this gap stops being a passive coupling region\n"
            "and multiplies in its own right, like a parallel-plate stage. That gain is\n"
            "real and shows up in the total avalanche size and the collected charge, but\n"
            "it belongs to neither stage: electrons born there are reported separately as\n"
            "n_born_transfer, and counted in the mesh-transparency denominator.\n"
            "Note also that the mesh's electron transparency scales with\n"
            "E_amplification / E_transfer — raising this field lowers it.\n"
            "The shipped Magboltz table already reaches 400 kV/cm, so no gas\n"
            "regeneration is needed; raise gas.transport_max_energy_eV for high-field\n"
            "runs to stop the collision-rate table ratcheting up during transport."
        )
        self.e_amplification = self._dspin(0.0, 100.0, 1.0, 2, 45.0)
        self.e_amplification.setToolTip(
            "Amplification-gap field [kV/cm], mesh → anode.\n"
            "This IS the micromegas gain knob: the mesh is a single conductor, so its\n"
            "stage has no voltage across it and its gain comes from this field over the\n"
            "gap (40-60 kV/cm over 64-320 μm in a real device).\n"
            "It also sets the mesh's electron transparency, through the ratio\n"
            "E_amplification / E_transfer — below about 20 the mesh collects most of\n"
            "the charge the THGEM produced instead of passing it through.")

        fld_form.addRow("E_drift [kV/cm]",         self.e_drift)
        fld_form.addRow("ΔV_THGEM [V]",            self.delta_v1)
        fld_form.addRow("E_transfer [kV/cm]",      self.e_transfer)
        fld_form.addRow("E_amplification [kV/cm]", self.e_amplification)

        self.derived_v_label = QLabel("—")
        self.derived_v_label.setWordWrap(True)
        self.derived_v_label.setStyleSheet("font-size: 10px; color: grey;")
        fld_form.addRow("Electrode V (derived)", self.derived_v_label)
        self.transparency_label = QLabel("—")
        self.transparency_label.setWordWrap(True)
        self.transparency_label.setStyleSheet("font-size: 10px; color: grey;")
        fld_form.addRow("Mesh transparency", self.transparency_label)
        root_layout.addWidget(fld_box)

        # ── Readout electrodes ────────────────────────────────────────────
        ro_box  = QGroupBox("Readout electrodes")
        ro_lay  = QVBoxLayout(ro_box)
        ro_hint = QLabel(
            "Each electrode costs one neBEM weighting-field solve and one cached "
            "grid (tens of MB), so keep this to what you actually plot.")
        ro_hint.setWordWrap(True)
        ro_hint.setStyleSheet("font-size: 10px; color: grey;")
        ro_lay.addWidget(ro_hint)
        self.readout_checks = {}
        for eid in ELECTRODE_IDS:
            cb = QCheckBox(ELECTRODE_LABELS[eid])
            cb.setChecked(eid == "anode")
            self.readout_checks[eid] = cb
            ro_lay.addWidget(cb)
        root_layout.addWidget(ro_box)

        # ── Advanced (neBEM discretisation / transport grid) ──────────────
        # "Element" here means a boundary element of the neBEM discretisation —
        # not the physical mesh electrode above.
        adv_box  = QGroupBox("Advanced (neBEM discretisation)")
        adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv_form = self._left_form_layout(adv_box)

        self.target_element = self._dspin(20.0, 1000.0, 10.0, 0, 120.0)
        self.target_element.setToolTip("neBEM target boundary-element size [μm]")
        self.grid_nx = self._spin(10, 401, 41)
        self.grid_nx.setToolTip(
            "Transport-grid nodes across the cell in x.\n"
            "The cell is holes_per_wire pitches wide, so raise this with that number\n"
            "to keep the resolution.")
        self.grid_ny = self._spin(10, 401, 41)
        self.grid_ny.setToolTip("Transport-grid nodes across the cell in y (one pitch)")
        self.grid_nz = self._spin(10, 801, 221)
        self.grid_nz.setToolTip(
            "Transport-grid nodes along z (the whole stack).\n"
            "This grid is what has to resolve the mesh: if no node lands inside a mesh\n"
            "wire, nothing absorbs charge there and the measured electron transparency\n"
            "reads 1.00 no matter what the fields are. The run's field validation prints\n"
            "the nodes-across-the-mesh budget and the values needed to fix it.")
        self.periodic_copies = self._spin(0, 16, 9)
        self.periodic_copies.setToolTip(
            "neBEM periodic copies (uses 2n+1).\n"
            "Too few leaves the on-axis field REVERSED in a gap, which traps every\n"
            "drifting charge — the run's field validation is what catches it.")
        self.hole_sectors = self._spin(2, 16, 4)
        self.hole_sectors.setToolTip("Hole polygon sectors (2=square, 3=octagon, …)")
        self.min_elements = self._spin(1, 20, 2)
        self.max_elements = self._spin(1, 40, 4)
        self.max_elements_budget = self._spin(1000, 200000, 8000)
        self.max_elements_budget.setSingleStep(1000)
        self.max_elements_budget.setToolTip(
            "Refuse to start a solve projected above this many boundary elements.\n"
            "neBEM inverts a dense N×N matrix — cost O(N³), stored inverse N² doubles —\n"
            "so a perforated mesh at a fine pitch goes from minutes to days with no\n"
            "warning in between. The reference double-THGEM solve is ~2300 elements.")

        _elem_recipe = (
            "Per-solid neBEM element size [μm]; 0 = auto (use the target above).\n\n"
            "The min/max element clamp is GLOBAL, so a per-solid size can only make a\n"
            "solid coarser, never finer. To spend elements on the mesh and not on the\n"
            "plate the recipe is therefore inverted: raise 'Max elements' to ~8, set a\n"
            "LARGE THGEM element size (pinning the plate at min elements) and a SMALL\n"
            "mesh element size (driving each wire to max elements).")
        self.thgem_element_size = self._dspin(0.0, 20000.0, 100.0, 0, 0.0)
        self.thgem_element_size.setToolTip(_elem_recipe)
        self.mesh_element_size = self._dspin(0.0, 20000.0, 5.0, 0, 0.0)
        self.mesh_element_size.setToolTip(_elem_recipe)

        adv_form.addRow("Target element [μm]",  self.target_element)
        adv_form.addRow("THGEM element [μm]",   self.thgem_element_size)
        adv_form.addRow("Mesh element [μm]",    self.mesh_element_size)
        adv_form.addRow("Grid nx",              self.grid_nx)
        adv_form.addRow("Grid ny",              self.grid_ny)
        adv_form.addRow("Grid nz",              self.grid_nz)
        adv_form.addRow("Periodic copies",      self.periodic_copies)
        adv_form.addRow("Hole sectors",         self.hole_sectors)
        adv_form.addRow("Min elements",         self.min_elements)
        adv_form.addRow("Max elements",         self.max_elements)
        adv_form.addRow("Max elements budget",  self.max_elements_budget)
        root_layout.addWidget(adv_box)

        # ── Source ────────────────────────────────────────────────────────
        src_box  = QGroupBox("Source")
        src_form = self._left_form_layout(src_box)

        self.energy_kev = self._dspin(0.1, 100.0, 0.1, 2, 5.9)

        self.dist_random = QCheckBox("Random (uniform over drift gap)")
        self.dist_random.setChecked(False)
        self.distances  = QLineEdit("1.0")
        self.distances.setToolTip(
            "Comma-separated primary-electron heights above the THGEM's top surface, "
            "in the drift gap [mm] (0 = at the top copper).")
        self.dist_random.toggled.connect(lambda on: self.distances.setEnabled(not on))

        self.x_random = QCheckBox("Random (uniform over cell)")
        self.x_random.setChecked(True)
        self.x_positions = QLineEdit("0.0")
        self.x_positions.setEnabled(False)
        self.x_positions.setToolTip(
            "Comma-separated fixed x-positions in the cell [cm]; a fixed x pins y to "
            "the THGEM's hole row, so a scan in x crosses both the hole and the wire. "
            "Random samples both x and y over the cell.")
        self.x_random.toggled.connect(lambda on: self.x_positions.setEnabled(not on))

        src_form.addRow("Energy [keV]",        self.energy_kev)
        src_form.addRow("Drift-gap height",    self.dist_random)
        src_form.addRow("  fixed height [mm]", self.distances)
        src_form.addRow("Cell position",       self.x_random)
        src_form.addRow("  fixed x [cm]",      self.x_positions)
        root_layout.addWidget(src_box)
        # ── Gas ───────────────────────────────────────────────────────────
        gas_box  = QGroupBox("Gas")
        gas_form = self._left_form_layout(gas_box)

        # — Composition rows —
        _GAS_LIST = ["ar", "co2", "cf4", "ch4", "c2h6", "n2", "he", "ne"]
        _ION_LIST  = ["ar", "co2", "cf4", "he", "ne"]  # species with IonMobility files

        self.gas1_combo = QComboBox()
        self.gas1_combo.addItems(_GAS_LIST)
        self.gas1_combo.setEditable(True)
        self.gas1_combo.setCurrentText("ar")
        self.gas1_combo.setToolTip("First gas component (Magboltz species name, lowercase)")

        self.frac1_spin = QDoubleSpinBox()
        self.frac1_spin.setRange(1.0, 99.0)
        self.frac1_spin.setSingleStep(1.0)
        self.frac1_spin.setDecimals(1)
        self.frac1_spin.setValue(70.0)
        self.frac1_spin.setSuffix(" %")

        gas1_row = QWidget()
        gas1_h   = self._left_row_layout(gas1_row)
        gas1_h.addWidget(self.gas1_combo)
        gas1_h.addWidget(self.frac1_spin)
        gas_form.addRow("Gas 1 [%]", gas1_row)

        self.gas2_combo = QComboBox()
        self.gas2_combo.addItems(_GAS_LIST)
        self.gas2_combo.setEditable(True)
        self.gas2_combo.setCurrentText("co2")
        self.gas2_combo.setToolTip("Second gas component (Magboltz species name, lowercase)")

        self.gas2_frac_lbl = QLabel("30.0 %")
        self.gas2_frac_lbl.setToolTip("Fraction of gas 2 = 100% − gas 1 fraction (auto-computed)")

        gas2_row = QWidget()
        gas2_h   = self._left_row_layout(gas2_row)
        gas2_h.addWidget(self.gas2_combo)
        gas2_h.addWidget(self.gas2_frac_lbl)
        gas_form.addRow("Gas 2 [%]", gas2_row)

        self.ion_combo = QComboBox()
        self.ion_combo.addItems(_ION_LIST)
        self.ion_combo.setCurrentText("co2")
        self.ion_combo.setToolTip(
            "Ion species for the mobility table (IonMobility_X+_X.txt).\n"
            "Available: ar, co2, cf4, he, ne.\n"
            "Should match the dominant drifting ion in the mixture."
        )
        gas_form.addRow("Ion species", self.ion_combo)

        self.temperature = self._dspin(200.0, 500.0, 1.0, 2, 293.15)
        self.pressure    = self._dspin(100.0, 3000.0, 10.0, 1, 760.0)

        self.penning = QCheckBox()
        self.penning.setChecked(True)
        self.ncoll = self._spin(1, 100, 2)
        self.ncoll.setToolTip("Magboltz collision cycles per field point (higher = more accurate)")
        self.w_value = self._dspin(10.0, 100.0, 0.5, 1, 26.0)
        self.w_value.setToolTip("Effective ionisation energy W [eV per ion pair] for primary electron count")

        self.max_electron_energy = self._dspin(100.0, 100_000.0, 100.0, 0, 2000.0)
        self.max_electron_energy.setToolTip(
            "EFINAL of the Magboltz transport table [eV].\n"
            "Also keys the cached .gas filename."
        )
        self.transport_max_energy = self._dspin(10.0, 100_000.0, 50.0, 0, 200.0)
        self.transport_max_energy.setToolTip(
            "Ceiling of the microscopic collision-rate table [eV].\n"
            "Applied after the gas table is loaded, so it does not change the\n"
            "cached .gas file. Lower values give finer energy bins; it only needs\n"
            "to exceed the energies electrons actually reach (a THGEM hole at 24 kV/cm\n"
            "peak field keeps them at a few eV). Garfield raises it automatically\n"
            "if an electron ever exceeds it."
        )
        self.n_field_pts = self._spin(5, 500, 10)
        self.n_field_pts.setToolTip(
            "Number of log-spaced E-field points for the Magboltz transport table.\n"
            "More points → smoother interpolation; fewer → faster gas generation."
        )
        self.e_field_min = self._dspin(10.0, 100_000.0, 100.0, 0, 100.0)
        self.e_field_min.setToolTip(
            "Minimum E-field in the Magboltz table [V/cm].\n"
            "100 V/cm is suitable for most TGC operating conditions."
        )
        self.e_field_max = self._dspin(10_000.0, 1_000_000.0, 10_000.0, 0, 400_000.0)
        self.e_field_max.setToolTip(
            "Maximum E-field in the Magboltz table [V/cm].\n"
            "Must exceed the largest field in the detector — the run prints it and\\n"
            "warns when the table falls short."
        )

        self.gas_file_label = QLabel()
        self.gas_file_label.setWordWrap(True)
        self.gas_file_label.setStyleSheet("font-size: 10px;")

        gas_form.addRow("Temperature [K]",     self.temperature)
        gas_form.addRow("Pressure [Torr]",     self.pressure)
        gas_form.addRow("Penning transfer",    self.penning)
        gas_form.addRow("Magboltz ncoll",      self.ncoll)
        gas_form.addRow("W-value [eV]",        self.w_value)
        gas_form.addRow("Max e⁻ energy [eV]",  self.max_electron_energy)
        gas_form.addRow("Transport ceiling [eV]", self.transport_max_energy)
        gas_form.addRow("Field points",        self.n_field_pts)
        gas_form.addRow("E-field min [V/cm]", self.e_field_min)
        gas_form.addRow("E-field max [V/cm]", self.e_field_max)
        gas_form.addRow("Gas file (auto)",     self.gas_file_label)
        root_layout.addWidget(gas_box)

        # Update gas file label whenever a gas or geometry parameter changes
        self.gas1_combo.currentTextChanged.connect(self._update_gas2_frac_label)
        self.frac1_spin.valueChanged.connect(self._update_gas2_frac_label)
        self.gas1_combo.currentTextChanged.connect(self._update_gas_file_label)
        self.frac1_spin.valueChanged.connect(self._update_gas_file_label)
        self.gas2_combo.currentTextChanged.connect(self._update_gas_file_label)
        self.ion_combo.currentTextChanged.connect(self._update_gas_file_label)
        self.temperature.valueChanged.connect(self._update_gas_file_label)
        self.pressure.valueChanged.connect(self._update_gas_file_label)
        self.penning.toggled.connect(self._update_gas_file_label)
        self.ncoll.valueChanged.connect(self._update_gas_file_label)
        self.max_electron_energy.valueChanged.connect(self._update_gas_file_label)
        self.n_field_pts.valueChanged.connect(self._update_gas_file_label)
        self.e_field_min.valueChanged.connect(self._update_gas_file_label)
        self.e_field_max.valueChanged.connect(self._update_gas_file_label)

        # ── Simulation ────────────────────────────────────────────────────
        sim_box  = QGroupBox("Simulation")
        sim_form = self._left_form_layout(sim_box)

        self.n_events    = self._spin(0, 100000, 1000)
        self.n_events.setSpecialValueText("0  (field only)")
        self.n_events.setToolTip(
            "Events per source point.\n"
            "0 = field only: solve the field, sample it and every read-out electrode's\n"
            "weighting field onto the grid, dump the maps and stop before any transport.\n"
            "The cheap way to check a new geometry — the field solve is the expensive part,\n"
            "and the E-Field / Weighting Field tabs are populated either way."
        )
        self.max_aval    = self._spin(1000, 10000000, 500000)
        self.time_window = self._dspin(10.0, 100000.0, 10.0, 1, 300.0)
        self.time_step   = self._dspin(0.1, 10.0, 0.1, 2, 0.5)
        self.enable_ion_drift = QCheckBox()
        self.enable_ion_drift.setChecked(False)
        self.enable_ion_drift.setToolTip(
            "Drift positive ions after each avalanche (AvalancheMC, bounded).\n"
            "Off by default: ions induce ~nothing on the anode, so the signal is\n"
            "unchanged, and ions are ~1000× slower to transport. Enable for the\n"
            "3D ion-path view or ion-backflow studies."
        )
        self.store_drift_lines = QCheckBox()
        self.store_drift_lines.setChecked(False)
        self.store_drift_lines.setToolTip(
            "Store intermediate drift-line steps for the 3D track view.\n"
            "Off: primary electron shown as straight start→end line, no avalanche paths.\n"
            "On: full curved primary + avalanche-electron trajectories, "
            "but adds ~15 % CPU time per event."
        )
        self.ion_max_step = self._dspin(0.0, 100.0, 1.0, 1, 5.0)
        self.ion_max_step.setToolTip(
            "AvalancheMC ion drift distance step [µm].\n"
            "Smaller steps resolve the early ion signal and curved paths but cost\n"
            "proportionally more CPU. 0 uses the AvalancheMC default step."
        )
        self.ion_time_window = self._dspin(1000.0, 1.0e7, 1000.0, 0, 1.0e6)
        self.ion_time_window.setToolTip(
            "Ion-drift time-window backstop [ns].\n"
            "Terminates an ion trapped at a field stagnation point so it can never\n"
            "loop indefinitely (the old DriftLineRKF had no such bound → hang/OOM).\n"
            "1 ms comfortably exceeds a full drift-gap ion transit (~0.6 ms)."
        )
        self.max_ions_drifted = self._spin(0, 10000000, 200)
        self.max_ions_drifted.setToolTip(
            "Max avalanche ions back-drifted per event (0 = all).\n"
            "Ions drift away from the anode and induce ~nothing on it, so the\n"
            "anode signal is unchanged; transporting all of them dominates the\n"
            "runtime at high gain. Keep ~200 for fast runs; raise or set 0 for\n"
            "ion-backflow studies."
        )
        self.random_seed = self._spin(0, 2147483647, 0)
        self.random_seed.setToolTip(
            "Random-number seed.\n"
            "0 = randomize each run; >0 = fixed seed for reproducible runs."
        )

        sim_form.addRow("Events",             self.n_events)
        sim_form.addRow("Max avalanche size", self.max_aval)
        sim_form.addRow("Time window [ns]",   self.time_window)
        sim_form.addRow("Time step [ns]",     self.time_step)
        sim_form.addRow("Ion transport",      self.enable_ion_drift)
        sim_form.addRow("Store drift lines",  self.store_drift_lines)
        sim_form.addRow("Ion max step [µm]",  self.ion_max_step)
        sim_form.addRow("Ion time window [ns]", self.ion_time_window)
        sim_form.addRow("Max ions drifted",   self.max_ions_drifted)
        sim_form.addRow("Random seed",        self.random_seed)
        root_layout.addWidget(sim_box)

        # ── Amplifier (front-end electronics) ──────────────────────────────
        amp_box  = QGroupBox("Amplifier (front-end)")
        amp_form = self._left_form_layout(amp_box)
        self.amp_enable = QCheckBox()
        self.amp_enable.setChecked(False)
        self.amp_enable.setToolTip(
            "Pass each electrode's induced current through a CIVIDEC C2-TCT broadband\n"
            "transimpedance amplifier to get a shaped output voltage [mV]. The\n"
            "Waveforms tab then offers an 'Amplifier' display mode.")
        self.amp_gain      = self._dspin(0.0, 80.0, 1.0, 1, 40.0)
        self.amp_gain.setToolTip("Voltage gain [dB] (40 dB = ×100).")
        self.amp_zin       = self._dspin(1.0, 10000.0, 10.0, 1, 50.0)
        self.amp_zin.setToolTip("Input impedance [Ω] (transimpedance I→V scale).")
        self.amp_bw_high   = self._dspin(1e6, 1e11, 1e8, 0, 2.0e9)
        self.amp_bw_high.setToolTip("Upper −3 dB bandwidth [Hz] → intrinsic low-pass.")
        self.amp_sample_ns = self._dspin(0.0, 100.0, 0.5, 2, 0.0)
        self.amp_sample_ns.setToolTip("Acquisition aperture / boxcar average [ns] (0 = off).")
        amp_form.addRow("Enable",               self.amp_enable)
        amp_form.addRow("Gain [dB]",            self.amp_gain)
        amp_form.addRow("Input impedance [Ω]",  self.amp_zin)
        amp_form.addRow("Bandwidth high [Hz]",  self.amp_bw_high)
        amp_form.addRow("Output sample [ns]",   self.amp_sample_ns)
        root_layout.addWidget(amp_box)

        # ── Output ────────────────────────────────────────────────────────
        out_box  = QGroupBox("Output")
        out_form = self._left_form_layout(out_box)

        out_row = QWidget()
        out_h   = self._left_row_layout(out_row)
        self.out_dir = QLineEdit("results")
        self.out_dir.setToolTip("Output base directory (relative paths resolve from projects/double_THGEM/)")
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(28)
        btn_out.clicked.connect(self._browse_out_dir)
        out_h.addWidget(self.out_dir)
        out_h.addWidget(btn_out)

        out_form.addRow("Directory", out_row)

        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("auto  (date + voltage + events)")
        self.run_name.setToolTip(
            "Optional label for the run subfolder.\n"
            "Leave blank: yymmdd_hh-mm__VφV__nη  (auto)\n"
            "Filled:      yymmdd_hh-mm__<your label>")
        out_form.addRow("Run name", self.run_name)

        root_layout.addWidget(out_box)

        root_layout.addStretch()
        self.setWidget(container)

        self._wire_widgets_connect()
        self._update_gas2_frac_label()
        self._update_gas_file_label()
        self.mesh.apply_model()
        self._update_derived_voltages()
        self._update_wire_pitch_label()
        self._update_mesh_derived()

    # ── signal wiring ────────────────────────────────────────────────────

    def _wire_widgets_connect(self):
        """Connect everything that feeds a live derived readout."""
        for w in (self.drift_gap, self.transfer_gap, self.amplification_gap,
                  self.e_drift, self.delta_v1, self.e_transfer,
                  self.e_amplification):
            w.valueChanged.connect(self._update_derived_voltages)
        for w in (self.hole_pitch, self.holes_per_wire):
            w.valueChanged.connect(self._update_wire_pitch_label)
        for w in (self.hole_pitch, self.holes_per_wire, self.mesh.pitch,
                  self.mesh.wire_diameter, self.mesh.aperture):
            w.valueChanged.connect(self._update_mesh_derived)
        self.mesh.aperture_shape.currentIndexChanged.connect(self._update_mesh_derived)

    def _on_mesh_model_changed(self, *_):
        self.mesh.apply_model()
        self._update_mesh_derived()

    def _update_wire_pitch_label(self):
        pitch = self.hole_pitch.value() * self.holes_per_wire.value()
        self.wire_pitch_label.setText(f"{pitch:.0f} μm  ({self.holes_per_wire.value()} hole(s) per wire)")

    def _update_mesh_derived(self):
        """Show the snapped mesh lattice and its optical transparency.

        Deliberately duplicates the arithmetic in ThgemMeshGeom/ComputeGeom
        (thgem_mesh_sim.cc) rather than reading it back from a run: a
        disagreement between this label and the binary's own "mesh :" line is
        then visible before anything is solved.
        """
        p = self.hole_pitch.value()
        m = self.mesh.pitch.value()
        if p <= 0.0 or m <= 0.0:
            self.mesh.snap_label.setText("—")
            return
        # cellX = holes_per_wire * cellY, so a pitch dividing cellY divides both.
        k = max(1, round(p / m))
        snapped = p / k
        nx, ny = k * self.holes_per_wire.value(), k

        if self.mesh.is_woven():
            d = self.mesh.wire_diameter.value()
            open_side = max(0.0, snapped - d)
            t_opt = (open_side / snapped) ** 2
            bad = d >= snapped
        else:
            r = self.mesh.aperture.value() / 2.0
            sectors = dict(APERTURE_SHAPES)[self.mesh.aperture_shape.currentText()]
            nv = 4 * (sectors - 1)
            area = 0.5 * nv * r * r * math.sin(2.0 * math.pi / nv)
            t_opt = area / (snapped * snapped)
            bad = 2.0 * r >= snapped

        txt = (f"{snapped:.1f} μm snapped ({k} per hole pitch) · "
               f"{nx}×{ny} per cell · T_opt {t_opt:.3f}")
        if abs(snapped - m) > 0.05:
            txt = f"{m:.1f} → " + txt
        if bad:
            colour, txt = "#c0392b", txt + "  — features merge at this pitch"
        elif k > 16:
            colour = "#e67e22"
            txt += "  — past hole_pitch/16; the transport grid will not resolve it"
        elif abs(snapped - m) > 0.5:
            colour = "#e67e22"
        else:
            colour = "#888"
        self.mesh.snap_label.setStyleSheet(f"color: {colour};")
        self.mesh.snap_label.setText(txt)

    # ── widget factories ─────────────────────────────────────────────────

    @staticmethod
    def _dspin(lo: float, hi: float, step: float, dec: int, val: float) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(dec)
        w.setValue(val)
        return w

    @staticmethod
    def _spin(lo: int, hi: int, val: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    @staticmethod
    def _left_form_layout(parent: QWidget) -> QFormLayout:
        form = QFormLayout(parent)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        return form

    @staticmethod
    def _left_row_layout(parent: QWidget) -> QHBoxLayout:
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return layout

    # ── gas file label ───────────────────────────────────────────────────

    def _update_gas2_frac_label(self):
        """Keep the gas-2 fraction label in sync with gas-1 fraction spinner."""
        self.gas2_frac_lbl.setText(f"{100.0 - self.frac1_spin.value():.1f} %")

    def _update_gas_file_label(self):
        gas = {
            "gas1":                   self.gas1_combo.currentText().strip().lower(),
            "gas1_fraction_pct":      self.frac1_spin.value(),
            "gas2":                   self.gas2_combo.currentText().strip().lower(),
            "temperature_K":          self.temperature.value(),
            "pressure_Torr":          self.pressure.value(),
            "enable_penning":         self.penning.isChecked(),
            "n_magboltz_collisions":  self.ncoll.value(),
            "max_electron_energy_eV": self.max_electron_energy.value(),
            "n_field_points":         self.n_field_pts.value(),
            "e_field_min_vcm":        self.e_field_min.value(),
            "e_field_max_vcm":        self.e_field_max.value(),
        }
        name = derive_gas_filename(gas)
        exists = (GAS_DIR / name).exists()
        status = "✓ exists" if exists else "will be generated"
        self.gas_file_label.setText(f"gas/{name}\n[{status}]")

    def _update_derived_voltages(self):
        """Show the electrode potentials derived from the physics fields.

        Mirrors ThgemMeshDetector::ComputeGeom in thgem_mesh_sim.cc: the anode is
        the 0 V reference and electrons drift −z toward it.  Keep the two in
        step — this label is the quickest check that a config means what you think.
        """
        d_amp_cm   = self.amplification_gap.value() * 0.1
        d_tr_cm    = self.transfer_gap.value() * 0.1
        d_drift_cm = self.drift_gap.value() * 0.1
        e_amp_vcm   = self.e_amplification.value() * 1000.0
        e_tr_vcm    = self.e_transfer.value() * 1000.0
        e_drift_vcm = self.e_drift.value() * 1000.0

        v_anode = 0.0
        v_mesh  = v_anode - e_amp_vcm * d_amp_cm
        v_bot   = v_mesh  - e_tr_vcm * d_tr_cm
        v_top   = v_bot   - self.delta_v1.value()
        v_wire  = v_top   - e_drift_vcm * d_drift_cm

        self.derived_v_label.setText(
            f"wire {v_wire:.0f} · thgem_top {v_top:.0f} · thgem_bot {v_bot:.0f} · "
            f"mesh {v_mesh:.0f} · anode {v_anode:.0f} V")

        # The mesh's electron transparency is driven by the ratio of the fields
        # on its two sides: the amplification field has to pull the drift lines
        # through the apertures against the transfer field's spread.
        e_tr = self.e_transfer.value()
        if e_tr <= 0.0:
            self.transparency_label.setStyleSheet("font-size: 10px; color: grey;")
            self.transparency_label.setText(
                "E_amp / E_transfer = ∞  (no transfer field: all charge is pulled through)")
            return
        ratio = self.e_amplification.value() / e_tr
        if ratio < 20.0:
            self.transparency_label.setStyleSheet("font-size: 10px; color: #e67e22;")
            self.transparency_label.setText(
                f"E_amp / E_transfer = {ratio:.1f} — too low for the mesh to be "
                f"electron-transparent. Charge leaving the THGEM will be collected on "
                f"the mesh rather than entering the amplification gap.")
        else:
            self.transparency_label.setStyleSheet("font-size: 10px; color: grey;")
            self.transparency_label.setText(f"E_amp / E_transfer = {ratio:.0f}")

    def _browse_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.out_dir.setText(path)

    # ── serialisation ─────────────────────────────────────────────────────

    def selected_electrodes(self) -> list:
        """Ticked readout electrodes, in physical stack order."""
        return [e for e in ELECTRODE_IDS if self.readout_checks[e].isChecked()]

    def to_config_dict(self) -> dict:
        """Assemble widget values into a config dict suitable for JSON dump."""
        if self.dist_random.isChecked():
            dists = None  # → JSON null → C++ random per-event
        else:
            raw = self.distances.text().strip()
            try:
                dists = [float(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                dists = [1.0]

        if self.x_random.isChecked():
            x_positions = None
        else:
            raw_x = self.x_positions.text().strip()
            try:
                x_positions = [float(v.strip()) for v in raw_x.split(",") if v.strip()]
            except ValueError:
                x_positions = [0.0]

        electrodes = self.selected_electrodes()
        if not electrodes:
            # thgem_mesh_sim rejects an empty list; fall back to what it would have
            # chosen itself rather than failing the run.
            electrodes = ["anode"]

        return {
            "geometry": {
                "hole_pitch_um":          self.hole_pitch.value(),
                "wire_diameter_um":       self.wire_diameter.value(),
                "holes_per_wire":         self.holes_per_wire.value(),
                "wire_between_holes":     self.wire_between_holes.isChecked(),
                "drift_gap_mm":           self.drift_gap.value(),
                "transfer_gap_mm":        self.transfer_gap.value(),
                "amplification_gap_mm":   self.amplification_gap.value(),
                "thgem":                  self.plate1.to_dict(),
                "mesh":                   dict(self.mesh.to_dict(),
                                               element_size_um=self.mesh_element_size.value()),
                "target_element_size_um": self.target_element.value(),
                "thgem_element_size_um":  self.thgem_element_size.value(),
                "min_elements":           self.min_elements.value(),
                "max_elements":           self.max_elements.value(),
                "max_elements_budget":    self.max_elements_budget.value(),
                "periodic_copies":        self.periodic_copies.value(),
                "hole_sectors":           self.hole_sectors.value(),
                "grid_nx":                self.grid_nx.value(),
                "grid_ny":                self.grid_ny.value(),
                "grid_nz":                self.grid_nz.value(),
            },
            "fields": {
                "e_drift_kvcm":         self.e_drift.value(),
                "delta_v_thgem_V":      self.delta_v1.value(),
                "e_transfer_kvcm":      self.e_transfer.value(),
                "e_amplification_kvcm": self.e_amplification.value(),
            },
            "readout": {
                "electrodes": electrodes,
            },
            "source": {
                "energy_keV":          self.energy_kev.value(),
                "source_distances_mm": dists,
                "x_positions_cm":      x_positions,
            },
            "gas": {
                "gas1":                   self.gas1_combo.currentText().strip().lower(),
                "gas1_fraction_pct":      self.frac1_spin.value(),
                "gas2":                   self.gas2_combo.currentText().strip().lower(),
                "ion_species":            self.ion_combo.currentText().strip().lower(),
                "temperature_K":          self.temperature.value(),
                "pressure_Torr":          self.pressure.value(),
                "enable_penning":         self.penning.isChecked(),
                "n_magboltz_collisions":  self.ncoll.value(),
                "w_value_eV":             self.w_value.value(),
                "max_electron_energy_eV": self.max_electron_energy.value(),
                "transport_max_energy_eV": self.transport_max_energy.value(),
                "n_field_points":         self.n_field_pts.value(),
                "e_field_min_vcm":        self.e_field_min.value(),
                "e_field_max_vcm":        self.e_field_max.value(),
            },
            "simulation": {
                "n_events":           self.n_events.value(),
                "max_avalanche_size": self.max_aval.value(),
                "time_window_ns":     self.time_window.value(),
                "time_step_ns":       self.time_step.value(),
                "enable_ion_drift":   self.enable_ion_drift.isChecked(),
                "store_drift_lines":  self.store_drift_lines.isChecked(),
                "ion_max_step_um":    self.ion_max_step.value(),
                "ion_time_window_ns": self.ion_time_window.value(),
                "max_ions_drifted":   self.max_ions_drifted.value(),
                "random_seed":        self.random_seed.value(),
            },
            "amplifier": {
                "enable":              self.amp_enable.isChecked(),
                "gain_db":             self.amp_gain.value(),
                "input_impedance_ohm": self.amp_zin.value(),
                "bandwidth_high_hz":   self.amp_bw_high.value(),
                "output_sample_ns":    self.amp_sample_ns.value(),
            },
        }

    def load_from_dict(self, d: dict):
        """Populate all widgets from a config dict (e.g. loaded from JSON)."""
        g = d.get("geometry", {})
        self.hole_pitch.setValue(     g.get("hole_pitch_um", 800.0))
        self.wire_diameter.setValue(  g.get("wire_diameter_um", 50.0))
        self.holes_per_wire.setValue(int(g.get("holes_per_wire", 1)))
        self.wire_between_holes.setChecked(bool(g.get("wire_between_holes", False)))
        self.drift_gap.setValue(      g.get("drift_gap_mm", 2.0))
        self.transfer_gap.setValue(   g.get("transfer_gap_mm", 1.0))
        self.amplification_gap.setValue(g.get("amplification_gap_mm", 0.2))

        self.plate1.load(g.get("thgem", {}), {})
        self.mesh.load(g.get("mesh", {}))

        self.target_element.setValue( g.get("target_element_size_um", 120.0))
        self.thgem_element_size.setValue(g.get("thgem_element_size_um", 0.0))
        self.mesh_element_size.setValue((g.get("mesh") or {}).get("element_size_um", 0.0))
        self.min_elements.setValue(   int(g.get("min_elements", 2)))
        self.max_elements.setValue(   int(g.get("max_elements", 4)))
        self.max_elements_budget.setValue(int(g.get("max_elements_budget", 8000)))
        self.periodic_copies.setValue(int(g.get("periodic_copies", 9)))
        self.hole_sectors.setValue(   int(g.get("hole_sectors", 4)))
        self.grid_nx.setValue(        int(g.get("grid_nx", 65)))
        self.grid_ny.setValue(        int(g.get("grid_ny", 65)))
        self.grid_nz.setValue(        int(g.get("grid_nz", 245)))

        fl = d.get("fields", {})
        self.e_drift.setValue(       fl.get("e_drift_kvcm", 0.5))
        self.delta_v1.setValue(      fl.get("delta_v_thgem_V", 1200.0))
        self.e_transfer.setValue(    fl.get("e_transfer_kvcm", 1.0))
        self.e_amplification.setValue(fl.get("e_amplification_kvcm", 45.0))

        ro = (d.get("readout") or {}).get("electrodes")
        if ro is None:
            # Same fallback thgem_mesh_sim applies to a null list.
            ro = ["anode"]
        for eid, cb in self.readout_checks.items():
            cb.setChecked(eid in ro)

        s = d.get("source", {})
        self.energy_kev.setValue(s.get("energy_keV", 5.9))
        dists = s.get("source_distances_mm", [1.0])
        if dists is None:
            self.dist_random.setChecked(True)
        else:
            self.dist_random.setChecked(False)
            self.distances.setText(",".join(str(v) for v in dists))
        x_positions = s.get("x_positions_cm", None)
        if x_positions is None:
            self.x_random.setChecked(True)
        else:
            self.x_random.setChecked(False)
            self.x_positions.setText(",".join(str(v) for v in x_positions))

        gas = d.get("gas", {})
        self.gas1_combo.setCurrentText(gas.get("gas1", "ar"))
        self.frac1_spin.setValue(       gas.get("gas1_fraction_pct", 70.0))
        self.gas2_combo.setCurrentText(gas.get("gas2", "co2"))
        self.ion_combo.setCurrentText( gas.get("ion_species", "co2"))
        self._update_gas2_frac_label()
        self.temperature.setValue(gas.get("temperature_K", 293.15))
        self.pressure.setValue(   gas.get("pressure_Torr", 760.0))
        self.penning.setChecked(  gas.get("enable_penning", True))
        self.ncoll.setValue(      gas.get("n_magboltz_collisions", 2))
        self.w_value.setValue(    gas.get("w_value_eV", 26.0))
        self.max_electron_energy.setValue(gas.get("max_electron_energy_eV", 2000.0))
        self.transport_max_energy.setValue(gas.get("transport_max_energy_eV", 200.0))
        self.n_field_pts.setValue(        gas.get("n_field_points", 10))
        self.e_field_min.setValue(        gas.get("e_field_min_vcm", 100.0))
        self.e_field_max.setValue(        gas.get("e_field_max_vcm", 400_000.0))

        sim = d.get("simulation", {})
        self.n_events.setValue(        sim.get("n_events", 10))
        self.max_aval.setValue(        sim.get("max_avalanche_size", 200000))
        self.time_window.setValue(     sim.get("time_window_ns", 1200.0))
        self.time_step.setValue(       sim.get("time_step_ns", 1.0))
        self.enable_ion_drift.setChecked(sim.get("enable_ion_drift", False))
        self.store_drift_lines.setChecked(sim.get("store_drift_lines", True))
        self.ion_max_step.setValue(      sim.get("ion_max_step_um", 5.0))
        self.ion_time_window.setValue(   sim.get("ion_time_window_ns", 1.0e6))
        self.max_ions_drifted.setValue(  int(sim.get("max_ions_drifted", 200)))
        self.random_seed.setValue(       int(sim.get("random_seed", 0)))

        amp = d.get("amplifier", {})
        self.amp_enable.setChecked(bool(amp.get("enable", False)))
        self.amp_gain.setValue(      amp.get("gain_db", 40.0))
        self.amp_zin.setValue(       amp.get("input_impedance_ohm", 50.0))
        self.amp_bw_high.setValue(   amp.get("bandwidth_high_hz", 2.0e9))
        self.amp_sample_ns.setValue( amp.get("output_sample_ns", 0.0))

        self.mesh.apply_model()
        self._update_derived_voltages()
        self._update_wire_pitch_label()
        self._update_mesh_derived()
        self._update_gas_file_label()


# ---------------------------------------------------------------------------
# Matplotlib canvas wrapper
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib Figure embedded in a Qt widget."""

    def __init__(self, nrows: int = 1, ncols: int = 1,
                 figsize: tuple | None = None, parent=None):
        self.figure = Figure(figsize=figsize or (6, 4))
        self.axes: list = []
        self.set_grid(nrows, ncols)
        super().__init__(self.figure)

    def set_grid(self, nrows: int, ncols: int):
        """Rebuild the subplot grid.

        The Plots tab shows one panel per read-out electrode plus the cascade
        diagnostics, and the electrode set is a config knob — so the grid has to
        follow the data rather than being fixed at construction.
        """
        if len(self.axes) == nrows * ncols:
            return
        self.figure.clf()
        self.axes = [self.figure.add_subplot(nrows, ncols, i + 1)
                     for i in range(nrows * ncols)]



# ---------------------------------------------------------------------------
# Results panel (right side)
# ---------------------------------------------------------------------------

class ResultsPanel(QTabWidget):
    """Results browser with summary tables, plots, waveforms, integrals, and tracks."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Log tab ───────────────────────────────────────────────────────
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(11)
        self.log.setFont(mono)
        self.addTab(self.log, "Log")

        # ── Summary tab ───────────────────────────────────────────────────
        _sum_widget = QWidget()
        _sum_vbox   = QVBoxLayout(_sum_widget)
        _sum_vbox.setContentsMargins(4, 4, 4, 4)

        _sum_btn_row = QWidget()
        _sum_btn_h   = QHBoxLayout(_sum_btn_row)
        _sum_btn_h.setContentsMargins(0, 0, 0, 0)
        self.summary_export_btn = QPushButton("Export CSV …")
        self.summary_export_btn.setEnabled(False)
        _sum_btn_h.addWidget(self.summary_export_btn)
        _sum_btn_h.addStretch()
        _sum_vbox.addWidget(_sum_btn_row)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        _sum_vbox.addWidget(self.table)

        self._tab_summary = self.addTab(_sum_widget, "Summary")
        self.summary_export_btn.clicked.connect(self._on_summary_export_csv)

        # ── Plots tab: 2×3 matplotlib figure ─────────────────────────────
        self.plots_canvas = MplCanvas(nrows=2, ncols=3, figsize=(11, 5))
        self.addTab(self.plots_canvas, "Plots")

        # ── Waveforms tab: ROOT TCanvas browser ──────────────────────────
        self._waveform_data: dict = {}
        # Keeps the per-canvas TPyDispatcher objects (Closed() → null the ref) alive.
        self._canvas_dispatchers: dict = {}
        self._root_canvas  = None    # ROOT TCanvas (kept alive between events)
        self._root_objects: list = []  # TGraph/TLegend objects (prevent Python GC)
        self._charge_canvas  = None   # ROOT TCanvas for charge integrals
        self._charge_objects: list = []
        self._track_data:   dict = {}  # {dist_label: {xpos_label: data_dict}}
        self._track_geom:   dict | None = None
        self._tracks_canvas  = None   # ROOT TCanvas for 3D tracks
        self._tracks_objects: list = []
        self._trk_legend_objects: list = []   # TLegend + proxy objects (prevent Python GC)
        self._trk_zoom_scale: float = 1.0   # <1 zoomed in, >1 zoomed out
        self._trk_view_phi:   float = 32.0  # TPad azimuthal angle (32° avoids Y-label inside box)
        self._trk_view_theta: float = 30.0  # TPad elevation angle (ROOT default)
        self._trk_pan_x:      float = 0.0   # cm offset of X visible centre
        self._trk_pan_y:      float = 0.0   # cm offset of Y visible centre
        self._trk_pan_z:      float = 0.0   # cm offset of Z visible centre
        self._trk_n_holes:    int   = 4     # N×N block of holes drawn in the 3D view
        self._trk_n_aval_paths: int = 50    # avalanche e⁻ drift lines drawn (0 = hide)
        self._map_n_holes:    int   = 4     # holes tiled across the E/W field maps in x
        self._efield_cache: dict | None = None   # {x, y, Ex, Ey} computed arrays
        self._efield_root_canvas  = None   # ROOT TCanvas for E-field maps
        self._efield_objects: list = []
        self._wfield_cache: dict | None = None   # {x, y, W, Wx, Wy, electrode} arrays
        self._wfield_root_canvas  = None   # ROOT TCanvas for weighting-field maps
        self._wfield_objects: list = []
        self._gas_canvas  = None   # ROOT TCanvas for Magboltz gas properties
        self._gas_objects: list = []
        self._garfield = None              # cached ROOT.Garfield once libs are loaded
        self._garfield_failed = False      # don't retry the load after a failure
        self.config_panel = None           # set by MainWindow; live readout settings
        self._amp_mode_available = False

        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 4, 0)
        mode_layout.setSpacing(6)
        mode_layout.addWidget(QLabel("Display:"))
        self.signal_mode_combo = QComboBox()
        self.signal_mode_combo.addItem("Amplifier", "amp")
        self.signal_mode_combo.addItem("Raw", "raw")
        self.signal_mode_combo.setToolTip(
            "Shared display mode for the Waveforms and Integrals tabs.\n"
            "Amplifier mode plots the front-end output branches when they are available;\n"
            "Raw mode shows the Garfield induced-current waveforms."
        )
        mode_layout.addWidget(self.signal_mode_combo)
        self.setCornerWidget(mode_widget, Qt.TopRightCorner)
        self._signal_mode_widget = mode_widget
        self._signal_mode_widget.hide()
        self.signal_mode_combo.currentIndexChanged.connect(self._on_signal_mode_changed)

        wave_widget = QWidget()
        wave_layout = QVBoxLayout(wave_widget)
        wave_layout.setContentsMargins(8, 6, 8, 6)
        wave_layout.setSpacing(6)

        # — selector row —
        sel_row = QWidget()
        sel_h   = QHBoxLayout(sel_row)
        sel_h.setContentsMargins(0, 0, 0, 0)
        sel_h.addWidget(QLabel("Distance:"))
        self.wave_dist_combo = QComboBox()
        sel_h.addWidget(self.wave_dist_combo)
        sel_h.addSpacing(8)
        sel_h.addWidget(QLabel("X pos:"))
        self.wave_xpos_combo = QComboBox()
        sel_h.addWidget(self.wave_xpos_combo)
        sel_h.addSpacing(16)
        sel_h.addWidget(QLabel("Event:"))
        self.wave_event_slider = QSlider(Qt.Horizontal)
        self.wave_event_slider.setMinimum(0)
        self.wave_event_slider.setMaximum(0)
        self.wave_event_slider.setSingleStep(1)
        sel_h.addWidget(self.wave_event_slider)
        self.wave_event_label = QLabel("— / —")
        self.wave_event_label.setMinimumWidth(55)
        sel_h.addWidget(self.wave_event_label)
        sel_h.addSpacing(8)
        self.wave_split_cb = QCheckBox("e⁻/ion components")
        self.wave_split_cb.setToolTip(
            "Overlay the electron and ion contributions to the induced current\n"
            "(requires a ROOT file produced with the component-split branches).")
        sel_h.addWidget(self.wave_split_cb)
        wave_layout.addWidget(sel_row)

        # — per-event info —
        info_box  = QGroupBox("Current event")
        info_form = QFormLayout(info_box)
        info_form.setVerticalSpacing(2)
        self.wave_qa_title_lbl = QLabel("Q anode [fC]:")
        self.wave_qc_title_lbl = QLabel("Q top [fC]:")
        self.wave_ratio_title_lbl = QLabel("Q ratio:")
        self.wave_qa_lbl    = QLabel("—")
        self.wave_qc_lbl    = QLabel("—")
        self.wave_ratio_lbl = QLabel("—")
        info_form.addRow(self.wave_qa_title_lbl,   self.wave_qa_lbl)
        info_form.addRow(self.wave_qc_title_lbl,   self.wave_qc_lbl)
        info_form.addRow(self.wave_ratio_title_lbl, self.wave_ratio_lbl)
        wave_layout.addWidget(info_box)

        # — hint —
        hint = QLabel(
            "ROOT canvas opens automatically when results are loaded.\n"
            "Right-click inside the ROOT window to zoom, change axes, or save."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey; font-size: 11px;")
        wave_layout.addWidget(hint)
        wave_layout.addStretch()

        self.wave_dist_combo.currentIndexChanged.connect(self._on_wave_dist_changed)
        self.wave_xpos_combo.currentIndexChanged.connect(self._on_wave_xpos_changed)
        self.wave_event_slider.valueChanged.connect(self._update_waveform_plot)
        self.wave_split_cb.toggled.connect(self._update_waveform_plot)

        self.addTab(wave_widget, "Waveforms")

        # ── Integrals tab: cumulative integral of the displayed waveforms ────
        charge_widget  = QWidget()
        charge_layout  = QVBoxLayout(charge_widget)
        charge_layout.setContentsMargins(8, 6, 8, 6)
        charge_layout.setSpacing(6)

        # — selector row —
        ch_sel_row = QWidget()
        ch_sel_h   = QHBoxLayout(ch_sel_row)
        ch_sel_h.setContentsMargins(0, 0, 0, 0)
        ch_sel_h.addWidget(QLabel("Distance:"))
        self.charge_dist_combo = QComboBox()
        ch_sel_h.addWidget(self.charge_dist_combo)
        ch_sel_h.addSpacing(8)
        ch_sel_h.addWidget(QLabel("X pos:"))
        self.charge_xpos_combo = QComboBox()
        ch_sel_h.addWidget(self.charge_xpos_combo)
        ch_sel_h.addSpacing(16)
        ch_sel_h.addWidget(QLabel("Event:"))
        self.charge_event_slider = QSlider(Qt.Horizontal)
        self.charge_event_slider.setMinimum(0)
        self.charge_event_slider.setMaximum(0)
        self.charge_event_slider.setSingleStep(1)
        ch_sel_h.addWidget(self.charge_event_slider)
        self.charge_event_label = QLabel("— / —")
        self.charge_event_label.setMinimumWidth(55)
        ch_sel_h.addWidget(self.charge_event_label)
        charge_layout.addWidget(ch_sel_row)

        # — hint —
        ch_hint = QLabel(
            "ROOT canvas opens automatically when results are loaded.\n"
            "Right-click inside the ROOT window to zoom, change axes, or save."
        )
        ch_hint.setWordWrap(True)
        ch_hint.setStyleSheet("color: grey; font-size: 11px;")
        charge_layout.addWidget(ch_hint)
        charge_layout.addStretch()

        self.charge_dist_combo.currentIndexChanged.connect(self._on_charge_dist_changed)
        self.charge_xpos_combo.currentIndexChanged.connect(self._on_charge_xpos_changed)
        self.charge_event_slider.valueChanged.connect(self._update_charge_plot)

        self.addTab(charge_widget, "Integrals")

        # ── 3D Tracks tab ─────────────────────────────────────────────────────
        tracks_widget = QWidget()
        tracks_layout = QVBoxLayout(tracks_widget)
        tracks_layout.setContentsMargins(8, 6, 8, 6)
        tracks_layout.setSpacing(6)

        # — selector row —
        trk_sel_row = QWidget()
        trk_sel_h   = QHBoxLayout(trk_sel_row)
        trk_sel_h.setContentsMargins(0, 0, 0, 0)
        trk_sel_h.addWidget(QLabel("Distance:"))
        self.trk_dist_combo = QComboBox()
        trk_sel_h.addWidget(self.trk_dist_combo)
        trk_sel_h.addSpacing(12)
        trk_sel_h.addWidget(QLabel("X pos:"))
        self.trk_xpos_combo = QComboBox()
        trk_sel_h.addWidget(self.trk_xpos_combo)
        trk_sel_h.addSpacing(12)
        trk_sel_h.addWidget(QLabel("Event:"))
        self.trk_event_slider = QSlider(Qt.Horizontal)
        self.trk_event_slider.setMinimum(0)
        self.trk_event_slider.setMaximum(0)
        self.trk_event_slider.setSingleStep(1)
        trk_sel_h.addWidget(self.trk_event_slider)
        self.trk_event_label = QLabel("— / —")
        self.trk_event_label.setMinimumWidth(55)
        trk_sel_h.addWidget(self.trk_event_label)
        rel_hint = QLabel("(release slider to update)")
        rel_hint.setStyleSheet("color: grey; font-size: 10px;")
        trk_sel_h.addWidget(rel_hint)
        trk_sel_h.addStretch()
        tracks_layout.addWidget(trk_sel_row)

        # — view controls (preset orientations + zoom) —
        trk_ctrl_row = QWidget()
        trk_ctrl_h   = QHBoxLayout(trk_ctrl_row)
        trk_ctrl_h.setContentsMargins(0, 0, 0, 0)

        for _label, _phi, _theta, _rz in [
            ("Gap XY",   0,  90, False),  # theta=90 → camera along Z → sees X-Y
            ("Top XZ",   0,   0, False),  # phi=0   → top down (along Y) → sees X-Z
            ("Side YZ",  90,  0, False),  # phi=90  → camera along X → sees Y-Z
            ("3D",      32,  30, True),   # default perspective + reset zoom
        ]:
            _btn = QPushButton(_label)
            _btn.setMaximumWidth(72)
            _btn.clicked.connect(
                (lambda p, t, rz: lambda: self._trk_preset_view(p, t, rz))
                (_phi, _theta, _rz))
            trk_ctrl_h.addWidget(_btn)

        trk_ctrl_h.addSpacing(16)
        trk_zoom_in_btn  = QPushButton("Zoom +")
        trk_zoom_out_btn = QPushButton("Zoom −")
        trk_zoom_in_btn.setMaximumWidth(65)
        trk_zoom_out_btn.setMaximumWidth(65)
        trk_zoom_in_btn.clicked.connect(lambda: self._trk_adjust_zoom(0.7))
        trk_zoom_out_btn.clicked.connect(lambda: self._trk_adjust_zoom(1.0 / 0.7))
        trk_ctrl_h.addWidget(trk_zoom_in_btn)
        trk_ctrl_h.addWidget(trk_zoom_out_btn)

        trk_ctrl_h.addSpacing(16)
        trk_ctrl_h.addWidget(QLabel("Holes:"))
        self.trk_holes_spin = QSpinBox()
        self.trk_holes_spin.setRange(1, 15)
        self.trk_holes_spin.setValue(self._trk_n_holes)
        self.trk_holes_spin.setMaximumWidth(52)
        self.trk_holes_spin.setToolTip(
            "Number of holes drawn per side (N×N) in the 3D view.\n"
            "The simulated hole sits at the centre; this is display-only.")
        self.trk_holes_spin.valueChanged.connect(self._on_trk_holes_changed)
        trk_ctrl_h.addWidget(self.trk_holes_spin)

        trk_ctrl_h.addSpacing(10)
        trk_ctrl_h.addWidget(QLabel("Aval paths:"))
        self.trk_aval_spin = QSpinBox()
        self.trk_aval_spin.setRange(0, 200)
        self.trk_aval_spin.setValue(self._trk_n_aval_paths)
        self.trk_aval_spin.setMaximumWidth(58)
        self.trk_aval_spin.setToolTip(
            "Number of avalanche-electron drift lines drawn in the 3D view (0 hides them).\n"
            "Requires a run with 'Store drift lines' enabled.")
        self.trk_aval_spin.valueChanged.connect(self._on_trk_n_aval_changed)
        trk_ctrl_h.addWidget(self.trk_aval_spin)

        trk_ctrl_h.addStretch()
        tracks_layout.addWidget(trk_ctrl_row)

        # — pan controls —
        trk_pan_row = QWidget()
        trk_pan_h   = QHBoxLayout(trk_pan_row)
        trk_pan_h.setContentsMargins(0, 0, 0, 0)
        trk_pan_h.addWidget(QLabel("Pan:"))
        for _ax, _d, _lbl in [
            ("x", -1, "X-"), ("x", +1, "X+"),
            ("y", -1, "Y-"), ("y", +1, "Y+"),
            ("z", -1, "Z-"), ("z", +1, "Z+"),
        ]:
            _btn = QPushButton(_lbl)
            _btn.setMaximumWidth(42)
            _btn.clicked.connect(
                (lambda a, d: lambda: self._trk_pan(a, d))(_ax, _d))
            trk_pan_h.addWidget(_btn)
        trk_pan_h.addStretch()
        tracks_layout.addWidget(trk_pan_row)

        trk_hint = QLabel(
            "ROOT canvas opens automatically when results are loaded.\n"
            "Left-click drag: rotate  ·  Zoom / Pan / View buttons above  ·  Right-click: save."
        )
        trk_hint.setWordWrap(True)
        trk_hint.setStyleSheet("color: grey; font-size: 11px;")
        tracks_layout.addWidget(trk_hint)
        tracks_layout.addStretch()

        self.trk_dist_combo.currentIndexChanged.connect(self._on_trk_dist_changed)
        self.trk_xpos_combo.currentIndexChanged.connect(self._on_trk_xpos_changed)
        self.trk_event_slider.sliderReleased.connect(self._update_track_plot)

        self.addTab(tracks_widget, "3D Tracks")

        # ── E-Field tab (neBEM field map, loaded from the run) ────────────
        efield_widget = QWidget()
        efield_layout = QVBoxLayout(efield_widget)
        efield_layout.setContentsMargins(8, 6, 8, 6)
        efield_layout.setSpacing(6)

        ef_ctrl = QWidget()
        ef_h = QHBoxLayout(ef_ctrl)
        ef_h.setContentsMargins(0, 0, 0, 0)
        ef_h.addWidget(QLabel("Quantity:"))
        self.ef_quantity = QComboBox()
        self.ef_quantity.addItems([
            "|E| [kV/cm]  (x–z)", "Potential [V]  (x–z)",
            "Ez [kV/cm]  (x–z)", "Ex [kV/cm]  (x–z)",
            "|E| [kV/cm]  (y–z, along a wire)", "Potential [V]  (y–z)"])
        self.ef_quantity.setToolTip(
            "The x–z slice cuts across the wires (they appear as points); the y–z slice "
            "runs along one wire. Unlike a single-plate cell the two are not equivalent — "
            "the wires break the x/y symmetry.")
        self.ef_quantity.currentIndexChanged.connect(self._redraw_field)
        ef_h.addWidget(self.ef_quantity)
        ef_h.addSpacing(12)
        ef_h.addWidget(QLabel("Palette:"))
        self.ef_cmap = QComboBox()
        self.ef_cmap.addItems(list(_ROOT_PALETTE))
        self.ef_cmap.currentIndexChanged.connect(self._redraw_field)
        ef_h.addWidget(self.ef_cmap)
        ef_h.addSpacing(12)
        ef_h.addWidget(QLabel("Cells:"))
        self.ef_holes = QSpinBox()
        self.ef_holes.setRange(1, 15)
        self.ef_holes.setValue(self._map_n_holes)
        self.ef_holes.setMaximumWidth(52)
        self.ef_holes.setToolTip(
            "Number of periodic cells shown across x. The field is periodic, so this "
            "tiles the one simulated cell — exact, not an approximation. A cell is "
            "holes_per_wire hole pitches wide and holds one wire.")
        self.ef_holes.valueChanged.connect(self._on_map_holes_changed)
        ef_h.addWidget(self.ef_holes)
        ef_h.addStretch()
        efield_layout.addWidget(ef_ctrl)

        ef_hint = QLabel(
            "The field is solved by neBEM in the simulation and dumped to the run ROOT "
            "file — run a simulation to populate this tab.\n"
            "The ROOT canvas opens automatically: right-click inside it to zoom, rescale "
            "the axes, or save. Left pad = the colour map with the stack overlaid (the "
            "THGEM's copper and hole walls, the mesh, the cathode wires, the anode); "
            "right pad = the on-hole-axis profile. Unless the mesh lattice lines up with "
            "the THGEM hole, that axis runs through mesh metal, so the profile also "
            "carries the mesh's own aperture axis.")
        ef_hint.setWordWrap(True)
        ef_hint.setStyleSheet("color: grey; font-size: 11px;")
        efield_layout.addWidget(ef_hint)
        efield_layout.addStretch()

        self._field_maps = None   # cached ROOT objects from load_field_maps
        self._field_geom  = None   # run_config.json "derived" block of the loaded run
        self._tab_efield = self.addTab(efield_widget, "E-Field")

        # ── Weighting Field tab (true neBEM Shockley–Ramo field per electrode) ──
        wfield_widget = QWidget()
        wfield_layout = QVBoxLayout(wfield_widget)
        wfield_layout.setContentsMargins(8, 6, 8, 6)
        wfield_layout.setSpacing(6)

        wf_ctrl = QWidget()
        wf_h = QHBoxLayout(wf_ctrl)
        wf_h.setContentsMargins(0, 0, 0, 0)
        wf_h.addWidget(QLabel("Electrode:"))
        self.wf_electrode = QComboBox()
        self.wf_electrode.addItems(["all electrodes"] + ELECTRODE_IDS)
        self.wf_electrode.currentIndexChanged.connect(self._redraw_wfield)
        wf_h.addWidget(self.wf_electrode)
        wf_h.addSpacing(12)
        wf_h.addWidget(QLabel("Quantity:"))
        self.wf_quantity = QComboBox()
        self.wf_quantity.addItems(["W (potential)", "|E_w| [1/cm]"])
        self.wf_quantity.currentIndexChanged.connect(self._redraw_wfield)
        wf_h.addWidget(self.wf_quantity)
        wf_h.addSpacing(12)
        wf_h.addWidget(QLabel("Palette:"))
        self.wf_cmap = QComboBox()
        self.wf_cmap.addItems(list(_ROOT_PALETTE))
        self.wf_cmap.currentIndexChanged.connect(self._redraw_wfield)
        wf_h.addWidget(self.wf_cmap)
        wf_h.addSpacing(12)
        wf_h.addWidget(QLabel("Cells:"))
        self.wf_holes = QSpinBox()
        self.wf_holes.setRange(1, 15)
        self.wf_holes.setValue(self._map_n_holes)
        self.wf_holes.setMaximumWidth(52)
        self.wf_holes.setToolTip(
            "Number of periodic cells shown across x. The weighting field is periodic, "
            "so this tiles the one simulated cell — exact, not an approximation.")
        self.wf_holes.valueChanged.connect(self._on_map_holes_changed)
        wf_h.addWidget(self.wf_holes)
        wf_h.addStretch()
        wfield_layout.addWidget(wf_ctrl)

        wf_hint = QLabel(
            "Exact Shockley–Ramo weighting field of each read-out electrode, solved by "
            "neBEM (1 V on it, 0 V on all the others) and cached per geometry — it does "
            "not depend on the applied voltages, so a ΔV scan reuses one solve. These are "
            "the same maps the induced signals are integrated from.\n"
            "Only the electrodes this run read out have a map. \"all electrodes\" overlays "
            "their on-axis W(z): each should peak at its own position in the stack, which "
            "is the direct check that the signals are not swapped. Right-click the ROOT "
            "canvas to zoom.")
        wf_hint.setWordWrap(True)
        wf_hint.setStyleSheet("color: grey; font-size: 11px;")
        wfield_layout.addWidget(wf_hint)
        wfield_layout.addStretch()

        self._weighting_maps = None   # cached ROOT objects from load_weighting_maps
        self.addTab(wfield_widget, "Weighting Field")


        gas_widget = QWidget()
        gas_layout = QVBoxLayout(gas_widget)
        gas_layout.setContentsMargins(8, 6, 8, 6)
        gas_layout.setSpacing(6)

        exp_row = QWidget()
        exp_h = QHBoxLayout(exp_row)
        exp_h.setContentsMargins(0, 0, 0, 0)
        self.gas_export_root_btn = QPushButton("Export as ROOT …")
        self.gas_export_csv_btn  = QPushButton("Export as CSV …")
        self.gas_export_root_btn.setEnabled(False)
        self.gas_export_csv_btn.setEnabled(False)
        exp_h.addWidget(self.gas_export_root_btn)
        exp_h.addWidget(self.gas_export_csv_btn)
        exp_h.addStretch()
        gas_layout.addWidget(exp_row)

        gas_hint = QLabel(
            "ROOT canvas opens automatically when gas properties are available.\n"
            "Right-click inside the ROOT window to zoom, change axes, or save."
        )
        gas_hint.setWordWrap(True)
        gas_hint.setStyleSheet("color: grey; font-size: 11px;")
        gas_layout.addWidget(gas_hint)
        gas_layout.addStretch()

        self.gas_export_root_btn.clicked.connect(self._on_gas_export_root)
        self.gas_export_csv_btn.clicked.connect(self._on_gas_export_csv)

        self.addTab(gas_widget, "Magboltz")

        # — timer to keep ROOT canvas responsive —
        self._root_timer = QTimer(self)
        self._root_timer.setInterval(100)
        self._root_timer.timeout.connect(self._process_root_events)

    # ── Log ───────────────────────────────────────────────────────────────

    def append_log(self, line: str):
        self.log.appendPlainText(line)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.log.clear()

    # ── Summary table ─────────────────────────────────────────────────────

    def populate_table(self, csv_path: str):
        self._summary_csv_path = csv_path
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Could not read CSV: {exc}")
            return

        df.columns = [c.strip() for c in df.columns]
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(list(df.columns))

        for r_idx, row in df.iterrows():
            for c_idx, val in enumerate(row):
                if isinstance(val, float) and not pd.isna(val):
                    text = f"{val:.4g}"
                elif isinstance(val, float):  # NaN = random x-position
                    text = "—"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r_idx, c_idx, item)

        self.table.resizeColumnsToContents()
        self.summary_export_btn.setEnabled(True)

    # ── Plots ─────────────────────────────────────────────────────────────

    def draw_plots(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            return

        df.columns = [c.strip() for c in df.columns]
        axes = self.plots_canvas.axes
        for ax in axes:
            ax.cla()

        x = df["source_distance_mm"].to_numpy()

        def _errplot(ax, y_col: str, yerr_col, ylabel: str, title: str):
            if y_col not in df.columns:
                ax.set_title(title + " (no data)")
                ax.axis("off")
                return
            y  = df[y_col].to_numpy()
            ye = df[yerr_col].to_numpy() if (yerr_col and yerr_col in df.columns) else None
            ax.errorbar(x, y, yerr=ye, fmt="o-", capsize=4, lw=1.5, ms=5)
            ax.set_xlabel("Source distance [mm]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Which electrodes the run actually read out is a config choice, so the
        # panels follow the CSV's own columns rather than a fixed list.
        panels = [(f"mean_{e}_charge_fC", f"sem_{e}_charge_fC",
                   f"⟨Q⟩ [fC]", f"{ELECTRODE_LABELS[e]} charge")
                  for e in ELECTRODE_IDS
                  if f"mean_{e}_charge_fC" in df.columns]
        # The cascade diagnostics are what a two-stage stack is actually tuned on.
        panels += [
            ("mean_avalanche_size", "sem_avalanche_size",
             "⟨N_e,total⟩", "Total avalanche size"),
            ("mean_gain_thgem",        None, "⟨G_THGEM⟩", "THGEM gain"),
            ("mean_gain_amp",          None, "⟨G_amp⟩", "Amplification-gap gain"),
            ("mean_mesh_transparency", None, "⟨ε_mesh⟩", "Mesh electron transparency"),
        ]

        # Lay out enough panels for this run's electrode set, at most 4 per row.
        ncols = min(4, max(1, len(panels)))
        nrows = (len(panels) + ncols - 1) // ncols
        self.plots_canvas.set_grid(nrows, ncols)
        axes = self.plots_canvas.axes
        for ax in axes:
            ax.cla()

        for ax, (col, ecol, ylab, title) in zip(axes, panels):
            _errplot(ax, col, ecol, ylab, title)
        for ax in axes[len(panels):]:
            ax.axis("off")

        self.plots_canvas.figure.tight_layout()
        self.plots_canvas.draw()

    def draw_gas_props(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path, comment='#')
        except Exception:  # noqa: BLE001
            return
        df.columns = [c.strip() for c in df.columns]
        self._gas_props_csv = csv_path

        try:
            import ROOT  # noqa: PLC0415
            ROOT.gROOT.SetBatch(False)

            self._gas_canvas = self._ensure_canvas(
                "_gas_canvas", "thgem_mesh_magboltz", "Magboltz Gas Properties", 1200, 900)
            self._gas_canvas.Clear()
            self._gas_canvas.Divide(3, 3)
            self._gas_objects.clear()
            ROOT.TGaxis.SetMaxDigits(0)   # force scientific notation for all values < 1

            e = (df["e_field_Vcm"] / 1000.0).to_numpy().astype("f8")   # kV/cm
            n = len(e)

            # Derive ion species and carrier gas from the "# ion_mobility: <file>"
            # comment written by ExportGasProps at the top of the props CSV.
            # File naming convention: IonMobility_<ION>_<GAS>.txt
            _ion_label, _gas_suffix = "ion", ""
            try:
                with open(csv_path) as _f:
                    _first = _f.readline().strip()
                    if _first.startswith("# ion_mobility:"):
                        _stem = Path(_first.split(":", 1)[1].strip()).stem  # "IonMobility_CO2+_CO2"
                        _parts = _stem[len("IonMobility_"):].split("_", 1)  # ["CO2+", "CO2"]
                        if len(_parts) == 2:
                            _ion_label  = _parts[0]            # e.g. "CO2+"
                            _gas_suffix = " in " + _parts[1]   # e.g. " in CO2"
                    else:
                        # Old CSV (no comment) — binary always uses IonMobility_CO2+_CO2.txt
                        _default = (GARFIELD_INSTALL / "share" / "Garfield"
                                    / "Data" / "IonMobility_CO2+_CO2.txt")
                        if _default.exists():
                            _stem = _default.stem  # "IonMobility_CO2+_CO2"
                            _parts = _stem[len("IonMobility_"):].split("_", 1)
                            if len(_parts) == 2:
                                _ion_label  = _parts[0]
                                _gas_suffix = " in " + _parts[1]
            except Exception:  # noqa: BLE001
                pass
            _ion_title_v  = f"{_ion_label} drift velocity{_gas_suffix}"
            _ion_title_mu = f"{_ion_label} mobility{_gas_suffix}"

            panels = [
                # (pad, col,               take_abs, logy, title, xlabel, ylabel)
                (1, "vd_cm_per_us",     True,  False,
                 "Electron drift velocity",
                 "E [kV/cm]", "|v_{d}| [cm/#mus]"),
                (2, "alpha_per_cm",     False, True,
                 "Townsend #alpha",
                 "E [kV/cm]", "#alpha [cm^{-1}]"),
                (3, "eta_per_cm",       False, True,
                 "Attachment #eta",
                 "E [kV/cm]", "#eta [cm^{-1}]"),
                (4, "dl_sqrtcm",        False, False,
                 "Long. diffusion D_{L}",
                 "E [kV/cm]", "D_{L} [cm^{0.5}]"),
                (5, "dt_sqrtcm",        False, False,
                 "Trans. diffusion D_{T}",
                 "E [kV/cm]", "D_{T} [cm^{0.5}]"),
                (6, None,               False, True,
                 "Effective gain (#alpha-#eta)",
                 "E [kV/cm]", "(#alpha-#eta) [cm^{-1}]"),
                (7, "v_ion_cm_per_us",  True,  False,
                 _ion_title_v,
                 "E [kV/cm]", "|v_{ion}| [cm/#mus]"),
                (8, "mu_ion_cm2_per_Vus", False, False,
                 _ion_title_mu,
                 "E [kV/cm]", "#mu [cm^{2}/(V#upoint#mus)]"),
            ]
            for pad_num, col, take_abs, logy, title, xlabel, ylabel in panels:
                self._gas_canvas.cd(pad_num)
                ROOT.gPad.SetLeftMargin(0.18)
                ROOT.gPad.SetBottomMargin(0.16)
                ROOT.gPad.SetRightMargin(0.04)
                ROOT.gPad.SetTopMargin(0.08)
                ROOT.gPad.SetLogx()
                ROOT.gPad.SetGrid()
                if logy:
                    ROOT.gPad.SetLogy()

                if col is None:
                    # effective gain = alpha - eta
                    if "alpha_per_cm" not in df.columns or "eta_per_cm" not in df.columns:
                        continue
                    yraw = (df["alpha_per_cm"] - df["eta_per_cm"]).to_numpy()
                else:
                    if col not in df.columns:
                        continue
                    yraw = df[col].to_numpy()
                    if take_abs:
                        yraw = np.abs(yraw)

                mask = (yraw > 0) if logy else np.ones(n, dtype=bool)
                xe = e[mask].astype("f8")
                ye = yraw[mask].astype("f8")
                if len(xe) == 0:
                    continue

                g = ROOT.TGraph(len(xe), xe, ye)
                g.SetTitle(f"{title};{xlabel};{ylabel}")
                g.SetLineColor(ROOT.kBlue + 1)
                g.SetMarkerColor(ROOT.kBlue + 1)
                g.SetMarkerStyle(20)
                g.SetMarkerSize(0.5)
                g.SetLineWidth(2)
                g.Draw("ALP")
                for _axis in (g.GetXaxis(), g.GetYaxis()):
                    _axis.SetTitleSize(_axis.GetTitleSize() + 0.03)
                    _axis.SetLabelSize(_axis.GetLabelSize() + 0.03)
                self._gas_objects.append(g)

            self._gas_canvas.cd(9)   # leave pad 9 empty

            self._gas_canvas.Update()
            ROOT.TGaxis.SetMaxDigits(5)   # restore ROOT default so other tabs are unaffected
            self._root_timer.start()

            self.gas_export_root_btn.setEnabled(True)
            self.gas_export_csv_btn.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Magboltz ROOT canvas error: {exc}")

    def _on_gas_export_root(self):
        if not self._gas_props_csv:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Magboltz as ROOT", str(PROJ_DIR), "ROOT files (*.root)")
        if not path:
            return
        try:
            import ROOT  # noqa: PLC0415
            df = pd.read_csv(self._gas_props_csv, comment='#')
            df.columns = [c.strip() for c in df.columns]
            e = (df["e_field_Vcm"] / 1000.0).to_numpy().astype("f8")
            f = ROOT.TFile(path, "RECREATE")
            spec = [
                ("vd",    "vd_cm_per_us",       True,
                 "Electron drift velocity;E [kV/cm];|v_{d}| [cm/#mus]"),
                ("alpha", "alpha_per_cm",        False,
                 "Townsend #alpha;E [kV/cm];#alpha [cm^{-1}]"),
                ("eta",   "eta_per_cm",          False,
                 "Attachment #eta;E [kV/cm];#eta [cm^{-1}]"),
                ("dl",    "dl_sqrtcm",           False,
                 "Long. diffusion;E [kV/cm];D_{L} [cm^{0.5}]"),
                ("dt",    "dt_sqrtcm",           False,
                 "Trans. diffusion;E [kV/cm];D_{T} [cm^{0.5}]"),
                ("v_ion", "v_ion_cm_per_us",     True,
                 "Ion drift velocity;E [kV/cm];|v_{ion}| [cm/#mus]"),
                ("mu",    "mu_ion_cm2_per_Vus",  False,
                 "Ion mobility;E [kV/cm];#mu [cm^{2}/(V#upoint#mus)]"),
            ]
            for gname, col, take_abs, title in spec:
                if col not in df.columns:
                    continue
                y = df[col].to_numpy().astype("f8")
                if take_abs:
                    y = np.abs(y)
                g = ROOT.TGraph(len(e), e, y)
                g.SetTitle(title)
                g.SetName(gname)
                g.Write()
            if "alpha_per_cm" in df.columns and "eta_per_cm" in df.columns:
                eff = (df["alpha_per_cm"] - df["eta_per_cm"]).to_numpy().astype("f8")
                g = ROOT.TGraph(len(e), e, eff)
                g.SetTitle("Effective gain;E [kV/cm];(#alpha-#eta) [cm^{-1}]")
                g.SetName("eff_gain")
                g.Write()
            f.Close()
            self.append_log(f"[GUI] Magboltz data exported to {path}")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] ROOT export failed: {exc}")
            QMessageBox.warning(self.parent(), "Export failed", str(exc))

    def _on_gas_export_csv(self):
        if not self._gas_props_csv or not Path(self._gas_props_csv).exists():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Magboltz as CSV", str(PROJ_DIR), "CSV files (*.csv)")
        if not path:
            return
        import shutil
        shutil.copy2(self._gas_props_csv, path)
        self.append_log(f"[GUI] Magboltz CSV exported to {path}")

    def _on_summary_export_csv(self) -> None:
        csv_path = getattr(self, "_summary_csv_path", None)
        if not csv_path or not Path(csv_path).exists():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export summary as CSV", "summary.csv", "CSV files (*.csv)")
        if not path:
            return
        import shutil
        shutil.copy2(csv_path, path)
        self.append_log(f"[GUI] Summary CSV exported to {path}")

    def _save_plots_root(self, run_dir) -> None:
        """Write all currently-rendered ROOT canvases to thgem_mesh_plots.root in run_dir."""
        try:
            import ROOT  # noqa: PLC0415
            out = Path(run_dir) / "thgem_mesh_plots.root"
            tf = ROOT.TFile.Open(str(out), "RECREATE")
            if not tf or tf.IsZombie():
                self.append_log("[GUI] Warning: could not create thgem_mesh_plots.root")
                return
            saved = []
            for canvas, key in [
                (self._root_canvas,        "waveforms"),
                (self._charge_canvas,      "charge"),
                (self._tracks_canvas,      "tracks_3d"),
                (self._gas_canvas,         "magboltz"),
                (self._efield_root_canvas, "efield"),
                (self._wfield_root_canvas, "wfield"),
            ]:
                if canvas is None:        # closed canvases have a None ref
                    continue
                tf.cd()
                canvas.Write(key)
                saved.append(key)
            tf.Close()
            if saved:
                self.append_log(
                    f"[GUI] Plots saved → {out.name}  ({', '.join(saved)})")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Could not save plots ROOT file: {exc}")

    # ── Waveforms ─────────────────────────────────────────────────────────

    def _ensure_canvas(self, attr, name, title, w, h):
        """Return a live ROOT TCanvas stored on ``self.<attr>``, creating it if needed.

        Robust to the user closing the window: each canvas's ``Closed()`` signal nulls
        ``self.<attr>`` (via a kept-alive TPyDispatcher), and pending close events are
        flushed first, so a stale dead-window canvas is never reused — which on macOS
        otherwise draws to a freed Cocoa drawable and hard-crashes the process.  The
        canvas list itself is unreliable here: a closed canvas lingers in
        ``GetListOfCanvases()`` even though its native window is gone.
        """
        import ROOT  # noqa: PLC0415
        ROOT.gSystem.ProcessEvents()          # deliver any pending window-close → Closed()
        canvas = getattr(self, attr)
        if canvas is not None:
            return canvas                     # still open (a close would have nulled it)
        canvas = ROOT.TCanvas(name, title, w, h)
        disp = ROOT.TPyDispatcher(lambda a=attr: setattr(self, a, None))
        canvas.Connect("Closed()", "TPyDispatcher", disp, "Dispatch()")
        self._canvas_dispatchers[attr] = disp  # keep the dispatcher alive
        setattr(self, attr, canvas)
        return canvas

    def _process_root_events(self):
        """Keep the ROOT TCanvas window responsive while Qt runs."""
        try:
            import ROOT  # noqa: PLC0415
            ROOT.gSystem.ProcessEvents()
        except Exception:  # noqa: BLE001
            pass

    def load_waveform_data(self, root_path: str):
        """Load per-event TTree waveforms (and mean TProfiles) from the ROOT file.

        Which electrodes were read out is a per-run config choice, so the series
        are stored under their electrode ids rather than in fixed anode/cathode
        slots, and every plot downstream iterates over that list.
        """
        try:
            import uproot  # noqa: PLC0415
        except ImportError:
            self.append_log("[GUI] uproot not available — Waveforms/Integrals tabs will be empty")
            return

        if not Path(root_path).exists():
            self.append_log(f"[GUI] ROOT file not found: {root_path}")
            return

        # Force canvas recreation for the new run
        self._root_canvas   = None
        self._charge_canvas = None
        self._warned_no_split = False

        self._waveform_data.clear()
        self._set_signal_mode_available(False)
        self.wave_dist_combo.blockSignals(True)
        self.wave_dist_combo.clear()
        self.wave_xpos_combo.blockSignals(True)
        self.wave_xpos_combo.clear()
        self.charge_dist_combo.blockSignals(True)
        self.charge_dist_combo.clear()
        self.charge_xpos_combo.blockSignals(True)
        self.charge_xpos_combo.clear()

        run_config_amp = False
        run_cfg_electrodes = None
        run_config_path = Path(root_path).with_name("run_config.json")
        if run_config_path.exists():
            try:
                with run_config_path.open("r", encoding="utf-8") as fh:
                    run_cfg = json.load(fh)
                run_config_amp = bool(run_cfg.get("amplifier", {}).get("enable", False))
                run_cfg_electrodes = (run_cfg.get("readout") or {}).get("electrodes")
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[GUI] Could not read {run_config_path.name}: {exc}")

        try:
            amp_available_any = False
            with uproot.open(root_path) as f:
                dist_keys = sorted({
                    k.split("/")[0] for k in f.keys(cycle=False)
                    if k.split("/")[0].startswith("dist_")
                })
                for key in dist_keys:
                    rest = key.removeprefix("dist_")
                    dist_raw, sep, xpos_raw = rest.partition("_x")
                    dist_label = "—" if dist_raw == "rnd" else dist_raw.replace("p", ".").replace("mm", " mm")
                    xpos_label = xpos_raw.replace("p", ".").replace("mm", " mm") if sep else "—"
                    try:
                        tree = f[f"{key}/t_signals"]
                        branches = set(tree.keys())
                        # Trust the run config when it is there, but fall back to
                        # whatever branches the file actually carries.
                        ids = [e for e in (run_cfg_electrodes or ELECTRODE_IDS)
                               if e in branches]
                        if not ids:
                            self.append_log(f"[GUI] Waveforms: no electrode branches in {key}")
                            continue

                        times = f[f"{key}/p_{ids[0]}_signal"].axis(0).centers()
                        elec, amp_here = {}, False
                        for eid in ids:
                            # std::vector<float> branches → object array of 1D arrays;
                            # np.stack() converts to a proper (n_evt, nBins) 2D array.
                            ch = {"sig":  np.stack(tree[eid].array(library="np")),
                                  "mean": f[f"{key}/p_{eid}_signal"].values()}
                            for suffix in ("e", "i"):
                                br = f"{eid}_{suffix}"
                                if br in branches:
                                    ch[suffix] = np.stack(tree[br].array(library="np"))
                            if f"{eid}_amp" in branches:
                                ch["amp"] = np.stack(tree[f"{eid}_amp"].array(library="np"))
                                ch["mean_amp"] = f[f"{key}/p_{eid}_amp"].values()
                                amp_here = amp_here or self._array_has_signal(ch["mean_amp"])
                            elec[eid] = ch

                        data = {
                            "times": times,
                            "ids":   ids,
                            "elec":  elec,
                            "amp_available": run_config_amp or amp_here,
                        }
                        # Cascade diagnostics, shown alongside the waveform metrics.
                        for br in ("gain_thgem", "gain_amp", "mesh_transparency"):
                            if br in branches:
                                data[br] = tree[br].array(library="np")
                        amp_available_any = amp_available_any or data["amp_available"]
                        self._waveform_data.setdefault(dist_label, {})[xpos_label] = data
                    except Exception as exc:  # noqa: BLE001
                        self.append_log(f"[GUI] Waveforms: could not read {key}: {exc}")
                # Populate dist combos in sorted order (random "—" first)
                for dl in self._sorted_dists(self._waveform_data.keys()):
                    self.wave_dist_combo.addItem(dl)
                    self.charge_dist_combo.addItem(dl)
                self._set_signal_mode_available(amp_available_any)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Could not open ROOT file: {exc}")

        self.wave_dist_combo.blockSignals(False)
        self.wave_xpos_combo.blockSignals(False)
        self.charge_dist_combo.blockSignals(False)
        self.charge_xpos_combo.blockSignals(False)
        # Triggering wave syncs charge combos + slider; then explicitly draw charge canvas
        if self.wave_dist_combo.count():
            self._on_wave_dist_changed(0)   # syncs charge combos + creates wave canvas
            self._update_charge_plot()      # create charge canvas on initial load
            self._root_timer.start()

    # ── Waveform/Charge sync helpers ──────────────────────────────────────────

    @staticmethod
    def _sorted_xpos(xpos_dict: dict) -> list:
        """Return x-position labels sorted numerically; '—' (random) sorts first."""
        def _key(s):
            try:
                return float(s.replace(" mm", ""))
            except ValueError:
                return -1e9
        return sorted(xpos_dict.keys(), key=_key)

    @staticmethod
    def _sorted_dists(dists_iterable) -> list:
        """Return distance labels sorted numerically; '—' (random) sorts first."""
        def _key(s):
            try:
                return float(s.replace(" mm", ""))
            except ValueError:
                return -1e9
        return sorted(dists_iterable, key=_key)

    @staticmethod
    def _array_has_signal(arr) -> bool:
        return bool(np.any(np.abs(np.asarray(arr, dtype="f8")) > 0.0))

    def _set_signal_mode_available(self, amp_available: bool):
        self._amp_mode_available = amp_available
        self.signal_mode_combo.blockSignals(True)
        self.signal_mode_combo.setCurrentIndex(0 if amp_available else 1)
        self.signal_mode_combo.blockSignals(False)
        self._signal_mode_widget.setVisible(amp_available)

    def _display_amp_mode(self, data: dict | None = None) -> bool:
        if not self._amp_mode_available or self.signal_mode_combo.currentData() != "amp":
            return False
        return True if data is None else bool(data.get("amp_available", False))

    @staticmethod
    def _display_label_meta(amp_mode: bool) -> dict:
        """Mode-specific labels for the Qt metrics and ROOT canvases."""
        if amp_mode:
            return {
                "wave_units": "V [mV]",
                "root_integral_units": "#int V dt [mV#upointns]",
                "metric_unit": "mV·ns",
            }
        return {
            "wave_units": "i [fC/ns]",
            "root_integral_units": "Q [fC]",
            "metric_unit": "fC",
        }

    def _select_display_series(self, data: dict, evt_idx: int) -> dict:
        """Per-electrode series for one event, in the selected display mode.

        Falls back to the raw current for any electrode without an amplifier
        branch, so a mixed file still plots rather than raising.
        """
        amp_mode = self._display_amp_mode(data)
        series = {}
        for eid in data["ids"]:
            ch = data["elec"][eid]
            use_amp = amp_mode and "amp" in ch
            series[eid] = {
                "evt":  (ch["amp"] if use_amp else ch["sig"])[evt_idx].astype("f8"),
                "mean": (ch["mean_amp"] if use_amp else ch["mean"]).astype("f8"),
                "e":    ch["e"][evt_idx].astype("f8") if "e" in ch else None,
                "i":    ch["i"][evt_idx].astype("f8") if "i" in ch else None,
            }
        return {"amp_mode": amp_mode, "ids": data["ids"], "series": series,
                **self._display_label_meta(amp_mode)}

    @staticmethod
    def _collecting_electrode(ids: list) -> str:
        """The electrode that collects the avalanche, if it is being read out.

        Its integral is negated for a conventionally positive collected charge —
        the same convention thgem_mesh_sim writes into the *_charge_fC branches.
        """
        return "anode" if "anode" in ids else ""

    def _on_signal_mode_changed(self, _index: int):
        self._update_waveform_plot()
        self._update_charge_plot()

    def _rebuild_charge_xpos(self):
        """Repopulate charge_xpos_combo to match the current charge_dist selection."""
        dist_label = self.charge_dist_combo.currentText()
        xpos_dict  = self._waveform_data.get(dist_label, {})
        self.charge_xpos_combo.blockSignals(True)
        self.charge_xpos_combo.clear()
        for xp in self._sorted_xpos(xpos_dict):
            self.charge_xpos_combo.addItem(xp)
        self.charge_xpos_combo.blockSignals(False)

    def _rebuild_wave_xpos(self):
        """Repopulate wave_xpos_combo to match the current wave_dist selection."""
        dist_label = self.wave_dist_combo.currentText()
        xpos_dict  = self._waveform_data.get(dist_label, {})
        self.wave_xpos_combo.blockSignals(True)
        self.wave_xpos_combo.clear()
        for xp in self._sorted_xpos(xpos_dict):
            self.wave_xpos_combo.addItem(xp)
        self.wave_xpos_combo.blockSignals(False)

    def _sync_charge_slider(self):
        """Update charge_event_slider range for the current (charge_dist, charge_xpos)."""
        dist_label = self.charge_dist_combo.currentText()
        xpos_label = self.charge_xpos_combo.currentText()
        data = self._waveform_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        n = len(data["elec"][data["ids"][0]]["sig"])
        self.charge_event_slider.blockSignals(True)
        self.charge_event_slider.setMaximum(max(0, n - 1))
        self.charge_event_slider.setValue(0)
        self.charge_event_slider.blockSignals(False)
        self.charge_event_label.setText(f"1 / {n}")

    def _sync_wave_slider(self):
        """Update wave_event_slider range for the current (wave_dist, wave_xpos)."""
        dist_label = self.wave_dist_combo.currentText()
        xpos_label = self.wave_xpos_combo.currentText()
        data = self._waveform_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        n = len(data["elec"][data["ids"][0]]["sig"])
        self.wave_event_slider.blockSignals(True)
        self.wave_event_slider.setMaximum(max(0, n - 1))
        self.wave_event_slider.setValue(0)
        self.wave_event_slider.blockSignals(False)
        self.wave_event_label.setText(f"1 / {n}")

    def _on_wave_dist_changed(self, index: int):
        dist_label = self.wave_dist_combo.currentText()
        xpos_dict  = self._waveform_data.get(dist_label, {})

        # Rebuild xpos combo for this distance
        self.wave_xpos_combo.blockSignals(True)
        self.wave_xpos_combo.clear()
        for xp in self._sorted_xpos(xpos_dict):
            self.wave_xpos_combo.addItem(xp)
        self.wave_xpos_combo.blockSignals(False)

        # Sync charge tab (blocked)
        self.charge_dist_combo.blockSignals(True)
        self.charge_dist_combo.setCurrentIndex(index)
        self.charge_dist_combo.blockSignals(False)
        self._rebuild_charge_xpos()

        self._on_wave_xpos_changed(0)

    def _on_wave_xpos_changed(self, index: int):
        dist_label = self.wave_dist_combo.currentText()
        xpos_label = self.wave_xpos_combo.currentText()
        data = self._waveform_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        n = len(data["elec"][data["ids"][0]]["sig"])
        self.wave_event_slider.blockSignals(True)
        self.wave_event_slider.setMaximum(max(0, n - 1))
        self.wave_event_slider.setValue(0)
        self.wave_event_slider.blockSignals(False)
        self.wave_event_label.setText(f"1 / {n}")
        # Sync charge xpos (blocked) and update its slider
        self.charge_xpos_combo.blockSignals(True)
        self.charge_xpos_combo.setCurrentIndex(index)
        self.charge_xpos_combo.blockSignals(False)
        self._sync_charge_slider()
        self._update_waveform_plot()

    def _update_waveform_plot(self):
        """Draw the selected event in a ROOT TCanvas, one pad per read-out electrode."""
        dist_label = self.wave_dist_combo.currentText()
        xpos_label = self.wave_xpos_combo.currentText()
        data  = self._waveform_data.get(dist_label, {}).get(xpos_label)
        label = f"{dist_label}  x={xpos_label}" if xpos_label != "—" else dist_label
        if data is None:
            return

        evt_idx = self.wave_event_slider.value()
        ids     = data["ids"]
        n       = len(data["elec"][ids[0]]["sig"])
        self.wave_event_label.setText(f"{evt_idx + 1} / {n}")

        times = data["times"].astype("f8")
        nbins = len(times)
        dt    = float(times[1] - times[0]) if nbins > 1 else 1.0

        series   = self._select_display_series(data, evt_idx)
        amp_mode = series["amp_mode"]
        units    = series["wave_units"]
        unit_lbl = series["metric_unit"]

        # Metrics: the collected charge on the collecting electrode, and this
        # event's cascade gains — which is what a two-stage stack is read on.
        coll = self._collecting_electrode(ids)
        if coll:
            q = float(-np.sum(series["series"][coll]["evt"]) * dt)
            self.wave_qa_title_lbl.setText(f"{ELECTRODE_LABELS[coll]} [{unit_lbl}]:")
            self.wave_qa_lbl.setText(f"{q:.4g}")
        else:
            self.wave_qa_title_lbl.setText(f"{ELECTRODE_LABELS[ids[0]]} [{unit_lbl}]:")
            self.wave_qa_lbl.setText(f"{float(np.sum(series['series'][ids[0]]['evt']) * dt):.4g}")

        def _evt_scalar(key):
            arr = data.get(key)
            return float(arr[evt_idx]) if arr is not None and evt_idx < len(arr) else float("nan")

        g_thgem = _evt_scalar("gain_thgem")
        g_amp   = _evt_scalar("gain_amp")
        eps     = _evt_scalar("mesh_transparency")
        self.wave_qc_title_lbl.setText("Gain G_THGEM · G_amp:")
        self.wave_qc_lbl.setText(f"{g_thgem:.3g} · {g_amp:.3g}")
        self.wave_ratio_title_lbl.setText("Mesh ε:")
        self.wave_ratio_lbl.setText(f"{eps:.3g}")

        # Optional e⁻/ion component overlay (current mode only)
        has_split = any("e" in data["elec"][e] for e in ids)
        self.wave_split_cb.setEnabled(not amp_mode and has_split)
        split = self.wave_split_cb.isChecked() and not amp_mode and has_split
        if self.wave_split_cb.isChecked() and not has_split:
            if not getattr(self, "_warned_no_split", False):
                self.append_log(
                    "[GUI] This ROOT file has no e⁻/ion component branches — overlay disabled.")
                self._warned_no_split = True

        try:
            import ROOT  # noqa: PLC0415
            ROOT.gROOT.SetBatch(False)

            self._root_canvas = self._ensure_canvas(
                "_root_canvas", "thgem_mesh_waveforms", "THGEM + mesh Waveforms", 950, 700)

            self._root_canvas.Clear()
            ncols = 1 if len(ids) <= 3 else 2
            nrows = (len(ids) + ncols - 1) // ncols
            self._root_canvas.Divide(ncols, nrows)
            self._root_objects.clear()   # release previous objects

            for pad_idx, eid in enumerate(ids, start=1):
                ser = series["series"][eid]
                self._root_canvas.cd(pad_idx)
                ROOT.gPad.SetGrid()
                ga = ROOT.TGraph(nbins, times, ser["evt"])
                ga.SetTitle(
                    f"{ELECTRODE_LABELS[eid]} - {label}, event {evt_idx + 1};"
                    f"Time [ns];{units}"
                )
                ga.SetLineColor(electrode_color(eid))
                ga.SetLineWidth(2)
                gm = ROOT.TGraph(nbins, times, ser["mean"])
                gm.SetLineColor(ROOT.kGray + 1)
                gm.SetLineWidth(1)
                gm.SetLineStyle(2)    # dashed mean
                ga.Draw("AL")
                gm.Draw("L same")
                leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
                leg.AddEntry(ga, "this event", "L")
                leg.AddEntry(gm, "mean",       "L")
                if split and ser["e"] is not None:
                    ge = ROOT.TGraph(nbins, times, ser["e"])
                    ge.SetLineColor(ROOT.kGreen + 2)
                    ge.SetLineWidth(1)
                    gi = ROOT.TGraph(nbins, times, ser["i"])
                    gi.SetLineColor(ROOT.kMagenta + 1)
                    gi.SetLineWidth(1)
                    ge.Draw("L same")
                    gi.Draw("L same")
                    leg.AddEntry(ge, "e^{-} component", "L")
                    leg.AddEntry(gi, "ion component",   "L")
                    self._root_objects.extend([ge, gi])
                leg.SetBorderSize(0)
                leg.Draw()
                self._root_objects.extend([ga, gm, leg])

            self._root_canvas.Update()

        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] ROOT canvas error: {exc}")

    # ── Integrals (cumulative integral of the displayed mode) ─────────────────

    def _on_charge_dist_changed(self, index: int):
        dist_label = self.charge_dist_combo.currentText()
        xpos_dict  = self._waveform_data.get(dist_label, {})

        # Rebuild xpos combo for this distance
        self.charge_xpos_combo.blockSignals(True)
        self.charge_xpos_combo.clear()
        for xp in self._sorted_xpos(xpos_dict):
            self.charge_xpos_combo.addItem(xp)
        self.charge_xpos_combo.blockSignals(False)

        # Sync wave tab (blocked)
        self.wave_dist_combo.blockSignals(True)
        self.wave_dist_combo.setCurrentIndex(index)
        self.wave_dist_combo.blockSignals(False)
        self._rebuild_wave_xpos()

        self._on_charge_xpos_changed(0)

    def _on_charge_xpos_changed(self, index: int):
        dist_label = self.charge_dist_combo.currentText()
        xpos_label = self.charge_xpos_combo.currentText()
        data = self._waveform_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        n = len(data["elec"][data["ids"][0]]["sig"])
        self.charge_event_slider.blockSignals(True)
        self.charge_event_slider.setMaximum(max(0, n - 1))
        self.charge_event_slider.setValue(0)
        self.charge_event_slider.blockSignals(False)
        self.charge_event_label.setText(f"1 / {n}")
        # Sync wave xpos (blocked) and update its slider
        self.wave_xpos_combo.blockSignals(True)
        self.wave_xpos_combo.setCurrentIndex(index)
        self.wave_xpos_combo.blockSignals(False)
        self._sync_wave_slider()
        self._update_charge_plot()

    def _update_charge_plot(self):
        """Draw cumulative integrals of the displayed waveform mode in a ROOT TCanvas."""
        dist_label = self.charge_dist_combo.currentText()
        xpos_label = self.charge_xpos_combo.currentText()
        data  = self._waveform_data.get(dist_label, {}).get(xpos_label)
        label = f"{dist_label}  x={xpos_label}" if xpos_label != "—" else dist_label
        if data is None:
            return

        evt_idx = self.charge_event_slider.value()
        ids     = data["ids"]
        n       = len(data["elec"][ids[0]]["sig"])
        self.charge_event_label.setText(f"{evt_idx + 1} / {n}")

        times  = data["times"].astype("f8")
        series = self._select_display_series(data, evt_idx)
        nbins  = len(times)
        dt     = float(times[1] - times[0]) if nbins > 1 else 1.0
        coll   = self._collecting_electrode(ids)

        try:
            import ROOT  # noqa: PLC0415
            ROOT.gROOT.SetBatch(False)

            self._charge_canvas = self._ensure_canvas(
                "_charge_canvas", "thgem_mesh_charges", "THGEM + mesh Integrals", 950, 700)

            self._charge_canvas.Clear()
            ncols = 1 if len(ids) <= 3 else 2
            nrows = (len(ids) + ncols - 1) // ncols
            self._charge_canvas.Divide(ncols, nrows)
            self._charge_objects.clear()

            for pad_idx, eid in enumerate(ids, start=1):
                ser = series["series"][eid]
                # Negate the collecting electrode so its integral runs positive —
                # the same convention as the *_charge_fC branches.
                sign = -1.0 if eid == coll else 1.0
                y_evt  = sign * np.cumsum(ser["evt"]) * dt
                y_mean = sign * np.cumsum(ser["mean"]) * dt

                self._charge_canvas.cd(pad_idx)
                ROOT.gPad.SetGrid()
                ga = ROOT.TGraph(nbins, times, y_evt)
                ga.SetTitle(
                    f"{ELECTRODE_LABELS[eid]} integral - {label}, event {evt_idx + 1};"
                    f"Time [ns];{series['root_integral_units']}"
                )
                ga.SetLineColor(electrode_color(eid))
                ga.SetLineWidth(2)
                gm = ROOT.TGraph(nbins, times, y_mean)
                gm.SetLineColor(ROOT.kGray + 1)
                gm.SetLineWidth(1)
                gm.SetLineStyle(2)
                ga.Draw("AL")
                gm.Draw("L same")
                leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
                leg.AddEntry(ga, "this event", "L")
                leg.AddEntry(gm, "mean",       "L")
                leg.SetBorderSize(0)
                leg.Draw()
                self._charge_objects.extend([ga, gm, leg])

            self._charge_canvas.Update()

        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] ROOT charge canvas error: {exc}")

    # ── 3D Tracks ──────────────────────────────────────────────────────────────

    def load_track_data(self, root_path: str, run_dir: str):
        """Load per-event track branches and geometry from the ROOT file."""
        try:
            import uproot  # noqa: PLC0415
        except ImportError:
            self.append_log("[GUI] uproot not available — 3D Tracks tab disabled")
            return

        self._track_data.clear()
        self._track_geom = None
        self._tracks_canvas  = None   # force recreation for the new run
        self._trk_zoom_scale = 1.0
        self._trk_view_phi   = 32.0
        self._trk_view_theta = 30.0
        self._trk_pan_x = 0.0
        self._trk_pan_y = 0.0
        self._trk_pan_z = 0.0
        self.trk_dist_combo.blockSignals(True)
        self.trk_dist_combo.clear()
        self.trk_xpos_combo.blockSignals(True)
        self.trk_xpos_combo.clear()

        # Geometry comes from run_config.json written by the binary
        cfg_path = Path(run_dir) / "run_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    rc = json.load(f)
                g  = rc.get("geometry", {})
                dv = rc.get("derived", {})
                # thgem_mesh_sim echoes the whole resolved stack under "derived", so the
                # GUI never re-derives geometry from the config panel (which may
                # have been edited since the run) and cannot drift out of sync.
                self._track_geom = {
                    "pitch_cm":    dv.get("pitch_cm", g.get("hole_pitch_um", 800.0) * 1e-4),
                    "cell_x_cm":   dv.get("cell_x_cm"),
                    "cell_y_cm":   dv.get("cell_y_cm"),
                    "n_holes_x":   int(dv.get("n_holes_x", 1)),
                    "hole_x_thgem_cm": dv.get("hole_x_thgem_cm", [0.0]),
                    "z_wire_cm":   dv.get("z_wire_cm"),
                    "r_wire_cm":   dv.get("r_wire_cm", 25e-4),
                    "z_anode_cm":  dv.get("z_anode_cm"),
                    "z_min_cm":    dv.get("z_min_cm"),
                    "z_max_cm":    dv.get("z_max_cm"),
                    "thgem":       dv.get("thgem", {}),
                    "mesh":        dv.get("mesh", {}),
                }
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[GUI] run_config.json read error: {exc}")

        if not Path(root_path).exists():
            self.trk_dist_combo.blockSignals(False)
            return

        try:
            with uproot.open(root_path) as f:
                # Folder names: dist_0p1mm_x0p18mm  (or dist_0p1mm for old files)
                all_keys = sorted({
                    k.split("/")[0] for k in f.keys(cycle=False)
                    if k.split("/")[0].startswith("dist_")
                })
                for key in all_keys:
                    rest = key.removeprefix("dist_")
                    dist_raw, sep, xpos_raw = rest.partition("_x")
                    dist_label = "—" if dist_raw == "rnd" else dist_raw.replace("p", ".").replace("mm", " mm")
                    xpos_label = (xpos_raw.replace("p", ".").replace("mm", " mm")
                                  if sep else "—")
                    try:
                        tree = f[f"{key}/t_signals"]
                        if "primary_x" not in tree.keys():
                            continue   # pre-feature ROOT file — skip silently
                        data_dict = {
                            "primary_x": tree["primary_x"].array(library="np"),
                            "primary_y": tree["primary_y"].array(library="np"),
                            "primary_z": tree["primary_z"].array(library="np"),
                            "cloud_x":   tree["cloud_x"].array(library="np"),
                            "cloud_y":   tree["cloud_y"].array(library="np"),
                            "cloud_z":   tree["cloud_z"].array(library="np"),
                            "ion_x":     tree["ion_x"].array(library="np"),
                            "ion_y":     tree["ion_y"].array(library="np"),
                            "ion_z":     tree["ion_z"].array(library="np"),
                            "ion_npts":  tree["ion_npts"].array(library="np"),
                        }
                        # Avalanche-electron drift lines (added later; older ROOT
                        # files lack them → fall back to empty per-event arrays).
                        n_ev = len(data_dict["primary_x"])
                        if "aval_x" in tree.keys():
                            for br in ("aval_x", "aval_y", "aval_z", "aval_npts"):
                                data_dict[br] = tree[br].array(library="np")
                        else:
                            for br in ("aval_x", "aval_y", "aval_z", "aval_npts"):
                                data_dict[br] = [np.empty(0) for _ in range(n_ev)]
                        self._track_data.setdefault(dist_label, {})[xpos_label] = data_dict
                    except Exception as exc:  # noqa: BLE001
                        self.append_log(f"[GUI] Track load error for {key}: {exc}")
                # Populate dist combo in sorted order (random "—" first)
                for dl in self._sorted_dists(self._track_data.keys()):
                    self.trk_dist_combo.addItem(dl)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Could not open ROOT file for tracks: {exc}")

        self.trk_dist_combo.blockSignals(False)
        self.trk_xpos_combo.blockSignals(False)
        if self.trk_dist_combo.count():
            self._on_trk_dist_changed(0)
            self._root_timer.start()   # keep tracks window responsive

    def _on_trk_dist_changed(self, index: int):
        dist_label = self.trk_dist_combo.currentText()
        xpos_dict  = self._track_data.get(dist_label, {})

        self.trk_xpos_combo.blockSignals(True)
        self.trk_xpos_combo.clear()
        for xpos_label in sorted(
                xpos_dict.keys(),
                key=lambda s: float(s.replace(" mm", "")) if s != "—" else 0.0):
            self.trk_xpos_combo.addItem(xpos_label)
        self.trk_xpos_combo.blockSignals(False)

        self._on_trk_xpos_changed(0)

    def _on_trk_xpos_changed(self, index: int):
        dist_label = self.trk_dist_combo.currentText()
        xpos_label = self.trk_xpos_combo.currentText()
        data = self._track_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        n = len(data["primary_x"])
        self.trk_event_slider.blockSignals(True)
        self.trk_event_slider.setMaximum(max(0, n - 1))
        self.trk_event_slider.setValue(0)
        self.trk_event_slider.blockSignals(False)
        self.trk_event_label.setText(f"1 / {n}")
        self._update_track_plot()

    def _update_track_plot(self):
        """Render detector geometry and per-event tracks in a ROOT TCanvas."""
        dist_label = self.trk_dist_combo.currentText()
        xpos_label = self.trk_xpos_combo.currentText()
        data = self._track_data.get(dist_label, {}).get(xpos_label)
        if data is None:
            return
        label = f"{dist_label}  x={xpos_label}" if xpos_label != "—" else dist_label

        ev = self.trk_event_slider.value()
        n  = len(data["primary_x"])
        self.trk_event_label.setText(f"{ev + 1} / {n}")

        geom    = self._track_geom or {}
        pitch   = geom.get("pitch_cm", 0.08)
        cell_x  = geom.get("cell_x_cm") or pitch
        cell_y  = geom.get("cell_y_cm") or pitch
        p1      = geom.get("thgem", {})
        mesh    = geom.get("mesh", {})
        z_wire  = geom.get("z_wire_cm")
        r_wire  = geom.get("r_wire_cm", 25e-4)
        z_anode = geom.get("z_anode_cm")
        z_min   = geom.get("z_min_cm", 0.0)
        z_max   = geom.get("z_max_cm", z_wire if z_wire is not None else 0.6)
        # The drift axis is z and the stack runs from the bottom electrode up to
        # the wire plane — unlike the single-plate sibling it is not centred on
        # z = 0, so the view cube is centred on the stack instead of the origin.
        z_centre = 0.5 * (z_min + z_max)
        gap      = 0.5 * max(z_max - z_min, 1e-3)
        x_half   = 1.5 * cell_x
        max_half = max(x_half, gap)

        try:
            import ROOT  # noqa: PLC0415
            ROOT.gROOT.SetBatch(False)

            self._tracks_canvas = self._ensure_canvas(
                "_tracks_canvas", "thgem_mesh_tracks", "THGEM + mesh 3D Tracks", 900, 700)

            self._tracks_canvas.cd()
            self._tracks_canvas.Clear()
            self._tracks_objects.clear()
            self._tracks_canvas.SetPhi(self._trk_view_phi)
            self._tracks_canvas.SetTheta(self._trk_view_theta)

            # TH3F frame — defines axes and 3D coordinate range
            s  = self._trk_zoom_scale
            px = self._trk_pan_x
            py = self._trk_pan_y
            pz = z_centre + self._trk_pan_z
            frame = ROOT.TH3F(
                "trk_frame",
                f"THGEM + mesh 3D Tracks - {label}, event {ev + 1};"
                "x [cm];y [cm];z [cm]",
                1, px - max_half * s, px + max_half * s,
                1, py - max_half * s, py + max_half * s,
                1, pz - max_half * s, pz + max_half * s,
            )
            frame.SetStats(0)
            # Dynamic margins: make the inner pad area square so ROOT maps the
            # equal-range 3D cube without horizontal stretch, regardless of the
            # actual Qt-determined canvas pixel size.
            _cw = self._tracks_canvas.GetWw() or 900
            _ch = self._tracks_canvas.GetWh() or 700
            _top, _bottom = 0.05, 0.10
            _ih  = _ch * (1.0 - _top - _bottom)         # inner height in px
            _lr  = max(0.18, 1.0 - _ih / _cw)           # left+right fraction needed
            self._tracks_canvas.SetTopMargin(_top)
            self._tracks_canvas.SetBottomMargin(_bottom)
            self._tracks_canvas.SetLeftMargin(_lr * 0.55)   # more left for y-label
            self._tracks_canvas.SetRightMargin(_lr * 0.45)
            # Push all axis titles away from their axis lines.
            frame.GetXaxis().SetTitleOffset(1.6)
            frame.GetYaxis().SetTitleOffset(2.5)
            frame.GetZaxis().SetTitleOffset(1.6)
            frame.Draw()
            self._tracks_objects.append(frame)

            def _clip(xs, ys, zs):
                """Return contiguous sub-segments whose points lie within the
                visible cube [centre ± max_half*s].  Point-mask only — no exact
                boundary intersection, but avoids drawing far outside the frame."""
                hr = max_half * s
                cx_ = px
                cy_ = py
                cz_ = pz
                mask = ((xs >= cx_ - hr) & (xs <= cx_ + hr) &
                        (ys >= cy_ - hr) & (ys <= cy_ + hr) &
                        (zs >= cz_ - hr) & (zs <= cz_ + hr))
                segs, i, n = [], 0, len(xs)
                while i < n:
                    if mask[i]:
                        j = i + 1
                        while j < n and mask[j]:
                            j += 1
                        if j - i >= 2:
                            segs.append((xs[i:j], ys[i:j], zs[i:j]))
                        i = j
                    else:
                        i += 1
                return segs

            def _pl3(xs, ys, zs, color, width=1, alpha=1.0):
                ln = ROOT.TPolyLine3D(
                    len(xs), xs.astype("f4"), ys.astype("f4"), zs.astype("f4"))
                if alpha < 1.0:
                    ln.SetLineColorAlpha(color, alpha)
                else:
                    ln.SetLineColor(color)
                ln.SetLineWidth(width)
                ln.Draw("SAME")
                self._tracks_objects.append(ln)

            # ── Stack geometry (electrode planes, plates, wires) ───────────────
            _hr = max_half * s   # visible half-range (cube half-size at this zoom)

            def _draw_plane_z(z, color, width=2):
                """Rectangle in the x,y plane at constant z, clipped to the view."""
                if z is None or not (pz - _hr <= z <= pz + _hr):
                    return
                x0, x1 = px - _hr, px + _hr
                y0, y1 = py - _hr, py + _hr
                xs = np.array([x0, x1, x1, x0, x0], "f4")
                ys = np.array([y0, y0, y1, y1, y0], "f4")
                zs = np.full(5, z, "f4")
                pl = ROOT.TPolyLine3D(5, xs, ys, zs)
                pl.SetLineColor(color)
                pl.SetLineWidth(width)
                pl.Draw("SAME")
                self._tracks_objects.append(pl)

            def _draw_cylinder(cx, cy, z0, z1, r_cm, color, alpha, n_sides=12):
                """Wireframe cylinder (a hole) of radius r_cm between z0 and z1."""
                ang = np.linspace(0.0, 2.0 * np.pi, n_sides + 1)
                xs = (cx + r_cm * np.cos(ang)).astype("f4")
                ys = (cy + r_cm * np.sin(ang)).astype("f4")
                for zc in (z0, z1):                       # top and bottom rings
                    zs = np.full(n_sides + 1, zc, "f4")
                    ring = ROOT.TPolyLine3D(n_sides + 1, xs, ys, zs)
                    ring.SetLineColorAlpha(color, alpha)
                    ring.SetLineWidth(1)
                    ring.Draw("SAME")
                    self._tracks_objects.append(ring)
                for a in ang[:-1:2]:                      # every other longitudinal edge
                    xe = float(cx + r_cm * np.cos(a))
                    ye = float(cy + r_cm * np.sin(a))
                    ln = ROOT.TPolyLine3D(
                        2, np.array([xe, xe], "f4"), np.array([ye, ye], "f4"),
                        np.array([z0, z1], "f4"))
                    ln.SetLineColorAlpha(color, alpha)
                    ln.SetLineWidth(1)
                    ln.Draw("SAME")
                    self._tracks_objects.append(ln)

            def _draw_wire_y(cx, zc, rad, color, alpha, n_sides=8):
                """A wire running along y across the view (cathode, or mesh layer)."""
                y0, y1 = py - _hr, py + _hr
                ang = np.linspace(0.0, 2.0 * np.pi, n_sides + 1)
                for a in ang[:-1]:
                    xe = float(cx + rad * np.cos(a))
                    ze = float(zc + rad * np.sin(a))
                    ln = ROOT.TPolyLine3D(
                        2, np.array([xe, xe], "f4"), np.array([y0, y1], "f4"),
                        np.array([ze, ze], "f4"))
                    ln.SetLineColorAlpha(color, alpha)
                    ln.SetLineWidth(1)
                    ln.Draw("SAME")
                    self._tracks_objects.append(ln)

            def _draw_wire_x(cy, zc, rad, color, alpha, n_sides=8):
                """The mirror of _draw_wire_y: a wire running along x."""
                x0, x1 = px - _hr, px + _hr
                ang = np.linspace(0.0, 2.0 * np.pi, n_sides + 1)
                for a in ang[:-1]:
                    ye = float(cy + rad * np.cos(a))
                    ze = float(zc + rad * np.sin(a))
                    ln = ROOT.TPolyLine3D(
                        2, np.array([x0, x1], "f4"), np.array([ye, ye], "f4"),
                        np.array([ze, ze], "f4"))
                    ln.SetLineColorAlpha(color, alpha)
                    ln.SetLineWidth(1)
                    ln.Draw("SAME")
                    self._tracks_objects.append(ln)

            def _draw_mesh(mg, color):
                """The mesh: its two bounding planes plus the wires or apertures."""
                z_t = mg.get("z_top_cm")
                z_b = mg.get("z_bot_cm")
                if z_t is None or z_b is None:
                    return
                _draw_plane_z(z_t, ROOT.kGreen + 1, 1)
                _draw_plane_z(z_b, ROOT.kGreen + 1, 1)
                if not (pz - _hr <= z_t and z_b <= pz + _hr):
                    return
                n = max(1, int(self._trk_n_holes))
                half = n // 2
                if mg.get("model") == "woven":
                    r = mg.get("r_wire_cm", 25e-4)
                    for i in range(-half, n - half):
                        for xw in mg.get("wire_x_cm", []):
                            _draw_wire_y(xw + i * cell_x, mg.get("z_lower_cm", z_b),
                                         r, color, 0.8)
                        for yw in mg.get("wire_y_cm", []):
                            _draw_wire_x(yw + i * cell_y, mg.get("z_upper_cm", z_t),
                                         r, color, 0.8)
                else:
                    r_a = mg.get("r_aperture_cm", 0.01)
                    for i in range(-half, n - half):
                        for j in range(-half, n - half):
                            for xa in mg.get("wire_x_cm", []):
                                for ya in mg.get("wire_y_cm", []):
                                    _draw_cylinder(xa + i * cell_x, ya + j * cell_y,
                                                   z_b, z_t, r_a, color, 0.55)

            def _draw_plate(pg, hole_xs, color):
                """The THGEM: its two copper faces plus a block of hole cylinders."""
                z_t = pg.get("z_top_cu_top_cm")
                z_b = pg.get("z_bot_cu_bot_cm")
                r_h = pg.get("r_hole_cm", 0.02)
                if z_t is None or z_b is None:
                    return
                _draw_plane_z(z_t, ROOT.kGray + 2, 1)
                _draw_plane_z(z_b, ROOT.kGray + 2, 1)
                if not (pz - _hr <= z_t and z_b <= pz + _hr):
                    return
                # Tile the cell's own hole columns outward, so a staggered plate
                # keeps its true x positions rather than being redrawn on a
                # generic lattice.
                n = max(1, int(self._trk_n_holes))
                half = n // 2
                for i in range(-half, n - half):
                    for xh in hole_xs:
                        cx = xh + i * cell_x
                        for j in range(-half, n - half):
                            _draw_cylinder(cx, j * pitch, z_b, z_t, r_h, color, 0.55)

            # Bounding electrode planes.
            _draw_plane_z(z_wire,  ROOT.kCyan - 7)    # wire-cathode plane (top)
            if z_anode is not None:
                _draw_plane_z(z_anode, ROOT.kRed - 7)  # anode pad (bottom)

            # The wire cathode itself: one wire per cell, running along y.
            if z_wire is not None and pz - _hr <= z_wire <= pz + _hr:
                n = max(1, int(self._trk_n_holes))
                half = n // 2
                for i in range(-half, n - half):
                    _draw_wire_y(i * cell_x, z_wire, r_wire, ROOT.kCyan + 2, 0.9)

            # The two multiplying stages, in distinct colours.
            _draw_plate(p1, geom.get("hole_x_thgem_cm", [0.0]), ROOT.kOrange + 7)
            _draw_mesh(mesh, ROOT.kGreen + 2)

            # ── Primary electron drift (blue) ─────────────────────────────────
            px = np.asarray(data["primary_x"][ev])
            py = np.asarray(data["primary_y"][ev])
            pz = np.asarray(data["primary_z"][ev])
            # No Python-level clipping — always draw the full track so it
            # remains visible at any zoom level. ROOT's 3D→2D projector handles
            # clipping at the pad boundary automatically.
            if len(px) >= 2:
                _pl3(px, py, pz, ROOT.kBlue + 1, 2, alpha=0.65)

            # ── Avalanche cloud (orange markers) ──────────────────────────────
            cx_ = np.asarray(data["cloud_x"][ev])
            cy_ = np.asarray(data["cloud_y"][ev])
            cz_ = np.asarray(data["cloud_z"][ev])
            if len(cx_) > 0:
                p = np.empty(3 * len(cx_), "f4")
                p[0::3] = cx_; p[1::3] = cy_; p[2::3] = cz_
                mrk = ROOT.TPolyMarker3D(len(cx_), p, 7)
                mrk.SetMarkerColorAlpha(ROOT.kOrange + 1, 0.50)
                mrk.SetMarkerSize(0.4)
                mrk.Draw("SAME")
                self._tracks_objects.append(mrk)

            # ── Avalanche-electron transport (orange drift lines) ─────────────
            # Draw the first self._trk_n_aval_paths stored secondary drift lines so
            # the cascade's transport through the hole to the anode is visible.
            av_x = np.asarray(data["aval_x"][ev])
            av_y = np.asarray(data["aval_y"][ev])
            av_z = np.asarray(data["aval_z"][ev])
            av_np = np.asarray(data["aval_npts"][ev])
            n_draw = min(int(self._trk_n_aval_paths), len(av_np))
            off = 0
            for k in range(len(av_np)):
                n_seg = int(av_np[k])
                if k < n_draw and n_seg >= 2:
                    xs = av_x[off:off + n_seg]
                    ys = av_y[off:off + n_seg]
                    zs = av_z[off:off + n_seg]
                    for _seg in _clip(xs, ys, zs):
                        _pl3(*_seg, ROOT.kOrange + 1, 1, alpha=0.40)
                off += n_seg

            # ── Ion drift paths (colour-coded by destination) ─────────────────
            ion_x = np.asarray(data["ion_x"][ev])
            ion_y = np.asarray(data["ion_y"][ev])
            ion_z = np.asarray(data["ion_z"][ev])
            inpts = np.asarray(data["ion_npts"][ev])
            tol   = 0.05 * gap
            off   = 0
            for n_seg in inpts:
                xs = ion_x[off:off + n_seg]
                ys = ion_y[off:off + n_seg]
                zs = ion_z[off:off + n_seg]
                off += n_seg
                if len(ys) < 2:
                    # single start-point: ion immediately absorbed — draw as marker
                    if len(ys) == 1:
                        pt = np.array([float(xs[0]), float(ys[0]), float(zs[0])], "f4")
                        dot = ROOT.TPolyMarker3D(1, pt, 7)
                        dot.SetMarkerColorAlpha(ROOT.kGray + 2, 0.5)
                        dot.SetMarkerSize(0.3)
                        dot.Draw("SAME")
                        self._tracks_objects.append(dot)
                    continue
                z_end = float(zs[-1])
                if z_wire is not None and abs(z_end - z_wire) <= tol:
                    col = ROOT.kGreen + 2    # → wire cathode (ions drift up)
                elif z_anode is not None and abs(z_end - z_anode) <= tol:
                    col = ROOT.kMagenta      # → anode pad
                else:
                    col = ROOT.kGray + 1     # absorbed on a plate / out of window
                for _seg in _clip(xs, ys, zs):
                    _pl3(*_seg, col, 1, alpha=0.55)

            # ── Legend ────────────────────────────────────────────────────────
            self._trk_legend_objects.clear()

            def _mk_line(color: int, width: int = 1) -> object:
                ln = ROOT.TLine()
                ln.SetLineColor(color)
                ln.SetLineWidth(width)
                return ln

            def _mk_marker(color: int) -> object:
                mk = ROOT.TMarker()
                mk.SetMarkerColor(color)
                mk.SetMarkerStyle(7)
                mk.SetMarkerSize(1.2)
                return mk

            _leg_proxies = [
                (_mk_line(ROOT.kBlue + 1, 2),  "Primary e^{-}",           "L"),
                (_mk_marker(ROOT.kOrange + 1), "Avalanche e^{-} (birth)", "P"),
                (_mk_line(ROOT.kOrange + 1),   "Avalanche e^{-} (path)",  "L"),
                (_mk_line(ROOT.kGreen + 2),    "Ion #rightarrow wires",   "L"),
                (_mk_line(ROOT.kMagenta),      "Ion #rightarrow anode",   "L"),
                (_mk_line(ROOT.kGray + 1),     "Ion (absorbed)",          "L"),
                (_mk_line(ROOT.kOrange + 7),   "THGEM holes",             "L"),
                (_mk_line(ROOT.kGreen + 2),    "Mesh",                    "L"),
                (_mk_line(ROOT.kCyan + 2),     "Cathode wires",           "L"),
                (_mk_line(ROOT.kRed - 7, 2),   "Anode plane",             "L"),
            ]
            leg = ROOT.TLegend(0.70, 0.56, 0.99, 0.97)
            leg.SetBorderSize(0)
            leg.SetFillColorAlpha(ROOT.kWhite, 0.75)
            leg.SetTextSize(0.024)
            for _proxy, _label, _opt in _leg_proxies:
                leg.AddEntry(_proxy, _label, _opt)
            leg.Draw()
            self._trk_legend_objects = [leg] + [p for p, _, _ in _leg_proxies]
            # ──────────────────────────────────────────────────────────────────

            self._tracks_canvas.Update()

        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] ROOT 3D tracks error: {exc}")

    def _trk_adjust_zoom(self, factor: float) -> None:
        """Scale the visible axis range and redraw (factor < 1 = zoom in)."""
        self._trk_zoom_scale = max(0.005, min(20.0, self._trk_zoom_scale * factor))
        self._update_track_plot()

    def _on_trk_holes_changed(self, value: int) -> None:
        """Redraw with an N×N block of holes (display-only)."""
        self._trk_n_holes = int(value)
        self._update_track_plot()

    def _on_trk_n_aval_changed(self, value: int) -> None:
        """Redraw with the given number of avalanche-electron drift lines."""
        self._trk_n_aval_paths = int(value)
        self._update_track_plot()

    def _trk_pan(self, axis: str, direction: int) -> None:
        """Shift the visible centre by 30 % of the current visible half-range."""
        geom    = self._track_geom or {}
        cell_x  = geom.get("cell_x_cm") or geom.get("pitch_cm", 0.08)
        z_min   = geom.get("z_min_cm", 0.0)
        z_max   = geom.get("z_max_cm", 0.6)
        gap     = 0.5 * max(z_max - z_min, 1e-3)
        max_half = max(1.5 * cell_x, gap)        # matches the TH3F frame in _update_track_plot
        step = max_half * self._trk_zoom_scale * 0.3 * direction
        if   axis == "x": self._trk_pan_x += step
        elif axis == "y": self._trk_pan_y += step
        else:             self._trk_pan_z += step
        self._update_track_plot()

    def _trk_preset_view(self, phi: float, theta: float,
                         reset_zoom: bool = False) -> None:
        """Set pad view angles and redraw."""
        self._trk_view_phi   = phi
        self._trk_view_theta = theta
        if reset_zoom:
            self._trk_zoom_scale = 1.0
            self._trk_pan_x = self._trk_pan_y = self._trk_pan_z = 0.0
        self._update_track_plot()

    # ── E-Field ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clone_root(obj):
        """Detach a histogram/graph from its TFile so it survives the file being closed."""
        c = obj.Clone()
        try:
            c.SetDirectory(0)      # TH*; TGraph has no SetDirectory
        except AttributeError:
            pass
        return c

    def _load_run_geometry(self, root_path: str):
        """Read the run's resolved stack from run_config.json's "derived" block.

        The overlays used to re-derive geometry from the config panel, which is
        wrong the moment the panel is edited after a run — and silently so.  The
        binary echoes everything needed, so read that instead.
        """
        cfg_path = Path(root_path).with_name("run_config.json")
        if not cfg_path.exists():
            self._field_geom = None
            return
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                self._field_geom = json.load(fh).get("derived", {})
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Could not read run_config.json: {exc}")
            self._field_geom = None

    def load_field_maps(self, root_path: str):
        """Load the neBEM field-map dump (field/ dir) from the run ROOT file."""
        try:
            import ROOT  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return
        self._load_run_geometry(root_path)
        f = ROOT.TFile.Open(root_path)
        if not f or f.IsZombie():
            return
        try:
            hmag = f.Get("field/h_field_mag")
            hpot = f.Get("field/h_potential")
            hez  = f.Get("field/h_field_ez")
            hex_ = f.Get("field/h_field_ex")
            hmyz = f.Get("field/h_field_mag_yz")     # along the wires
            hpyz = f.Get("field/h_potential_yz")
            gE   = f.Get("field/g_axis_field")
            gV   = f.Get("field/g_axis_potential")
            gE2  = f.Get("field/g_axis_field_mesh")   # the mesh's own aperture axis
            gV2  = f.Get("field/g_axis_potential_mesh")
            if not hmag:
                return
            self._field_maps = dict(
                mag=self._clone_root(hmag),
                pot=self._clone_root(hpot) if hpot else None,
                ez=self._clone_root(hez) if hez else None,
                ex=self._clone_root(hex_) if hex_ else None,
                mag_yz=self._clone_root(hmyz) if hmyz else None,
                pot_yz=self._clone_root(hpyz) if hpyz else None,
                axE=self._clone_root(gE) if gE else None,
                axV=self._clone_root(gV) if gV else None,
                axE2=self._clone_root(gE2) if gE2 else None,
                axV2=self._clone_root(gV2) if gV2 else None)
        finally:
            f.Close()
        self._redraw_field()

    def load_weighting_maps(self, root_path: str):
        """Load the neBEM weighting map of every read-out electrode (field/h_wpot_<id>, …).

        Runs produced before the per-electrode weighting solve existed simply have no
        such objects; the tab then stays empty rather than raising.
        """
        try:
            import ROOT  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return
        f = ROOT.TFile.Open(root_path)
        if not f or f.IsZombie():
            return
        try:
            maps = {}
            for eid in ELECTRODE_IDS:
                hw = f.Get(f"field/h_wpot_{eid}")
                if not hw:
                    continue
                hwe = f.Get(f"field/h_wfield_mag_{eid}")
                gw  = f.Get(f"field/g_axis_wpot_{eid}")
                maps[eid] = dict(
                    wpot=self._clone_root(hw),
                    wmag=self._clone_root(hwe) if hwe else None,
                    axW=self._clone_root(gw) if gw else None)
            if not maps:
                # Older run: clear any map still shown from a previously loaded run.
                self._weighting_maps = None
                self._wfield_objects.clear()
                if self._wfield_root_canvas is not None:
                    self._wfield_root_canvas.Clear()
                    self._wfield_root_canvas.Update()
                self.append_log(
                    "[GUI] This run has no weighting maps — re-run to populate "
                    "the Weighting Field tab.")
                return
            self._weighting_maps = maps
        finally:
            f.Close()
        self._redraw_wfield()

    def _draw_field_canvas(self, attr, cname, ctitle, h2, prof, palette, objects,
                           overlay: bool = True):
        """Draw a TH2 colour map (+ geometry overlay) and an on-axis profile into an
        interactive ROOT canvas — right-click to zoom, rescale the axes, or save."""
        import ROOT  # noqa: PLC0415
        ROOT.gROOT.SetBatch(False)
        canvas = self._ensure_canvas(attr, cname, ctitle, 1150, 540)
        canvas.cd()
        canvas.Clear()
        objects.clear()
        canvas.Divide(2, 1)

        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(_ROOT_PALETTE.get(palette, 112))

        canvas.cd(1)
        ROOT.gPad.SetRightMargin(0.16)
        ROOT.gPad.SetLeftMargin(0.13)
        if h2:
            # The dump spans exactly two periodic cells in x, so half the map's
            # x-range is the cell period — read it from the map itself rather than
            # from the config panel, which may have drifted since the run.
            cell_x = (h2.GetXaxis().GetXmax() - h2.GetXaxis().GetXmin()) / 2.0
            if overlay:
                h2 = self._tile_map_x(h2, self._map_n_holes, cell_x)
            h2.Draw("COLZ")
            objects.append(h2)
            if overlay:
                for ln in self._root_geometry_lines(h2, cell_x):
                    ln.Draw("SAME")
                    objects.append(ln)

        canvas.cd(2)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        ROOT.gPad.SetLeftMargin(0.15)
        if prof:
            prof.SetLineWidth(2)
            prof.SetLineColor(ROOT.kAzure + 2)
            prof.Draw("AL")
            objects.append(prof)

        canvas.Modified()
        canvas.Update()
        return canvas

    def _tile_map_x(self, h2, n_cells, cell_x):
        """Return a copy of the x–z map widened to `n_cells` periodic cells across x.

        The dumped map already spans two full periods, and the field is periodic, so
        this is an exact tiling — not an interpolation.  z binning and the per-cell x
        resolution are preserved (nbinsx scales with the width).

        The map is centred on the reference hole axis, which is not necessarily
        x = 0 (a `wire_between_holes` lattice puts it at half a pitch), so the wrap
        is taken about the map's own centre.
        """
        import ROOT  # noqa: PLC0415
        n = max(1, int(n_cells))
        if cell_x <= 0 or n == 1:
            return h2

        srcx = h2.GetXaxis()
        dx = (srcx.GetXmax() - srcx.GetXmin()) / srcx.GetNbins()   # source x bin width
        xc = 0.5 * (srcx.GetXmin() + srcx.GetXmax())               # reference hole axis
        half = n * cell_x / 2.0
        nx = max(1, int(round(2.0 * half / dx)))                   # preserve resolution
        nz = h2.GetNbinsY()
        zax = h2.GetYaxis()
        wide = ROOT.TH2D(h2.GetName() + "_wide", h2.GetTitle(),
                         nx, xc - half, xc + half, nz, zax.GetXmin(), zax.GetXmax())
        wide.SetDirectory(0)
        wide.GetZaxis().SetTitle(h2.GetZaxis().GetTitle())
        for iz in range(1, nz + 1):
            z = wide.GetYaxis().GetBinCenter(iz)
            jz = zax.FindBin(z)
            for ix in range(1, nx + 1):
                x = wide.GetXaxis().GetBinCenter(ix)
                # wrap x into one period centred on the map's reference axis
                xw = xc + ((x - xc + cell_x / 2.0) % cell_x) - cell_x / 2.0
                wide.SetBinContent(ix, iz, h2.GetBinContent(srcx.FindBin(xw), jz))
        return wide

    def _on_map_holes_changed(self, value: int) -> None:
        """Re-tile both field maps; keep the two 'Holes' spinboxes in sync."""
        self._map_n_holes = int(value)
        for spin in (getattr(self, "ef_holes", None), getattr(self, "wf_holes", None)):
            if spin is not None and spin.value() != self._map_n_holes:
                spin.blockSignals(True)
                spin.setValue(self._map_n_holes)
                spin.blockSignals(False)
        self._redraw_field()
        self._redraw_wfield()

    def _root_geometry_lines(self, h2, cell_x):
        """TLine overlay of the stack: the THGEM's copper surfaces and hole walls,
        the mesh, the cathode wires, and the bounding electrode planes.

        Everything comes from the loaded run's own "derived" block, so the overlay
        always describes the run on screen — not whatever the config panel happens
        to hold now.

        Note what the slice does to a woven mesh: this is an x–z cut, so it crosses
        the lower layer (wires ∥ y) transversely — those are drawn as ticks — while
        the upper layer (wires ∥ x) is cut lengthwise and cannot be resolved in x
        at all.  The upper layer is therefore drawn as a single dashed line marking
        its plane; it is not a solid sheet.
        """
        import ROOT  # noqa: PLC0415
        lines = []
        dv = self._field_geom
        if not dv:
            return lines

        x0, x1 = h2.GetXaxis().GetXmin(), h2.GetXaxis().GetXmax()
        zmin, zmax = h2.GetYaxis().GetXmin(), h2.GetYaxis().GetXmax()
        pitch = dv.get("pitch_cm", cell_x)

        def _hline(z, colour, width=1, style=1):
            if not (zmin <= z <= zmax):
                return
            ln = ROOT.TLine(x0, z, x1, z)
            ln.SetLineColor(colour)
            ln.SetLineWidth(width)
            ln.SetLineStyle(style)
            lines.append(ln)

        def _vline(x, za, zb, colour, style=2):
            if not (x0 <= x <= x1):
                return
            ln = ROOT.TLine(x, za, x, zb)
            ln.SetLineColor(colour)
            ln.SetLineStyle(style)
            ln.SetLineWidth(1)
            lines.append(ln)

        ncells = int(math.ceil((x1 - x0) / cell_x)) + 2 if cell_x > 0 else 1

        # The THGEM: copper faces as solid white lines, hole walls dashed.
        pg = dv.get("thgem") or {}
        z_t = pg.get("z_top_cu_top_cm")
        z_b = pg.get("z_bot_cu_bot_cm")
        zdh = pg.get("z_diel_half_cm")
        if z_t is not None and z_b is not None and zdh is not None:
            zc  = pg.get("z_cen_cm", 0.5 * (z_t + z_b))
            r   = pg.get("r_hole_cm", 0.02)
            rcu = pg.get("r_cu_cm", r)
            for z in (z_t, zc + zdh, zc - zdh, z_b):
                _hline(z, ROOT.kWhite)
            hole_xs = dv.get("hole_x_thgem_cm") or [0.0]
            for c in range(-ncells, ncells + 1):
                for xh in hole_xs:
                    xcen = xh + c * cell_x
                    for xw in (xcen - r, xcen + r):
                        _vline(xw, zc - zdh, zc + zdh, ROOT.kWhite)
                    for xw in (xcen - rcu, xcen + rcu):
                        _vline(xw, zc + zdh, z_t, ROOT.kWhite)
                        _vline(xw, z_b, zc - zdh, ROOT.kWhite)

        # The mesh.
        mg = dv.get("mesh") or {}
        mz_t, mz_b = mg.get("z_top_cm"), mg.get("z_bot_cm")
        if mz_t is not None and mz_b is not None:
            _hline(mz_t, ROOT.kGreen + 2)
            _hline(mz_b, ROOT.kGreen + 2)
            xs = mg.get("wire_x_cm") or [0.0]
            if mg.get("model") == "woven":
                # The dashed line is the upper layer's *plane*: this slice runs
                # along those wires, so there is nothing to draw in x for them.
                _hline(mg.get("z_upper_cm", mz_t), ROOT.kGreen + 2, 1, 2)
                r_mw = mg.get("r_wire_cm", 25e-4)
                z_lo = mg.get("z_lower_cm", mz_b)
                for c in range(-ncells, ncells + 1):
                    for xw in xs:
                        xc = xw + c * cell_x
                        if not (x0 <= xc <= x1):
                            continue
                        ln = ROOT.TLine(xc, z_lo - 3 * r_mw, xc, z_lo + 3 * r_mw)
                        ln.SetLineColor(ROOT.kGreen + 2)
                        ln.SetLineWidth(3)
                        lines.append(ln)
            else:
                r_a = mg.get("r_aperture_cm", 0.01)
                for c in range(-ncells, ncells + 1):
                    for xa in xs:
                        xc = xa + c * cell_x
                        for xw in (xc - r_a, xc + r_a):
                            _vline(xw, mz_b, mz_t, ROOT.kGreen + 2)

        # Cathode wires: one per cell, drawn as a short vertical tick at the wire
        # plane (the map slice cuts each wire lengthwise, so it is a point in x).
        z_wire = dv.get("z_wire_cm")
        r_wire = dv.get("r_wire_cm", 25e-4)
        if z_wire is not None and zmin <= z_wire <= zmax:
            _hline(z_wire, ROOT.kCyan + 1, 2, 2)
            for c in range(-ncells, ncells + 1):
                xw = c * cell_x
                if x0 <= xw <= x1:
                    # A wire is thinner than a pixel here; mark it with a tick
                    # spanning a few radii so it is visible at all.
                    ln = ROOT.TLine(xw, z_wire - 4 * r_wire, xw, z_wire + 4 * r_wire)
                    ln.SetLineColor(ROOT.kCyan + 1)
                    ln.SetLineWidth(3)
                    lines.append(ln)

        # The anode pad bounds the map in z.  It is always present here — a mesh
        # cannot terminate the volume — so it is drawn unconditionally.
        _hline(dv.get("z_anode_cm", zmin), ROOT.kRed - 4, 3)
        return lines

    def _axis_profile(self, h2, name, ytitle):
        """On-hole-axis profile (the x = 0 column of the map) as a TGraph."""
        import ROOT  # noqa: PLC0415
        ib = h2.GetXaxis().FindBin(0.0)
        n  = h2.GetNbinsY()
        z  = np.empty(n, "f8"); y = np.empty(n, "f8")
        for j in range(1, n + 1):
            z[j - 1] = h2.GetYaxis().GetBinCenter(j)
            y[j - 1] = h2.GetBinContent(ib, j)
        g = ROOT.TGraph(n, z, y)
        g.SetName(name)
        g.SetTitle(f"On hole axis;z [cm];{ytitle}")
        return g

    def _redraw_field(self):
        d = self._field_maps
        if not d:
            return
        idx = self.ef_quantity.currentIndex()
        if idx == 1:
            h2, prof = d["pot"], d["axV"]
        elif idx == 2 and d.get("ez"):
            h2 = d["ez"]; prof = self._axis_profile(h2, "g_axis_ez", "E_{z} [kV/cm]")
        elif idx == 3 and d.get("ex"):
            h2 = d["ex"]; prof = self._axis_profile(h2, "g_axis_ex", "E_{x} [kV/cm]")
        elif idx == 4 and d.get("mag_yz"):
            # The y–z slice runs along a wire; its profile is the plate-2 axis
            # only in name, so pair it with the same on-axis |E| for comparison.
            h2, prof = d["mag_yz"], d["axE"]
        elif idx == 5 and d.get("pot_yz"):
            h2, prof = d["pot_yz"], d["axV"]
        else:
            h2, prof = d["mag"], d["axE"]
        # A y–z slice is along the wires, so the x-tiling and the x-based stack
        # overlay do not apply to it.
        along_wire = idx in (4, 5)
        try:
            self._draw_field_canvas(
                "_efield_root_canvas", "thgem_mesh_efield", "THGEM + mesh E-Field",
                h2, prof, self.ef_cmap.currentText(), self._efield_objects,
                overlay=not along_wire)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] E-field canvas error: {exc}")

    def _redraw_wfield(self):
        maps = self._weighting_maps
        if not maps:
            return
        eid = self.wf_electrode.currentText()
        if eid == "all electrodes":
            self._draw_wfield_overlay(maps)
            return
        d = maps.get(eid)
        if d is None:
            self.append_log(f"[GUI] This run did not read out '{eid}' — no weighting map.")
            return
        want_mag = self.wf_quantity.currentIndex() == 1
        h2   = d["wmag"] if want_mag else d["wpot"]
        prof = None if want_mag else d["axW"]
        try:
            self._draw_field_canvas(
                "_wfield_root_canvas", "thgem_mesh_wfield",
                f"THGEM + mesh {eid} Weighting Field",
                h2, prof, self.wf_cmap.currentText(), self._wfield_objects)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[GUI] Weighting-field canvas error: {exc}")

    def _draw_wfield_overlay(self, maps):
        """Overlay every read-out electrode's on-axis weighting potential W(z).

        Each should peak at its own place in the stack — the direct check that the
        signals are attributed to the right electrodes.
        """
        import ROOT  # noqa: PLC0415
        ROOT.gROOT.SetBatch(False)
        canvas = self._ensure_canvas("_wfield_root_canvas", "thgem_mesh_wfield",
                                     "THGEM + mesh Weighting Potentials", 1000, 620)
        canvas.cd()
        canvas.Clear()
        self._wfield_objects.clear()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gPad.SetGrid()
        leg = ROOT.TLegend(0.68, 0.68, 0.99, 0.93)
        leg.SetBorderSize(0)
        frame = None
        for eid in ELECTRODE_IDS:
            d = maps.get(eid)
            if d is None or d.get("axW") is None:
                continue
            g = d["axW"].Clone()
            try:
                g.SetDirectory(0)
            except AttributeError:
                pass
            g.SetLineColor(electrode_color(eid))
            g.SetLineWidth(2)
            g.SetTitle("On-axis weighting potentials;z [cm];W")
            g.Draw("AL" if frame is None else "L same")
            if frame is None:
                g.GetYaxis().SetRangeUser(-0.1, 1.05)
                frame = g
            leg.AddEntry(g, ELECTRODE_LABELS.get(eid, eid), "L")
            self._wfield_objects.append(g)
        leg.Draw()
        self._wfield_objects.append(leg)
        canvas.Modified()
        canvas.Update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._runner: SimRunner | None = None
        self._last_loaded_config_path: str | None = None

        binary_ok = BINARY.exists()
        suffix = "" if binary_ok else "  ⚠ Binary not found — build first"
        self.setWindowTitle(f"THGEM + mesh Simulation{suffix}")
        self.resize(1200, 740)

        # ── Toolbar ───────────────────────────────────────────────────────
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_run  = tb.addAction("▶  Run")
        self.act_stop = tb.addAction("■  Stop")
        tb.addSeparator()
        self.act_load = tb.addAction("Load Config")
        self.act_save = tb.addAction("Save Config")

        self.act_run.setEnabled(binary_ok)
        self.act_stop.setEnabled(False)

        self.act_run.triggered.connect(self._on_run)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_load.triggered.connect(self._on_load_config)
        self.act_save.triggered.connect(self._on_save_config)

        # ── Central splitter ─────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        self.config_panel  = ConfigPanel()
        # Open on the shipped default configuration so the GUI is a single source
        # of truth with config/default_thgem_mesh.json (avoids widget defaults drifting
        # from the tuned config).  Falls back to the widget defaults if it is absent.
        _default_cfg = PROJ_DIR / "config" / "default_thgem_mesh.json"
        if _default_cfg.exists():
            try:
                with open(_default_cfg) as _f:
                    self.config_panel.load_from_dict(json.load(_f))
            except Exception:  # noqa: BLE001
                pass
        self.results_panel = ResultsPanel()
        # Give the E-Field tab live access to the geometry for its overlay.
        self.results_panel.config_panel = self.config_panel

        splitter.addWidget(self.config_panel)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 850])

        self.setCentralWidget(splitter)

        # ── Status bar ────────────────────────────────────────────────────
        if binary_ok:
            self.statusBar().showMessage(f"Ready — binary: {BINARY}")
        else:
            self.statusBar().showMessage(
                f"Binary not found at {BINARY} — run cmake to build"
            )

    # ── Toolbar actions ───────────────────────────────────────────────────

    def _on_run(self):
        if not BINARY.exists():
            QMessageBox.critical(
                self, "Binary not found",
                f"thgem_mesh_sim binary not found at:\n{BINARY}\n\n"
                "Build the project first:\n"
                "  cmake -S projects/double_THGEM -B projects/double_THGEM/build ...\n"
                "  cmake --build projects/double_THGEM/build -j4"
            )
            return

        cfg     = self.config_panel.to_config_dict()
        out_str = self.config_panel.out_dir.text().strip() or "results"

        # Resolve relative paths from the tgc project directory
        out_path = Path(out_str)
        if not out_path.is_absolute():
            out_path = (PROJ_DIR / out_str).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Build the run subfolder name: yymmdd_hh-mm__ + user tag or auto params
        date_pfx = datetime.now().strftime("%y%m%d_%H-%M")
        tag      = self.config_panel.run_name.text().strip()
        if tag:
            subdir = f"{date_pfx}__{tag}"
        else:
            # Mirror BuildRunFolderName in thgem_mesh_sim.cc: the THGEM voltage and
            # the amplification field, then the event count.  .get() with defaults
            # so a hand-edited config can never crash the run before it starts.
            fl = cfg.get("fields", {})
            v1 = int(fl.get("delta_v_thgem_V", 0))
            ea = _file_safe_number(fl.get("e_amplification_kvcm", 0.0))
            n  = cfg.get("simulation", {}).get("n_events", 0)
            # n_events 0 is a field-only run; "__field" reads better than "__n0".
            suffix = "field" if n == 0 else f"n{n}"
            subdir = f"{date_pfx}__dV{v1}V_Ea{ea}__{suffix}"

        self.results_panel.clear_log()
        self.results_panel.setCurrentIndex(0)   # show Log tab while running
        self.statusBar().showMessage("Running…")

        self._runner = SimRunner(cfg, str(out_path), run_name=subdir)
        self._runner.log_line.connect(self.results_panel.append_log)
        self._runner.finished.connect(self._on_run_finished)
        self._runner.failed.connect(self._on_run_failed)
        self._runner.stopped.connect(self._on_run_stopped)

        self.act_run.setEnabled(False)
        self.act_stop.setEnabled(True)

        if self._last_loaded_config_path:
            self.results_panel.append_log(
                f"[GUI] Config based on: {self._last_loaded_config_path}"
            )
        else:
            self.results_panel.append_log("[GUI] Config from widget defaults")

        self._runner.start()

    def _on_stop(self):
        if self._runner:
            self._runner.stop()
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped by user")

    def _on_run_finished(self, run_dir: str):
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        warned = getattr(self._runner, "exit_code", 0) == SimRunner.EXIT_WARNINGS
        self.statusBar().showMessage(
            (f"Done, with field-validation warnings — output in {run_dir}"
             if warned else f"Done — output in {run_dir}"))

        self.results_panel.append_log(
            f"\n[GUI] Simulation complete{' (with warnings)' if warned else ''}."
            f"  Output: {run_dir}")

        csv_path  = str(Path(run_dir) / "summary.csv")
        root_path = str(Path(run_dir) / "thgem_mesh_sim.root")

        # A field-only run (n_events = 0, or --field-only from outside the GUI)
        # transports nothing, so there is no summary.csv and no per-event data.
        # Asking for it anyway would just log a read error and leave the user on
        # an empty Summary tab; go straight to what the run did produce.
        field_only = (self.config_panel.n_events.value() == 0
                      or not Path(csv_path).exists())

        if not field_only:
            self.results_panel.populate_table(csv_path)
            self.results_panel.draw_plots(csv_path)
        self.results_panel.load_waveform_data(root_path)
        self.results_panel.load_track_data(root_path, run_dir)
        self.results_panel.load_field_maps(root_path)
        self.results_panel.load_weighting_maps(root_path)
        self._try_load_gas_props()
        if field_only:
            self.results_panel.append_log(
                "[GUI] Field-only run: no events were transported, so the Summary, "
                "Plots, Waveforms, Integrals and 3D Tracks tabs stay empty.")
        self.results_panel.setCurrentIndex(
            self.results_panel._tab_efield if field_only
            else self.results_panel._tab_summary)
        self.results_panel._save_plots_root(run_dir)

    def _try_load_gas_props(self):
        """Load Magboltz properties CSV if it exists for the current gas config."""
        gas_cfg = self.config_panel.to_config_dict().get("gas", {})
        props_path = GAS_DIR / derive_gas_props_filename(gas_cfg)
        if props_path.exists():
            self.results_panel.draw_gas_props(str(props_path))

    def _on_run_failed(self, msg: str):
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.statusBar().showMessage(f"Failed: {msg}")
        self.results_panel.append_log(f"\n[GUI] ERROR: {msg}")
        QMessageBox.warning(self, "Simulation failed", msg)

    def _on_run_stopped(self):
        """User clicked Stop (or closed the window) — not an error."""
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.statusBar().showMessage("Run stopped")
        self.results_panel.append_log("\n[GUI] Run stopped by user.")

    # ── Config load/save ──────────────────────────────────────────────────

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", str(PROJ_DIR / "config"),
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
            self.config_panel.load_from_dict(d)
            self._last_loaded_config_path = path
            self.statusBar().showMessage(f"Config loaded from {path}")
            self._try_load_gas_props()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load failed", str(exc))

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", str(PROJ_DIR / "config"),
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            cfg = self.config_panel.to_config_dict()
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self.statusBar().showMessage(f"Config saved to {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))

    # ── Window lifecycle ──────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._runner and self._runner.isRunning():
            self._runner.stop()
            self._runner.wait(3000)

        # Stop ROOT timer and close all ROOT TCanvas windows before Qt tears
        # down its macOS Cocoa layer — prevents "drawable not found" crash.
        rp = self.results_panel
        rp._root_timer.stop()
        try:
            import ROOT  # noqa: PLC0415
            for _canvas in [rp._root_canvas, rp._charge_canvas,
                            rp._tracks_canvas, rp._efield_root_canvas,
                            rp._wfield_root_canvas, rp._gas_canvas]:
                try:
                    if (_canvas is not None and
                            ROOT.gROOT.GetListOfCanvases()
                                      .FindObject(_canvas.GetName())):
                        _canvas.Close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("THGEM + mesh Simulation")
    win = MainWindow()
    win.show()
    code = app.exec_()
    # os._exit bypasses Python/ROOT atexit destructors that crash on macOS
    # when ROOT's Cocoa layer outlives Qt's autorelease pool.
    os._exit(code)


if __name__ == "__main__":
    main()
