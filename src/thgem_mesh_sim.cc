// Garfield++ THGEM + micromegas-mesh cascade simulation
//
// Geometry : a THGEM feeding a mesh, under a wire cathode.  The electrodes lie
//            in the x–y plane, the drift axis is z and electrons drift down (−z):
//
//                ○   ○   ○   ○   ○      wire cathode  V_wire   (wires ∥ y)
//                      drift gap
//              ▓▓▓▓░░▓▓▓▓▓▓░░▓▓▓▓       THGEM  (top Cu / dielectric / bot Cu)
//                     transfer gap
//              ╪═╪═╪═╪═╪═╪═╪═╪═╪═       mesh  V_mesh  (woven wires, or a
//                  amplification gap          perforated sheet)
//              ═══════════════════      anode pad  0 V
//
//            The mesh + amplification gap + anode is a micromegas.  The mesh is
//            a single conductor and so an equipotential — it has no voltage
//            across *itself*, unlike a THGEM's two copper faces — but the gap
//            below it is an ordinary two-electrode gap, set by
//            delta_v_mesh_anode_V.  The field E = ΔV / d_amp is derived.
//            The anode is therefore mandatory here — unlike the double-THGEM
//            sibling, whose second plate's bottom copper could terminate the
//            volume, a mesh is 35-65 % open and cannot.
//
//            One periodic unit cell is modelled: cellX = holes_per_wire ×
//            hole_pitch (one wire, N holes), cellY = hole_pitch.  The cell is
//            tiled by neBEM with translation periodicity.  The mesh lattice has
//            to fit that cell a whole number of times, so its pitch is snapped
//            to hole_pitch / k; since cellX is always an integer multiple of
//            cellY, one snap satisfies both periods.
// Field    : solved in-process by ComponentNeBem3d (native boundary-element
//            method) — neither the hole field nor the mesh field has a usable
//            closed form here.  Electrode potentials are derived from the
//            physics fields, with the anode as the reference:
//              V_anode     = 0
//              V_mesh      = V_anode      − ΔV_mesh→anode
//              V_thgem_bot = V_mesh       − E_transfer · d_transfer
//              V_thgem_top = V_thgem_bot  − ΔV_THGEM
//              V_wire      = V_thgem_top  − E_drift    · d_drift
//            so electrons drift toward the most-positive bottom electrode.
// Gas      : Ar:CO2 (configurable), 1 atm, 293.15 K, transported by Magboltz.
// Source   : N = E/W primary electrons placed at a configurable height in the
//            drift gap below the wire plane; a single representative electron
//            is transported by AvalancheMicroscopic and the result scaled by N.
// Readout  : any subset of five electrodes — wire_cathode, thgem_top,
//            thgem_bottom, mesh, anode.  Induced signals come from each
//            electrode's true Shockley–Ramo weighting potential, solved
//            natively by neBEM and sampled onto the transport grid.  Every mesh
//            wire (or aperture tile) shares the "mesh" label, and neBEM groups
//            same-labelled solids into one readout group, so the whole mesh
//            costs a single weighting-field solve.
//
// Ported from projects/double_THGEM/src/dthgem_sim.cc, whose second THGEM this
// replaces with a mesh; the gas, avalanche, signal, track and I/O machinery is
// shared, and the wire cathode is unchanged.
//
#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TRandom.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Garfield/AvalancheMC.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/Component.hh"
#include "Garfield/ComponentGrid.hh"
#include "Garfield/ComponentNeBem3d.hh"
#include "Garfield/FundamentalConstants.hh"
#include "Garfield/GarfieldConstants.hh"
#include "Garfield/GeometrySimple.hh"
#include "Garfield/MediumConductor.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/MediumPlastic.hh"
#include "Garfield/Random.hh"
#include "Garfield/RandomEngineRoot.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/Solid.hh"
#include "Garfield/SolidBox.hh"
#include "Garfield/SolidHole.hh"
#include "Garfield/SolidWire.hh"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace {

using Garfield::AvalancheMC;
using Garfield::AvalancheMicroscopic;
using Garfield::ComponentGrid;
using Garfield::ComponentNeBem3d;
using Garfield::GeometrySimple;
using Garfield::MediumConductor;
using Garfield::MediumMagboltz;
using Garfield::MediumPlastic;
using Garfield::Sensor;
using Garfield::Solid;
using Garfield::SolidBox;
using Garfield::SolidHole;
using Garfield::SolidWire;
using json = nlohmann::json;

// ─── Readout electrode identifiers ────────────────────────────────────────────
// These strings are the Solid labels neBEM solves weighting fields for, the
// Sensor electrode names, and the ROOT branch-name prefixes — one vocabulary
// shared end to end, including by the GUI.

constexpr const char* kElecWire     = "wire_cathode";
constexpr const char* kElecThgemTop = "thgem_top";
constexpr const char* kElecThgemBot = "thgem_bottom";
constexpr const char* kElecMesh     = "mesh";
constexpr const char* kElecAnode    = "anode";

// There is one THGEM here, so its faces carry no index: a "thgem1_top" would
// imply a second plate that does not exist.  The mesh is a single conductor —
// every one of its wires (or aperture tiles) carries this one label, and neBEM
// groups solids sharing a label into a single weighting readout group, so the
// whole electrode gets exactly one weighting-field solve.
const std::array<std::string, 5> kAllElectrodeIds{
    kElecWire, kElecThgemTop, kElecThgemBot, kElecMesh, kElecAnode};

// ─── Configuration structs ────────────────────────────────────────────────────

// The THGEM plate.  The hole *pitch* is deliberately not here: it sets the
// periodic cell, which the mesh lattice below must also fit into, so it lives in
// GeometryConfig::holePitchUm.  There is no lattice offset either — there is no
// second plate to stagger against, and the mesh carries its own offsets.
struct PlateConfig {
  double holeDiameterUm    = 500.0;   // hole diameter [µm]
  double plateThicknessUm  = 400.0;   // dielectric foil thickness [µm]
  double copperThicknessUm = 35.0;    // copper cladding thickness per face [µm]
  double rimUm             = 0.0;     // copper etched back from the hole edge [µm]
  std::string dielectric   = "fr4";   // "fr4" | "kapton"
};

// The mesh electrode: a micromegas micromesh, or a thick coarse mesh.  Two
// models, because the two ends of the pitch range are physically different
// objects and cost wildly different amounts to solve:
//
//   "woven"      two orthogonal layers of wires.  neBEM turns a SolidWire into a
//                single line-charge primitive, so a whole woven mesh costs a
//                handful of boundary elements.  This is the model that reaches
//                true micromegas pitch (~50-65 µm, 400/500 LPI).
//   "perforated" a thin conductor sheet with a lattice of apertures (an
//                electroformed micromesh, or a thick calendered mesh).  Each
//                aperture is a SolidHole ≈ 390 elements, so the element count
//                grows as (cell/pitch)² and this model is only viable down to
//                roughly hole_pitch / 3.
struct MeshConfig {
  std::string model      = "woven";   // "woven" | "perforated"
  // Requested lattice pitch [µm].  Snapped in ComputeGeom to hole_pitch / k so
  // that a whole number of mesh cells fits the periodic cell; the snapped value
  // is what is simulated, echoed and keyed on.
  double pitchUm         = 250.0;
  double wireDiameterUm  = 50.0;      // woven only
  // Mesh thickness [µm].  0 = auto: woven → 2 × wire diameter (the two layers
  // tangent, which is the physical weave).  Required (> 0) for "perforated".
  double thicknessUm     = 0.0;
  double apertureUm      = 300.0;     // perforated: aperture size across corners
  int    apertureSectors = 4;         // perforated: 2 = square, 3 = octagon, 4 = 12-gon
  double offsetXUm       = 0.0;       // lattice stagger in x [µm]
  double offsetYUm       = 0.0;       // lattice stagger in y [µm]
  // Per-solid neBEM discretisation target [µm]; 0 = auto (use
  // geometry.target_element_size_um).  See geometry.thgem_element_size_um for
  // why this exists and how the two are meant to be used together.
  double elementSizeUm   = 0.0;
};

struct GeometryConfig {
  double holePitchUm = 1000.0;  // THGEM hole lattice pitch, x and y [µm]
  PlateConfig thgem;
  MeshConfig  mesh;

  // Wire cathode: wires run along y, spaced along x.  The wire pitch is an
  // integer multiple of the hole pitch so that one periodic cell holds exactly
  // one wire and holesPerWire holes per plate — the cell x-period is
  // holesPerWire × holePitchUm.
  double wireDiameterUm  = 50.0;
  int    holesPerWire    = 1;
  bool   wireBetweenHoles = false;   // shift the hole lattice by half a pitch so
                                     // the wire sits between holes, not over one

  // Gaps, surface to surface [mm]
  double driftGapMm         = 2.0;   // wire plane      → THGEM top copper
  double transferGapMm      = 1.0;   // THGEM bottom Cu → mesh top surface
  double amplificationGapMm = 0.2;   // mesh bottom     → anode pad
  // There is no anode toggle.  The sibling double-THGEM could drop its anode
  // because THGEM 2's bottom copper is a continuous conductor that legitimately
  // terminates the volume; a mesh is 35-65 % open and cannot.  Without an anode
  // there is no amplification gap and no second stage at all.

  // neBEM discretisation / solver controls (rarely touched; coarser = faster,
  // less accurate).  Note minElements/maxElements dominate: neBEM's
  // NbOfSegments() is clamp(floor(length / target), min, max), so on primitives
  // shorter than min × target the target size does nothing at all.
  double targetElementSizeUm = 120.0;  // target boundary-element size [µm]
  // Per-solid discretisation target for the THGEM [µm]; 0 = auto.  Because the
  // min/max element clamp is *global*, a per-solid level can only make a solid
  // coarser, never finer.  So the way to spend elements on the mesh and not on
  // the plate is inverted: raise max_elements, set a *large*
  // thgem_element_size_um (pinning the plate at min_elements) and a *small*
  // mesh.element_size_um (driving the wires to max_elements).
  double thgemElementSizeUm  = 0.0;
  int    minElements         = 2;      // min elements along a primitive edge
  int    maxElements         = 4;      // max elements along a primitive edge
  // Refuse to start a solve projected to exceed this many boundary elements.
  // neBEM inverts a dense N×N matrix: the cost is O(N³) and the stored inverse
  // is N² doubles, so an accidentally fine "perforated" mesh goes from minutes
  // to days without any intermediate warning.
  int    maxElementsBudget   = 8000;
  int    periodicCopies      = 9;      // neBEM periodic copies; the tiled wire /
                                       // anode patches only approximate infinite
                                       // structures, and too few copies leaves the
                                       // on-axis drift field *reversed* mid-gap.
                                       // The Ez-sign check in ValidateField() is
                                       // what tells you this number is too small.
  int    holeSectors         = 4;      // hole circle approximation (2=square, 3=octagon, …)

  // The neBEM field is sampled once onto this grid (fast trilinear interpolation
  // during the avalanche).  It spans exactly one periodic cell in x and y, and
  // the full electrode stack in z.
  //
  // This grid, not neBEM, is what bounds the geometry here: ComponentGrid is
  // uniform-only, so it must resolve a 25-50 µm mesh wire in x and y *and* a
  // sub-millimetre amplification gap in z while still spanning the drift gap.
  // If it cannot resolve the mesh, every electron passes through it, the
  // measured electron transparency is exactly 1.00, and the run looks perfectly
  // healthy while being meaningless.  ValidateField() checks both budgets.
  int gridNx = 61;    // nodes across the cell in x (cellX = holesPerWire × pitch)
  int gridNy = 61;    // nodes across the cell in y (cellY = pitch)
  int gridNz = 301;   // nodes along z (anode → wire plane)
};

struct FieldConfig {
  double eDriftKvcm    = 0.5;     // *average* drift-gap field [kV/cm] — the local
                                  // field at a wire surface is far higher
  double deltaVThgemV  = 1200.0;  // voltage across the THGEM (top→bottom Cu) [V]
  double eTransferKvcm = 1.0;     // THGEM → mesh gap field [kV/cm]
  // Voltage across the amplification gap, mesh -> anode [V].
  //
  // Deliberately not "delta_v_mesh": a THGEM is two copper faces with a
  // dielectric between them, so delta_v_thgem_V is a drop across one object,
  // whereas a mesh is a single conductor and therefore an equipotential —
  // asking for "the voltage across the mesh" is not a question.  What the
  // second stage has is an ordinary two-electrode gap, and the name says which
  // two electrodes it is measured between.
  //
  // The gap is set by its voltage rather than its field because that is what a
  // supply is set to, and because it is the quantity that stays put when the
  // geometry moves: change amplification_gap_mm and the mesh potential is
  // unchanged.  The field E = dV / d_amp is derived (MeshGeom::eAmpKvcm) and is
  // what the physics runs on — the micromegas literature's 40-60 kV/cm over
  // 64-320 µm is 256-1920 V, and 512-768 V at the usual 128 µm gap.
  double deltaVMeshAnodeV = 900.0;
};

// Which electrodes are read out.  Every extra electrode costs one neBEM
// weighting-field sampling pass and one cached grid (tens of MB), so the
// default is the single collecting electrode.
struct ReadoutConfig {
  // nullopt → "anode", the collecting electrode.
  std::optional<std::vector<std::string>> electrodes;
};

struct SourceConfig {
  double energyKeV = 5.9;
  // Height of the primary electrons above the THGEM's top copper, in the drift
  // gap [mm] — the same reference as the sibling THGEM projects.
  // nullopt → uniform random over the drift gap per event.
  std::optional<std::vector<double>> fixedDistMm = std::vector<double>{1.0};
  // Fixed x-position within the unit cell [cm]; nullopt → random over the cell.
  // (y is always sampled uniformly over the cell.)
  std::optional<std::vector<double>> fixedXCmList;
};

struct GasConfig {
  std::string gas1           = "ar";
  double      frac1          = 70.0;
  std::string gas2           = "co2";
  std::string ionSpecies     = "co2";
  double temperatureK        = 293.15;
  double pressureTorr        = 760.0;
  bool   enablePenning       = true;
  int    nCollisions         = 2;
  double maxElectronEnergyEV = 2000.0;  // EFINAL of the Magboltz table (keys the .gas file name)
  // Ceiling of the microscopic collision-rate table.  AvalancheMicroscopic samples
  // every transport step against the *maximum* collision rate over the whole energy
  // grid (MediumMagboltz::m_cfNull), rejecting the rest as null collisions.  A ceiling
  // far above the energies the electrons actually reach therefore costs a proportional
  // number of wasted steps.  Applied after LoadGasFile, which rebuilds the rate table
  // without touching the transport tables.
  double transportMaxEnergyEV = 200.0;
  int    nFieldPoints        = 10;
  double eFieldMinVcm        = 100.0;
  double eFieldMaxVcm        = 400000.0;
  double wValueEV            = 26.0;
};

struct SimulationConfig {
  std::size_t nEvents          = 100;
  std::size_t maxAvalancheSize = 200000;
  double      timeWindowNs     = 200.0;
  double      timeStepNs       = 0.5;
  bool        enableIonDrift   = false;
  bool        storeDriftLines  = false;
  double      ionMaxStepUm     = 5.0;
  // Wall-clock safety bound for ion drift (AvalancheMC).  Ions are ~1000x slower
  // than electrons; distance-stepping bounds a normal ion by geometry, and this
  // time window is the backstop that terminates an ion trapped at a field
  // stagnation point.
  double      ionTimeWindowNs  = 1.0e6;
  // Cap on the number of avalanche ions actually back-drifted per event (0 =
  // no cap).  Ions drift up, away from the anode, so they contribute ~nothing
  // to its signal — but transporting all of them dominates the runtime at the
  // high gain a cascade reaches.  Raise it (or set 0) for ion-backflow studies.
  std::size_t maxIonsDrifted   = 200;
  int         randomSeed       = 0;
};

// Front-end amplifier (CIVIDEC C2-TCT broadband transimpedance current amp): turns a
// binned induced current [fC/ns ≡ µA] into an output voltage [mV].  Datasheet defaults.
struct AmplifierConfig {
  bool   enable            = false;
  double gainDb            = 40.0;    // voltage gain [dB] (40 dB = ×100)
  double inputImpedanceOhm = 50.0;    // input impedance [Ω]
  double bandwidthHighHz   = 2.0e9;   // upper −3 dB edge → low-pass τ = 1/(2π f)
  double outputSampleNs    = 0.0;     // finite acquisition aperture / boxcar [ns]
};

// ─── 3D-visualisation display limits ─────────────────────────────────────────
constexpr std::size_t kMaxDispIonPaths      = 100;  // ion drift paths saved per event
constexpr std::size_t kMaxDispCloudPts      = 500;  // avalanche-cloud points saved per event
constexpr std::size_t kMaxDispElectronPaths = 200;  // avalanche e⁻ drift lines saved per event

// ─── Field / weighting map dump resolution ───────────────────────────────────
// Two slices are dumped: x–z (across the wires, through a hole axis) and y–z
// (along a wire, through the same hole axis).  Unlike the single-plate sibling
// these are *not* equivalent — the wires break the x/y symmetry.  Shared by the
// E-field and weighting maps so both land on an identical grid.  nx/ny are odd
// so the centre column sits exactly on the reference hole axis.
constexpr int kMapNx = 81;
constexpr int kMapNy = 81;
constexpr int kMapNz = 241;

// Lateral half-span of the Sensor drift area, in periodic cells.  The field is
// supplied for any cell by the grid's periodicity, so this only has to be wide
// enough that diffusion and the avalanche cone stay inside it.
constexpr double kSensorSpanCells = 3.0;

struct Config {
  GeometryConfig   geometry;
  FieldConfig      fields;
  ReadoutConfig    readout;
  SourceConfig     source;
  GasConfig        gas;
  SimulationConfig simulation;
  AmplifierConfig  amplifier;
};

// ─── Per-electrode and per-distance summaries ────────────────────────────────

struct ElectrodeStats {
  std::string id;
  double meanChargeFC = 0.;
  double rmsChargeFC  = 0.;
  double semChargeFC  = 0.;
};

struct DistanceSummary {
  std::optional<double> distanceMm;    // nullopt = random per event
  std::optional<double> xPositionCm;   // nullopt = random per event
  std::size_t nEvents             = 0;
  std::size_t nInteracted         = 0;
  double      interactionFraction = 0.;
  std::vector<ElectrodeStats> electrodes;
  double      meanPrimaryElectrons = 0.;
  double      meanAvalancheSize    = 0.;
  double      rmsAvalancheSize     = 0.;
  double      semAvalancheSize     = 0.;
  // Cascade diagnostics: electrons *born* in each multiplying stage, and the
  // per-stage gains they imply.  gainThgem = 1 + n_born(THGEM); gainAmp is the
  // multiplication seen by the charge that threaded the mesh.
  double      meanBornThgem        = 0.;
  double      meanBornTransfer     = 0.;   // born between the THGEM and the mesh
  double      meanBornAmp          = 0.;
  double      meanGainThgem        = 0.;
  double      meanGainAmp          = 0.;
  double      meanMeshTransparency = 0.;   // fraction of THGEM charge entering the amp gap
};
// ─── Utility ──────────────────────────────────────────────────────────────────

std::string FormatNumber(double v, int precision = 4) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << v;
  std::string s = ss.str();
  // Trim trailing zeros only *after* a decimal point.  With precision 0 the
  // fixed form has no point at all, and an unguarded trim eats the number's own
  // trailing zeros — 600 prints as "6", 530 as "53".
  if (s.find('.') != std::string::npos) {
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
  }
  return s.empty() ? "0" : s;
}

std::string FileSafeNumber(double v) {
  std::string s = FormatNumber(v);
  std::replace(s.begin(), s.end(), '.', 'p');
  std::replace(s.begin(), s.end(), '-', 'm');
  return s;
}

std::string DeriveGasFileName(const GasConfig& g) {
  auto I = [](double v) {
    return std::to_string(static_cast<long long>(std::llround(v)));
  };
  const int    efKv    = static_cast<int>(std::llround(g.eFieldMaxVcm / 1000.0));
  const int    efMinV  = static_cast<int>(std::llround(g.eFieldMinVcm));
  const double frac2 = 100.0 - g.frac1;
  const std::string prefix = g.gas1 + I(g.frac1) + "_" + g.gas2 + "_" + I(frac2);
  return prefix
       + "_T"  + I(g.temperatureK)
       + "_P"  + I(g.pressureTorr)
       + "_Ee" + I(g.maxElectronEnergyEV)
       + "_Ef" + std::to_string(efMinV) + "v-" + std::to_string(efKv) + "k"
       + "_n"  + std::to_string(g.nFieldPoints)
       + "_c"  + std::to_string(g.nCollisions)
       + (g.enablePenning ? "_pen" : "_nopen")
       + ".gas";
}

void EnsureDirectory(const fs::path& p) { fs::create_directories(p); }

/// Relative permittivity of a named dielectric foil material.
double DielectricEpsR(const std::string& material) {
  if (material == "fr4") return 4.6;
  if (material == "kapton") return 3.5;
  return 4.0;  // generic plastic fallback
}

template <typename T>
double Mean(const std::vector<T>& v) {
  if (v.empty()) return 0.;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

template <typename T>
double Rms(const std::vector<T>& v, double mean) {
  if (v.size() < 2) return 0.;
  double var = 0.;
  for (const auto& x : v) { double d = static_cast<double>(x) - mean; var += d * d; }
  return std::sqrt(var / static_cast<double>(v.size()));
}

double Sem(double rms, std::size_t n) {
  return n < 2 ? 0. : rms / std::sqrt(static_cast<double>(n));
}

// ─── JSON helpers ─────────────────────────────────────────────────────────────

[[noreturn]] void ThrowJsonTypeError(const std::initializer_list<std::string_view>& path,
                                     std::string_view expect) {
  std::string msg = "JSON error at '";
  bool first = true;
  for (auto p : path) { if (!first) msg += '.'; first = false; msg += std::string(p); }
  msg += "': expected " + std::string(expect) + ".";
  throw std::runtime_error(msg);
}

const json* FindMember(const json& obj, std::string_view key,
                       const std::initializer_list<std::string_view>& path) {
  if (!obj.is_object()) ThrowJsonTypeError(path, "an object");
  auto it = obj.find(std::string(key));
  return it == obj.end() ? nullptr : &(*it);
}

const json* FindSection(const json& obj, std::string_view key) {
  auto* p = FindMember(obj, key, {key});
  if (p && !p->is_object()) ThrowJsonTypeError({key}, "an object");
  return p;
}

double ReadDouble(const json& obj, std::string_view sec, std::string_view key, double fb) {
  auto* v = FindMember(obj, key, {sec, key});
  if (!v) return fb;
  if (!v->is_number()) ThrowJsonTypeError({sec, key}, "a number");
  return v->get<double>();
}

int ReadInt(const json& obj, std::string_view sec, std::string_view key, int fb) {
  auto* v = FindMember(obj, key, {sec, key});
  if (!v) return fb;
  if (!v->is_number_integer()) ThrowJsonTypeError({sec, key}, "an integer");
  return v->get<int>();
}

std::size_t ReadSizeT(const json& obj, std::string_view sec, std::string_view key, std::size_t fb) {
  auto* v = FindMember(obj, key, {sec, key});
  if (!v) return fb;
  if (!v->is_number_integer() && !v->is_number_unsigned())
    ThrowJsonTypeError({sec, key}, "a non-negative integer");
  auto val = v->get<long long>();
  if (val < 0) throw std::runtime_error("Expected non-negative integer at key '" + std::string(key) + "'");
  return static_cast<std::size_t>(val);
}

bool ReadBool(const json& obj, std::string_view sec, std::string_view key, bool fb) {
  auto* v = FindMember(obj, key, {sec, key});
  if (!v) return fb;
  if (!v->is_boolean()) ThrowJsonTypeError({sec, key}, "a boolean");
  return v->get<bool>();
}

std::string ReadString(const json& obj, std::string_view sec, std::string_view key,
                       const std::string& fb) {
  auto* v = FindMember(obj, key, {sec, key});
  if (!v) return fb;
  if (!v->is_string()) ThrowJsonTypeError({sec, key}, "a string");
  return v->get<std::string>();
}

json ReadJsonFile(const fs::path& p) {
  std::ifstream s(p);
  if (!s) throw std::runtime_error("Cannot open JSON file: " + p.string());
  try { return json::parse(s); }
  catch (const json::parse_error& e) {
    throw std::runtime_error("JSON parse error in '" + p.string() + "': " + e.what());
  }
}

void WriteJsonFile(const fs::path& p, const json& payload) {
  std::ofstream s(p);
  if (!s) throw std::runtime_error("Cannot write JSON file: " + p.string());
  s << std::setw(2) << payload << '\n';
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct CliOptions {
  fs::path configPath{"config/default_thgem_mesh.json"};
  fs::path outDir{"results"};
  std::string runName;                    // empty = auto-generate from config
  std::optional<double> singleDistanceMm;
  // Solve the field, sample it, dump the field/weighting maps and stop before
  // any transport.  The cheap way to validate a new geometry — a bad field
  // solve is the failure mode that costs the most time to diagnose downstream.
  bool fieldOnly = false;
};

[[noreturn]] void PrintUsageAndExit(const char* prog, int code) {
  std::ostream& out = code == 0 ? std::cout : std::cerr;
  out << "Usage: " << prog << " [options]\n"
         "  --config <path>    JSON config file (default: config/default_thgem_mesh.json)\n"
         "  --out    <dir>     Output directory (default: results)\n"
         "  --run-name <name>  Subdirectory name under --out (default: auto)\n"
         "  --distance <mm>    Run only this drift-gap height (overrides config list)\n"
         "  --field-only       Solve and dump the field maps, then exit (no transport).\n"
         "                     simulation.n_events = 0 in the config does the same.\n"
         "  --help             Show this message\n";
  std::exit(code);
}

CliOptions ParseCli(int argc, char* argv[]) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) PrintUsageAndExit(argv[0], 1);
      opts.configPath = argv[++i];
    } else if (arg == "--out") {
      if (i + 1 >= argc) PrintUsageAndExit(argv[0], 1);
      opts.outDir = argv[++i];
    } else if (arg == "--run-name") {
      if (i + 1 >= argc) PrintUsageAndExit(argv[0], 1);
      opts.runName = argv[++i];
    } else if (arg == "--distance") {
      if (i + 1 >= argc) PrintUsageAndExit(argv[0], 1);
      opts.singleDistanceMm = std::stod(argv[++i]);
    } else if (arg == "--field-only") {
      opts.fieldOnly = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsageAndExit(argv[0], 0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return opts;
}

// ─── Config loading ───────────────────────────────────────────────────────────

// Read the THGEM plate's keys, each defaulting to the matching value in `fb`.
PlateConfig ReadPlate(const json& parent, const char* secName, const PlateConfig& fb) {
  PlateConfig p = fb;
  const auto* o = FindMember(parent, secName, {"geometry", secName});
  if (!o) return p;
  if (!o->is_object()) ThrowJsonTypeError({"geometry", secName}, "an object");
  p.holeDiameterUm    = ReadDouble(*o, secName, "hole_diameter_um",    p.holeDiameterUm);
  p.plateThicknessUm  = ReadDouble(*o, secName, "plate_thickness_um",  p.plateThicknessUm);
  p.copperThicknessUm = ReadDouble(*o, secName, "copper_thickness_um", p.copperThicknessUm);
  p.rimUm             = ReadDouble(*o, secName, "rim_um",              p.rimUm);
  p.dielectric        = ReadString(*o, secName, "dielectric_material", p.dielectric);
  return p;
}

// Read the mesh section.  Every key is optional; which of them actually matter
// depends on `model`, and LoadConfig validates only the ones the chosen model
// uses so that switching models does not require deleting the other's keys.
MeshConfig ReadMesh(const json& parent, const MeshConfig& fb) {
  MeshConfig m = fb;
  const auto* o = FindMember(parent, "mesh", {"geometry", "mesh"});
  if (!o) return m;
  if (!o->is_object()) ThrowJsonTypeError({"geometry", "mesh"}, "an object");
  m.model           = ReadString(*o, "mesh", "model",             m.model);
  m.pitchUm         = ReadDouble(*o, "mesh", "pitch_um",          m.pitchUm);
  m.wireDiameterUm  = ReadDouble(*o, "mesh", "wire_diameter_um",  m.wireDiameterUm);
  m.thicknessUm     = ReadDouble(*o, "mesh", "thickness_um",      m.thicknessUm);
  m.apertureUm      = ReadDouble(*o, "mesh", "aperture_um",       m.apertureUm);
  m.apertureSectors = ReadInt   (*o, "mesh", "aperture_sectors",  m.apertureSectors);
  m.offsetXUm       = ReadDouble(*o, "mesh", "offset_x_um",       m.offsetXUm);
  m.offsetYUm       = ReadDouble(*o, "mesh", "offset_y_um",       m.offsetYUm);
  m.elementSizeUm   = ReadDouble(*o, "mesh", "element_size_um",   m.elementSizeUm);
  return m;
}

// True if `id` names one of the five electrodes in this geometry.  Unlike the
// double-THGEM sibling every electrode always exists — the anode is mandatory.
bool IsKnownElectrode(const std::string& id) {
  return std::find(kAllElectrodeIds.begin(), kAllElectrodeIds.end(), id) !=
         kAllElectrodeIds.end();
}

Config LoadConfig(const fs::path& path) {
  if (!fs::exists(path))
    throw std::runtime_error("Configuration file not found: " + path.string());
  const json root = ReadJsonFile(path);
  if (!root.is_object()) ThrowJsonTypeError({"<root>"}, "a JSON object");

  Config cfg;

  if (const auto* g = FindSection(root, "geometry")) {
    auto& gm = cfg.geometry;
    gm.holePitchUm      = ReadDouble(*g, "geometry", "hole_pitch_um",       gm.holePitchUm);
    gm.wireDiameterUm   = ReadDouble(*g, "geometry", "wire_diameter_um",    gm.wireDiameterUm);
    gm.holesPerWire     = ReadInt   (*g, "geometry", "holes_per_wire",      gm.holesPerWire);
    gm.wireBetweenHoles = ReadBool  (*g, "geometry", "wire_between_holes",  gm.wireBetweenHoles);
    gm.driftGapMm         = ReadDouble(*g, "geometry", "drift_gap_mm",         gm.driftGapMm);
    gm.transferGapMm      = ReadDouble(*g, "geometry", "transfer_gap_mm",      gm.transferGapMm);
    gm.amplificationGapMm = ReadDouble(*g, "geometry", "amplification_gap_mm", gm.amplificationGapMm);
    // A config written for the double-THGEM sibling must not be silently
    // half-accepted: these two keys have no meaning here and ignoring them
    // would run a geometry the user did not ask for.
    if (FindMember(*g, "anode_enabled", {"geometry", "anode_enabled"}))
      throw std::runtime_error(
          "geometry.anode_enabled is not supported: a mesh is 35-65 % open and cannot "
          "terminate the volume, so the anode is always present. Set "
          "geometry.amplification_gap_mm and fields.delta_v_mesh_anode_V instead.");
    if (FindMember(*g, "induction_gap_mm", {"geometry", "induction_gap_mm"}))
      throw std::runtime_error(
          "geometry.induction_gap_mm is not supported: the gap below the mesh is where "
          "multiplication happens, not induction. Use geometry.amplification_gap_mm.");
    gm.targetElementSizeUm = ReadDouble(*g, "geometry", "target_element_size_um", gm.targetElementSizeUm);
    gm.thgemElementSizeUm  = ReadDouble(*g, "geometry", "thgem_element_size_um",  gm.thgemElementSizeUm);
    gm.minElements        = ReadInt   (*g, "geometry", "min_elements",         gm.minElements);
    gm.maxElements        = ReadInt   (*g, "geometry", "max_elements",         gm.maxElements);
    gm.maxElementsBudget  = ReadInt   (*g, "geometry", "max_elements_budget",  gm.maxElementsBudget);
    gm.periodicCopies     = ReadInt   (*g, "geometry", "periodic_copies",      gm.periodicCopies);
    gm.holeSectors        = ReadInt   (*g, "geometry", "hole_sectors",         gm.holeSectors);
    gm.gridNx             = ReadInt   (*g, "geometry", "grid_nx",              gm.gridNx);
    gm.gridNy             = ReadInt   (*g, "geometry", "grid_ny",              gm.gridNy);
    gm.gridNz             = ReadInt   (*g, "geometry", "grid_nz",              gm.gridNz);
    gm.thgem = ReadPlate(*g, "thgem", gm.thgem);
    gm.mesh  = ReadMesh (*g, gm.mesh);
  }

  if (const auto* f = FindSection(root, "fields")) {
    cfg.fields.eDriftKvcm    = ReadDouble(*f, "fields", "e_drift_kvcm",         cfg.fields.eDriftKvcm);
    cfg.fields.deltaVThgemV  = ReadDouble(*f, "fields", "delta_v_thgem_V",      cfg.fields.deltaVThgemV);
    cfg.fields.eTransferKvcm = ReadDouble(*f, "fields", "e_transfer_kvcm",      cfg.fields.eTransferKvcm);
    cfg.fields.deltaVMeshAnodeV = ReadDouble(*f, "fields", "delta_v_mesh_anode_V",
                                             cfg.fields.deltaVMeshAnodeV);
    // This project's own retired key.  Refused rather than converted so a run's
    // config says exactly one thing; the message does the arithmetic.
    if (const auto* dead = FindMember(*f, "e_amplification_kvcm",
                                      {"fields", "e_amplification_kvcm"})) {
      const double kvcm = dead->is_number() ? dead->get<double>() : 0.;
      throw std::runtime_error(
          "fields.e_amplification_kvcm is retired: the amplification gap is now set by its "
          "voltage, not its field, so that changing geometry.amplification_gap_mm does not "
          "silently move the mesh potential. Use fields.delta_v_mesh_anode_V = "
          + FormatNumber(kvcm * 1000. * cfg.geometry.amplificationGapMm * 0.1, 1) +
          " V (" + FormatNumber(kvcm, 3) + " kV/cm over " +
          FormatNumber(cfg.geometry.amplificationGapMm, 3) + " mm).");
    }
    for (const char* dead : {"delta_v_thgem1_V", "delta_v_thgem2_V", "e_induction_kvcm"})
      if (FindMember(*f, dead, {"fields", dead}))
        throw std::runtime_error(
            std::string("fields.") + dead + " is not supported: there is one THGEM "
            "(fields.delta_v_thgem_V) and one mesh stage "
            "(fields.delta_v_mesh_anode_V) here.");
  }

  if (const auto* r = FindSection(root, "readout")) {
    auto* el = FindMember(*r, "electrodes", {"readout", "electrodes"});
    if (el && el->is_array()) {
      std::vector<std::string> ids;
      for (auto& v : *el) {
        if (!v.is_string()) ThrowJsonTypeError({"readout", "electrodes"}, "an array of strings");
        ids.push_back(v.get<std::string>());
      }
      cfg.readout.electrodes = std::move(ids);
    } else {
      cfg.readout.electrodes = std::nullopt;   // resolved after validation
    }
  }

  if (const auto* s = FindSection(root, "source")) {
    cfg.source.energyKeV    = ReadDouble(*s, "source", "energy_keV", cfg.source.energyKeV);
    {
      auto* dl = FindMember(*s, "source_distances_mm", {"source", "source_distances_mm"});
      if (dl && dl->is_array()) {
        cfg.source.fixedDistMm = std::vector<double>{};
        for (auto& v : *dl) cfg.source.fixedDistMm->push_back(v.get<double>());
      } else {
        cfg.source.fixedDistMm = std::nullopt;  // random per event
      }
    }
    auto* xpList = FindMember(*s, "x_positions_cm", {"source", "x_positions_cm"});
    if (xpList && xpList->is_array()) {
      cfg.source.fixedXCmList = std::vector<double>{};
      for (auto& v : *xpList)
        cfg.source.fixedXCmList->push_back(v.get<double>());
    } else {
      cfg.source.fixedXCmList = std::nullopt;
    }
  }

  if (const auto* g = FindSection(root, "gas")) {
    cfg.gas.gas1       = ReadString(*g, "gas", "gas1",               cfg.gas.gas1);
    cfg.gas.frac1      = ReadDouble(*g, "gas", "gas1_fraction_pct",  cfg.gas.frac1);
    cfg.gas.gas2       = ReadString(*g, "gas", "gas2",               cfg.gas.gas2);
    cfg.gas.ionSpecies = ReadString(*g, "gas", "ion_species",        cfg.gas.ionSpecies);
    cfg.gas.temperatureK  = ReadDouble(*g, "gas", "temperature_K",       cfg.gas.temperatureK);
    cfg.gas.pressureTorr  = ReadDouble(*g, "gas", "pressure_Torr",       cfg.gas.pressureTorr);
    cfg.gas.enablePenning = ReadBool  (*g, "gas", "enable_penning",       cfg.gas.enablePenning);
    cfg.gas.nCollisions         = ReadInt   (*g, "gas", "n_magboltz_collisions",  cfg.gas.nCollisions);
    cfg.gas.maxElectronEnergyEV = ReadDouble(*g, "gas", "max_electron_energy_eV", cfg.gas.maxElectronEnergyEV);
    cfg.gas.transportMaxEnergyEV = ReadDouble(*g, "gas", "transport_max_energy_eV",
                                              cfg.gas.transportMaxEnergyEV);
    cfg.gas.nFieldPoints        = ReadInt   (*g, "gas", "n_field_points",         cfg.gas.nFieldPoints);
    cfg.gas.eFieldMinVcm        = ReadDouble(*g, "gas", "e_field_min_vcm",        cfg.gas.eFieldMinVcm);
    cfg.gas.eFieldMaxVcm        = ReadDouble(*g, "gas", "e_field_max_vcm",        cfg.gas.eFieldMaxVcm);
    cfg.gas.wValueEV            = ReadDouble(*g, "gas", "w_value_eV",             cfg.gas.wValueEV);
  }

  if (const auto* s = FindSection(root, "simulation")) {
    cfg.simulation.nEvents          = ReadSizeT (*s, "simulation", "n_events",          cfg.simulation.nEvents);
    cfg.simulation.maxAvalancheSize = ReadSizeT (*s, "simulation", "max_avalanche_size", cfg.simulation.maxAvalancheSize);
    cfg.simulation.timeWindowNs     = ReadDouble(*s, "simulation", "time_window_ns",     cfg.simulation.timeWindowNs);
    cfg.simulation.timeStepNs       = ReadDouble(*s, "simulation", "time_step_ns",       cfg.simulation.timeStepNs);
    cfg.simulation.enableIonDrift   = ReadBool  (*s, "simulation", "enable_ion_drift",   cfg.simulation.enableIonDrift);
    cfg.simulation.storeDriftLines  = ReadBool  (*s, "simulation", "store_drift_lines",  cfg.simulation.storeDriftLines);
    cfg.simulation.ionMaxStepUm     = ReadDouble(*s, "simulation", "ion_max_step_um",    cfg.simulation.ionMaxStepUm);
    cfg.simulation.ionTimeWindowNs  = ReadDouble(*s, "simulation", "ion_time_window_ns", cfg.simulation.ionTimeWindowNs);
    cfg.simulation.maxIonsDrifted   = ReadSizeT (*s, "simulation", "max_ions_drifted",   cfg.simulation.maxIonsDrifted);
    cfg.simulation.randomSeed       = ReadInt   (*s, "simulation", "random_seed",        cfg.simulation.randomSeed);
  }

  if (const auto* a = FindSection(root, "amplifier")) {
    cfg.amplifier.enable            = ReadBool  (*a, "amplifier", "enable",             cfg.amplifier.enable);
    cfg.amplifier.gainDb            = ReadDouble(*a, "amplifier", "gain_db",            cfg.amplifier.gainDb);
    cfg.amplifier.inputImpedanceOhm = ReadDouble(*a, "amplifier", "input_impedance_ohm", cfg.amplifier.inputImpedanceOhm);
    cfg.amplifier.bandwidthHighHz   = ReadDouble(*a, "amplifier", "bandwidth_high_hz",  cfg.amplifier.bandwidthHighHz);
    cfg.amplifier.outputSampleNs    = ReadDouble(*a, "amplifier", "output_sample_ns",   cfg.amplifier.outputSampleNs);
  }

  // ── Validation ───────────────────────────────────────────────────────────────
  auto& gm = cfg.geometry;
  auto checkPlate = [&gm](const PlateConfig& p, const char* name) {
    const std::string s = std::string("geometry.") + name + ".";
    if (p.holeDiameterUm <= 0.)
      throw std::runtime_error(s + "hole_diameter_um must be positive");
    if (p.holeDiameterUm >= gm.holePitchUm)
      throw std::runtime_error(s + "hole_diameter_um must be smaller than geometry.hole_pitch_um");
    if (p.plateThicknessUm <= 0.)
      throw std::runtime_error(s + "plate_thickness_um must be positive");
    if (p.copperThicknessUm < 0.)
      throw std::runtime_error(s + "copper_thickness_um must be >= 0");
    if (p.rimUm < 0.)
      throw std::runtime_error(s + "rim_um must be >= 0");
    // The copper opening is hole + rim; it must still fit inside the cell.
    if (p.holeDiameterUm + 2. * p.rimUm >= gm.holePitchUm)
      throw std::runtime_error(s + "hole_diameter_um + 2*rim_um must be smaller than "
                                   "geometry.hole_pitch_um (copper openings would merge)");
    if (p.dielectric != "fr4" && p.dielectric != "kapton")
      throw std::runtime_error(s + "dielectric_material must be 'fr4' or 'kapton'");
  };

  if (gm.holePitchUm <= 0.) throw std::runtime_error("geometry.hole_pitch_um must be positive");
  checkPlate(gm.thgem, "thgem");
  if (gm.wireDiameterUm <= 0.) throw std::runtime_error("geometry.wire_diameter_um must be positive");
  if (gm.holesPerWire < 1) throw std::runtime_error("geometry.holes_per_wire must be >= 1");
  if (gm.wireDiameterUm >= gm.holesPerWire * gm.holePitchUm)
    throw std::runtime_error("geometry.wire_diameter_um must be smaller than the wire pitch "
                             "(holes_per_wire * hole_pitch_um)");
  if (gm.driftGapMm         <= 0.) throw std::runtime_error("geometry.drift_gap_mm must be positive");
  if (gm.transferGapMm      <= 0.) throw std::runtime_error("geometry.transfer_gap_mm must be positive");
  if (gm.amplificationGapMm <= 0.) throw std::runtime_error("geometry.amplification_gap_mm must be positive");

  // Mesh.  The lattice checks that depend on the *snapped* pitch cannot live
  // here — LoadConfig does not know the cell — and are done by ValidateGeometry
  // once ComputeGeom has run.  These are the model-independent ones.
  auto& mc = gm.mesh;
  if (mc.model != "woven" && mc.model != "perforated")
    throw std::runtime_error("geometry.mesh.model must be 'woven' or 'perforated' (got '" +
                             mc.model + "')");
  if (mc.pitchUm <= 0.) throw std::runtime_error("geometry.mesh.pitch_um must be positive");
  if (mc.thicknessUm < 0.) throw std::runtime_error("geometry.mesh.thickness_um must be >= 0");
  if (mc.elementSizeUm < 0.) throw std::runtime_error("geometry.mesh.element_size_um must be >= 0");
  if (mc.model == "woven") {
    if (mc.wireDiameterUm <= 0.)
      throw std::runtime_error("geometry.mesh.wire_diameter_um must be positive for the woven model");
    // thickness 0 means "auto = 2 x wire diameter"; anything smaller than one
    // wire diameter cannot hold two crossed layers.
    if (mc.thicknessUm > 0. && mc.thicknessUm < mc.wireDiameterUm)
      throw std::runtime_error("geometry.mesh.thickness_um must be >= mesh.wire_diameter_um "
                               "(two crossed wire layers do not fit in a thinner mesh); "
                               "use 0 for the default 2 x wire diameter");
  } else {
    if (mc.thicknessUm <= 0.)
      throw std::runtime_error("geometry.mesh.thickness_um must be positive for the perforated "
                               "model (there is no wire diameter to derive it from)");
    if (mc.apertureUm <= 0.)
      throw std::runtime_error("geometry.mesh.aperture_um must be positive for the perforated model");
    if (mc.apertureSectors < 2)
      throw std::runtime_error("geometry.mesh.aperture_sectors must be >= 2 "
                               "(2 = square, 3 = octagon, 4 = 12-gon)");
  }

  if (gm.targetElementSizeUm <= 0.) throw std::runtime_error("geometry.target_element_size_um must be positive");
  if (gm.thgemElementSizeUm < 0.) throw std::runtime_error("geometry.thgem_element_size_um must be >= 0");
  if (gm.minElements < 1 || gm.maxElements < gm.minElements)
    throw std::runtime_error("geometry.min_elements/max_elements invalid (need 1 <= min <= max)");
  if (gm.maxElementsBudget < 1000)
    throw std::runtime_error("geometry.max_elements_budget must be >= 1000 (a THGEM plate alone "
                             "needs roughly 1200 boundary elements)");
  if (gm.periodicCopies < 0) throw std::runtime_error("geometry.periodic_copies must be >= 0");
  if (gm.holeSectors < 2) throw std::runtime_error("geometry.hole_sectors must be >= 2");
  if (gm.gridNx < 4 || gm.gridNy < 4 || gm.gridNz < 4)
    throw std::runtime_error("geometry.grid_nx/grid_ny/grid_nz must be >= 4");

  if (cfg.fields.deltaVThgemV <= 0.) throw std::runtime_error("fields.delta_v_thgem_V must be positive");
  if (cfg.fields.eDriftKvcm    < 0.) throw std::runtime_error("fields.e_drift_kvcm must be >= 0");
  if (cfg.fields.eTransferKvcm < 0.) throw std::runtime_error("fields.e_transfer_kvcm must be >= 0");
  // >= 0, not > 0: unlike delta_v_thgem_V (a THGEM at 0 V stops being a THGEM),
  // an amplification gap at 0 V is an ordinary collection gap that still
  // transports charge, and ValidateField's starved-gap branch exists to explain
  // exactly that case.  The symmetry is with the other two gap knobs.
  if (cfg.fields.deltaVMeshAnodeV < 0.)
    throw std::runtime_error("fields.delta_v_mesh_anode_V must be >= 0");
  // The sparking bound is on the *derived* field, so it follows the gap.  It
  // lives here rather than in ValidateGeometry because LoadConfig already knows
  // the gap (parsed and validated > 0 above) and runs before SetupGas — a typo'd
  // voltage should not cost a Magboltz table first.
  {
    const double eAmpKvcm = cfg.fields.deltaVMeshAnodeV /
                            (cfg.geometry.amplificationGapMm * 0.1) * 1.e-3;
    if (eAmpKvcm > 100.)
      throw std::runtime_error(
          "fields.delta_v_mesh_anode_V = " + FormatNumber(cfg.fields.deltaVMeshAnodeV, 1) +
          " V over geometry.amplification_gap_mm = " +
          FormatNumber(cfg.geometry.amplificationGapMm, 3) + " mm is " +
          FormatNumber(eAmpKvcm, 1) + " kV/cm across the amplification gap, past any real "
          "micromegas sparking limit. Lower the voltage or widen the gap.");
  }

  // Resolve the readout list: an explicit list is validated against the geometry,
  // an absent one falls back to the single collecting electrode.
  if (!cfg.readout.electrodes.has_value())
    cfg.readout.electrodes = std::vector<std::string>{std::string(kElecAnode)};
  auto& ids = *cfg.readout.electrodes;
  if (ids.empty())
    throw std::runtime_error("readout.electrodes must name at least one electrode");
  for (const auto& id : ids) {
    if (!IsKnownElectrode(id)) {
      std::string msg = "readout.electrodes contains unknown electrode '" + id + "'. Valid: ";
      for (const auto& v : kAllElectrodeIds) msg += v + " ";
      throw std::runtime_error(msg);
    }
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::runtime_error("readout.electrodes contains a duplicate entry");
  // Restore the physical top-to-bottom order; sorting above was only for the
  // duplicate check, and the stack order is what makes plots readable.
  std::vector<std::string> ordered;
  for (const auto& canonical : kAllElectrodeIds)
    if (std::find(ids.begin(), ids.end(), canonical) != ids.end()) ordered.push_back(canonical);
  ids = std::move(ordered);

  if (cfg.source.fixedDistMm.has_value() && cfg.source.fixedDistMm->empty())
    throw std::runtime_error("source.source_distances_mm must not be empty when set");
  // A release height at (or above) the cathode plane is not merely unphysical:
  // the Sensor's drift area is inset by one transport-grid cell, so it lands
  // outside the active area and AvalancheMicroscopic refuses the track with a
  // message that says nothing about the config key responsible.
  if (cfg.source.fixedDistMm.has_value())
    for (const double d : *cfg.source.fixedDistMm)
      if (d <= 0. || d >= gm.driftGapMm)
        throw std::runtime_error(
            "source.source_distances_mm entry " + FormatNumber(d) +
            " mm must lie strictly inside the drift gap (0 .. " +
            FormatNumber(gm.driftGapMm) + " mm); it is measured upward from the "
            "THGEM's top copper.");
  if (cfg.gas.frac1 <= 0. || cfg.gas.frac1 >= 100.)
    throw std::runtime_error("gas.gas1_fraction_pct must be in (0, 100)");
  if (cfg.gas.temperatureK     <= 0.)  throw std::runtime_error("gas.temperature_K must be positive");
  if (cfg.gas.pressureTorr     <= 0.)  throw std::runtime_error("gas.pressure_Torr must be positive");
  if (cfg.gas.transportMaxEnergyEV <= 0.)
    throw std::runtime_error("gas.transport_max_energy_eV must be positive");
  // n_events == 0 is legal: it means "field only" — solve, sample, dump the
  // maps and stop before any transport, exactly as --field-only does.
  if (cfg.simulation.timeWindowNs <= 0.) throw std::runtime_error("simulation.time_window_ns must be positive");
  if (cfg.simulation.timeStepNs   <= 0.) throw std::runtime_error("simulation.time_step_ns must be positive");
  if (cfg.simulation.randomSeed   <  0)  throw std::runtime_error("simulation.random_seed must be >= 0");

  return cfg;
}
// ─── Gas setup ────────────────────────────────────────────────────────────────

static void ExportGasProps(MediumMagboltz& gas, const std::string& outPath,
                           const std::string& ionMobFile = "") {
  std::vector<double> efields, bfields, angles;
  gas.GetFieldGrid(efields, bfields, angles);

  std::ofstream f(outPath);
  if (!f) {
    std::cerr << "  Warning: could not write gas properties to " << outPath << "\n";
    return;
  }
  if (!ionMobFile.empty()) {
    const auto sep = ionMobFile.find_last_of("/\\");
    const std::string base = (sep == std::string::npos)
                             ? ionMobFile : ionMobFile.substr(sep + 1);
    f << "# ion_mobility: " << base << "\n";
  }
  f << "e_field_Vcm,vd_cm_per_us,alpha_per_cm,eta_per_cm,"
       "dl_sqrtcm,dt_sqrtcm,v_ion_cm_per_us,mu_ion_cm2_per_Vus\n";

  for (double E : efields) {
    double vx = 0, vy = 0, vz = 0;
    gas.ElectronVelocity(E, 0, 0, 0, 0, 0, vx, vy, vz);
    double alpha = 0, eta = 0, dl = 0, dt = 0;
    gas.ElectronTownsend(E, 0, 0, 0, 0, 0, alpha);
    gas.ElectronAttachment(E, 0, 0, 0, 0, 0, eta);
    gas.ElectronDiffusion(E, 0, 0, 0, 0, 0, dl, dt);

    double v_ion = 0, mu_ion = 0;
    double vix = 0, viy = 0, viz = 0;
    if (gas.IonVelocity(E, 0, 0, 0, 0, 0, vix, viy, viz)) {
      v_ion  = std::abs(vix) * 1.e3;
      mu_ion = (E > 0.) ? v_ion / E : 0.;
    }

    f << std::scientific << std::setprecision(6)
      << E       << ","
      << vx * 1.e3 << ","
      << alpha   << ","
      << eta     << ","
      << dl      << ","
      << dt      << ","
      << v_ion   << ","
      << mu_ion  << "\n";
  }
  std::cout << "  Gas properties exported to: " << outPath << "\n";
}

std::string DriftStatusToString(const int st) {
  switch (st) {
    case Garfield::StatusAlive: return "alive";
    case Garfield::StatusLeftDriftArea: return "left drift area";
    case Garfield::StatusTooManySteps: return "too many steps";
    case Garfield::StatusCalculationAbandoned: return "calculation abandoned";
    case Garfield::StatusLeftDriftMedium: return "left drift medium";
    case Garfield::StatusAttached: return "attached";
    case Garfield::StatusSharpKink: return "sharp kink";
    case Garfield::StatusRecombined: return "recombined";
    case Garfield::StatusHitPlane: return "hit plane";
    case Garfield::StatusBelowTransportCut: return "below transport cut";
    case Garfield::StatusOutsideTimeWindow: return "outside time window";
    default: return "status " + std::to_string(st);
  }
}

std::string SetupGas(MediumMagboltz& gas, const GasConfig& cfg,
                     const bool requireIonMobility) {
  gas.SetTemperature(cfg.temperatureK);
  gas.SetPressure(cfg.pressureTorr);

  // Gas tables and their _props.csv sidecars live in gas/ (keeps the project root tidy).
  EnsureDirectory("gas");
  const std::string gasFile = (fs::path("gas") / DeriveGasFileName(cfg)).string();

  if (fs::exists(gasFile)) {
    std::cout << "  Loading gas table from: " << gasFile << "\n";
    gas.LoadGasFile(gasFile);
  } else {
    std::cout << "  Gas file not found: " << gasFile << "\n"
              << "  Running Magboltz for " << cfg.nFieldPoints
              << " field points from " << static_cast<int>(cfg.eFieldMinVcm)
              << " to " << static_cast<int>(cfg.eFieldMaxVcm) << " V/cm ...\n";
    gas.SetMaxElectronEnergy(cfg.maxElectronEnergyEV);   // EFINAL of the generated table
    gas.SetFieldGrid(cfg.eFieldMinVcm, cfg.eFieldMaxVcm, cfg.nFieldPoints, /*logspacing=*/true);
    gas.GenerateGasTable(cfg.nCollisions, /*verbose=*/false);
    gas.WriteGasFile(gasFile);
    std::cout << "  Gas table saved to: " << gasFile << "\n";
  }

  // Cap the microscopic collision-rate table well below the transport-table EFINAL.
  // AvalancheMicroscopic draws every step from the *maximum* rate over the whole energy
  // grid and rejects the surplus as null collisions, so an oversized ceiling costs a
  // proportional number of wasted steps.  Re-run after LoadGasFile: SetMaxElectronEnergy
  // only forces the rate table to be rebuilt; the loaded transport tables are untouched.
  // If an electron ever exceeds the ceiling Garfield raises it automatically.
  gas.SetMaxElectronEnergy(cfg.transportMaxEnergyEV);
  std::cout << "  Collision-rate ceiling: " << cfg.transportMaxEnergyEV
            << " eV (table EFINAL " << cfg.maxElectronEnergyEV << " eV)"
            << ", null-collision rate = " << gas.GetElectronNullCollisionRate(0) << " /ns\n";

  if (cfg.enablePenning) {
    if (!gas.EnablePenningTransfer())
      std::cerr << "  Warning: Penning transfer could not be enabled.\n";
    else
      std::cout << "  Penning transfer enabled.\n";
  }

  const char* garfieldInstall = std::getenv("GARFIELD_INSTALL");
  std::string loadedMob;
  std::string ionUpper = cfg.ionSpecies;
  std::transform(ionUpper.begin(), ionUpper.end(), ionUpper.begin(), ::toupper);
  if (garfieldInstall) {
    const std::string mobFile = std::string(garfieldInstall) +
                                "/share/Garfield/Data/IonMobility_"
                                + ionUpper + "+_" + ionUpper + ".txt";
    if (fs::exists(mobFile)) {
      gas.LoadIonMobility(mobFile);
      std::cout << "  " << ionUpper << "+ ion mobility loaded.\n";
      loadedMob = mobFile;
    } else {
      std::cerr << "  Warning: IonMobility_" << ionUpper << "+_" << ionUpper
                << ".txt not found at " << mobFile << "\n";
    }
  } else {
    std::cerr << "  Warning: GARFIELD_INSTALL not set; ion mobility not loaded.\n";
  }

  if (requireIonMobility && loadedMob.empty()) {
    throw std::runtime_error(
        "simulation.enable_ion_drift=true requires an ion mobility table for "
        + ionUpper + "+. Set GARFIELD_INSTALL so Garfield++ can find "
        + "share/Garfield/Data/IonMobility_" + ionUpper + "+_" + ionUpper
        + ".txt, or disable simulation.enable_ion_drift.");
  }

  const std::string propsFile = gasFile.substr(0, gasFile.size() - 4) + "_props.csv";
  ExportGasProps(gas, propsFile, loadedMob);
  return loadedMob;
}


// ─── THGEM + mesh geometry (neBEM) ───────────────────────────────────────────

enum class MeshModel { Woven, Perforated };

// The THGEM plate's computed dimensions [cm] and potentials [V].
struct PlateGeom {
  double rHoleCm   = 0.;   // dielectric hole radius
  double rCuCm     = 0.;   // copper opening radius (= rHole + rim)
  double tDielCm   = 0.;
  double tCuCm     = 0.;
  double epsDiel   = 4.6;
  double zCen      = 0.;   // dielectric mid-plane
  double zDielHalf = 0.;
  double zTopCuTop = 0.;   // outer surface of the top copper
  double zBotCuBot = 0.;   // outer surface of the bottom copper
  double vTop      = 0.;
  double vBot      = 0.;
  // Copies of the shared lattice parameters, so InPlate() is a pure function of
  // the plate.  `sectors` is the order of the polygon neBEM actually meshes the
  // hole as — the in-solid test has to use the same polygon, not a circle.
  double pitchCm    = 0.;
  double latShiftCm = 0.;
  std::size_t sectors = 4;
};

// The mesh electrode's computed lattice, z planes and potential [cm, V].
//
// The lattice pitch here is always the *snapped* one: one periodic cell has to
// hold a whole number of mesh cells, so ComputeGeom rounds the requested pitch
// to hole_pitch / k.  Everything downstream — the solids, the in-solid test, the
// cache key, the echoed config — uses this value and never the requested one.
struct MeshGeom {
  MeshModel model = MeshModel::Woven;
  double pitchCm    = 0.;
  double pitchUm    = 0.;   // snapped, in µm (reports and cache keys)
  double pitchReqUm = 0.;   // what the config asked for
  int    wiresPerHolePitch = 1;   // k
  int    nX = 1;            // wires (or apertures) per cell along x = k · nHolesX
  int    nY = 1;            // wires (or apertures) per cell along y = k
  double offXCm = 0.;       // lattice stagger, already wrapped into (−pitch/2, pitch/2]
  double offYCm = 0.;
  double phaseXCm = 0.;     // lattice phase: WrapToCell(X(0), pitch)
  double phaseYCm = 0.;
  // z planes.  For the woven model the two wire layers sit at zLower / zUpper,
  // placed so their outer surfaces land exactly on zBot / zTop.
  double tMeshCm = 0.;
  double zCen = 0., zTop = 0., zBot = 0.;
  double zUpper = 0., zLower = 0.;
  double rWireCm     = 0.;  // woven
  double rApertureCm = 0.;  // perforated: circumradius handed to SolidHole
  std::size_t sectors = 4;  // perforated: 2 = square, 3 = octagon, 4 = 12-gon
  double opticalTransparency = 0.;
  double vMesh = 0.;
  // The amplification-gap field, derived once from delta_v_mesh_anode_V and the
  // gap.  Kept here so the validation, the banner and the transparency ratio
  // all read one number rather than each re-dividing.
  double eAmpKvcm = 0.;

  // Centre of wire / aperture i along x, j along y.  Canonically centred on the
  // cell like the hole lattice, so the extremes are ±(cell − pitch)/2 and every
  // centre stays strictly inside one period — neBEM adds wire primitives after
  // it has reduced panels to the basic period, so a centre outside the cell
  // would not be wrapped for us.
  double X(int i) const { return (i - 0.5 * (nX - 1)) * pitchCm + offXCm; }
  double Y(int j) const { return (j - 0.5 * (nY - 1)) * pitchCm + offYCm; }
};

// Computed cell dimensions [cm] and derived electrode potentials [V].
struct ThgemMeshGeom {
  double pitchCm   = 0.;   // THGEM hole lattice pitch, x and y
  double cellXCm   = 0.;   // periodic cell x-period = holesPerWire × pitch
  double cellYCm   = 0.;   // periodic cell y-period = pitch
  int    nHolesX   = 1;    // holes per cell in x
  double latShiftCm = 0.;  // hole-lattice shift (half a pitch when the wire is
                           // asked to sit between holes rather than over one)
  double rWireCm   = 0.;   // cathode wire
  double zWire     = 0.;
  double vWire     = 0.;
  PlateGeom thgem;
  MeshGeom  mesh;
  double zAnode    = 0.;
  double vAnode    = 0.;
  double dDriftCm = 0., dTransferCm = 0., dAmpCm = 0.;
  double zMin = 0.;        // bottom of the modelled volume (= zAnode)
  double zMax = 0.;        // top of the modelled volume (= zWire)

  // x of hole m (0 … nHolesX−1).  The canonical lattice is centred on the cell,
  // so every hole box lies strictly inside one period and the boxes tile it
  // exactly.
  double HoleX(int m) const {
    return (m - 0.5 * (nHolesX - 1)) * pitchCm + latShiftCm;
  }
};

// Shortest signed offset of `d` under a period — i.e. the distance to the
// nearest lattice point, in (−period/2, +period/2].
double WrapToCell(double d, double period) {
  if (period <= 0.) return d;
  d = std::fmod(d, period);
  if (d >  0.5 * period) d -= period;
  if (d < -0.5 * period) d += period;
  return d;
}

constexpr double kPi = 3.14159265358979323846;

// Inside the aperture of a SolidHole with `sectors` = n?
//
// SolidHole does not mesh a circle: it meshes a regular 4(n−1)-gon of
// *circumradius* r with its first vertex at −45°.  Testing a plain circle of
// radius r instead is 3 % wrong in area at n = 4 and 29 % wrong at n = 2, where
// the "square" has half-side r/√2 rather than r.  The in-solid flag has to
// describe the polygon neBEM actually solved, so this reproduces it exactly.
bool InHolePolygon(const double dx, const double dy, const double r,
                   const std::size_t sectors) {
  if (r <= 0.) return false;
  const std::size_t nv = 4 * (sectors - 1);
  const double dphi = 2. * kPi / static_cast<double>(nv);
  const double rIn  = r * std::cos(0.5 * dphi);   // inradius
  const double rho  = std::hypot(dx, dy);
  if (rho <= rIn) return true;
  if (rho >= r)   return false;
  // Fold the azimuth into one edge's sector, measured from that edge's normal.
  double th = std::atan2(dy, dx) + 0.25 * kPi - 0.5 * dphi;
  th -= dphi * std::floor(th / dphi + 0.5);
  return rho * std::cos(th) <= rIn;
}

// Area of the regular 4(n−1)-gon of circumradius r that SolidHole meshes.
double HolePolygonArea(const double r, const std::size_t sectors) {
  const double nv = static_cast<double>(4 * (sectors - 1));
  return 0.5 * nv * r * r * std::sin(2. * kPi / nv);
}

// Owns the media, solids, geometry and neBEM component of the unit cell.
// Lives for the whole run (neBEM references the geometry during field lookups).
class ThgemMeshDetector {
 public:
  ThgemMeshDetector(const GeometryConfig& g, const FieldConfig& f, MediumMagboltz& gas) {
    geom_ = ComputeGeom(g, f);
    diel_.SetDielectricConstant(geom_.thgem.epsDiel);
    thgemElemCm_ = g.thgemElementSizeUm * 1.e-4;
    meshElemCm_  = g.mesh.elementSizeUm * 1.e-4;

    const double hy  = geom_.cellYCm / 2.0;   // cell half-width in y
    const double hxCell = geom_.cellXCm / 2.0;
    const auto sec = static_cast<std::size_t>(g.holeSectors);

    // Wire cathode: one wire per cell at x = 0, running along y.  Its half-length
    // is half the cell, so the periodic copies concatenate into a continuous
    // wire.  neBEM meshes a SolidWire as a line-charge primitive (Solid::IsWire),
    // which is what makes a 50 µm conductor affordable next to millimetre plates
    // — and is the same reason a whole woven mesh costs almost nothing below.
    auto wire = std::make_unique<SolidWire>(0., 0., geom_.zWire, geom_.rWireCm, hy,
                                            0., 1., 0.);
    wire->SetBoundaryPotential(geom_.vWire);
    wire->SetLabel(kElecWire);
    geo_.AddSolid(wire.get(), &cu_);
    solids_.push_back(std::move(wire));

    AddPlate(geom_.thgem, sec);

    if (geom_.mesh.model == MeshModel::Woven) AddMeshWoven(geom_.mesh);
    else                                      AddMeshPerforated(geom_.mesh);

    // Anode readout plate one amplification gap below the mesh.  Labelled so
    // neBEM solves its true Shockley–Ramo weighting field (1 V here, 0 V on all
    // others).  Unlike the double-THGEM sibling it is not optional: a mesh is
    // 35-65 % open and cannot terminate the volume, and without an anode there
    // is no amplification gap at all.
    auto anode = std::make_unique<SolidBox>(0., 0., geom_.zAnode, hxCell, hy, 0.);
    anode->SetBoundaryPotential(geom_.vAnode);
    anode->SetLabel(kElecAnode);
    geo_.AddSolid(anode.get(), &cu_);
    solids_.push_back(std::move(anode));

    // The gas fills all space not occupied by a solid.
    geo_.SetMedium(&gas);

    // neBEM solver: one periodic cell tiled in x and y.  Translation (not
    // mirror) periodicity, because a staggered mesh lattice or a wire sitting
    // between holes leaves the cell without a mirror plane.
    nebem_.SetGeometry(&geo_);
    nebem_.SetTargetElementSize(g.targetElementSizeUm * 1.e-4);
    nebem_.SetMinMaxNumberOfElements(static_cast<std::size_t>(g.minElements),
                                     static_cast<std::size_t>(g.maxElements));
    nebem_.SetPeriodicityX(geom_.cellXCm);
    nebem_.SetPeriodicityY(geom_.cellYCm);
    const auto pc = static_cast<std::size_t>(g.periodicCopies);
    nebem_.SetPeriodicCopies(pc, pc, 0);
    // Performance: evaluate full elements only for the central cell / first ring,
    // and cheap primitive-averaged properties for the more distant periodic
    // copies.  Without this neBEM evaluates every element for every field call
    // (the default), which makes the field-map sampling intractably slow.  The
    // near field — inside the holes and the amplification gap, where the
    // avalanche lives — is unaffected.
    nebem_.SetPrimAfter(1);
    nebem_.SetWtFldPrimAfter(1);
    nebem_.UseLUInversion();
  }

  bool Initialise() { return nebem_.Initialise(); }
  ComponentNeBem3d& Component() { return nebem_; }
  const ThgemMeshGeom& Geom() const { return geom_; }

  // True if (x, y, z) is in the gas, i.e. not inside a conductor or dielectric.
  // Used to tag absorbing nodes when sampling the transport grid (ComponentGrid
  // carries no material information of its own).
  //
  // This is an explicit analytic test rather than a GeometrySimple::GetMedium
  // lookup, because GeometrySimple only knows the solids at their *literal*
  // positions.  The field is periodic and the sampled grid covers one cell, so
  // a staggered mesh — or a hole lattice shifted to put the wire between holes
  // — puts material across the cell edge where no literal solid sits, and the
  // lookup would report gas inside copper.  Wrapping to the nearest lattice
  // point is correct for any offset, and is also far cheaper than a solid walk
  // over the millions of nodes a sampling pass visits.
  bool InGas(const double x, const double y, const double z) const {
    const auto& g = geom_;
    // Cathode wire (infinite along y, one per cell in x).
    const double dzw = z - g.zWire;
    if (std::abs(dzw) <= g.rWireCm) {
      const double dxw = WrapToCell(x, g.cellXCm);
      if (dxw * dxw + dzw * dzw <= g.rWireCm * g.rWireCm) return false;
    }
    if (InPlate(g.thgem, x, y, z)) return false;
    if (InMesh(g.mesh, x, y, z)) return false;
    // The anode is a zero-thickness patch; flag the node layer that lands on it
    // so the grid has no gas there.
    if (z <= g.zAnode) return false;
    return true;
  }

  // Inside the mesh electrode's metal?
  static bool InMesh(const MeshGeom& m, const double x, const double y, const double z) {
    if (z > m.zTop || z < m.zBot) return false;
    if (m.model == MeshModel::Woven) {
      const double r2 = m.rWireCm * m.rWireCm;
      // Upper layer: wires ∥ x, spaced along y.
      const double dzU = z - m.zUpper;
      if (std::abs(dzU) <= m.rWireCm) {
        const double dy = WrapToCell(y - m.phaseYCm, m.pitchCm);
        if (dy * dy + dzU * dzU <= r2) return true;
      }
      // Lower layer: wires ∥ y, spaced along x.
      const double dzL = z - m.zLower;
      if (std::abs(dzL) <= m.rWireCm) {
        const double dx = WrapToCell(x - m.phaseXCm, m.pitchCm);
        if (dx * dx + dzL * dzL <= r2) return true;
      }
      return false;
    }
    // Perforated: metal everywhere in the z-band outside an aperture.
    const double dx = WrapToCell(x - m.phaseXCm, m.pitchCm);
    const double dy = WrapToCell(y - m.phaseYCm, m.pitchCm);
    return !InHolePolygon(dx, dy, m.rApertureCm, m.sectors);
  }

  // True where the mesh is open — the complement of InMesh() within its z-band,
  // used by the cascade diagnostics to say whether a charge threaded the mesh or
  // landed on it.  For the woven model "open" is the square window between four
  // wires, not merely "not in a wire": a point level with the upper layer but
  // shadowed by a lower wire is not a clear path.
  static bool InMeshAperture(const MeshGeom& m, const double x, const double y) {
    const double dx = WrapToCell(x - m.phaseXCm, m.pitchCm);
    const double dy = WrapToCell(y - m.phaseYCm, m.pitchCm);
    if (m.model == MeshModel::Woven)
      return std::abs(dx) > m.rWireCm && std::abs(dy) > m.rWireCm;
    return InHolePolygon(dx, dy, m.rApertureCm, m.sectors);
  }

 private:
  // Inside the plate's copper or dielectric?  Material everywhere in the plate's
  // z-band except within the hole polygon of the nearest lattice point.
  static bool InPlate(const PlateGeom& p, const double x, const double y, const double z) {
    if (z > p.zTopCuTop || z < p.zBotCuBot) return false;
    const double dx = WrapToCell(x - p.latShiftCm, p.pitchCm);
    const double dy = WrapToCell(y, p.pitchCm);
    // Copper faces use the etched-back opening; the dielectric uses the drilled hole.
    const bool inCopper = (z >= p.zCen + p.zDielHalf) || (z <= p.zCen - p.zDielHalf);
    const double rOpen = inCopper ? p.rCuCm : p.rHoleCm;
    return !InHolePolygon(dx, dy, rOpen, p.sectors);
  }

  // Emit the plate's three solids (top Cu / dielectric / bottom Cu) as a row of
  // hole cells tiling the periodic cell in x.
  void AddPlate(const PlateGeom& p, const std::size_t sec) {
    const double hx = geom_.pitchCm / 2.0;   // one hole cell, not the whole periodic cell
    const double hy = geom_.pitchCm / 2.0;
    const double rCu = p.rCuCm;
    const double zTopCu = p.zCen + p.zDielHalf + p.tCuCm / 2.0;
    const double zBotCu = p.zCen - p.zDielHalf - p.tCuCm / 2.0;
    for (int m = 0; m < geom_.nHolesX; ++m) {
      const double xc = geom_.HoleX(m);
      if (p.tCuCm > 0.) {
        auto topCu = std::make_unique<SolidHole>(xc, 0., zTopCu, rCu, rCu, hx, hy,
                                                 p.tCuCm / 2.0);
        topCu->SetSectors(sec);
        topCu->SetBoundaryPotential(p.vTop);
        topCu->SetLabel(kElecThgemTop);
        if (thgemElemCm_ > 0.) topCu->SetDiscretisationLevel(thgemElemCm_);
        geo_.AddSolid(topCu.get(), &cu_);
        solids_.push_back(std::move(topCu));

        auto botCu = std::make_unique<SolidHole>(xc, 0., zBotCu, rCu, rCu, hx, hy,
                                                 p.tCuCm / 2.0);
        botCu->SetSectors(sec);
        botCu->SetBoundaryPotential(p.vBot);
        botCu->SetLabel(kElecThgemBot);
        if (thgemElemCm_ > 0.) botCu->SetDiscretisationLevel(thgemElemCm_);
        geo_.AddSolid(botCu.get(), &cu_);
        solids_.push_back(std::move(botCu));
      }
      auto foil = std::make_unique<SolidHole>(xc, 0., p.zCen, p.rHoleCm, p.rHoleCm,
                                              hx, hy, p.zDielHalf);
      foil->SetSectors(sec);
      foil->SetBoundaryDielectric();
      if (thgemElemCm_ > 0.) foil->SetDiscretisationLevel(thgemElemCm_);
      geo_.AddSolid(foil.get(), &diel_);
      solids_.push_back(std::move(foil));
    }
  }

  // Two orthogonal layers of wires.  Every wire carries the same label, which is
  // what makes this affordable: neBEM groups solids sharing a label into one
  // readout group, so the whole mesh gets a single weighting-field solve, and a
  // SolidWire is one line-charge primitive worth at most max_elements elements.
  void AddMeshWoven(const MeshGeom& m) {
    const double hx = geom_.cellXCm / 2.0;
    const double hy = geom_.cellYCm / 2.0;
    // Upper layer: wires ∥ x, spaced along y.
    for (int j = 0; j < m.nY; ++j) {
      auto w = std::make_unique<SolidWire>(0., m.Y(j), m.zUpper, m.rWireCm, hx,
                                           1., 0., 0.);
      w->SetBoundaryPotential(m.vMesh);
      w->SetLabel(kElecMesh);
      if (meshElemCm_ > 0.) w->SetDiscretisationLevel(meshElemCm_);
      geo_.AddSolid(w.get(), &cu_);
      solids_.push_back(std::move(w));
    }
    // Lower layer: wires ∥ y, spaced along x.
    for (int i = 0; i < m.nX; ++i) {
      auto w = std::make_unique<SolidWire>(m.X(i), 0., m.zLower, m.rWireCm, hy,
                                           0., 1., 0.);
      w->SetBoundaryPotential(m.vMesh);
      w->SetLabel(kElecMesh);
      if (meshElemCm_ > 0.) w->SetDiscretisationLevel(meshElemCm_);
      geo_.AddSolid(w.get(), &cu_);
      solids_.push_back(std::move(w));
    }
  }

  // A thin conductor sheet with a lattice of apertures, tiled as one SolidHole
  // per aperture (half-widths pitch/2, so the tiles cover the cell exactly).
  // Each aperture costs ~390 boundary elements, which is why this model does not
  // scale to micromegas pitch — see geometry.max_elements_budget.
  void AddMeshPerforated(const MeshGeom& m) {
    const double h = m.pitchCm / 2.0;
    for (int i = 0; i < m.nX; ++i) {
      for (int j = 0; j < m.nY; ++j) {
        auto tile = std::make_unique<SolidHole>(m.X(i), m.Y(j), m.zCen,
                                                m.rApertureCm, m.rApertureCm,
                                                h, h, m.tMeshCm / 2.0);
        tile->SetSectors(m.sectors);
        tile->SetBoundaryPotential(m.vMesh);
        tile->SetLabel(kElecMesh);
        if (meshElemCm_ > 0.) tile->SetDiscretisationLevel(meshElemCm_);
        geo_.AddSolid(tile.get(), &cu_);
        solids_.push_back(std::move(tile));
      }
    }
  }

  static ThgemMeshGeom ComputeGeom(const GeometryConfig& g, const FieldConfig& f) {
    ThgemMeshGeom o;
    o.pitchCm    = g.holePitchUm * 1.e-4;
    o.nHolesX    = g.holesPerWire;
    o.cellXCm    = o.nHolesX * o.pitchCm;
    o.cellYCm    = o.pitchCm;
    // Where the canonical lattice (m − (N−1)/2)·p already puts a hole depends on
    // the parity of N: odd N has a hole at x = 0 (under the wire), even N has them
    // at ±p/2 (between wires).  So the half-pitch shift has to follow that parity,
    // or `wire_between_holes` means the opposite of what it says for even N.
    const bool evenN = (o.nHolesX % 2 == 0);
    o.latShiftCm = (g.wireBetweenHoles != evenN) ? 0.5 * o.pitchCm : 0.;
    o.rWireCm    = g.wireDiameterUm * 0.5e-4;
    o.dDriftCm    = g.driftGapMm         * 0.1;
    o.dTransferCm = g.transferGapMm      * 0.1;
    o.dAmpCm      = g.amplificationGapMm * 0.1;

    PlateGeom& p = o.thgem;
    {
      const PlateConfig& c = g.thgem;
      p.rHoleCm   = c.holeDiameterUm * 0.5e-4;
      p.rCuCm     = p.rHoleCm + c.rimUm * 1.e-4;
      p.tDielCm   = c.plateThicknessUm  * 1.e-4;
      p.tCuCm     = c.copperThicknessUm * 1.e-4;
      p.epsDiel   = DielectricEpsR(c.dielectric);
      p.zDielHalf = p.tDielCm / 2.0;
      p.pitchCm   = o.pitchCm;
      p.latShiftCm = o.latShiftCm;
      p.sectors   = static_cast<std::size_t>(g.holeSectors);
    }

    // ── Mesh lattice: snap the pitch to fit the periodic cell ────────────────
    // cellX = nHolesX · cellY, so cellY is a divisor of cellX and a pitch that
    // divides cellY automatically divides cellX too.  The two commensurability
    // constraints therefore collapse into one, and holes_per_wire > 1 needs no
    // special handling at all.
    MeshGeom& m = o.mesh;
    {
      const MeshConfig& c = g.mesh;
      m.model      = (c.model == "perforated") ? MeshModel::Perforated : MeshModel::Woven;
      m.pitchReqUm = c.pitchUm;
      const double reqCm = c.pitchUm * 1.e-4;
      m.wiresPerHolePitch =
          static_cast<int>(std::max<long long>(1, std::llround(o.cellYCm / reqCm)));
      m.pitchCm = o.cellYCm / m.wiresPerHolePitch;
      m.pitchUm = m.pitchCm * 1.e4;
      m.nY = m.wiresPerHolePitch;
      m.nX = m.wiresPerHolePitch * o.nHolesX;
      // Wrap the offsets first: a stagger by a whole mesh pitch is a no-op, and
      // must not perturb the lattice phase or the cache key.
      m.offXCm = WrapToCell(c.offsetXUm * 1.e-4, m.pitchCm);
      m.offYCm = WrapToCell(c.offsetYUm * 1.e-4, m.pitchCm);
      m.phaseXCm = WrapToCell(m.X(0), m.pitchCm);
      m.phaseYCm = WrapToCell(m.Y(0), m.pitchCm);
      m.rWireCm     = c.wireDiameterUm * 0.5e-4;
      m.rApertureCm = c.apertureUm     * 0.5e-4;
      m.sectors     = static_cast<std::size_t>(c.apertureSectors);
      // Thickness: 0 = auto.  Two wire diameters puts the crossed layers exactly
      // tangent, which is the physical weave.
      m.tMeshCm = c.thicknessUm > 0. ? c.thicknessUm * 1.e-4 : 4.0 * m.rWireCm;
      if (m.model == MeshModel::Woven) {
        const double open = std::max(0., m.pitchCm - 2. * m.rWireCm);
        m.opticalTransparency = (open * open) / (m.pitchCm * m.pitchCm);
      } else {
        m.opticalTransparency =
            HolePolygonArea(m.rApertureCm, m.sectors) / (m.pitchCm * m.pitchCm);
      }
    }

    // ── z stack, anchored at the anode ───────────────────────────────────────
    o.zAnode   = 0.;
    m.zBot     = o.zAnode + o.dAmpCm;
    m.zTop     = m.zBot + m.tMeshCm;
    m.zCen     = 0.5 * (m.zBot + m.zTop);
    // Place the two woven layers so their outer surfaces land exactly on the
    // mesh's own z-band: separation = (thickness − wire diameter) / 2.
    const double sep = 0.5 * std::max(0., m.tMeshCm - 2. * m.rWireCm);
    m.zLower   = m.zCen - sep;
    m.zUpper   = m.zCen + sep;
    p.zBotCuBot = m.zTop + o.dTransferCm;
    p.zCen      = p.zBotCuBot + p.tCuCm + p.zDielHalf;
    p.zTopCuTop = p.zCen + p.zDielHalf + p.tCuCm;
    o.zWire     = p.zTopCuTop + o.dDriftCm;
    o.zMin      = o.zAnode;
    o.zMax      = o.zWire;

    // ── Physics fields → electrode potentials ────────────────────────────────
    // The anode is the 0 V reference and electrons drift −z toward it.  The mesh
    // is one conductor, so it takes one potential and its stage's gain comes
    // from the field across the amplification gap, not from a ΔV across it.
    const double eDriftVcm    = f.eDriftKvcm    * 1000.0;
    const double eTransferVcm = f.eTransferKvcm * 1000.0;
    o.vAnode = 0.0;
    m.vMesh  = o.vAnode - f.deltaVMeshAnodeV;
    m.eAmpKvcm = o.dAmpCm > 0. ? f.deltaVMeshAnodeV / o.dAmpCm * 1.e-3 : 0.;
    p.vBot   = m.vMesh  - eTransferVcm * o.dTransferCm;
    p.vTop   = p.vBot   - f.deltaVThgemV;
    o.vWire  = p.vTop   - eDriftVcm    * o.dDriftCm;
    return o;
  }

  MediumConductor cu_;
  MediumPlastic   diel_;
  GeometrySimple  geo_;
  ComponentNeBem3d nebem_;
  std::vector<std::unique_ptr<Solid>> solids_;
  ThgemMeshGeom geom_;
  double thgemElemCm_ = 0.;
  double meshElemCm_  = 0.;
};

// Lattice checks that need the *snapped* mesh pitch, and so cannot live in
// LoadConfig — it does not know the periodic cell.  Returns the number of
// warnings raised (hard errors throw).
std::size_t ValidateGeometry(const ThgemMeshGeom& g, const GeometryConfig& gc) {
  const MeshGeom& m = g.mesh;
  std::size_t warnings = 0;

  if (m.model == MeshModel::Woven && 2. * m.rWireCm >= m.pitchCm)
    throw std::runtime_error(
        "geometry.mesh.wire_diameter_um (" + FormatNumber(2. * m.rWireCm * 1.e4, 1) +
        " µm) must be smaller than the snapped mesh pitch (" +
        FormatNumber(m.pitchUm, 1) + " µm): the wires would touch and the mesh "
        "would have no optical transparency.");
  if (m.model == MeshModel::Perforated && 2. * m.rApertureCm >= m.pitchCm)
    throw std::runtime_error(
        "geometry.mesh.aperture_um (" + FormatNumber(2. * m.rApertureCm * 1.e4, 1) +
        " µm) must be smaller than the snapped mesh pitch (" +
        FormatNumber(m.pitchUm, 1) + " µm): the apertures would merge.");

  if (m.pitchReqUm > gc.holePitchUm * (1. + 1.e-9)) {
    std::cout << "\n  WARNING: geometry.mesh.pitch_um (" << FormatNumber(m.pitchReqUm, 1)
              << " µm) exceeds the cell y-period (" << FormatNumber(gc.holePitchUm, 1)
              << " µm).\n           A mesh coarser than the THGEM hole pitch cannot fit "
                 "one periodic cell; it\n           was snapped to "
              << FormatNumber(m.pitchUm, 1) << " µm (1 mesh cell per hole pitch). Raise "
                 "geometry.hole_pitch_um\n           if you want a genuinely coarser mesh.\n";
    ++warnings;
  }
  if (m.wiresPerHolePitch > 16) {
    std::cout << "\n  WARNING: " << m.wiresPerHolePitch
              << " mesh cells per hole pitch is past the practical resolution limit.\n"
                 "           Below hole_pitch / 16 the transport grid can no longer resolve "
                 "the mesh\n           (see the grid check below): raise geometry.hole_pitch_um, "
                 "or treat this\n           run as a field study rather than a transparency "
                 "study.\n";
    ++warnings;
  }
  return warnings;
}

// Projected boundary-element count, before neBEM builds anything.  Calibrated on
// the double-THGEM reference solve (569 primitives / ~2300 elements, one
// SolidHole at sectors 4 ≈ 390 elements).  The point is to fail in a second
// rather than in a day: neBEM inverts a dense N×N matrix, so the cost is O(N³)
// and the stored inverse is N² doubles.
std::size_t EstimateElements(const ThgemMeshGeom& g, const GeometryConfig& gc) {
  auto holeElems = [](const std::size_t sectors) -> std::size_t {
    // Panel count per SolidHole scales as 4 + 12·(sectors − 1); 40 panels at
    // sectors = 4 measured 390 elements.
    const double panels = 4. + 12. * static_cast<double>(sectors - 1);
    return static_cast<std::size_t>(std::llround(390. * panels / 40.));
  };
  const auto sec = static_cast<std::size_t>(gc.holeSectors);
  const auto wireElems = static_cast<std::size_t>(gc.maxElements);

  std::size_t n = wireElems;                                   // cathode wire
  n += static_cast<std::size_t>(g.nHolesX) *
       ((g.thgem.tCuCm > 0. ? 2 : 0) + 1) * holeElems(sec);    // THGEM
  if (g.mesh.model == MeshModel::Woven)
    n += static_cast<std::size_t>(g.mesh.nX + g.mesh.nY) * wireElems;
  else
    n += static_cast<std::size_t>(g.mesh.nX) *
         static_cast<std::size_t>(g.mesh.nY) * holeElems(g.mesh.sectors);
  n += 40;                                                     // anode box
  return n;
}

// ─── Field maps and validation ────────────────────────────────────────────────

// The reference hole axis: the THGEM hole nearest x = 0.  All on-axis profiles
// and the map slices are taken through it.
void ReferenceHoleAxis(const ThgemMeshGeom& g, double& xRef, double& yRef) {
  xRef = g.HoleX(0);
  for (int m = 1; m < g.nHolesX; ++m) {
    const double xm = g.HoleX(m);
    if (std::abs(xm) < std::abs(xRef)) xRef = xm;
  }
  yRef = 0.;
}

// The mesh's open cell nearest the THGEM hole axis: where a transferred electron
// actually crosses the mesh plane, and therefore where the amplification gap and
// the mesh's own z-band have to be sampled.
//
// Probing those two zones on the THGEM hole axis instead would, whenever the
// mesh lattice does not happen to line up with it, land the probe *on a mesh
// wire* and report a meaningless "reversed" field — the same trap the
// double-THGEM sibling documents for a staggered second plate.
void MeshApertureAxis(const ThgemMeshGeom& g, double& x, double& y) {
  double xRef = 0., yRef = 0.;
  ReferenceHoleAxis(g, xRef, yRef);
  const MeshGeom& m = g.mesh;
  // For the perforated model the open cell is centred on a lattice point; for
  // the woven model it is the window midway between four wires.
  const double half = (m.model == MeshModel::Woven) ? 0.5 * m.pitchCm : 0.;
  x = xRef - WrapToCell(xRef - (m.phaseXCm + half), m.pitchCm);
  y = yRef - WrapToCell(yRef - (m.phaseYCm + half), m.pitchCm);
}

// Sample the solved field on two slices through the reference hole axis — x–z
// (across the wires) and y–z (along them) — plus the on-axis profile, and write
// them to the ROOT file for the GUI's E-Field tab.  Unlike the single-plate
// sibling the two slices are not equivalent: the wires break the x/y symmetry.
void DumpFieldMap(Garfield::Component& cmp, const ThgemMeshGeom& g, TDirectory* dir) {
  if (!dir) return;
  double xRef = 0., yRef = 0., xM = 0., yM = 0.;
  ReferenceHoleAxis(g, xRef, yRef);
  MeshApertureAxis(g, xM, yM);
  const double xLo = xRef - g.cellXCm, xHi = xRef + g.cellXCm;   // two cells (periodicity)
  const double yLo = yRef - g.cellYCm, yHi = yRef + g.cellYCm;
  const double zLo = g.zMin, zHi = g.zMax;

  TH2D hMag("h_field_mag", "|E| (x-z through the hole axis);x [cm];z [cm]",
            kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  TH2D hPot("h_potential", "Potential (x-z);x [cm];z [cm]",
            kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  TH2D hEz("h_field_ez", "E_{z} (x-z);x [cm];z [cm]",
           kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  TH2D hEx("h_field_ex", "E_{x} (x-z);x [cm];z [cm]",
           kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  TH2D hMagYz("h_field_mag_yz", "|E| (y-z along the wire);y [cm];z [cm]",
              kMapNy, yLo, yHi, kMapNz, zLo, zHi);
  TH2D hPotYz("h_potential_yz", "Potential (y-z);y [cm];z [cm]",
              kMapNy, yLo, yHi, kMapNz, zLo, zHi);
  hMag.SetDirectory(nullptr);   hPot.SetDirectory(nullptr);
  hEz.SetDirectory(nullptr);    hEx.SetDirectory(nullptr);
  hMagYz.SetDirectory(nullptr); hPotYz.SetDirectory(nullptr);

  std::vector<double> axZ, axE, axV, axEz;
  axZ.reserve(kMapNz); axE.reserve(kMapNz); axV.reserve(kMapNz); axEz.reserve(kMapNz);
  // Unless the mesh lattice happens to line up with the THGEM hole, the hole
  // axis passes through mesh metal.  Dump a second set of profiles on the mesh's
  // own open cell — the path a transferred electron actually takes into the
  // amplification gap; when the two lattices align they simply coincide.
  std::vector<double> ax2E, ax2V, ax2Ez;
  ax2E.reserve(kMapNz); ax2V.reserve(kMapNz); ax2Ez.reserve(kMapNz);

  for (int iz = 0; iz < kMapNz; ++iz) {
    const double z = zLo + (iz + 0.5) * (zHi - zLo) / kMapNz;
    for (int ix = 0; ix < kMapNx; ++ix) {
      const double x = xLo + (ix + 0.5) * (xHi - xLo) / kMapNx;
      double ex = 0, ey = 0, ez = 0, v = 0;
      int status = 0;
      Garfield::Medium* m = nullptr;
      cmp.ElectricField(x, yRef, z, ex, ey, ez, v, m, status);
      hMag.SetBinContent(ix + 1, iz + 1,
                         std::sqrt(ex * ex + ey * ey + ez * ez) / 1000.0);  // kV/cm
      hPot.SetBinContent(ix + 1, iz + 1, v);
      hEz.SetBinContent(ix + 1, iz + 1, ez / 1000.0);   // kV/cm, signed
      hEx.SetBinContent(ix + 1, iz + 1, ex / 1000.0);
    }
    for (int iy = 0; iy < kMapNy; ++iy) {
      const double y = yLo + (iy + 0.5) * (yHi - yLo) / kMapNy;
      double ex = 0, ey = 0, ez = 0, v = 0;
      int status = 0;
      Garfield::Medium* m = nullptr;
      cmp.ElectricField(xRef, y, z, ex, ey, ez, v, m, status);
      hMagYz.SetBinContent(iy + 1, iz + 1,
                           std::sqrt(ex * ex + ey * ey + ez * ez) / 1000.0);
      hPotYz.SetBinContent(iy + 1, iz + 1, v);
    }
    // On-axis profile through the reference hole.
    double ex = 0, ey = 0, ez = 0, v = 0; int status = 0;
    Garfield::Medium* m = nullptr;
    cmp.ElectricField(xRef, yRef, z, ex, ey, ez, v, m, status);
    axZ.push_back(z);
    axE.push_back(std::sqrt(ex * ex + ey * ey + ez * ez) / 1000.0);
    axEz.push_back(ez / 1000.0);
    axV.push_back(v);

    cmp.ElectricField(xM, yM, z, ex, ey, ez, v, m, status);
    ax2E.push_back(std::sqrt(ex * ex + ey * ey + ez * ez) / 1000.0);
    ax2Ez.push_back(ez / 1000.0);
    ax2V.push_back(v);
  }

  const int n = static_cast<int>(axZ.size());
  TGraph gAxisE(n, axZ.data(), axE.data());
  gAxisE.SetName("g_axis_field");
  gAxisE.SetTitle("On-axis |E|;z [cm];|E| [kV/cm]");
  TGraph gAxisEz(n, axZ.data(), axEz.data());
  gAxisEz.SetName("g_axis_ez");
  gAxisEz.SetTitle("On-axis E_{z};z [cm];E_{z} [kV/cm]");
  TGraph gAxisV(n, axZ.data(), axV.data());
  gAxisV.SetName("g_axis_potential");
  gAxisV.SetTitle("On-axis potential;z [cm];V [V]");
  TGraph gAxisE2(n, axZ.data(), ax2E.data());
  gAxisE2.SetName("g_axis_field_mesh");
  gAxisE2.SetTitle("|E| on the mesh aperture axis;z [cm];|E| [kV/cm]");
  TGraph gAxisEz2(n, axZ.data(), ax2Ez.data());
  gAxisEz2.SetName("g_axis_ez_mesh");
  gAxisEz2.SetTitle("E_{z} on the mesh aperture axis;z [cm];E_{z} [kV/cm]");
  TGraph gAxisV2(n, axZ.data(), ax2V.data());
  gAxisV2.SetName("g_axis_potential_mesh");
  gAxisV2.SetTitle("Potential on the mesh aperture axis;z [cm];V [V]");

  dir->cd();
  hMag.Write("h_field_mag");
  hPot.Write("h_potential");
  hEz.Write("h_field_ez");
  hEx.Write("h_field_ex");
  hMagYz.Write("h_field_mag_yz");
  hPotYz.Write("h_potential_yz");
  gAxisE.Write("g_axis_field");
  gAxisEz.Write("g_axis_ez");
  gAxisV.Write("g_axis_potential");
  gAxisE2.Write("g_axis_field_mesh");
  gAxisEz2.Write("g_axis_ez_mesh");
  gAxisV2.Write("g_axis_potential_mesh");
  std::cout << "  Field maps dumped (" << kMapNx << "x" << kMapNz << " x-z, "
            << kMapNy << "x" << kMapNz << " y-z, on-axis profiles).\n";
}

// Field-surface estimate at a cathode wire (Sauli 1977 eq. 2.3): the drift-gap
// voltage divided by the wire's capacitance per unit length.  Only an estimate —
// it assumes a wire plane between two ground planes — but it is the right order
// of magnitude and answers the question the Magboltz table cares about: does the
// tabulated field range reach the largest field an electron will ever see?
double WireSurfaceFieldVcm(const ThgemMeshGeom& g) {
  const double sCm = g.cellXCm;                 // wire pitch
  const double rCm = g.rWireCm;
  const double dV  = std::abs(g.vWire - g.thgem.vTop);
  const double cap = std::log(sCm / (2. * kPi * rCm)) + kPi * g.dDriftCm / sCm;
  if (cap <= 0. || rCm <= 0.) return 0.;
  return dV / (rCm * cap);
}

// The same estimate for a mesh wire facing the anode across the amplification
// gap.  In a micromegas this is the largest field anywhere in the detector — it
// is what sets the sparking limit — and it is also the field most likely to run
// off the end of the Magboltz table, so ValidateField folds it into the ceiling
// check.  Meaningless for the perforated model, which has no wires.
double MeshWireSurfaceFieldVcm(const ThgemMeshGeom& g) {
  const MeshGeom& m = g.mesh;
  if (m.model != MeshModel::Woven || m.rWireCm <= 0.) return 0.;
  const double sCm = m.pitchCm;
  const double dV  = std::abs(m.vMesh - g.vAnode);
  const double cap = std::log(sCm / (2. * kPi * m.rWireCm)) + kPi * g.dAmpCm / sCm;
  if (cap <= 0.) return 0.;
  return dV / (m.rWireCm * cap);
}

const char* MeshModelName(const MeshModel m) {
  return m == MeshModel::Woven ? "woven" : "perforated";
}

// Can the uniform transport grid actually see the mesh and the amplification gap?
//
// This is the one check whose failure is otherwise invisible.  ComponentGrid has
// no material map of its own — the in-solid flag stamped by SampleFieldToFile is
// all that stops an electron drifting through metal — and that flag is sampled at
// grid nodes.  If no node lands inside a mesh wire, nothing absorbs charge there:
// every electron threads the mesh, the measured electron transparency comes out
// at exactly 1.00, and the run looks entirely healthy while telling you nothing.
//
// The tempting cheap fix — flagging a node absorbing when a wire passes within
// half a grid cell — is deliberately not done: it over-blocks, and biases the
// transparency downward by an amount that depends on the grid rather than on the
// physics, which is worse than a known-wrong 1.00.  Resolve the mesh instead.
std::size_t ValidateGridResolution(const ThgemMeshGeom& g, const GeometryConfig& gc) {
  const double dx = g.cellXCm / (gc.gridNx - 1.);
  const double dy = g.cellYCm / (gc.gridNy - 1.);
  const double dz = (g.zMax - g.zMin) / (gc.gridNz - 1.);
  const double dMax = std::max({dx, dy, dz});
  // The narrowest metal feature the grid has to catch: a wire, or the web
  // between two apertures.
  const MeshGeom& m = g.mesh;
  const double feature = (m.model == MeshModel::Woven)
                             ? 2. * m.rWireCm
                             : std::max(0., m.pitchCm - 2. * m.rApertureCm);
  const double nodesAcross   = dMax > 0. ? feature / dMax : 0.;
  const double cellsPerAmpGap = dz > 0. ? g.dAmpCm / dz : 0.;

  std::cout << "    transport grid  dx " << FormatNumber(dx * 1.e4, 1)
            << " µm, dy " << FormatNumber(dy * 1.e4, 1)
            << " µm, dz " << FormatNumber(dz * 1.e4, 1) << " µm"
            << "   nodes across the mesh " << FormatNumber(nodesAcross, 1)
            << ", cells across the amp gap " << FormatNumber(cellsPerAmpGap, 1) << "\n";

  std::size_t warnings = 0;
  if (nodesAcross < 3.) {
    // Required node counts to reach 3 across the feature, per axis.
    const double need = feature / 3.;
    const int nx = static_cast<int>(std::ceil(g.cellXCm / need)) + 1;
    const int ny = static_cast<int>(std::ceil(g.cellYCm / need)) + 1;
    const int nz = static_cast<int>(std::ceil((g.zMax - g.zMin) / need)) + 1;
    std::cout << "\n  WARNING: the transport grid cannot resolve the mesh ("
              << FormatNumber(nodesAcross, 1) << " nodes across a "
              << FormatNumber(feature * 1.e4, 1) << " µm feature).\n";
    if (nodesAcross < 2.)
      std::cout << "           Below two nodes essentially every electron passes through "
                   "the mesh, so the\n           measured electron transparency will read "
                   "~1.00 whatever the field ratio is.\n";
    std::cout << "           Raise geometry.grid_nx/grid_ny/grid_nz to at least "
              << nx << "/" << ny << "/" << nz
              << ", or coarsen geometry.mesh.pitch_um.\n";
    ++warnings;
  }
  if (cellsPerAmpGap < 10.) {
    const int nz = static_cast<int>(
        std::ceil((g.zMax - g.zMin) / (g.dAmpCm / 10.))) + 1;
    std::cout << "\n  WARNING: only " << FormatNumber(cellsPerAmpGap, 1)
              << " grid cells span the amplification gap, where all of the\n"
                 "           second-stage gain happens.  The Sensor's drift area is inset "
                 "by one grid\n           cell at each end, so at this resolution the inset "
                 "alone eats a sizeable\n           fraction of the gap. Raise "
                 "geometry.grid_nz to at least " << nz << ".\n";
    ++warnings;
  }
  return warnings;
}

// Walk the reference hole axis and check the field is physically sane before any
// charge is transported.  The dominant failure mode of a periodic neBEM solve is
// a *reversed* on-axis field in one of the gaps: the tiled wire and anode patches
// only approximate infinite structures, and a truncated periodic sum flips the
// sign mid-gap, trapping every drifting charge.  It is silent, and it wastes
// hours downstream, so it is checked explicitly and loudly.
//
// Sign convention: the bottom electrode is the 0 V reference and the wire is the
// most negative, so the potential falls with z and E_z = -dV/dz is POSITIVE
// everywhere in the gas.  Electrons (charge -e) are pushed to -z, toward the
// readout.  Returns false if any gap is reversed.
std::size_t ValidateField(Garfield::Component& cmp, const ThgemMeshGeom& g,
                          const GasConfig& gas, const FieldConfig& f,
                          const GeometryConfig& gc, const bool onGrid) {
  double xRef = 0., yRef = 0., xM = 0., yM = 0.;
  ReferenceHoleAxis(g, xRef, yRef);
  MeshApertureAxis(g, xM, yM);

  // Each zone is sampled on an axis that is actually *gas* there.  Unless the
  // mesh lattice lines up with the THGEM hole, the hole axis runs straight
  // through mesh metal — sampling the amplification field there would report a
  // meaningless (and, since the field inside a conductor is zero and numerically
  // noisy, spuriously "reversed") result.
  // `appliedKvcm` / `key` are set for the *gaps*, whose field is dialled in
  // directly.  The THGEM hole is driven by its plate's dV, and the mesh's own
  // z-band by the funnel between two gaps, so neither carries a key — a reversal
  // there is never a "you asked for zero field" case.
  // `applied` is the field dialled into a gap.  `appliedText` spells out what
  // was actually *set*, which for the amplification gap is a voltage — without
  // it the starved-gap warning below would report a volt-named key "is 0.000
  // kV/cm", which reads as a contradiction.
  struct Zone {
    const char* name; double zLo, zHi, x, y;
    double appliedKvcm; const char* key; std::string appliedText;
  };
  std::vector<Zone> zones;
  zones.push_back({"amplification", g.zAnode, g.mesh.zBot, xM, yM,
                   g.mesh.eAmpKvcm, "fields.delta_v_mesh_anode_V",
                   FormatNumber(f.deltaVMeshAnodeV, 1) + " V (" +
                   FormatNumber(g.mesh.eAmpKvcm, 3) + " kV/cm over " +
                   FormatNumber(g.dAmpCm * 10., 3) + " mm)"});
  zones.push_back({"mesh aperture", g.mesh.zBot, g.mesh.zTop, xM, yM, -1., nullptr, ""});
  zones.push_back({"transfer",      g.mesh.zTop, g.thgem.zBotCuBot, xRef, yRef,
                   f.eTransferKvcm, "fields.e_transfer_kvcm",
                   FormatNumber(f.eTransferKvcm, 3) + " kV/cm"});
  zones.push_back({"THGEM hole",    g.thgem.zBotCuBot, g.thgem.zTopCuTop, xRef, yRef,
                   -1., nullptr, ""});
  zones.push_back({"drift",         g.thgem.zTopCuTop, g.zWire, xRef, yRef,
                   f.eDriftKvcm, "fields.e_drift_kvcm",
                   FormatNumber(f.eDriftKvcm, 3) + " kV/cm"});

  const bool offset = (std::abs(xM - xRef) > 1e-9 || std::abs(yM - yRef) > 1e-9);
  std::cout << "\n  Field validation on the THGEM hole axis (x = "
            << FormatNumber(xRef, 5) << " cm, y = " << FormatNumber(yRef, 5) << " cm)";
  if (offset)
    std::cout << ", mesh on its aperture axis (x = " << FormatNumber(xM, 5)
              << " cm, y = " << FormatNumber(yM, 5) << " cm)";
  std::cout << ":\n";
  std::size_t warnings = 0;
  double peakOverall = 0.;
  // Gaps whose reversal is explained by their own applied field being ~0.
  std::vector<Zone> starvedGaps, otherReversed;
  constexpr int kSamples = 60;
  for (const auto& z : zones) {
    // Inset by 2 % of the zone so a sample never lands exactly on a conductor.
    const double pad = 0.02 * (z.zHi - z.zLo);
    double ezMin = 1e30, ezMax = -1e30, ePeak = 0., vLo = 0., vHi = 0.;
    for (int i = 0; i < kSamples; ++i) {
      const double zz = z.zLo + pad +
                        (z.zHi - z.zLo - 2. * pad) * i / (kSamples - 1.);
      double ex = 0, ey = 0, ez = 0, v = 0; int status = 0;
      Garfield::Medium* m = nullptr;
      cmp.ElectricField(z.x, z.y, zz, ex, ey, ez, v, m, status);
      ezMin = std::min(ezMin, ez);
      ezMax = std::max(ezMax, ez);
      ePeak = std::max(ePeak, std::sqrt(ex * ex + ey * ey + ez * ez));
      if (i == 0) vLo = v;
      if (i == kSamples - 1) vHi = v;
    }
    peakOverall = std::max(peakOverall, ePeak);
    const bool reversed  = ezMin <= 0.;
    const bool vFalling  = vHi < vLo;   // potential must fall as z rises
    std::cout << "    " << std::setw(12) << std::left << z.name << std::right
              << "  Ez " << std::setw(9) << FormatNumber(ezMin / 1000., 3)
              << " .. " << std::setw(9) << FormatNumber(ezMax / 1000., 3) << " kV/cm"
              << "   peak |E| " << std::setw(8) << FormatNumber(ePeak / 1000., 2) << " kV/cm"
              << "   dV " << std::setw(9) << FormatNumber(vHi - vLo, 1) << " V";
    if (reversed || !vFalling) {
      std::cout << "   <-- REVERSED";
      ++warnings;
      // A gap with no potential across it is reversed *by construction*: the
      // neighbouring hole fields are the only thing acting there.
      constexpr double kStarvedKvcm = 0.05;
      if (z.key && z.appliedKvcm >= 0. && z.appliedKvcm < kStarvedKvcm)
        starvedGaps.push_back(z);
      else
        otherReversed.push_back(z);
    }
    std::cout << "\n";
  }

  const double eWire = WireSurfaceFieldVcm(g);
  const double eMeshWire = MeshWireSurfaceFieldVcm(g);
  std::cout << "    cathode wire surface (estimate)  |E| ~ "
            << FormatNumber(eWire / 1000., 2) << " kV/cm\n";
  if (eMeshWire > 0.)
    std::cout << "    mesh wire surface (estimate)     |E| ~ "
              << FormatNumber(eMeshWire / 1000., 2) << " kV/cm\n";
  std::cout << "    mesh optical transparency "
            << FormatNumber(g.mesh.opticalTransparency, 3) << " ("
            << MeshModelName(g.mesh.model) << ", geometric)\n";

  // Electron transparency: the fraction of transferred charge that threads the
  // mesh rather than landing on it.  It is driven by the ratio of the fields on
  // the two sides — the amplification field has to pull the drift lines through
  // the apertures against the transfer field's spread.  Below ~20 the mesh
  // collects most of the charge the THGEM produced.
  const double ratio = f.eTransferKvcm > 0. ? g.mesh.eAmpKvcm / f.eTransferKvcm : 0.;
  std::cout << "    amplification field E_amp = "
            << FormatNumber(g.mesh.eAmpKvcm, 2) << " kV/cm ("
            << FormatNumber(f.deltaVMeshAnodeV, 1) << " V over "
            << FormatNumber(g.dAmpCm * 10., 3) << " mm)\n";
  std::cout << "    field ratio E_amp / E_transfer = "
            << (f.eTransferKvcm > 0. ? FormatNumber(ratio, 1) : std::string("inf")) << "\n";
  if (f.eTransferKvcm > 0. && ratio < 20.) {
    std::cout << "\n  WARNING: E_amp / E_transfer = " << FormatNumber(ratio, 1)
              << " is too low for the mesh to be electron-transparent.\n"
                 "           Charge leaving the THGEM will be collected on the mesh "
                 "rather than entering\n           the amplification gap. Raise "
                 "fields.delta_v_mesh_anode_V, narrow geometry.amplification_gap_mm,\n"
                 "           or lower fields.e_transfer_kvcm.\n";
    ++warnings;
  }

  // Grid-resolution budget.  Only meaningful on the sampled transport grid —
  // the raw neBEM pass has no grid — and this is the check that stands between
  // a believable transparency number and a silently meaningless one.
  if (onGrid) warnings += ValidateGridResolution(g, gc);

  const double needed = std::max({peakOverall, eWire, eMeshWire});
  if (needed > gas.eFieldMaxVcm) {
    std::cout << "\n  WARNING: the largest field in the detector ("
              << FormatNumber(needed / 1000., 1) << " kV/cm) exceeds the Magboltz "
              << "table ceiling gas.e_field_max_vcm = "
              << FormatNumber(gas.eFieldMaxVcm / 1000., 1) << " kV/cm.\n"
              << "           Transport there is extrapolated; raise e_field_max_vcm to "
              << "at least " << FormatNumber(1.5 * needed / 1000., 0)
              << " kV/cm (a new gas table will be generated).\n";
    ++warnings;
  }
  for (const Zone& z : starvedGaps) {
    std::cout << "\n  WARNING: the on-axis field is reversed in the " << z.name
              << " gap, whose applied\n           " << z.key << " is "
              << z.appliedText << ".  With no potential across "
                 "the gap the\n           neighbouring hole fields are the only thing "
                 "acting there, so they reverse it\n           near the boundary.  That "
                 "is the configuration, not the solve: raise\n           "
              << z.key << " to sweep charge through the gap.\n";
  }
  if (!otherReversed.empty()) {
    std::cout << "\n  WARNING: the on-axis field is reversed in:";
    for (const Zone& z : otherReversed) std::cout << " " << z.name;
    std::cout << ".\n           With a field applied across it that is almost always too "
                 "few periodic\n           copies: raise geometry.periodic_copies (currently "
              << gc.periodicCopies << ") and re-run.\n";
  }
  if (warnings == 0) {
    std::cout << "  Field validation passed: E_z > 0 in every zone, potential falls "
                 "monotonically toward the wire,\n                            the mesh is "
                 "resolved and electron-transparent.\n";
  }
  return warnings;
}

// Estimate how long an electron needs to cross the whole stack, from just below
// the wire plane to the bottom electrode, by integrating dz / v_drift(|E|).  This
// is precisely the quantity simulation.time_window_ns has to cover, and getting it
// wrong is silent: every primary ends as StatusOutsideTimeWindow in the drift gap
// and the run reports no avalanche at all, which reads like a broken field rather
// than a too-short window.
//
// A wire cathode makes this easy to underestimate twice over.  "e_drift_kvcm" is
// the gap *average*, while the local field sags well below it in the bulk of the
// gap; and the field is strongly non-uniform across the cell — weakest at the
// corners, far from both a wire and a hole, where it can be several times smaller
// than on the hole axis.  So the drift gap is scanned at several lateral positions
// and the slowest one is taken; below the THGEM the charge is inside the funnel
// and the hole / mesh-aperture axes are representative.
//
// Ignores diffusion and the sideways path into a hole, so it remains a lower bound.
struct TransitEstimate {
  double axisNs  = 0.;   // hole axis, all the way down
  double worstNs = 0.;   // slowest lateral start in the drift gap, then the axis
};

TransitEstimate EstimateTransitTimeNs(Garfield::Component& cmp, const ThgemMeshGeom& g,
                                      MediumMagboltz& gas) {
  double xRef = 0., yRef = 0., xM = 0., yM = 0.;
  ReferenceHoleAxis(g, xRef, yRef);
  MeshApertureAxis(g, xM, yM);

  // Time to fall from zHi to zLo along a fixed (x, y), or along the stage-wise
  // axes when `followAxes` is set: the THGEM hole above the mesh, the mesh's own
  // open cell at and below it.
  auto integrate = [&](double x, double y, double zHi, double zLo,
                       bool followAxes) {
    const int steps = 1200;
    const double dz = (zHi - zLo) / steps;
    double t = 0.;
    for (int i = 0; i < steps; ++i) {
      const double z = zHi - (i + 0.5) * dz;
      double px = x, py = y;
      if (followAxes) {
        const bool belowMesh = (z <= g.mesh.zTop);
        px = belowMesh ? xM : xRef;
        py = belowMesh ? yM : yRef;
      }
      double ex = 0, ey = 0, ez = 0, v = 0; int status = 0;
      Garfield::Medium* m = nullptr;
      cmp.ElectricField(px, py, z, ex, ey, ez, v, m, status);
      double vx = 0., vy = 0., vz = 0.;
      if (!gas.ElectronVelocity(ex, ey, ez, 0., 0., 0., vx, vy, vz)) continue;
      const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);   // cm/ns
      if (speed <= 0.) continue;
      t += dz / speed;
    }
    return t;
  };

  TransitEstimate out;
  out.axisNs = integrate(0., 0., g.zMax, g.zMin, /*followAxes=*/true);

  // Below the THGEM every path is in the funnel; only the drift gap differs.
  const double tBelow = integrate(0., 0., g.thgem.zTopCuTop, g.zMin, /*followAxes=*/true);
  const double hx = g.cellXCm / 2.0, hy = g.cellYCm / 2.0;
  const std::array<std::pair<double, double>, 5> probes{{
      {xRef, yRef},                    // on the hole axis
      {hx, hy}, {hx, 0.}, {0., hy},    // cell edges and the corner
      {0.5 * hx, 0.5 * hy},
  }};
  double slowestDrift = 0.;
  for (const auto& [px, py] : probes) {
    slowestDrift = std::max(
        slowestDrift,
        integrate(px, py, g.zMax, g.thgem.zTopCuTop, /*followAxes=*/false));
  }
  out.worstNs = slowestDrift + tBelow;
  return out;
}

// ─── Electrode weighting fields (true neBEM solve, sampled to the grid) ───────
// Each read-out electrode's Shockley-Ramo weighting field (1 V on it, 0 V on the
// others) is solved by neBEM, sampled onto the transport grid via
// ComponentGrid::SaveWeightingField, and used both for the induced signal and for
// the Weighting Field display.  ComponentGrid holds one weighting field per
// instance, so there is one grid per electrode.

// Write one electrode's weighting map (x-z slice through the reference hole axis)
// into the run's ROOT file for the GUI's Weighting Field tab, read back from its
// loaded grid.
void DumpWeightingMap(TDirectory* dir, const std::string& id,
                      Garfield::Component& wgrid, const ThgemMeshGeom& g) {
  if (!dir) return;
  double xRef = 0., yRef = 0.;
  ReferenceHoleAxis(g, xRef, yRef);
  const double xLo = xRef - g.cellXCm, xHi = xRef + g.cellXCm;
  const double zLo = g.zMin, zHi = g.zMax;

  TH2D hW(("h_wpot_" + id).c_str(),
          (id + " weighting potential (neBEM);x [cm];z [cm]").c_str(),
          kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  TH2D hWE(("h_wfield_mag_" + id).c_str(),
           (id + " |E_{w}| (neBEM);x [cm];z [cm]").c_str(),
           kMapNx, xLo, xHi, kMapNz, zLo, zHi);
  hW.SetDirectory(nullptr);
  hWE.SetDirectory(nullptr);

  std::vector<double> axZ, axW;
  axZ.reserve(kMapNz);
  axW.reserve(kMapNz);

  for (int iz = 0; iz < kMapNz; ++iz) {
    const double z = zLo + (iz + 0.5) * (zHi - zLo) / kMapNz;
    for (int ix = 0; ix < kMapNx; ++ix) {
      const double x = xLo + (ix + 0.5) * (xHi - xLo) / kMapNx;
      double wx = 0., wy = 0., wz = 0.;
      wgrid.WeightingField(x, yRef, z, wx, wy, wz, id);      // label ignored by the grid
      hW.SetBinContent(ix + 1, iz + 1, wgrid.WeightingPotential(x, yRef, z, id));
      hWE.SetBinContent(ix + 1, iz + 1, std::sqrt(wx * wx + wy * wy + wz * wz));
    }
    axZ.push_back(z);
    axW.push_back(wgrid.WeightingPotential(xRef, yRef, z, id));
  }

  TGraph gAxisW(static_cast<int>(axZ.size()), axZ.data(), axW.data());
  gAxisW.SetName(("g_axis_wpot_" + id).c_str());
  gAxisW.SetTitle((id + " on-axis weighting potential;z [cm];W").c_str());

  dir->cd();
  hW.Write();
  hWE.Write();
  gAxisW.Write();
}

// ─── neBEM -> ComponentGrid sampling and caching ──────────────────────────────

// Sample the neBEM field onto the grid nodes and write a ComponentGrid "xyz"
// cache file with a trailing in-solid flag (1 = gas, 0 = inside copper/dielectric
// -> absorbing).  The flag is what lets electrons terminate on the electrodes:
// ComponentGrid has no material map, so without it charges drift through metal.
void SampleFieldToFile(ComponentNeBem3d& cmp, const ThgemMeshDetector& det,
                       const ThgemMeshGeom& g, const GeometryConfig& gc,
                       const std::string& file) {
  // Write atomically via a temp file + rename: an interrupted or crashing sample
  // must never leave a truncated file that looks like a valid cache on the next run.
  const std::string tmp = file + ".part";
  {
    std::ofstream out(tmp);
    if (!out) throw std::runtime_error("Cannot write field cache: " + tmp);
    out << std::setprecision(8);
    const int NX = gc.gridNx, NY = gc.gridNy, NZ = gc.gridNz;
    const double hx = g.cellXCm / 2.0, hy = g.cellYCm / 2.0;
    const double dx = 2. * hx / std::max(NX - 1, 1);
    const double dy = 2. * hy / std::max(NY - 1, 1);
    const double dz = (g.zMax - g.zMin) / std::max(NZ - 1, 1);
    const std::size_t total =
        static_cast<std::size_t>(NX) * static_cast<std::size_t>(NY) *
        static_cast<std::size_t>(NZ);
    std::size_t done = 0, nextPct = 10, nNonFinite = 0;
    for (int i = 0; i < NX; ++i) {
      const double x = -hx + i * dx;
      for (int j = 0; j < NY; ++j) {
        const double y = -hy + j * dy;
        for (int k = 0; k < NZ; ++k) {
          const double z = g.zMin + k * dz;
          double ex = 0, ey = 0, ez = 0, v = 0; int status = 0;
          Garfield::Medium* m = nullptr;
          cmp.ElectricField(x, y, z, ex, ey, ez, v, m, status);
          const int flag = det.InGas(x, y, z) ? 1 : 0;
          // A node landing on the wire axis sits on the 1/r singularity of
          // neBEM's line-charge primitive and comes back non-finite.  Such a
          // node is *inside* the conductor (InGas already flags it absorbing),
          // so zero is the right field there — and a bare "nan" in the file
          // makes ComponentGrid::LoadData reject the whole cache.
          if (!std::isfinite(ex) || !std::isfinite(ey) ||
              !std::isfinite(ez) || !std::isfinite(v)) {
            ex = ey = ez = 0.;
            if (!std::isfinite(v)) v = 0.;
            ++nNonFinite;
          }
          out << x << ' ' << y << ' ' << z << ' '
              << ex << ' ' << ey << ' ' << ez << ' ' << v << ' ' << flag << '\n';
          if (++done * 100 >= nextPct * total) {
            std::cout << "    sampling " << nextPct << "%\n";
            nextPct += 10;
          }
        }
      }
    }
    out.flush();
    if (!out) throw std::runtime_error("Write error while sampling field cache: " + tmp);
    if (nNonFinite > 0) {
      std::cout << "    " << nNonFinite << " node(s) on a conductor singularity "
                   "(wire axis) zeroed and marked absorbing.\n";
    }
  }  // close the stream before renaming
  fs::rename(tmp, file);
}

// Replace any non-finite token (nan / inf) in a whitespace-separated numeric file
// with 0, in place.  Needed for the files ComponentGrid::SaveWeightingField writes:
// a grid node landing exactly on the wire axis sits on the 1/r singularity of
// neBEM's line-charge primitive and is sampled as "nan", which then makes
// ComponentGrid::LoadData reject the *whole* file.  Such a node is inside the
// conductor, where the weighting field is zero and no charge ever reaches, so
// zeroing it is both correct and local.  (The transport-field sampler guards the
// same case inline; this is the version for a file Garfield wrote itself.)
// Returns the number of tokens scrubbed.
std::size_t ScrubNonFinite(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("Cannot read for scrubbing: " + file);
  const std::string tmp = file + ".part";
  std::ofstream out(tmp);
  if (!out) throw std::runtime_error("Cannot write while scrubbing: " + tmp);

  std::size_t nFixed = 0;
  std::string line, token;
  while (std::getline(in, line)) {
    if (line.find_first_of("nNiI") == std::string::npos) {   // fast path: no nan/inf
      out << line << '\n';
      continue;
    }
    std::istringstream ls(line);
    bool first = true;
    while (ls >> token) {
      // Match the literal spelling rather than round-tripping through operator>>:
      // libc++'s numeric extraction *rejects* "nan"/"inf" instead of parsing them,
      // so a parse-then-isfinite test silently passes them straight through.
      std::string lower = token;
      std::transform(lower.begin(), lower.end(), lower.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (lower.find("nan") != std::string::npos ||
          lower.find("inf") != std::string::npos) { token = "0"; ++nFixed; }
      if (!first) out << ' ';
      out << token;
      first = false;
    }
    out << '\n';
  }
  out.flush();
  if (!out) throw std::runtime_error("Write error while scrubbing: " + tmp);
  out.close();
  in.close();
  fs::rename(tmp, file);
  return nFixed;
}

// Count the lines in a file.  Used to detect a truncated field cache: a short
// file is not rejected by ComponentGrid::LoadElectricField (it silently leaves the
// missing nodes at zero, which would run the avalanche in a bogus ~zero field), so
// the node count must be validated against the grid before the cache is trusted.
std::size_t CountFileLines(const std::string& file) {
  std::ifstream in(file);
  std::size_t n = 0;
  std::string line;
  while (std::getline(in, line)) ++n;
  return n;
}

// The geometry half of a cache key: everything that changes the *shape* of the
// problem.  Shared by the transport-field and weighting-field keys so the two
// can never disagree about which geometry they describe.
std::string GeometryKey(const GeometryConfig& g, const MeshGeom& mg) {
  auto I = [](double v) {
    return std::to_string(static_cast<long long>(std::llround(v)));
  };
  std::ostringstream ss;
  ss << "p" << I(g.holePitchUm)
     << "_n" << g.holesPerWire << (g.wireBetweenHoles ? "b" : "o")
     << "_w" << I(g.wireDiameterUm);
  // THGEM
  ss << "_Th" << I(g.thgem.holeDiameterUm)
     << "t" << I(g.thgem.plateThicknessUm)
     << "c" << I(g.thgem.copperThicknessUm)
     << "_" << g.thgem.dielectric;
  if (g.thgem.rimUm != 0.) ss << "_rim" << I(g.thgem.rimUm);
  // Mesh.  The *snapped* pitch, never the requested one: two configs that snap
  // to the same lattice are the same geometry and must share a cache, and one
  // that snaps somewhere else must not silently reuse it.  Likewise the offsets
  // come from MeshGeom, already wrapped, so a stagger by a whole mesh pitch —
  // which changes nothing — does not invalidate the cache.
  ss << "_M" << (mg.model == MeshModel::Woven ? "w" : "p")
     << "mp" << FileSafeNumber(mg.pitchUm);
  if (mg.model == MeshModel::Woven) ss << "mw" << I(g.mesh.wireDiameterUm);
  else ss << "ma" << I(g.mesh.apertureUm) << "ms" << g.mesh.apertureSectors;
  ss << "mt" << FileSafeNumber(mg.tMeshCm * 1.e4);
  if (mg.offXCm != 0. || mg.offYCm != 0.)
    ss << "_moff" << FileSafeNumber(mg.offXCm * 1.e4) << "x"
       << FileSafeNumber(mg.offYCm * 1.e4);
  ss << "_dg" << FileSafeNumber(g.driftGapMm)
     << "_tg" << FileSafeNumber(g.transferGapMm)
     << "_ag" << FileSafeNumber(g.amplificationGapMm)
     << "_g" << g.gridNx << "x" << g.gridNy << "x" << g.gridNz
     << "_s" << g.holeSectors << "_pc" << g.periodicCopies;
  // The discretisation belongs in the *shared* key, not only in the weighting
  // one: it changes the solved field, so a transport cache keyed without it
  // would be silently reused after the mesh density was changed.  (The
  // double-THGEM sibling has exactly that bug.)
  ss << "_e" << I(g.targetElementSizeUm)
     << "_n" << g.minElements << "-" << g.maxElements;
  if (g.thgemElementSizeUm  > 0.) ss << "_te" << I(g.thgemElementSizeUm);
  if (g.mesh.elementSizeUm  > 0.) ss << "_me" << I(g.mesh.elementSizeUm);
  return ss.str();
}

// Cache filename for the sampled transport field, keyed by the geometry *and* the
// applied fields (mirrors the .gas caching): identical parameters reuse the neBEM
// sample instead of re-solving.
std::string DeriveFieldCacheName(const GeometryConfig& g, const MeshGeom& mg,
                                 const FieldConfig& f) {
  auto I = [](double v) {
    return std::to_string(static_cast<long long>(std::llround(v)));
  };
  std::ostringstream ss;
  ss << "thgem_mesh_field_" << GeometryKey(g, mg)
     << "_dV" << I(f.deltaVThgemV)
     << "_Ed" << FileSafeNumber(f.eDriftKvcm)
     << "_Et" << FileSafeNumber(f.eTransferKvcm)
     << "_dVma" << I(f.deltaVMeshAnodeV)
     << "_v1.txt";
  return ss.str();
}

// Cache filename for one electrode's sampled weighting field.  Keyed by the
// geometry and discretisation *only* — a Shockley-Ramo weighting field depends
// on the electrode shapes, not on the applied voltages — so an entire dV scan
// reuses a single solve.
std::string DeriveWeightingCacheName(const GeometryConfig& g, const MeshGeom& mg,
                                     const std::string& id) {
  return "thgem_mesh_wpot_" + id + "_" + GeometryKey(g, mg) + "_v1.txt";
}

// ─── Sensor setup ─────────────────────────────────────────────────────────────

// One weighting grid per read-out electrode, each holding that electrode's neBEM
// weighting field sampled onto the transport grid.
void SetupSensor(Sensor& sensor, ComponentGrid& grid,
                 const std::vector<std::unique_ptr<ComponentGrid>>& wgrids,
                 const std::vector<std::string>& ids,
                 const ThgemMeshGeom& g, const SimulationConfig& sim, int gridNz) {
  sensor.AddComponent(&grid);            // transport field + drift medium
  for (std::size_t e = 0; e < ids.size(); ++e) {
    sensor.AddElectrode(wgrids[e].get(), ids[e]);
  }

  const std::size_t nBins =
      static_cast<std::size_t>(std::round(sim.timeWindowNs / sim.timeStepNs));
  sensor.SetTimeWindow(0., sim.timeStepNs, nBins);

  // Wide in x/y (periodicity supplies the field for any cell), bounded in z to
  // the bottom electrode -> wire plane span: charges are collected when they
  // leave in z.  Several cells of lateral room let the avalanche and diffusion
  // spread, as in the Garfield GEM examples.
  //
  // Inset the z-bounds by one transport-grid cell so a charge is collected the
  // moment it reaches the last cell in front of an electrode.  With zmin exactly
  // at the anode the trilinear field in that final cell is too weak to push the
  // electron across the boundary, so it would otherwise hover ~one cell above
  // the anode and diffuse sideways along it for the whole time window --
  // producing spurious "anode-slide" drift lines (and StatusOutsideTimeWindow
  // fates) instead of a clean collection.  AvalancheMicroscopic stops a charge
  // when a step lands outside the area (StatusLeftDriftArea); raising zmin a
  // cell makes the bottom electrode absorb on contact.
  //
  // The same inset at the top is what keeps charge off the wires: the wire is
  // thinner than a grid cell, so no grid node ever falls inside it and the
  // in-solid flag cannot stop a charge there.  Terminating one cell below the
  // wire plane instead is exact for electrons (which drift away from the
  // cathode) and approximate for back-drifting ions.
  const double xy      = kSensorSpanCells * std::max(g.cellXCm, g.cellYCm);
  const double zMargin = (g.zMax - g.zMin) / std::max(1, gridNz - 1);
  sensor.SetArea(-xy, -xy, g.zMin + zMargin, xy, xy, g.zMax - zMargin);
}


// ─── Front-end amplifier (ported from tgc_sim) ────────────────────────────────
// One-pole RC low-pass, applied in place: y[k] = b·y[k−1] + (1−b)·x[k], b = e^{−Δt/τ}.
void ApplyOnePoleLowPass(std::vector<double>& x, const double dtNs, const double tauNs) {
  if (tauNs <= 0.) return;
  const double b = std::exp(-dtNs / tauNs);
  double y = 0.;
  for (auto& s : x) { y = b * y + (1. - b) * s; s = y; }
}

// Centred boxcar average over a finite acquisition aperture (no-op if window ≤ Δt).
void ApplyBoxcarAverage(std::vector<double>& x, const double dtNs, const double windowNs) {
  if (windowNs <= dtNs || x.empty()) return;
  const std::size_t nWin =
      std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(windowNs / dtNs)));
  if (nWin <= 1) return;
  std::vector<double> prefix(x.size() + 1, 0.);
  for (std::size_t i = 0; i < x.size(); ++i) prefix[i + 1] = prefix[i] + x[i];
  const std::size_t left = (nWin - 1) / 2, right = nWin / 2;
  std::vector<double> y(x.size(), 0.);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::size_t lo = i > left ? i - left : 0;
    const std::size_t hi = std::min(x.size() - 1, i + right);
    y[i] = (prefix[hi + 1] - prefix[lo]) / static_cast<double>(hi + 1 - lo);
  }
  x.swap(y);
}

// Pass a binned induced current i [fC/ns ≡ µA] through the transimpedance amplifier and
// return the output voltage [mV].  The CIVIDEC C2-TCT follows the input current within its
// band (no differentiation): apply the intrinsic upper-bandwidth low-pass, the gain·R_in
// (I→V) scale, then an optional output aperture.  A conductive readout pad has no input-cap
// current sink, so (unlike a resistive readout) there is no extra input low-pass.
std::vector<double> AmplifierOutputMv(const std::vector<double>& iFcNs, const double dtNs,
                                      const AmplifierConfig& amp) {
  const double gain = std::pow(10., amp.gainDb / 20.);          // dB → linear voltage gain
  const double tauLpHighNs = 1e9 / (2. * M_PI * amp.bandwidthHighHz);
  const double mvPerFcPerNs = gain * amp.inputImpedanceOhm * 1e-3;  // i[µA]·gain·R → mV
  std::vector<double> v(iFcNs);
  ApplyOnePoleLowPass(v, dtNs, tauLpHighNs);
  for (auto& s : v) s *= mvPerFcPerNs;
  ApplyBoxcarAverage(v, dtNs, amp.outputSampleNs);
  return v;
}

// ─── Per-distance simulation loop ─────────────────────────────────────────────

// Everything one read-out electrode needs for a run: the per-event current
// buffers, the tree branches, the profiles and the accumulated charges.  The
// readout set is a config knob, so these are built from the resolved electrode
// list instead of being spelled out once per electrode.
struct ElectrodeChannel {
  std::string id;
  std::vector<double> buf, bufE, bufI;                  // raw [fC/ns], this event
  std::vector<float>  sig, sigE, sigI, amp, ampInt;     // tree branch payloads
  float               charge = 0.f;                     // integrated charge [fC]
  std::vector<double> charges;                          // over events, for the summary
  std::unique_ptr<TH1D>     hCharge;
  std::unique_ptr<TProfile> pSig, pSigE, pSigI, pAmp, pAmpInt;
  // The bottom-most electrode *collects* the avalanche, so its raw integral is
  // negative; negate it for a conventionally positive collected charge.  Every
  // other electrode sees a bipolar induced signal whose net charge is ~0 and is
  // kept as-is.
  bool collecting = false;
};

DistanceSummary RunDistancePoint(const Config& cfg, const ThgemMeshGeom& g,
                                 std::optional<double> distOptMm,
                                 Sensor& sensor, TDirectory* distDir,
                                 std::optional<double> fixedXCm = std::nullopt) {
  const auto& sim = cfg.simulation;
  const auto& ids = *cfg.readout.electrodes;

  const double halfXCm = g.cellXCm / 2.0;
  const double halfYCm = g.cellYCm / 2.0;
  const double driftGapMm = g.dDriftCm * 10.0;
  double xRef = 0., yRef = 0.;
  ReferenceHoleAxis(g, xRef, yRef);

  const std::size_t nBins =
      static_cast<std::size_t>(std::round(sim.timeWindowNs / sim.timeStepNs));

  // ── Histograms ──────────────────────────────────────────────────────────────
  TH1D hNprimary("h_n_primary_electrons",
                 "Primary electrons per event;N_{e,primary};Events", 400, -0.5, 399.5);
  TH1D hAvalSize("h_avalanche_size",
                 "Total avalanche size;N_{e,total};Events", 200, 0., 0.);
  TH1D hGainThgem("h_gain_thgem", "THGEM gain;G_{THGEM};Events", 200, 0., 0.);
  TH1D hGainAmp("h_gain_amp", "Amplification-gap gain;G_{amp};Events", 200, 0., 0.);
  TH1D hMeshTrans("h_mesh_transparency",
                  "Mesh electron transparency;#epsilon_{mesh};Events",
                  110, 0., 1.1);
  for (TH1* h : std::initializer_list<TH1*>{&hNprimary, &hAvalSize, &hGainThgem,
                                            &hGainAmp, &hMeshTrans})
    h->SetDirectory(nullptr);

  auto MkProf = [&](const std::string& name, const std::string& title) {
    auto p = std::make_unique<TProfile>(name.c_str(), title.c_str(),
                                        static_cast<int>(nBins), 0., sim.timeWindowNs);
    p->SetDirectory(nullptr);
    return p;
  };

  // ── Per-event signal tree ────────────────────────────────────────────────────
  TTree signalTree("t_signals", "Per-event signal waveforms");
  signalTree.SetDirectory(nullptr);

  std::vector<std::unique_ptr<ElectrodeChannel>> chans;
  chans.reserve(ids.size());
  // The bottom-most read-out electrode in the stack is the collecting one.
  const std::string collectingId = kElecAnode;
  for (const auto& id : ids) {
    auto ch = std::make_unique<ElectrodeChannel>();
    ch->id = id;
    ch->collecting = (id == collectingId);
    ch->buf.assign(nBins, 0.);  ch->bufE.assign(nBins, 0.); ch->bufI.assign(nBins, 0.);
    ch->sig.assign(nBins, 0.f); ch->sigE.assign(nBins, 0.f); ch->sigI.assign(nBins, 0.f);
    ch->amp.assign(nBins, 0.f); ch->ampInt.assign(nBins, 0.f);
    ch->charges.reserve(sim.nEvents);
    ch->hCharge = std::make_unique<TH1D>(
        ("h_" + id + "_charge").c_str(),
        ("Induced charge on " + id + ";Q [fC];Events").c_str(), 200, 0., 0.);
    ch->hCharge->SetDirectory(nullptr);
    ch->pSig    = MkProf("p_" + id + "_signal",
                         "Mean " + id + " signal;t [ns];#LTi#GT [fC/ns]");
    ch->pSigE   = MkProf("p_" + id + "_electron",
                         "Mean " + id + " e^{-} signal;t [ns];#LTi_{e}#GT [fC/ns]");
    ch->pSigI   = MkProf("p_" + id + "_ion",
                         "Mean " + id + " ion signal;t [ns];#LTi_{ion}#GT [fC/ns]");
    ch->pAmp    = MkProf("p_" + id + "_amp",
                         "Mean " + id + " amplifier output;t [ns];#LTV#GT [mV]");
    ch->pAmpInt = MkProf("p_" + id + "_amp_int",
                         "Integrated " + id + " amp output;t [ns];#LT#int V dt#GT [mV ns]");
    chans.push_back(std::move(ch));
  }

  int evtId = 0;
  float evtGainThgem = 0.f, evtGainAmp = 0.f, evtMeshTrans = 0.f;
  int   evtBornThgem = 0, evtBornTransfer = 0, evtBornAmp = 0, evtEnterAmp = 0;
  signalTree.Branch("event", &evtId, "event/I");
  for (auto& ch : chans) {
    signalTree.Branch((ch->id + "_charge_fC").c_str(), &ch->charge,
                      (ch->id + "_charge_fC/F").c_str());
    signalTree.Branch(ch->id.c_str(),               &ch->sig);
    signalTree.Branch((ch->id + "_e").c_str(),      &ch->sigE);
    signalTree.Branch((ch->id + "_i").c_str(),      &ch->sigI);
    signalTree.Branch((ch->id + "_amp").c_str(),    &ch->amp);
    signalTree.Branch((ch->id + "_amp_int").c_str(), &ch->ampInt);
  }
  // Cascade diagnostics, per event.
  signalTree.Branch("gain_thgem",       &evtGainThgem, "gain_thgem/F");
  signalTree.Branch("gain_amp",         &evtGainAmp,   "gain_amp/F");
  signalTree.Branch("mesh_transparency", &evtMeshTrans, "mesh_transparency/F");
  signalTree.Branch("n_born_thgem",     &evtBornThgem, "n_born_thgem/I");
  signalTree.Branch("n_born_transfer",  &evtBornTransfer, "n_born_transfer/I");
  signalTree.Branch("n_born_amp",       &evtBornAmp,   "n_born_amp/I");
  signalTree.Branch("n_entered_amp",    &evtEnterAmp,  "n_entered_amp/I");

  // ── 3D track branches ────────────────────────────────────────────────────────
  std::vector<float> primaryX, primaryY, primaryZ;
  std::vector<float> cloudX,   cloudY,   cloudZ;
  std::vector<float> avalX,    avalY,    avalZ;
  std::vector<int>   avalNpts;
  std::vector<float> ionX,     ionY,     ionZ;
  std::vector<int>   ionNpts;
  signalTree.Branch("primary_x", &primaryX);
  signalTree.Branch("primary_y", &primaryY);
  signalTree.Branch("primary_z", &primaryZ);
  signalTree.Branch("cloud_x",   &cloudX);
  signalTree.Branch("cloud_y",   &cloudY);
  signalTree.Branch("cloud_z",   &cloudZ);
  signalTree.Branch("aval_x",    &avalX);
  signalTree.Branch("aval_y",    &avalY);
  signalTree.Branch("aval_z",    &avalZ);
  signalTree.Branch("aval_npts", &avalNpts);
  signalTree.Branch("ion_x",     &ionX);
  signalTree.Branch("ion_y",     &ionY);
  signalTree.Branch("ion_z",     &ionZ);
  signalTree.Branch("ion_npts",  &ionNpts);
  // The avalanche drift-line branches can reach hundreds of kB per event.  With the
  // default 32 kB basket they spill into several baskets, and uproot (used by the GUI)
  // mis-parses the multi-basket TObjArrayOfTBaskets that ROOT then embeds in the TTree.
  // A large basket keeps each branch in a single basket, which uproot reads fine.
  signalTree.SetBasketSize("*", 24 * 1024 * 1024);

  // ── Transport objects ────────────────────────────────────────────────────────
  const int nPrimary = std::max(1,
      static_cast<int>(std::round(cfg.source.energyKeV * 1.e3 / cfg.gas.wValueEV)));

  AvalancheMicroscopic aval(&sensor);
  if (sim.maxAvalancheSize > 0) aval.EnableAvalancheSizeLimit(sim.maxAvalancheSize);
  if (sim.storeDriftLines) aval.EnableDriftLines(true);
  // Compute each electrode's induced signal from its neBEM weighting *potential*
  // (Q per step = q·ΔW), not the weighting field: the potential is smooth on the
  // sampled grid, so the signals and their integrals are accurate for all electrodes.
  aval.UseWeightingPotential(true);
  // Bound the *transport* in time.  Sensor::SetTimeWindow only bins the induced signal;
  // without this an electron that drifts slowly (or stalls) is tracked indefinitely.
  // Charges still in flight at the end of the window end as StatusOutsideTimeWindow.
  aval.SetTimeWindow(0., sim.timeWindowNs);

  // Ion drift uses AvalancheMC (Monte-Carlo drift) rather than DriftLineRKF: the
  // RKF integrator has no step-count or time bound, so a single ion near a field
  // stagnation point loops indefinitely (hang) with unbounded path storage (OOM).
  // AvalancheMC steps by a fixed distance (bounds a normal ion by geometry) and
  // honours a time window (SetTimeWindow), the backstop that terminates a trapped ion.
  std::optional<AvalancheMC> ionDrift;
  if (sim.enableIonDrift) {
    ionDrift.emplace(&sensor);
    ionDrift->EnableDriftLines(true);        // populate EndPoint::path for the 3D view
    ionDrift->UseWeightingPotential(true);   // signal from the weighting potential (as above)
    if (sim.ionMaxStepUm > 0.) ionDrift->SetDistanceSteps(sim.ionMaxStepUm * 1.e-4);
    ionDrift->SetTimeWindow(0., sim.ionTimeWindowNs);
    std::cout << "  Ion drift: AvalancheMC, " << sim.ionMaxStepUm
              << " um steps, " << sim.ionTimeWindowNs << " ns time window.\n";
  }

  std::vector<double> primaryCounts, avalancheSizes;
  std::vector<double> gainsThgem, gainsAmp, meshTrans, bornThgems, bornTransfers, bornAmps;

  std::size_t nInteracted = 0;
  const std::size_t progressStep = std::max<std::size_t>(1, sim.nEvents / 10);
  const std::string distLabel = distOptMm.has_value()
      ? FormatNumber(*distOptMm) + " mm" : "random";

  // Where a charge ended, in stack terms.  Collection efficiency and attachment
  // are the two loss channels a cascade lives or dies by, and in a two-stage
  // stack the *stage* a charge was lost in is the thing worth knowing.  For the
  // mesh the distinction that matters is aperture vs metal: charge stopped on a
  // mesh wire is exactly the electron-transparency loss.
  auto inHole = [](const PlateGeom& p, double x, double y) {
    const double dx = WrapToCell(x - p.latShiftCm, p.pitchCm);
    const double dy = WrapToCell(y, p.pitchCm);
    return InHolePolygon(dx, dy, p.rHoleCm, p.sectors);
  };
  const char* meshMetalZone =
      (g.mesh.model == MeshModel::Woven) ? "mesh-wire" : "mesh-plate";
  auto classifyZone = [&g, &inHole, meshMetalZone](double x, double y,
                                                   double z) -> std::string {
    if (z >  g.thgem.zTopCuTop) return "drift";
    if (z >= g.thgem.zBotCuBot) return inHole(g.thgem, x, y) ? "thgem-hole" : "thgem-plate";
    if (z >  g.mesh.zTop)       return "transfer";
    if (z >= g.mesh.zBot)
      return ThgemMeshDetector::InMeshAperture(g.mesh, x, y) ? "mesh-aperture"
                                                             : meshMetalZone;
    return "amplification";
  };

  std::size_t nMultiplied = 0;
  std::map<std::string, int> primaryFate;

  // ── Event loop ───────────────────────────────────────────────────────────────
  auto tEvent = std::chrono::steady_clock::now();
  for (std::size_t ev = 0; ev < sim.nEvents; ++ev) {
    sensor.ClearSignal();

    // Start position: (x, y) within the periodic cell, z at the configured height
    // above the THGEM's top-copper surface (the same reference as the sibling
    // THGEM projects, so the three are directly comparable).  A fixed x pins y to the
    // reference hole row, so a scan in x crosses the hole and the wire in one line.
    const double x0 = fixedXCm.has_value()
                          ? *fixedXCm
                          : gRandom->Uniform(-halfXCm, halfXCm);
    const double y0 = fixedXCm.has_value()
                          ? yRef
                          : gRandom->Uniform(-halfYCm, halfYCm);

    const double heightMm = distOptMm.has_value()
                                ? *distOptMm
                                : gRandom->Uniform(0., driftGapMm);
    double z0 = g.thgem.zTopCuTop + heightMm * 0.1;   // mm → cm above the THGEM
    z0 = std::max(g.thgem.zTopCuTop + 1.e-4, std::min(g.zWire - 1.e-4, z0));

    // Transport one representative electron and scale by nPrimary: the mean
    // per-electrode charge is exactly linear in the number of primaries, and
    // tracking all ~215 of them would cost that factor in runtime for nothing.
    ++nInteracted;
    hNprimary.Fill(nPrimary);
    primaryCounts.push_back(static_cast<double>(nPrimary));

    aval.AvalancheElectron(x0, y0, z0, 0., 0.1);  // 0.1 eV ≈ thermal
    int ne = 0, ni = 0;
    aval.GetAvalancheSize(ne, ni);
    const int totalAvalElectrons = ne * nPrimary;
    hAvalSize.Fill(static_cast<double>(totalAvalElectrons));
    avalancheSizes.push_back(static_cast<double>(totalAvalElectrons));

    const std::size_t nEp = aval.GetNumberOfElectronEndpoints();

    // Fate of this event's primary electron (endpoint 0 is track 0, the primary).
    if (ne > 1) ++nMultiplied;
    if (nEp > 0) {
      double fx0, fy0, fz0, ft0, fe0, fx1, fy1, fz1, ft1, fe1; int fst;
      aval.GetElectronEndpoint(0, fx0, fy0, fz0, ft0, fe0,
                                  fx1, fy1, fz1, ft1, fe1, fst);
      ++primaryFate[DriftStatusToString(fst) + " @ " + classifyZone(fx1, fy1, fz1)];
    }

    // ── Cascade diagnostics ──────────────────────────────────────────────────
    // Where each secondary was *born* says which stage multiplied it, and a
    // track that starts above the mesh and ends at or below its *bottom* surface
    // is charge that actually reached the amplification gap.  Together these
    // separate a poor total gain into "the THGEM is weak" and "the mesh is not
    // electron-transparent".
    //
    // The crossing plane is zMeshBot, not zMeshTop.  A track that lands on a
    // mesh wire ends somewhere inside the mesh's own z-band, so it is correctly
    // *not* counted; one that threads an aperture ends below the band and is.
    // Testing against zMeshTop would count both and report a transparency of 1.
    evtBornThgem = evtBornTransfer = evtBornAmp = evtEnterAmp = 0;
    for (std::size_t i = 0; i < nEp; ++i) {
      double bx, by, bz, bt, be, ex, ey, ez, et, ee; int st;
      aval.GetElectronEndpoint(i, bx, by, bz, bt, be, ex, ey, ez, et, ee, st);
      if (i > 0) {   // endpoint 0 is the primary, not a multiplication product
        if (bz <= g.thgem.zTopCuTop && bz >= g.thgem.zBotCuBot) ++evtBornThgem;
        else if (bz < g.mesh.zBot) ++evtBornAmp;
        else if (bz < g.thgem.zBotCuBot) ++evtBornTransfer;
      }
      if (bz > g.mesh.zBot && ez <= g.mesh.zBot) ++evtEnterAmp;
    }
    const double gainThgem = 1.0 + evtBornThgem;
    // Transparency is measured against *all* the charge that existed above the
    // mesh, not just the THGEM's own output.  The transfer gap and the mesh
    // funnel are not field-free — the funnel runs at tens of kV/cm — so at a
    // high transfer field they multiply too, and dividing by gainThgem alone
    // yields a "transparency" above 1, which is not a quantity.  Counting that
    // charge in the denominator keeps the ratio a fraction, and n_born_transfer
    // reports the multiplication separately, where it can be read.
    const double aboveMesh = 1.0 + evtBornThgem + evtBornTransfer;
    const double meshTransparency = aboveMesh > 0. ? evtEnterAmp / aboveMesh : 0.;
    const double gainAmp = evtEnterAmp > 0
                               ? (evtEnterAmp + evtBornAmp) / static_cast<double>(evtEnterAmp)
                               : 0.;
    evtGainThgem = static_cast<float>(gainThgem);
    evtGainAmp   = static_cast<float>(gainAmp);
    evtMeshTrans = static_cast<float>(meshTransparency);
    hGainThgem.Fill(gainThgem);
    if (evtEnterAmp > 0) hGainAmp.Fill(gainAmp);
    hMeshTrans.Fill(std::min(1.1, meshTransparency));
    gainsThgem.push_back(gainThgem);
    if (evtEnterAmp > 0) gainsAmp.push_back(gainAmp);
    meshTrans.push_back(meshTransparency);
    bornThgems.push_back(evtBornThgem);
    bornTransfers.push_back(evtBornTransfer);
    bornAmps.push_back(evtBornAmp);

    // Primary electron drift line (track 0).
    primaryX.clear(); primaryY.clear(); primaryZ.clear();
    {
      const std::size_t nPts = aval.GetNumberOfElectronDriftLinePoints(0);
      primaryX.reserve(nPts); primaryY.reserve(nPts); primaryZ.reserve(nPts);
      for (std::size_t ip = 0; ip < nPts; ++ip) {
        double px, py, pz, pt;
        aval.GetElectronDriftLinePoint(px, py, pz, pt, ip, /*track=*/0);
        primaryX.push_back(static_cast<float>(px));
        primaryY.push_back(static_cast<float>(py));
        primaryZ.push_back(static_cast<float>(pz));
      }
    }

    // Avalanche cloud: start positions of secondary electron tracks.
    cloudX.clear(); cloudY.clear(); cloudZ.clear();
    {
      const std::size_t nSec   = nEp > 0 ? nEp - 1 : 0;
      const std::size_t stride = (nSec > kMaxDispCloudPts && kMaxDispCloudPts > 0)
                                  ? nSec / kMaxDispCloudPts : 1;
      cloudX.reserve(std::min(nSec, kMaxDispCloudPts));
      for (std::size_t i = 1; i < nEp; i += stride) {
        double x0c, y0c, z0c, t0c, e0c, x1c, y1c, z1c, t1c, e1c; int stc;
        aval.GetElectronEndpoint(i, x0c, y0c, z0c, t0c, e0c,
                                    x1c, y1c, z1c, t1c, e1c, stc);
        cloudX.push_back(static_cast<float>(x0c));
        cloudY.push_back(static_cast<float>(y0c));
        cloudZ.push_back(static_cast<float>(z0c));
      }
    }

    // Avalanche-electron transport: full drift lines of a strided, capped sample of
    // the secondary electrons (requires store_drift_lines; otherwise these are just
    // endpoints, like the primary line above).  Stored as concatenated points with a
    // per-track length list, exactly like the ion paths, for the 3D transport view.
    avalX.clear(); avalY.clear(); avalZ.clear(); avalNpts.clear();
    {
      const std::size_t nSec   = nEp > 0 ? nEp - 1 : 0;
      const std::size_t stride = (nSec > kMaxDispElectronPaths && kMaxDispElectronPaths > 0)
                                  ? nSec / kMaxDispElectronPaths : 1;
      constexpr std::size_t kMaxPtsPerPath = 64;   // subsample: smooth line, small file
      for (std::size_t i = 1; i < nEp; i += stride) {
        const std::size_t nPts = aval.GetNumberOfElectronDriftLinePoints(i);
        if (nPts == 0) { avalNpts.push_back(0); continue; }
        const std::size_t pStride = std::max<std::size_t>(1, nPts / kMaxPtsPerPath);
        int written = 0;
        for (std::size_t ip = 0; ip < nPts; ip += pStride) {
          double ax, ay, az, at;
          aval.GetElectronDriftLinePoint(ax, ay, az, at, ip, i);
          avalX.push_back(static_cast<float>(ax));
          avalY.push_back(static_cast<float>(ay));
          avalZ.push_back(static_cast<float>(az));
          ++written;
        }
        // Always include the true endpoint so the line reaches its collection point.
        if ((nPts - 1) % pStride != 0) {
          double ax, ay, az, at;
          aval.GetElectronDriftLinePoint(ax, ay, az, at, nPts - 1, i);
          avalX.push_back(static_cast<float>(ax));
          avalY.push_back(static_cast<float>(ay));
          avalZ.push_back(static_cast<float>(az));
          ++written;
        }
        avalNpts.push_back(written);
      }
    }

    // Back-drift the avalanche ions from where they were created; AvalancheMC adds
    // the Ramo-induced current to the sensor and (for the first kMaxDispIonPaths)
    // the drift-line path is extracted for 3D display.  Only the first
    // sim.maxIonsDrifted are transported (0 = all): ions drift away from the
    // readout and induce ~nothing on it, so this bounds the high-gain runtime
    // without changing the signal.  DriftIon clears its ion container each call,
    // so GetIons() holds exactly the ion just drifted (with its full trajectory).
    ionX.clear(); ionY.clear(); ionZ.clear(); ionNpts.clear();
    if (sim.enableIonDrift) {
      const std::size_t nIons = (sim.maxIonsDrifted > 0)
          ? std::min(nEp, sim.maxIonsDrifted) : nEp;
      for (std::size_t i = 0; i < nIons; ++i) {
        double xi0, yi0, zi0, ti0, ei0, xi1, yi1, zi1, ti1, ei1; int st;
        aval.GetElectronEndpoint(i, xi0, yi0, zi0, ti0, ei0,
                                    xi1, yi1, zi1, ti1, ei1, st);
        const bool ok = ionDrift->DriftIon(xi0, yi0, zi0, ti0);
        const auto& ions = ionDrift->GetIons();
        if (!ok || ions.empty()) {
          const int driftStatus =
              ions.empty() ? Garfield::StatusCalculationAbandoned : ions.front().status;
          std::ostringstream msg;
          msg << "Ion drift failed for event " << ev << ", height " << distLabel
              << ", ion " << i << "/" << nEp
              << " from (" << xi0 << ", " << yi0 << ", " << zi0 << ") cm"
              << " at t=" << ti0 << " ns; end status "
              << DriftStatusToString(driftStatus) << " (" << driftStatus << "). "
              << "Ensure GARFIELD_INSTALL exposes an ion mobility table for "
              << cfg.gas.ionSpecies << "+ or disable simulation.enable_ion_drift.";
          throw std::runtime_error(msg.str());
        }
        if (i < kMaxDispIonPaths) {
          const auto& path = ions.front().path;
          ionNpts.push_back(static_cast<int>(path.size()));
          for (const auto& pt : path) {
            ionX.push_back(static_cast<float>(pt.x));
            ionY.push_back(static_cast<float>(pt.y));
            ionZ.push_back(static_cast<float>(pt.z));
          }
        }
      }
    }

    // Bin the induced current [fC/ns] on each read-out electrode, from its neBEM
    // weighting potential, and shape it through the front-end amplifier.
    for (auto& ch : chans) {
      double raw = 0.;
      for (std::size_t k = 0; k < nBins; ++k) {
        ch->buf[k]  = sensor.GetSignal(ch->id, k);
        ch->bufE[k] = sensor.GetElectronSignal(ch->id, k);
        ch->bufI[k] = sensor.GetIonSignal(ch->id, k);
        raw += ch->buf[k];
        const double t = (static_cast<double>(k) + 0.5) * sim.timeStepNs;
        ch->pSig->Fill(t,  ch->buf[k]  * nPrimary);
        ch->pSigE->Fill(t, ch->bufE[k] * nPrimary);
        ch->pSigI->Fill(t, ch->bufI[k] * nPrimary);
        ch->sig[k]  = static_cast<float>(ch->buf[k]  * nPrimary);
        ch->sigE[k] = static_cast<float>(ch->bufE[k] * nPrimary);
        ch->sigI[k] = static_cast<float>(ch->bufI[k] * nPrimary);
      }
      const double sign = ch->collecting ? -1.0 : 1.0;
      const double q = sign * raw * sim.timeStepNs * nPrimary;   // [fC]
      ch->charge = static_cast<float>(q);
      ch->charges.push_back(q);
      ch->hCharge->Fill(q);

      if (cfg.amplifier.enable) {
        std::vector<double> i(nBins);
        for (std::size_t k = 0; k < nBins; ++k) i[k] = ch->buf[k] * nPrimary;
        const auto v = AmplifierOutputMv(i, sim.timeStepNs, cfg.amplifier);
        double cum = 0.;
        for (std::size_t k = 0; k < nBins; ++k) {
          cum += v[k] * sim.timeStepNs;
          ch->amp[k]    = static_cast<float>(v[k]);
          ch->ampInt[k] = static_cast<float>(cum);
          const double t = (static_cast<double>(k) + 0.5) * sim.timeStepNs;
          ch->pAmp->Fill(t, v[k]);
          ch->pAmpInt->Fill(t, cum);
        }
      }
    }

    evtId = static_cast<int>(ev);
    signalTree.Fill();

    if (ev == 0) {
      const double dt =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - tEvent).count();
      std::cout << "  [timing] first event: " << FormatNumber(dt, 2) << " s"
                << " (" << ne << " e- tracked)\n";
    }

    if ((ev + 1) % progressStep == 0 || ev + 1 == sim.nEvents) {
      std::cout << "  height=" << distLabel << ": "
                << (ev + 1) << "/" << sim.nEvents << " events processed\n";
    }
  }

  // Primary-electron fate summary (diagnostic): multiply fraction + where/why each
  // primary ended, so a run that "shows no avalanche" can be read as collection loss
  // (on a plate), attachment, or something anomalous (e.g. outside time window).
  std::cout << "  [fate] multiplied " << nMultiplied << "/" << sim.nEvents
            << " events; primary endpoint:";
  for (const auto& [label, cnt] : primaryFate) {
    std::cout << " {" << label << ": " << cnt << "}";
  }
  std::cout << "\n";
  const double meanMeshTrans = Mean(meshTrans);
  const double meanBornTransfer = Mean(bornTransfers);
  std::cout << "  [cascade] <G_thgem> = " << FormatNumber(Mean(gainsThgem), 2)
            << ", <mesh transparency> = " << FormatNumber(meanMeshTrans, 3)
            << ", <G_amp> = " << FormatNumber(Mean(gainsAmp), 2)
            << ", <total e-> = " << FormatNumber(Mean(avalancheSizes), 1) << "\n";
  // The acceptance test for the whole mesh model, printed on every transport
  // run.  Measured ~1.00 means the transport grid is not resolving the mesh at
  // all (see the grid check in ValidateField); measured ~= optical means the
  // funnelling into the apertures is unresolved and the mesh is acting as a
  // pure geometric stop.  The physical answer sits between the two and rises
  // with fields.delta_v_mesh_anode_V.
  std::cout << "  [mesh] optical transparency "
            << FormatNumber(g.mesh.opticalTransparency, 3)
            << " (geometric) | electron transparency "
            << FormatNumber(meanMeshTrans, 3) << " (measured, N = "
            << sim.nEvents << ")\n";
  if (meanBornTransfer > 0.05 * Mean(gainsThgem))
    std::cout << "  [mesh] note: " << FormatNumber(meanBornTransfer, 1)
              << " e- per event are born between the THGEM and the mesh — the "
                 "transfer gap and the\n         mesh funnel are multiplying. They "
                 "are counted in the transparency denominator,\n         not in "
                 "G_thgem or G_amp.\n";

  // ── Write histograms ─────────────────────────────────────────────────────────
  if (distDir) {
    distDir->cd();
    hNprimary.Write("h_n_primary_electrons");
    hAvalSize.Write("h_avalanche_size");
    hGainThgem.Write("h_gain_thgem");
    hGainAmp.Write("h_gain_amp");
    hMeshTrans.Write("h_mesh_transparency");
    for (auto& ch : chans) {
      ch->hCharge->Write();
      ch->pSig->Write();
      ch->pSigE->Write();
      ch->pSigI->Write();
      ch->pAmp->Write();
      ch->pAmpInt->Write();
    }
    signalTree.Write("t_signals");
  }

  // ── Build summary ─────────────────────────────────────────────────────────────
  DistanceSummary s;
  s.distanceMm          = distOptMm;
  s.xPositionCm         = fixedXCm;
  s.nEvents             = sim.nEvents;
  s.nInteracted         = nInteracted;
  s.interactionFraction = sim.nEvents > 0
                              ? static_cast<double>(nInteracted) / static_cast<double>(sim.nEvents)
                              : 0.;
  for (auto& ch : chans) {
    ElectrodeStats es;
    es.id           = ch->id;
    es.meanChargeFC = Mean(ch->charges);
    es.rmsChargeFC  = Rms(ch->charges, es.meanChargeFC);
    es.semChargeFC  = Sem(es.rmsChargeFC, ch->charges.size());
    s.electrodes.push_back(std::move(es));
  }
  s.meanPrimaryElectrons = Mean(primaryCounts);
  s.meanAvalancheSize    = Mean(avalancheSizes);
  s.rmsAvalancheSize     = Rms(avalancheSizes, s.meanAvalancheSize);
  s.semAvalancheSize     = Sem(s.rmsAvalancheSize, avalancheSizes.size());
  s.meanBornThgem        = Mean(bornThgems);
  s.meanBornTransfer     = meanBornTransfer;
  s.meanBornAmp          = Mean(bornAmps);
  s.meanGainThgem        = Mean(gainsThgem);
  s.meanGainAmp          = Mean(gainsAmp);
  s.meanMeshTransparency = meanMeshTrans;
  return s;
}

// ─── Summary graphs ───────────────────────────────────────────────────────────

void WriteSummaryGraphs(const std::vector<DistanceSummary>& sums,
                        TDirectory* summaryDir, const fs::path& pngPath) {
  if (sums.empty()) return;
  const std::size_t n = sums.size();
  std::vector<double> x(n), xe(n, 0.);

  for (std::size_t i = 0; i < n; ++i)
    x[i] = sums[i].distanceMm.value_or(static_cast<double>(i));

  std::vector<TGraphErrors> graphs;
  auto MakeGraph = [&](const std::string& name, const std::string& title,
                       const std::vector<double>& y, const std::vector<double>& ye,
                       int marker) {
    TGraphErrors g(static_cast<int>(n), x.data(), y.data(), xe.data(), ye.data());
    g.SetName(name.c_str());
    g.SetTitle(title.c_str());
    g.SetMarkerStyle(marker);
    g.SetLineWidth(2);
    if (summaryDir) { summaryDir->cd(); g.Write(); }
    graphs.push_back(std::move(g));
  };

  // One charge graph per read-out electrode, in stack order, then the cascade
  // diagnostics.  The electrode set is a config knob, so the panel count follows it.
  const std::size_t nElec = sums.front().electrodes.size();
  for (std::size_t e = 0; e < nElec; ++e) {
    std::vector<double> q(n), qe(n);
    for (std::size_t i = 0; i < n; ++i) {
      q[i]  = sums[i].electrodes[e].meanChargeFC;
      qe[i] = sums[i].electrodes[e].semChargeFC;
    }
    const std::string& id = sums.front().electrodes[e].id;
    MakeGraph("g_" + id + "_charge",
              "Mean " + id + " charge;Drift-gap height [mm];Q [fC]",
              q, qe, 20 + static_cast<int>(e % 6));
  }
  {
    std::vector<double> gain(n), gainE(n), gThgem(n), gAmp(n), eps(n), zero(n, 0.);
    for (std::size_t i = 0; i < n; ++i) {
      gain[i]  = sums[i].meanAvalancheSize;
      gainE[i] = sums[i].semAvalancheSize;
      gThgem[i] = sums[i].meanGainThgem;
      gAmp[i]   = sums[i].meanGainAmp;
      eps[i]    = sums[i].meanMeshTransparency;
    }
    MakeGraph("g_avalanche_size",
              "Mean avalanche size;Drift-gap height [mm];N_{e,total}", gain, gainE, 22);
    MakeGraph("g_gain_thgem",
              "THGEM gain;Drift-gap height [mm];G_{THGEM}", gThgem, zero, 23);
    MakeGraph("g_gain_amp",
              "Amplification-gap gain;Drift-gap height [mm];G_{amp}", gAmp, zero, 24);
    MakeGraph("g_mesh_transparency",
              "Mesh electron transparency;Drift-gap height [mm];#epsilon_{mesh}",
              eps, zero, 25);
  }

  const int nPads = static_cast<int>(graphs.size());
  const int nCols = std::min(4, std::max(1, nPads));
  const int nRows = (nPads + nCols - 1) / nCols;
  TCanvas canvas("c_thgem_mesh_summary", "THGEM + mesh summary", 600 * nCols, 500 * nRows);
  canvas.Divide(nCols, nRows);
  for (int i = 0; i < nPads; ++i) {
    canvas.cd(i + 1);
    graphs[static_cast<std::size_t>(i)].Draw("APL");
  }
  EnsureDirectory(pngPath.parent_path());
  canvas.SaveAs(pngPath.string().c_str());
}

// ─── CSV summary ─────────────────────────────────────────────────────────────

void WriteSummaryCsv(const fs::path& path, const std::vector<DistanceSummary>& sums) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot write CSV: " + path.string());
  if (sums.empty()) return;

  f << "source_distance_mm,x_position_cm,n_events,n_interacted,interaction_fraction";
  for (const auto& e : sums.front().electrodes)
    f << ",mean_" << e.id << "_charge_fC"
      << ",rms_"  << e.id << "_charge_fC"
      << ",sem_"  << e.id << "_charge_fC";
  f << ",mean_primary_electrons,mean_avalanche_size,rms_avalanche_size,"
       "sem_avalanche_size,mean_gain_thgem,mean_gain_amp,mean_mesh_transparency,"
       "mean_born_thgem,mean_born_transfer,mean_born_amp\n";

  f << std::fixed << std::setprecision(6);
  for (const auto& s : sums) {
    if (s.distanceMm) f << *s.distanceMm; else f << "random";
    f << ',';
    if (s.xPositionCm.has_value()) f << *s.xPositionCm;
    f << ','
      << s.nEvents             << ','
      << s.nInteracted         << ','
      << s.interactionFraction;
    for (const auto& e : s.electrodes)
      f << ',' << e.meanChargeFC << ',' << e.rmsChargeFC << ',' << e.semChargeFC;
    f << ',' << s.meanPrimaryElectrons
      << ',' << s.meanAvalancheSize
      << ',' << s.rmsAvalancheSize
      << ',' << s.semAvalancheSize
      << ',' << s.meanGainThgem
      << ',' << s.meanGainAmp
      << ',' << s.meanMeshTransparency
      << ',' << s.meanBornThgem
      << ',' << s.meanBornTransfer
      << ',' << s.meanBornAmp << '\n';
  }
}

// ─── Config echo ──────────────────────────────────────────────────────────────

json PlateToJson(const PlateConfig& p) {
  return {
    {"hole_diameter_um",    p.holeDiameterUm},
    {"plate_thickness_um",  p.plateThicknessUm},
    {"copper_thickness_um", p.copperThicknessUm},
    {"rim_um",              p.rimUm},
    {"dielectric_material", p.dielectric}
  };
}

json MeshToJson(const MeshConfig& m) {
  return {
    {"model",            m.model},
    {"pitch_um",         m.pitchUm},
    {"wire_diameter_um", m.wireDiameterUm},
    {"thickness_um",     m.thicknessUm},
    {"aperture_um",      m.apertureUm},
    {"aperture_sectors", m.apertureSectors},
    {"offset_x_um",      m.offsetXUm},
    {"offset_y_um",      m.offsetYUm},
    {"element_size_um",  m.elementSizeUm}
  };
}

json PlateGeomToJson(const PlateGeom& p) {
  return {
    {"z_cen_cm",        p.zCen},
    {"z_top_cu_top_cm", p.zTopCuTop},
    {"z_bot_cu_bot_cm", p.zBotCuBot},
    {"z_diel_half_cm",  p.zDielHalf},
    {"r_hole_cm",       p.rHoleCm},
    {"r_cu_cm",         p.rCuCm},
    {"sectors",         p.sectors},
    {"v_top",           p.vTop},
    {"v_bot",           p.vBot}
  };
}

// The solved mesh lattice, echoed so the GUI's overlays and 3D view can draw it
// without re-deriving the snap.  `wire_x_cm` / `wire_y_cm` are the wire centres
// for the woven model and the aperture centres for the perforated one — one key
// pair, with `model` telling the GUI how to draw them.
json MeshGeomToJson(const MeshGeom& m) {
  std::vector<double> xs, ys;
  for (int i = 0; i < m.nX; ++i) xs.push_back(m.X(i));
  for (int j = 0; j < m.nY; ++j) ys.push_back(m.Y(j));
  return {
    {"model",                MeshModelName(m.model)},
    {"pitch_cm",             m.pitchCm},
    {"pitch_snapped_um",     m.pitchUm},
    {"pitch_requested_um",   m.pitchReqUm},
    {"wires_per_hole_pitch", m.wiresPerHolePitch},
    {"n_x",                  m.nX},
    {"n_y",                  m.nY},
    {"r_wire_cm",            m.rWireCm},
    {"r_aperture_cm",        m.rApertureCm},
    {"sectors",              m.sectors},
    {"thickness_cm",         m.tMeshCm},
    {"z_cen_cm",             m.zCen},
    {"z_top_cm",             m.zTop},
    {"z_bot_cm",             m.zBot},
    {"z_upper_cm",           m.zUpper},
    {"z_lower_cm",           m.zLower},
    {"off_x_cm",             m.offXCm},
    {"off_y_cm",             m.offYCm},
    {"phase_x_cm",           m.phaseXCm},
    {"phase_y_cm",           m.phaseYCm},
    {"wire_x_cm",            xs},
    {"wire_y_cm",            ys},
    {"optical_transparency", m.opticalTransparency},
    {"v_mesh",               m.vMesh}
  };
}

json ConfigToJson(const Config& cfg, const ThgemMeshGeom& g) {
  json jSrc = {{"energy_keV", cfg.source.energyKeV}};
  jSrc["source_distances_mm"] = cfg.source.fixedDistMm.has_value()
                                     ? json(*cfg.source.fixedDistMm) : json(nullptr);
  jSrc["x_positions_cm"] = cfg.source.fixedXCmList.has_value()
                               ? json(*cfg.source.fixedXCmList) : json(nullptr);

  // The hole lattice, echoed so the GUI's overlays and 3D view do not have to
  // re-derive it (and drift out of sync with the C++).
  std::vector<double> holeX;
  for (int m = 0; m < g.nHolesX; ++m) holeX.push_back(g.HoleX(m));

  // The transport-grid spacing, echoed alongside the two budgets ValidateField
  // checks, so a run's resolution is recoverable from its config alone.
  const double dxCm = g.cellXCm / (cfg.geometry.gridNx - 1.);
  const double dyCm = g.cellYCm / (cfg.geometry.gridNy - 1.);
  const double dzCm = (g.zMax - g.zMin) / (cfg.geometry.gridNz - 1.);
  const double featureCm = (g.mesh.model == MeshModel::Woven)
                               ? 2. * g.mesh.rWireCm
                               : std::max(0., g.mesh.pitchCm - 2. * g.mesh.rApertureCm);
  const double dMaxCm = std::max({dxCm, dyCm, dzCm});

  return {
    {"geometry", {
      {"hole_pitch_um",          cfg.geometry.holePitchUm},
      {"wire_diameter_um",       cfg.geometry.wireDiameterUm},
      {"holes_per_wire",         cfg.geometry.holesPerWire},
      {"wire_between_holes",     cfg.geometry.wireBetweenHoles},
      {"drift_gap_mm",           cfg.geometry.driftGapMm},
      {"transfer_gap_mm",        cfg.geometry.transferGapMm},
      {"amplification_gap_mm",   cfg.geometry.amplificationGapMm},
      {"thgem",                  PlateToJson(cfg.geometry.thgem)},
      // The *snapped* mesh pitch, so reloading this file reproduces the run
      // exactly rather than snapping a second time from the original request.
      {"mesh",                   [&] {
                                   json j = MeshToJson(cfg.geometry.mesh);
                                   j["pitch_um"] = g.mesh.pitchUm;
                                   return j;
                                 }()},
      {"target_element_size_um", cfg.geometry.targetElementSizeUm},
      {"thgem_element_size_um",  cfg.geometry.thgemElementSizeUm},
      {"min_elements",           cfg.geometry.minElements},
      {"max_elements",           cfg.geometry.maxElements},
      {"max_elements_budget",    cfg.geometry.maxElementsBudget},
      {"periodic_copies",        cfg.geometry.periodicCopies},
      {"hole_sectors",           cfg.geometry.holeSectors},
      {"grid_nx",                cfg.geometry.gridNx},
      {"grid_ny",                cfg.geometry.gridNy},
      {"grid_nz",                cfg.geometry.gridNz}
    }},
    {"fields", {
      {"e_drift_kvcm",         cfg.fields.eDriftKvcm},
      {"delta_v_thgem_V",      cfg.fields.deltaVThgemV},
      {"e_transfer_kvcm",      cfg.fields.eTransferKvcm},
      {"delta_v_mesh_anode_V",  cfg.fields.deltaVMeshAnodeV}
    }},
    {"readout", {
      {"electrodes", cfg.readout.electrodes.has_value()
                         ? json(*cfg.readout.electrodes) : json(nullptr)}
    }},
    // Derived electrode potentials [V] and cell geometry [cm], echoed for the GUI.
    {"derived", {
      {"v_wire",       g.vWire},
      {"v_thgem_top",  g.thgem.vTop},
      {"v_thgem_bot",  g.thgem.vBot},
      {"v_mesh",       g.mesh.vMesh},
      {"v_anode",      g.vAnode},
      {"z_wire_cm",    g.zWire},
      {"z_anode_cm",   g.zAnode},
      {"z_min_cm",     g.zMin},
      {"z_max_cm",     g.zMax},
      {"r_wire_cm",    g.rWireCm},
      {"pitch_cm",     g.pitchCm},
      {"cell_x_cm",    g.cellXCm},
      {"cell_y_cm",    g.cellYCm},
      {"n_holes_x",    g.nHolesX},
      {"lattice_shift_cm", g.latShiftCm},
      {"hole_x_thgem_cm",  holeX},
      {"thgem",        PlateGeomToJson(g.thgem)},
      {"mesh",         MeshGeomToJson(g.mesh)},
      {"grid", {
        {"dx_cm", dxCm}, {"dy_cm", dyCm}, {"dz_cm", dzCm},
        {"mesh_feature_cm", featureCm},
        {"nodes_across_mesh_feature", dMaxCm > 0. ? featureCm / dMaxCm : 0.},
        {"cells_per_amp_gap",         dzCm   > 0. ? g.dAmpCm / dzCm    : 0.}
      }}
    }},
    {"source", jSrc},
    {"gas", {
      {"gas1",                   cfg.gas.gas1},
      {"gas1_fraction_pct",      cfg.gas.frac1},
      {"gas2",                   cfg.gas.gas2},
      {"ion_species",            cfg.gas.ionSpecies},
      {"temperature_K",          cfg.gas.temperatureK},
      {"pressure_Torr",          cfg.gas.pressureTorr},
      {"enable_penning",         cfg.gas.enablePenning},
      {"n_magboltz_collisions",  cfg.gas.nCollisions},
      {"max_electron_energy_eV", cfg.gas.maxElectronEnergyEV},
      {"transport_max_energy_eV", cfg.gas.transportMaxEnergyEV},
      {"n_field_points",         cfg.gas.nFieldPoints},
      {"e_field_min_vcm",        cfg.gas.eFieldMinVcm},
      {"e_field_max_vcm",        cfg.gas.eFieldMaxVcm},
      {"w_value_eV",             cfg.gas.wValueEV}
    }},
    {"simulation", {
      {"n_events",           cfg.simulation.nEvents},
      {"max_avalanche_size", cfg.simulation.maxAvalancheSize},
      {"time_window_ns",     cfg.simulation.timeWindowNs},
      {"time_step_ns",       cfg.simulation.timeStepNs},
      {"enable_ion_drift",   cfg.simulation.enableIonDrift},
      {"store_drift_lines",  cfg.simulation.storeDriftLines},
      {"ion_max_step_um",    cfg.simulation.ionMaxStepUm},
      {"ion_time_window_ns", cfg.simulation.ionTimeWindowNs},
      {"max_ions_drifted",   cfg.simulation.maxIonsDrifted},
      {"random_seed",        cfg.simulation.randomSeed}
    }},
    {"amplifier", {
      {"enable",              cfg.amplifier.enable},
      {"gain_db",             cfg.amplifier.gainDb},
      {"input_impedance_ohm", cfg.amplifier.inputImpedanceOhm},
      {"bandwidth_high_hz",   cfg.amplifier.bandwidthHighHz},
      {"output_sample_ns",    cfg.amplifier.outputSampleNs}
    }}
  };
}

std::string BuildRunFolderName(const Config& cfg) {
  std::time_t now = std::time(nullptr);
  std::tm tm_local = *std::localtime(&now);
  std::ostringstream ss;
  ss << std::put_time(&tm_local, "%y%m%d_%H-%M__")
     << "dV" << static_cast<int>(cfg.fields.deltaVThgemV) << "V_"
     << "amp" << static_cast<int>(cfg.fields.deltaVMeshAnodeV) << "V__";
  if (cfg.simulation.nEvents == 0) ss << "field";      // field-only run
  else                             ss << "n" << cfg.simulation.nEvents;
  return ss.str();
}

} // namespace

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  try {
    std::cout << std::unitbuf;   // stream the log to the GUI in real time

    gROOT->SetBatch(true);
    gStyle->SetOptStat(1110);
    TH1::AddDirectory(false);
    TH1::StatOverflows(true);
    const auto opts = ParseCli(argc, argv);
    Config cfg = LoadConfig(opts.configPath);

    const auto seed = static_cast<UInt_t>(cfg.simulation.randomSeed);
    gRandom->SetSeed(seed);
    // AvalancheMicroscopic and AvalancheMC draw from Garfield's own engine, not
    // ROOT's gRandom, so both must be seeded for a reproducible run.  SetEngine
    // copies the engine by value — seed it first.
    Garfield::RandomEngineRoot rngEngine;
    rngEngine.SetSeed(seed);
    Garfield::Random::SetEngine(rngEngine);

    if (opts.singleDistanceMm)
      cfg.source.fixedDistMm = std::vector<double>{*opts.singleDistanceMm};

    // Field-only: solve, sample, dump the maps, and stop before any transport.
    // Reachable two ways — the CLI flag, and n_events: 0 in the config, which is
    // how the GUI asks for it.
    const bool fieldOnly = opts.fieldOnly || cfg.simulation.nEvents == 0;

    const fs::path runDir = opts.outDir /
        (opts.runName.empty() ? BuildRunFolderName(cfg) : opts.runName);
    EnsureDirectory(runDir);

    const auto& gc = cfg.geometry;
    auto plateLine = [](const PlateConfig& p) {
      std::ostringstream s;
      s << "hole " << p.holeDiameterUm << " um dia, " << p.plateThicknessUm << " um "
        << p.dielectric << " + 2x" << p.copperThicknessUm << " um Cu";
      if (p.rimUm != 0.) s << ", rim " << p.rimUm << " um";
      return s.str();
    };
    std::string elecList;
    for (const auto& id : *cfg.readout.electrodes)
      elecList += (elecList.empty() ? "" : ", ") + id;

    std::cout << "THGEM + mesh Garfield++ simulation (THGEM + micromegas mesh + wire cathode)\n"
              << "  config  : " << opts.configPath << "\n"
              << "  output  : " << runDir << "\n"
              << "  cell    : pitch " << gc.holePitchUm << " um, "
              << gc.holesPerWire << " hole(s) per wire -> wire pitch "
              << gc.holesPerWire * gc.holePitchUm << " um"
              << (gc.wireBetweenHoles ? " (wire between holes)" : " (wire over a hole)") << "\n"
              << "  wire    : " << gc.wireDiameterUm << " um diameter\n"
              << "  THGEM   : " << plateLine(gc.thgem) << "\n"
              << "  gaps    : drift " << gc.driftGapMm << " mm, transfer "
              << gc.transferGapMm << " mm, amplification "
              << gc.amplificationGapMm << " mm\n"
              << "  fields  : dV_THGEM " << cfg.fields.deltaVThgemV << " V, E_drift "
              << cfg.fields.eDriftKvcm << " kV/cm, E_transfer "
              << cfg.fields.eTransferKvcm << " kV/cm, dV_mesh-anode "
              << cfg.fields.deltaVMeshAnodeV << " V\n"
              << "  readout : " << elecList << "\n"
              << "  gas     : " << cfg.gas.gas1 << ":" << cfg.gas.gas2 << " "
              << static_cast<int>(cfg.gas.frac1) << ":"
              << static_cast<int>(100. - cfg.gas.frac1) << ", "
              << cfg.gas.temperatureK << " K, " << cfg.gas.pressureTorr << " Torr\n"
              << "  source  : " << cfg.source.energyKeV << " keV, "
              << (cfg.source.fixedDistMm.has_value()
                    ? std::to_string(cfg.source.fixedDistMm->size()) + " height point(s)"
                    : "random height")
              << "\n  events  : "
              << (cfg.simulation.nEvents == 0
                    ? std::string("0 (field only)")
                    : std::to_string(cfg.simulation.nEvents) + " per point")
              << "\n";
    if (fieldOnly)
      std::cout << "  mode    : field only ("
                << (opts.fieldOnly ? "--field-only" : "n_events = 0")
                << ") — solve and dump the maps, no transport\n";

    using Clock = std::chrono::steady_clock;
    auto secsSince = [](Clock::time_point t0) {
      return std::chrono::duration<double>(Clock::now() - t0).count();
    };

    // Gas (shared across all height points).  In --field-only mode the gas is
    // still needed: it is the drift medium the ComponentGrid is given, and its
    // tabulated field range is what the field validation is checked against.
    std::cout << "\nSetting up gas...\n";
    auto tGas = Clock::now();
    MediumMagboltz gas(cfg.gas.gas1, cfg.gas.frac1,
                       cfg.gas.gas2, 100. - cfg.gas.frac1);
    SetupGas(gas, cfg.gas, cfg.simulation.enableIonDrift && !fieldOnly);
    std::cout << "  [timing] gas setup: " << FormatNumber(secsSince(tGas), 1) << " s\n";

    // Detector cell + transport field.  neBEM solves the real field; it is sampled
    // once onto a ComponentGrid (fast trilinear interpolation during the
    // avalanche — direct neBEM lookups are ~10^3x too slow) and cached by
    // geometry/fields so repeat runs skip the solve entirely.
    ThgemMeshDetector detector(cfg.geometry, cfg.fields, gas);
    const ThgemMeshGeom& g = detector.Geom();
    // Lattice warnings raised before the solve (an over-coarse or over-fine mesh
     // pitch); they carry through to the exit code alongside the field ones.
    const std::size_t geomWarnings = ValidateGeometry(g, gc);

    // The mesh line: what was asked for, what is actually being solved, and the
    // geometric transparency that the measured electron transparency will later
    // be compared against.
    {
      const MeshGeom& mg = g.mesh;
      std::cout << "  mesh    : " << MeshModelName(mg.model) << ", pitch ";
      if (std::abs(mg.pitchUm - mg.pitchReqUm) > 1.e-9)
        std::cout << FormatNumber(mg.pitchReqUm, 1) << " um -> "
                  << FormatNumber(mg.pitchUm, 1) << " um (snapped to fit the cell; ";
      else
        std::cout << FormatNumber(mg.pitchUm, 1) << " um (";
      std::cout << mg.wiresPerHolePitch << " per hole pitch)\n"
                << "            ";
      if (mg.model == MeshModel::Woven)
        std::cout << FormatNumber(2. * mg.rWireCm * 1.e4, 1) << " um wire, ";
      else
        std::cout << FormatNumber(2. * mg.rApertureCm * 1.e4, 1) << " um aperture ("
                  << mg.sectors << " sectors), ";
      std::cout << FormatNumber(mg.tMeshCm * 1.e4, 1) << " um thick, "
                << mg.nX << " x " << mg.nY << " per cell, optical transparency "
                << FormatNumber(mg.opticalTransparency, 3) << "\n";
    }

    // Projected element count, before neBEM builds anything: the solve inverts a
    // dense N x N matrix, so an accidentally fine mesh must fail here rather
    // than after hours.
    {
      const std::size_t nEst = EstimateElements(g, gc);
      const double rel = static_cast<double>(nEst) / 2336.;
      std::cout << "  neBEM   : ~" << nEst << " boundary elements projected (LU cost ~"
                << FormatNumber(rel * rel * rel, 2)
                << "x the double-THGEM reference solve)\n";
      if (static_cast<int>(nEst) > gc.maxElementsBudget)
        throw std::runtime_error(
            "projected " + std::to_string(nEst) + " boundary elements exceeds "
            "geometry.max_elements_budget = " + std::to_string(gc.maxElementsBudget) +
            ". neBEM inverts a dense N x N matrix, so this solve would cost roughly " +
            FormatNumber(rel * rel * rel, 0) + "x the reference one. Coarsen "
            "geometry.mesh.pitch_um (a perforated mesh costs ~390 elements per "
            "aperture; a woven one costs ~" + std::to_string(gc.maxElements) +
            " per wire), lower geometry.hole_sectors, or raise the budget "
            "deliberately.");
    }

    std::cout << "\n  Electrode potentials: V_wire=" << FormatNumber(g.vWire, 1)
              << " V, V_thgem_top=" << FormatNumber(g.thgem.vTop, 1)
              << " V, V_thgem_bot=" << FormatNumber(g.thgem.vBot, 1)
              << " V, V_mesh=" << FormatNumber(g.mesh.vMesh, 1)
              << " V, V_anode=" << FormatNumber(g.vAnode, 1) << " V";
    std::cout << "\n  Stack z [cm]: wire " << FormatNumber(g.zWire, 4)
              << " | THGEM " << FormatNumber(g.thgem.zTopCuTop, 4) << ".."
              << FormatNumber(g.thgem.zBotCuBot, 4)
              << " | mesh " << FormatNumber(g.mesh.zTop, 4) << ".."
              << FormatNumber(g.mesh.zBot, 4)
              << " | anode " << FormatNumber(g.zAnode, 4) << "\n";

    const auto& ids = *cfg.readout.electrodes;

    ComponentGrid grid;
    const double hx = g.cellXCm / 2.0, hy = g.cellYCm / 2.0;
    grid.SetMesh(static_cast<std::size_t>(gc.gridNx),
                 static_cast<std::size_t>(gc.gridNy),
                 static_cast<std::size_t>(gc.gridNz),
                 -hx, hx, -hy, hy, g.zMin, g.zMax);
    // One-cell mesh tiled with *translation* periodicity, so charges diffusing
    // past a cell edge re-enter the neighbouring cell instead of being lost
    // (charges are collected only in z).  Translation, not mirror: a staggered
    // mesh lattice or a wire sitting between holes leaves the cell without a
    // mirror plane, and mirroring it would fabricate a geometry that is not
    // the one neBEM solved.
    grid.EnablePeriodicityX();
    grid.EnablePeriodicityY();

    // Sampled neBEM field caches live in field_cache/ (generated; gitignored).
    EnsureDirectory("field_cache");
    const std::string fieldCache =
        (fs::path("field_cache") / DeriveFieldCacheName(gc, g.mesh, cfg.fields)).string();
    // One weighting grid per read-out electrode, keyed by geometry alone (a
    // weighting field does not depend on the applied voltages), so a whole dV
    // scan reuses one solve.
    std::vector<std::string> wCaches;
    std::vector<std::unique_ptr<ComponentGrid>> wgrids;
    for (const auto& id : ids) {
      wCaches.push_back((fs::path("field_cache") /
                         DeriveWeightingCacheName(gc, g.mesh, id)).string());
      auto wg = std::make_unique<ComponentGrid>();
      wg->SetMesh(static_cast<std::size_t>(gc.gridNx),
                  static_cast<std::size_t>(gc.gridNy),
                  static_cast<std::size_t>(gc.gridNz),
                  -hx, hx, -hy, hy, g.zMin, g.zMax);
      wg->EnablePeriodicityX();
      wg->EnablePeriodicityY();
      wgrids.push_back(std::move(wg));
    }

    const std::size_t expectNodes = static_cast<std::size_t>(gc.gridNx) *
                                    static_cast<std::size_t>(gc.gridNy) *
                                    static_cast<std::size_t>(gc.gridNz);
    auto tField = Clock::now();
    bool haveField = false;
    if (fs::exists(fieldCache)) {
      std::cout << "\nLoading cached transport field from: " << fieldCache << "\n";
      const std::size_t nLines = CountFileLines(fieldCache);
      if (nLines == expectNodes &&
          grid.LoadElectricField(fieldCache, "xyz",
                                 /*withPotential=*/true, /*withFlag=*/true)) {
        haveField = true;
      } else {
        std::cout << "  Cached field is incomplete or unreadable (" << nLines << "/"
                  << expectNodes << " nodes) — deleting and regenerating.\n";
        std::error_code ec;
        fs::remove(fieldCache, ec);
      }
    }
    std::vector<bool> haveW(ids.size(), false);
    bool haveAllW = true;
    for (std::size_t e = 0; e < ids.size(); ++e) {
      if (fs::exists(wCaches[e]) &&
          wgrids[e]->LoadWeightingField(wCaches[e], "xyz", /*withPotential=*/true)) {
        haveW[e] = true;
      } else {
        haveAllW = false;
      }
    }
    if (haveField && haveAllW)
      std::cout << "Loaded cached electrode weighting fields (" << elecList << ").\n";

    // A single neBEM solve serves the transport field and every weighting field,
    // so initialise if *any* is missing.  (On a pure cache hit neBEM is never
    // initialised, which is why each weighting field needs its own cache.)
    if (!haveField || !haveAllW) {
      std::cout << (haveField
        ? "\nTransport field was cached but an electrode weighting field is not — "
          "solving neBEM once to build it...\n"
        : "\nSolving the field with neBEM and sampling the transport grid "
          "(one-time; cached)...\n");
      auto tSolve = Clock::now();
      if (!detector.Initialise())
        throw std::runtime_error("neBEM Initialise() failed — try a coarser mesh "
                                 "(larger target_element_size_um / fewer periodic_copies).");
      std::cout << "  neBEM solved (" << detector.Component().GetNumberOfElements()
                << " boundary elements) in " << FormatNumber(secsSince(tSolve), 1) << " s.\n";
      // Validate against the neBEM solution itself, before the grid can blur it.
      ValidateField(detector.Component(), g, cfg.gas, cfg.fields, gc, /*onGrid=*/false);
      if (!haveField) {
        std::cout << "  Sampling transport grid (" << expectNodes
                  << " nodes; be patient)...\n";
        SampleFieldToFile(detector.Component(), detector, g, gc, fieldCache);
        if (!grid.LoadElectricField(fieldCache, "xyz",
                                    /*withPotential=*/true, /*withFlag=*/true))
          throw std::runtime_error("Failed to load freshly sampled transport field: "
                                   + fieldCache);
      }
      for (std::size_t e = 0; e < ids.size(); ++e) {
        if (haveW[e]) continue;
        std::cout << "  Sampling " << ids[e] << " weighting field onto the grid...\n";
        if (!wgrids[e]->SaveWeightingField(&detector.Component(), ids[e],
                                           wCaches[e], "xyz"))
          throw std::runtime_error("Failed to sample the " + ids[e] +
                                   " weighting field.");
        const std::size_t nFixed = ScrubNonFinite(wCaches[e]);
        if (nFixed > 0)
          std::cout << "    " << nFixed << " value(s) on a conductor singularity "
                       "(wire axis) zeroed.\n";
        if (!wgrids[e]->LoadWeightingField(wCaches[e], "xyz", /*withPotential=*/true))
          throw std::runtime_error("Failed to load the freshly sampled " + ids[e] +
                                   " weighting field: " + wCaches[e]);
      }
    }
    grid.SetMedium(&gas);   // must follow SetMesh(): SetMesh() calls Reset().
    std::cout << "  [timing] transport + weighting fields: "
              << FormatNumber(secsSince(tField), 1) << " s\n";

    // Re-validate on the interpolated grid: this is the field the avalanche will
    // actually see, and a solve that was fine can still be degraded by too coarse
    // a mesh.
    // Exit status: 0 = clean, 2 = the run completed and wrote its output but the
    // field validation flagged something, 1 = a genuine error (thrown).  The grid
    // pass decides it, because that is the field the avalanche actually sees.
    const std::size_t gridWarnings =
        ValidateField(grid, g, cfg.gas, cfg.fields, gc, /*onGrid=*/true);

    const auto tTransit = EstimateTransitTimeNs(grid, g, gas);
    std::cout << "    full-stack electron transit time ~ "
              << FormatNumber(tTransit.axisNs, 0) << " ns on the hole axis, "
              << FormatNumber(tTransit.worstNs, 0) << " ns from the slowest corner "
              << "of the cell;\n                                   "
              << "simulation.time_window_ns = "
              << FormatNumber(cfg.simulation.timeWindowNs, 0) << " ns\n";
    if (cfg.simulation.timeWindowNs < tTransit.worstNs) {
      std::cout << "\n  WARNING: the time window is shorter than the worst-case transit "
                   "time.  Primaries released\n"
                   "           high in the drift gap, away from a hole, will end as "
                   "'outside time window'\n"
                   "           before reaching the THGEM and will not multiply.  Raise "
                   "simulation.time_window_ns\n"
                   "           to at least "
                << FormatNumber(1.2 * tTransit.worstNs, 0)
                << " ns, or release the source closer to the plate\n"
                   "           (source.source_distances_mm).\n";
    }

    Sensor sensor;
    SetupSensor(sensor, grid, wgrids, ids, g, cfg.simulation, gc.gridNz);

    // ROOT output.
    TFile rootFile((runDir / "thgem_mesh_sim.root").string().c_str(), "RECREATE");
    if (rootFile.IsZombie())
      throw std::runtime_error("Failed to create ROOT file in " + runDir.string());

    TDirectory* summaryDir = rootFile.mkdir("summary");
    TDirectory* fieldDir   = rootFile.mkdir("field");
    auto tDump = Clock::now();
    DumpFieldMap(grid, g, fieldDir);
    for (std::size_t e = 0; e < ids.size(); ++e)
      DumpWeightingMap(fieldDir, ids[e], *wgrids[e], g);
    std::cout << "  [timing] field-map dump: " << FormatNumber(secsSince(tDump), 1) << " s\n";

    if (fieldOnly) {
      rootFile.Write();
      rootFile.Close();
      WriteJsonFile(runDir / "run_config.json", ConfigToJson(cfg, g));
      std::cout << "\nField only: field and weighting maps written to " << runDir << "\n";
      return (gridWarnings + geomWarnings) == 0 ? 0 : 2;
    }

    std::vector<DistanceSummary> allSummaries;

    std::vector<std::optional<double>> dList;
    if (cfg.source.fixedDistMm.has_value() && !cfg.source.fixedDistMm->empty()) {
      for (double d : *cfg.source.fixedDistMm) dList.push_back(d);
    } else {
      dList.push_back(std::nullopt);
    }
    std::vector<std::optional<double>> xList;
    if (cfg.source.fixedXCmList.has_value() && !cfg.source.fixedXCmList->empty()) {
      for (double x : *cfg.source.fixedXCmList) xList.push_back(x);
    } else {
      xList.push_back(std::nullopt);
    }

    for (const auto& dOpt : dList) {
      for (const auto& xOpt : xList) {
        std::string label = dOpt.has_value() ? FormatNumber(*dOpt) + " mm" : "random";
        if (xOpt.has_value()) label += "  x=" + FormatNumber(*xOpt * 10.0) + " mm";
        std::cout << "\n--- Drift-gap height: " << label << " ---\n";

        std::string tag = dOpt.has_value()
            ? "dist_" + FileSafeNumber(*dOpt) + "mm" : "dist_rnd";
        if (xOpt.has_value())
          tag += "_x" + FileSafeNumber(*xOpt * 10.0) + "mm";
        TDirectory* distDir = rootFile.mkdir(tag.c_str());
        if (!distDir) throw std::runtime_error("Failed to create ROOT dir: " + tag);

        DistanceSummary summary = RunDistancePoint(cfg, g, dOpt, sensor, distDir, xOpt);
        allSummaries.push_back(summary);

        for (const auto& e : summary.electrodes)
          std::cout << "  <Q_" << e.id << "> = " << FormatNumber(e.meanChargeFC)
                    << " fC  +/-" << FormatNumber(e.semChargeFC) << " (SEM)\n";
        std::cout << "  <avalanche size> = "
                  << FormatNumber(summary.meanAvalancheSize, 0) << " electrons\n";
      }
    }

    WriteSummaryGraphs(allSummaries, summaryDir, runDir / "summary" / "thgem_mesh_summary.png");
    rootFile.Write();
    rootFile.Close();

    WriteSummaryCsv(runDir / "summary.csv", allSummaries);
    WriteJsonFile(runDir / "run_config.json", ConfigToJson(cfg, g));

    std::cout << "\nDone. Results written to " << runDir << "\n";
    return (gridWarnings + geomWarnings) == 0 ? 0 : 2;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
