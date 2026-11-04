# Mobilis 迁跃

[![tests](https://github.com/xiaoshentu/Mobilis/actions/workflows/tests.yml/badge.svg)](https://github.com/xiaoshentu/Mobilis/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23593488.svg)](https://doi.org/10.5281/zenodo.23593488)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**A charge-transport and photophysics toolkit for organic semiconductors** — from a crystal structure or an MD
morphology to carrier mobilities, with a Tkinter GUI and a full command-line interface.

Developer: **Yarui Shi**, Anyang Normal University (安阳师范学院). 中文说明见 [README_zh.md](README_zh.md).

*Mobilis* is Latin for "movable", the root of *mobility*; 迁跃 combines 迁移 (migration) and 跃迁 (transition).
The Python package is `mobility_tool`.

## What it does

| Area | Features |
|---|---|
| Site energies | Four-point reorganization energy, IP/EA/HEP/EEP, HOMO/LUMO; λ decomposition (Huang–Rhys factors, Duschinsky matrix, internal coordinates, atoms) built-in or via Gaussian FCHT |
| Transfer integrals | Gaussian dimer projection, ADF fragment orbitals, built-in extended Hückel (EHT), ZINDO/MOO; method comparison and DFT calibration of semi-empirical couplings |
| Polarization | Atomic charges + Thole induced dipoles: P±, solid-state IE/EA, site-energy differences and σ_E |
| Transport models | Marcus/Jortner rates, mobility tensors and angular mobility, crystal-graph KMC (with exact-solution self-check), transient localization theory (TLT) with effective masses |
| Dynamic disorder (CEDS) | Coupling–energy dual surface: DFT coupling surfaces over intermolecular coordinates; thermal sampling by dimer MC, U_eq-calibrated MC, or full-crystal rigid-body MD; ⟨V⟩, σ_V, τ_c → TLT, μ(T) |
| Active learning | Gaussian-process coupling surfaces with symmetry augmentation; batch IMSE selection of the next DFT configurations; posterior intervals on ⟨V⟩ and σ_V as a stopping criterion |
| Morphologies | GROMACS morphology → molecule mapping → neighbour list → couplings → site energies → KMC; CIF → .gro supercells |
| Photophysics | S0/S1/T1 inputs, absorption/fluorescence/phosphorescence, Stokes shift, vibronically resolved phosphorescence, energy-gap law |
| Workflow | Batch scan projects (levels, grids, validation sets, symmetry checks, reports, archives); local job runner |

External quantum-chemistry codes (Gaussian, ADF/AMS, xtb, GROMACS) are optional — Mobilis writes their inputs
and reads their outputs; EHT, KMC, TLT, CEDS sampling and the GP surfaces run inside Mobilis.

## Install

```bash
git clone https://github.com/xiaoshentu/Mobilis.git
cd Mobilis
pip install -r requirements.txt
python run_gui.py            # GUI (on Windows you can also double-click Mobilis.bat)
python -m mobility_tool -h   # CLI
```

Python ≥ 3.9. The GUI uses tkinter (bundled with the official Windows/macOS Python; on Linux
`sudo apt install python3-tk`). Reading ADF binary results needs PLAMS (`pip install plams`, or start Mobilis
with `$AMSBIN/amspython run_gui.py`). A Windows installer can be built with the scripts in `packaging/`.

## Quick examples

```bash
# Marcus hopping mobility from a CIF using the built-in EHT couplings
python -m mobility_tool kmc pentacene.cif --lam 0.092 --carrier hole
python -m mobility_tool tlt pentacene.cif --carrier hole

# CEDS with active learning: each round writes 10–20 ADF inputs; run them, then rerun the same command
python -m mobility_tool ceds al --cif pentacene.cif --project P --name pen --pairs 1,2,3,7
# ... when every pair reports converged:
python -m mobility_tool ceds md --cif pentacene.cif --project P --al-surfaces pen
```

Run `python -m mobility_tool <command> -h` for all options. Help and messages are in English; for Chinese use
`python -m mobility_tool --lang zh <command> ...` or set `MOBILIS_LANG=zh` (the GUI language is switched in the
interface). The full command reference is in [docs/USER_MANUAL.md](docs/USER_MANUAL.md).

## Worked example: mobility tensor of a molecular crystal

```bash
# 1  neighbour dimers within 4 Å (shortest atom-atom distance), grouped by symmetry, with Gaussian inputs
python -m mobility_tool crystal anthracene.cif --cutoff 4.0 -o dimers --program gaussian --method B3LYP --basis "6-31G(d)"
python -m mobility_tool reorg-gen dimers/G1_A.gjf -o reorg --name anthracene --carrier both
# 2  run the Gaussian jobs (dimers/run_ti.bat, reorg/run_reorg.bat), or locally:  python -m mobility_tool run dimers
# 3  reorganization energy and the mobility tensor
python -m mobility_tool reorg-analyze reorg --json anthracene_lambda.json
python -m mobility_tool mobility --dimers dimers --lam 0.137 --carrier hole --dim 3 --cif anthracene.cif
```

The angle-resolved mobility uses the projection of the random-walk diffusion tensor, μ(φ) = e/(2k_BT) Σ W r² cos²γ cos²(θ − φ).
The formula of Wen et al. (J. Phys. Chem. B 2009), which multiplies each pathway by its hopping probability, is kept as
`mobility2d --model wen` for comparison with published values; on a one-dimensional chain it gives half of the correct value
(`python -m mobility_tool mobility --lam 0.2 --path 5.0,30,2`). For cells whose molecules do not sit on inversion centres the
exact random walk on the crystal graph is solved from the homogenized rate equations; `kmc` converges to the same limit.

## Test run

`examples/test_run/run_test.py` runs a complete calculation without Gaussian or ADF (KMC against exact solutions,
dimer search and input generation, EHT/KMC/TLT, and the four-crystal anisotropy benchmark from stored B3LYP results)
in about 40 s and compares every number with `examples/test_run/expected_output/`.

## Repository layout

| Path | Contents |
|---|---|
| `mobility_tool/` | the Python package: `cli.py` (command line), `gui.py` (graphical interface), `crystal.py` (CIF, molecules, dimers, symmetry), `reorganization.py`, `transfer_integral.py`, `adf.py`, `eht.py`, `zindo.py` (quantum-chemistry inputs, outputs and built-in couplings), `mobility.py`, `transport.py`, `kmc.py`, `validation.py` (transport models; transient localization in `transport.py`), `polar.py` (site energies), `ceds.py`, `rigidmd.py`, `alsurface.py`, `phonons.py` (dynamic disorder), `morphology.py` (MD morphologies), plotting and analysis modules |
| `tests/` | test suite (`run_all.py`) with simulated Gaussian/ADF executables and test data |
| `examples/test_run/` | test run with inputs, driver script and expected output |
| `examples/` | small input examples (configuration file, dimer structures, pathway table) |
| `docs/USER_MANUAL.md` | user manual and command reference |
| `packaging/` | scripts for a Windows installer |
| `run_gui.py`, `Mobilis.bat` | start the graphical interface |

## Changes in 1.25.0

- `diagnose --uq`: the transport regime as a probability. The largest scaled EHT coupling is drawn from its error
  against B3LYP dimer projection in the four-crystal benchmark, λ and the phonon σ_J/|J| from log-normal distributions;
  the report gives P(hopping), P(transient localization), P(band-like), the 90 % interval of 2|J|/λ and the input that
  dominates the uncertainty (`--lam-rel`, `--uq-samples`, `--seed`).

## Changes in 1.24.0

- Command-line help and messages in English (Chinese with `--lang zh` or `MOBILIS_LANG=zh`).
- `kmc-validate --cif`: adds the crystal-graph case (KMC against the homogenized exact solution) for any crystal.
- `examples/test_run`: complete test run with expected output; `docs/USER_MANUAL.md`: user manual and command reference.
- `diagnose`: transport regime from a CIF and the two reorganization energies in a few seconds (built-in EHT couplings and harmonic rigid-body phonons): 2|J|/λ, σ_J/|J|, interlayer/in-plane coupling ratio and π character of the frontier orbitals, with the recommended model and command.
- π frontier orbitals: EHT reports the π character of HOMO/LUMO and warns when a lone-pair σ orbital lies at the frontier (aza-aromatics); `--pi` (`eht-ti`, `kmc`, `tlt`; `pi=True` in Python) uses the highest occupied / lowest unoccupied π orbital instead.
- CIF reader accepts blank lines inside loop headers.
- Molecular frames: when the monomer (for example a gas-phase optimized structure) fits a crystal molecule poorly on its principal axes (pseudo-symmetric molecules such as mer-Alq3, whose principal moments are nearly degenerate), the frame is refined by a least-squares fit of the heavy atoms. Before, such molecules could be placed rotated by tens of degrees in the crystal (positive MD lattice energy).
- ZINDO couplings of molecules that coincide with the reference only after an improper operation (inversion/mirror images of non-planar molecules) fall back to a full dimer INDO/S projection instead of stopping.

The full change history is in [README_zh.md](README_zh.md) (Chinese).

## Tests

```bash
python tests/run_all.py          # all tests (no Gaussian/ADF needed: fake programs are used)
python tests/run_all.py --quick  # skips the slow GUI smoke test and test_v18 (~6 min less)
```

## Citing

If you use Mobilis, please cite:

> Y.-R. Shi, *Mobilis: a charge-transport and photophysics toolkit for organic semiconductors*, version 1.25.0 (2026).
> Zenodo. https://doi.org/10.5281/zenodo.23593488

and the program paper (Y.-R. Shi, J. Yang, submitted to Computer Physics Communications).

Metadata for citation managers is in [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
