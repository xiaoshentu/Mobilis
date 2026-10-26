# Mobilis 迁跃

[![tests](https://github.com/xiaoshentu/Mobilis/actions/workflows/tests.yml/badge.svg)](https://github.com/xiaoshentu/Mobilis/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23593489.svg)](https://doi.org/10.5281/zenodo.23593489)
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

Run `python -m mobility_tool <command> -h` for all options.

## Tests

```bash
python tests/run_all.py          # all tests (no Gaussian/ADF needed: fake programs are used)
python tests/run_all.py --quick  # skips the slow GUI smoke test and test_v18 (~6 min less)
```

## Citing

If you use Mobilis, please cite:

> Y. Shi, *Mobilis: a charge-transport and photophysics toolkit for organic semiconductors*, version 1.16.0 (2026).
> Zenodo. https://doi.org/10.5281/zenodo.23593489

Metadata for citation managers is in [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
