# Mobilis user manual

Mobilis 1.25.0 (developer: Yarui Shi, Anyang Normal University (Anyang Normal University))

Mobilis computes charge-carrier mobilities of organic semiconductors from a crystal structure (CIF) or an MD
morphology. It prepares and reads Gaussian and ADF calculations, contains its own extended-Hückel (EHT), ZINDO/MOO and
polarizable site-energy models, and evaluates Marcus and Marcus–Levich–Jortner hopping (random-walk tensor, exact random
walk on the crystal graph, kinetic Monte Carlo) and transient localization theory. The graphical interface and the
command line call the same functions.

## 1 Installation

```bash
git clone https://github.com/xiaoshentu/Mobilis.git
cd Mobilis
pip install -r requirements.txt        # numpy, scipy, matplotlib (PyYAML for phonopy files, optional)
python run_gui.py                      # graphical interface
python -m mobility_tool -h             # command line
python tests/run_all.py --quick        # test suite (no Gaussian or ADF needed)
python examples/test_run/run_test.py   # test run with expected output (about 40 s)
```

Python ≥ 3.9 on Windows or Linux (tested in continuous integration); the GUI needs tkinter. Gaussian 16, ADF/AMS,
xtb and GROMACS are optional: Mobilis writes their inputs and batch scripts and reads their outputs.

## 2 Language

The command line prints help and messages in English. Chinese is selected with
`python -m mobility_tool --lang zh <subcommand> ...` or the environment variable `MOBILIS_LANG=zh`; a user who has
switched the GUI to Chinese keeps that setting on the command line. The GUI language is changed in the interface
(`python -m mobility_tool gui --lang en`).

## 3 Workflow: mobility tensor of a molecular crystal

| Step | Command | Result |
|---|---|---|
| Neighbour dimers | `crystal X.cif --cutoff 4.0 -o dimers --program gaussian` | `dimers/dimers.json` (groups, translations, point group) and Gaussian inputs `G*_A/B/AB.gjf` with `run_ti.bat/.sh` |
| Reorganization energy | `reorg-gen dimers/G1_A.gjf -o reorg --carrier both`, then `reorg-analyze reorg --json X_lambda.json` | four-point λ, IP, EA |
| Gaussian jobs | `run dimers` (or the batch files) | fchk files |
| Mobility | `mobility --dimers dimers --lam <λ> --carrier hole --dim 3 --cif X.cif --json X_mobility_hole.json` | couplings of each group, rates, random-walk D and μ, tensor symmetrized over the point group; the Deng–Goddard value for comparison |
| Angle-resolved mobility | GUI mobility window, or `mobility2d table.csv --model tensor` | μ(φ) from the random-walk tensor; `--model wen` gives the formula of Wen et al. |
| Crystal-graph KMC | `kmc X.cif --lam <λ>` | KMC tensor (with field, energetic disorder) |
| Check against exact solutions | `kmc-validate --cif X.cif` | 12 cases with exact mobilities |
| Transient localization | `tlt X.cif --carrier hole` | TLT mobility and effective masses |
| Polarization energies | `polar --cif X.cif --gjf esp/X`, then `polar --cif X.cif --neutral ... --cation ... --anion ...` | P±, solid-state IE/EA, site-energy differences |

Semiempirical couplings (`eht-ti`, `zindo`) can be calibrated against about 20 DFT dimers chosen across the |J|
distribution (`morph --export-dimers --calib-pairs 20`, then `--calibrate`).

## 4 Angle-resolved mobility

For hops of rate W_i along r_i the carrier stays on a site for τ = 1/ΣW and takes hop i with probability P_i = W_i τ.
The diffusion tensor is D = ½ Σ W_i r_i r_iᵀ, and its projection gives μ(φ) = e/(2k_BT) Σ W_i r_i² cos²γ_i cos²(θ_i − φ)
(default). The formula of Wen et al. replaces W_i by W_i P_i; on a one-dimensional chain it gives half of the correct
mobility (`mobility --lam 0.2 --path 5.0,30,2` prints both). The closed formula is exact when every hop has a partner
of the same rate in the opposite direction (molecules on inversion centres); otherwise Mobilis solves the homogenized
rate equations on the crystal graph, the long-time limit of the KMC.

## 5 Output files

* `dimers.json` — dimers of the reference molecule: group, centroid vector, translation, symmetry fingerprint; groups; crystal point group.
* `*_lambda.json` — four-point energies, λ for holes and electrons, IP and EA.
* `*_mobility_hole/electron.json` — for each pathway r, J, rate k and P; D and μ from the random walk; `mobility_DG_cm2_Vs` (Deng–Goddard, for comparison); the tensor and its principal values.
* `kmc_validation.csv` — case, exact μ, KMC μ, statistical error, deviation, z.

## 6 Tests

`tests/run_all.py` runs 38 scripts. External programs are replaced by simulated executables (`fake_programs.py`,
`fake_adf.py`, `fake_fchk.py`) that write outputs in the formats of Gaussian 16, ADF 2016 and formchk, so every workflow,
including job submission, resumption and collection, runs without licensed software. `examples/test_run` reproduces the
results of the program paper and compares them with stored output.

## 7 Command reference

```text
usage: mobility_tool [-h] [--version]
                     {reorg-gen,reorg-analyze,ti-gen,ti-calc,mobility,reorg-decomp,adf-gen,adf-read,crystal,ipea,orbitals,fcht-gen,fcht-read,mobility2d,mobility3d,excited-gen,excited-read,rmsd,orbcomp,run,eht-ti,compare-ti,kmc,strain-scan,tlt,vibronic,kmc-validate,scan,ceds,polar,morph,zindo,cif2gro,gui} ...

Mobilis — charge transport and photophysics of organic semiconductors

positional arguments:
  {reorg-gen,reorg-analyze,ti-gen,ti-calc,mobility,reorg-decomp,adf-gen,adf-read,crystal,ipea,orbitals,fcht-gen,fcht-read,mobility2d,mobility3d,excited-gen,excited-read,rmsd,orbcomp,run,eht-ti,compare-ti,kmc,strain-scan,tlt,vibronic,kmc-validate,scan,ceds,polar,morph,zindo,cif2gro,gui}
    reorg-gen           Generate Gaussian inputs for four-point reorganization energies
    reorg-analyze       Read Gaussian outputs and compute the reorganization energy
    ti-gen              Generate Gaussian inputs for transfer integrals (A, B, AB)
    ti-calc             Transfer integrals from fchk files
    mobility            Mobility from Marcus theory
    reorg-decomp        Reorganization-energy decomposition: normal modes (Huang-Rhys) / internal
                        coordinates / atoms
    adf-gen             Generate ADF fragment-orbital inputs for transfer integrals
    adf-read            Read ADF transfer integrals (adf.rkf / TAPE21 / dmpkf text)
    crystal             Find neighbour dimers in a CIF and write the inputs
    ipea                Compute IP/EA/HEP/EEP/reorganization energies
    orbitals            Read HOMO/LUMO from fchk/log
    fcht-gen            Generate Gaussian FCHT inputs (Huang-Rhys / Duschinsky) in a reorganization folder
    fcht-read           Parse FCHT outputs: reorganization decomposition / Huang-Rhys factors / Duschinsky
                        matrix
    mobility2d          2D angle-resolved mobility: reads a CSV table (name,d,theta,Ve,Vh)
    mobility3d          3D anisotropic mobility: reads a CSV table (name,x,y,z,Ve,Vh,count)
    excited-gen         Generate S0/S1/T1 optimization and TD single-point inputs
    excited-read        Read absorption/fluorescence/phosphorescence, Stokes shift, RMSD
    rmsd                RMSD of two structures after superposition
    orbcomp             Orbital composition by fragments (SCPA/Mulliken)
    run                 Run the Gaussian jobs of a folder one after another on this computer, with formchk
    eht-ti              Transfer integrals with the built-in EHT (no external program)
    compare-ti          Compare transfer integrals from EHT / Gaussian / ADF
    kmc                 CIF → EHT transfer integrals → crystal-graph KMC (VOTCA style)
    strain-scan         Cell scaling (strain) scan: EHT transfer integrals → mobility
    tlt                 CIF → EHT transfer integrals → transient-localization (TLT) mobility and effective
                        mass
    vibronic            Vibronically resolved phosphorescence spectrum and energy-gap law from FCHT Huang-
                        Rhys factors
    kmc-validate        KMC self-check: comparison with exact solutions case by case (optionally writes a
                        CSV and a comparison figure)
    scan                Batch scan workflow: project / level / grid / validation set / symmetry check /
                        results / report / archive
    ceds                CEDS dynamic disorder and mobility vs temperature (crystal local-potential MC +
                        coupling surfaces + TLT)
    polar               Polarizable site energies (atomic charges + Thole induced dipoles): P±, solid-state
                        IE/EA, site-energy differences and σ_E
    morph               Amorphous / MD morphology: GROMACS morphology → molecule mapping → neighbours →
                        transfer integrals → site energies → KMC
    zindo               ZINDO/MOO transfer integrals (VOTCA izindo): single dimer, test set against Gaussian
    cif2gro             Write a crystal supercell as a GROMACS .gro file (starting structure / morphology
                        workflow test)
    gui                 Start the graphical interface

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Chinese help and messages: python -m mobility_tool --lang zh <subcommand> ... (or MOBILIS_LANG=zh)
```

### reorg-gen

```text
usage: mobility_tool reorg-gen [-h] [-o OUTDIR] [--name NAME] [--carrier {hole,electron,both}]
                               [--charge CHARGE] [--mult MULT] [--ion-mult ION_MULT] [--no-freq]
                               [--sp-method SP_METHOD] [--sp-basis SP_BASIS] [--solvent SOLVENT]
                               [--ecp-basis ECP_BASIS] [--method METHOD] [--basis BASIS] [--nproc NPROC]
                               [--mem MEM] [--extra EXTRA] [--program PROGRAM]
                               geom

positional arguments:
  geom                  Structure file (xyz / gjf / log)

options:
  -h, --help            show this help message and exit
  -o, --outdir OUTDIR   Output folder
  --name NAME           Job name (prefix of the generated files)
  --carrier {hole,electron,both}
                        Charge carrier
  --charge CHARGE       Molecular charge (default 0)
  --mult MULT           Spin multiplicity of the neutral molecule (default 1)
  --ion-mult ION_MULT   Multiplicity of the ions (default 2)
  --no-freq             No frequency calculation in the optimizations
  --sp-method SP_METHOD
                        Functional for the single points (default: same as the optimization)
  --sp-basis SP_BASIS   Basis set for the single points, e.g. 6-311G(2d,2p)
  --solvent SOLVENT     PCM solvent, e.g. dichloromethane
  --ecp-basis ECP_BASIS
                        ECP basis for heavy atoms (automatic GenECP for Pople basis + Ir etc.)
  --method METHOD       Functional/method (default B3LYP)
  --basis BASIS         Basis set (default 6-31G(d))
  --nproc NPROC         Processors per Gaussian job (%nprocshared)
  --mem MEM             Memory per Gaussian job (%mem), e.g. 16GB
  --extra EXTRA         Extra keywords, e.g. "em=gd3bj"
  --program PROGRAM     Gaussian command in the run scripts (g16/g09)
```

### reorg-analyze

```text
usage: mobility_tool reorg-analyze [-h] [--json JSON] dir

positional arguments:
  dir          Folder written by reorg-gen (with the log files)

options:
  -h, --help   show this help message and exit
  --json JSON  Save the results as json
```

### ipea

```text
usage: mobility_tool ipea [-h] [--dir DIR] [--energies E E E E E E E] [--orbitals ORBITALS]

options:
  -h, --help            show this help message and exit
  --dir DIR             Finished folder written by reorg-gen
  --energies E E E E E E E
                        Give the 7 energies manually (Hartree): E0(N) E+(C) E-(A) E+(N) E-(N) E0(C) E0(A)
  --orbitals ORBITALS   fchk or log to read HOMO/LUMO from
```

### orbitals

```text
usage: mobility_tool orbitals [-h] file

positional arguments:
  file        Gaussian fchk or log

options:
  -h, --help  show this help message and exit
```

### reorg-decomp

```text
usage: mobility_tool reorg-decomp [-h] [--bond-scale BOND_SCALE] [--top TOP] [--csv CSV] neutral ion

positional arguments:
  neutral               fchk of the frequency calculation of the neutral molecule
  ion                   fchk of the frequency calculation of the ion

options:
  -h, --help            show this help message and exit
  --bond-scale BOND_SCALE
                        Bond detection scale factor for the internal coordinates
  --top TOP             Number of largest contributions to list
  --csv CSV             Folder for the exported CSV files
```

### fcht-gen

```text
usage: mobility_tool fcht-gen [-h] [--nproc NPROC] [--mem MEM] [--clear-low-freq CLEAR_LOW_FREQ] dir

positional arguments:
  dir                   Folder with the calculations

options:
  -h, --help            show this help message and exit
  --nproc NPROC         Processors per Gaussian job (%nprocshared)
  --mem MEM             Memory per Gaussian job (%mem), e.g. 16GB
  --clear-low-freq CLEAR_LOW_FREQ
                        Value of RedDim=ClearLowFreq, e.g. 100
```

### fcht-read

```text
usage: mobility_tool fcht-read [-h] [--freq-block FREQ_BLOCK] [-o OUT] log

positional arguments:
  log                   Gaussian output (log)

options:
  -h, --help            show this help message and exit
  --freq-block FREQ_BLOCK
                        Frequency group paired with the Huang-Rhys factors (default 0)
  -o, --out OUT         Folder for csv / Origin data and figures
```

### ti-gen

```text
usage: mobility_tool ti-gen [-h] [--dimer DIMER] [--na NA] [--mono-a MONO_A] [--mono-b MONO_B] [-o OUTDIR]
                            [--name NAME] [--charge CHARGE] [--mult MULT] [--method METHOD] [--basis BASIS]
                            [--nproc NPROC] [--mem MEM] [--extra EXTRA] [--program PROGRAM]

options:
  -h, --help           show this help message and exit
  --dimer DIMER        Dimer structure file; the first na atoms are monomer A
  --na NA              Number of atoms of monomer A
  --mono-a MONO_A      Structure file of monomer A (use with --mono-b)
  --mono-b MONO_B      Structure file of monomer B
  -o, --outdir OUTDIR  Output folder
  --name NAME          Job name (prefix of the generated files)
  --charge CHARGE      Molecular charge (default 0)
  --mult MULT          Spin multiplicity of the neutral molecule (default 1)
  --method METHOD      Functional/method (default B3LYP)
  --basis BASIS        Basis set (default 6-31G(d))
  --nproc NPROC        Processors per Gaussian job (%nprocshared)
  --mem MEM            Memory per Gaussian job (%mem), e.g. 16GB
  --extra EXTRA        Extra keywords, e.g. "em=gd3bj"
  --program PROGRAM    Gaussian command in the run scripts (g16/g09)
```

### ti-calc

```text
usage: mobility_tool ti-calc [-h] [--prefix PREFIX] [-a A] [-b B] [-ab AB] [--orbitals ORBITALS]
                             [--json JSON]

options:
  -h, --help           show this help message and exit
  --prefix PREFIX      e.g. ti/dimer → reads ti/dimer_A/B/AB.fchk
  -a A                 fchk of monomer A
  -b B                 fchk of monomer B
  -ab AB               fchk of the dimer
  --orbitals ORBITALS  e.g. HOMO-1,HOMO,LUMO,LUMO+1
  --json JSON          Write the results to this json file
```

### adf-gen

```text
usage: mobility_tool adf-gen [-h] [--dimer DIMER] [--na NA] [--mono-a MONO_A] [--mono-b MONO_B] [-o OUTDIR]
                             [--name NAME] [--functional FUNCTIONAL] [--basis BASIS] [--style {ams,classic}]
                             [--quality QUALITY]

options:
  -h, --help            show this help message and exit
  --dimer DIMER         Dimer structure file (the first --na atoms are monomer A)
  --na NA               Number of atoms of monomer A in the dimer file
  --mono-a MONO_A       Structure file of monomer A
  --mono-b MONO_B       Structure file of monomer B
  -o, --outdir OUTDIR   Output folder
  --name NAME           Job name (prefix of the generated files)
  --functional FUNCTIONAL
                        ADF functional
  --basis BASIS         Basis set
  --style {ams,classic}
                        ams: AMS2020+; classic: ADF2019-
  --quality QUALITY     ADF numerical quality
```

### adf-read

```text
usage: mobility_tool adf-read [-h] file

positional arguments:
  file        ADF output: adf.rkf, TAPE21 or a dmpkf text dump

options:
  -h, --help  show this help message and exit
```

### eht-ti

```text
usage: mobility_tool eht-ti [-h] [--dimer DIMER] [--na NA] [--mono-a MONO_A] [--mono-b MONO_B]
                            [--orbitals ORBITALS] [--pi]

options:
  -h, --help           show this help message and exit
  --dimer DIMER        Dimer structure file (the first --na atoms are monomer A)
  --na NA              Number of atoms of monomer A in the dimer file
  --mono-a MONO_A      Structure file of monomer A
  --mono-b MONO_B      Structure file of monomer B
  --orbitals ORBITALS  Orbitals to analyse, e.g. HOMO,LUMO
  --pi                 Use the highest occupied / lowest unoccupied π orbital instead of HOMO/LUMO
                       (for aza-aromatics and other molecules with lone-pair σ orbitals near the
                       HOMO)
```

### zindo

```text
usage: mobility_tool zindo [-h] [--dimer DIMER] [--na NA] [--orbitals ORBITALS] [--degeneracy DEGENERACY]
                           [--cif CIF] [--monomer MONOMER] [--frame {all,core}] [--cutoff CUTOFF]
                           [--level LEVEL] [--out OUT] [--dir DIR]
                           {pair,bench-inputs,bench}

positional arguments:
  {pair,bench-inputs,bench}
                        pair: one dimer; bench-inputs: write Gaussian ZINDO and DFT inputs of the test set;
                        bench: summarize the test

options:
  -h, --help            show this help message and exit
  --dimer DIMER         pair: dimer structure (the first --na atoms are A)
  --na NA               Number of atoms of monomer A in the dimer file
  --orbitals ORBITALS   pair: fchk of a Gaussian ZINDO monomer (default: built-in INDO/S)
  --degeneracy DEGENERACY
                        Near-degeneracy threshold / meV
  --cif CIF             bench-inputs: crystal (all inequivalent neighbour pairs)
  --monomer MONOMER     bench-inputs: reference monomer (DFT-optimized structure)
  --frame {all,core}    Molecular frame: all heavy atoms or the conjugated core
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
  --level LEVEL         bench-inputs: DFT method/basis (default B3LYP/6-31G(d))
  --out OUT             bench-inputs: output folder; bench: results CSV
  --dir DIR             bench: test-set folder
```

### compare-ti

```text
usage: mobility_tool compare-ti [-h] dir

positional arguments:
  dir         Dimer folder containing dimers.json

options:
  -h, --help  show this help message and exit
```

### crystal

```text
usage: mobility_tool crystal [-h] [--ref REF] [--cutoff CUTOFF] [--centroid] [-o OUTDIR]
                             [--program {gaussian,adf}] [--method METHOD] [--basis BASIS] [--nproc NPROC]
                             [--mem MEM]
                             cif

positional arguments:
  cif                   Crystal structure (CIF)

options:
  -h, --help            show this help message and exit
  --ref REF             Index of the reference molecule (from 1)
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
  --centroid            Use the centroid-distance criterion (default: shortest atom-atom distance)
  -o, --outdir OUTDIR   Output folder
  --program {gaussian,adf}
                        Write Gaussian or ADF inputs
  --method METHOD       Gaussian method or ADF functional
  --basis BASIS         Basis set
  --nproc NPROC         Processors per Gaussian job (%nprocshared)
  --mem MEM             Memory per Gaussian job (%mem), e.g. 16GB
```

### mobility

```text
usage: mobility_tool mobility [-h] [--config CONFIG] [--lam LAM] [--path PATH] [--temp TEMP] [--dim DIM]
                              [--dimers DIMERS] [--program {gaussian,adf}] [--carrier {hole,electron}]
                              [--cif CIF] [--json JSON]

options:
  -h, --help            show this help message and exit
  --config CONFIG       json configuration file (see examples/mobility_config.json)
  --lam LAM             Reorganization energy / eV
  --path PATH           Hopping pathway 'r(Å),J(meV)[,count]', repeatable
  --temp TEMP           Temperature / K
  --dim DIM             Dimensionality n in D = Σ k r² / 2n (default 3)
  --dimers DIMERS       Folder exported by the crystal subcommand (reads J of each group)
  --program {gaussian,adf}
                        Program of the dimer results read with --dimers
  --carrier {hole,electron}
                        Charge carrier
  --cif CIF             --dimers: take the crystal point group from this CIF to symmetrize the tensor (when
                        an old dimers.json has no point group)
  --json JSON           Write the results to this json file
```

### mobility2d

```text
usage: mobility_tool mobility2d [-h] [--lam-e LAM_E] [--lam-h LAM_H] [--temp TEMP] [--scale SCALE]
                                [--plot PLOT] [--model {tensor,wen}] [--label LABEL]
                                table

positional arguments:
  table                 CSV table of the hopping pathways

options:
  -h, --help            show this help message and exit
  --lam-e LAM_E         Electron reorganization energy / eV
  --lam-h LAM_H         Hole reorganization energy / eV
  --temp TEMP           Temperature / K
  --scale SCALE         Divisor converting V of the table to eV (1000 for meV)
  --plot PLOT           File name of the polar plot
  --model {tensor,wen}  tensor: corrected formula (projection of the random-walk tensor, default); wen:
                        original formula of Wen et al. (contains the hopping probability P, systematically
                        too low; for comparison with published values only)
  --label LABEL         Label of the curve in the plot
```

### mobility3d

```text
usage: mobility_tool mobility3d [-h] [--lam-e LAM_E] [--lam-h LAM_H] [--temp TEMP] [--scale SCALE]
                                [--model {wen,tensor}] [--cif CIF] [--plane PLANE] [--plot PLOT]
                                [--label LABEL]
                                table

positional arguments:
  table                 CSV table of the hopping pathways

options:
  -h, --help            show this help message and exit
  --lam-e LAM_E         Electron reorganization energy / eV
  --lam-h LAM_H         Hole reorganization energy / eV
  --temp TEMP           Temperature / K
  --scale SCALE         Divisor converting V of the table to eV (1000 for meV)
  --model {wen,tensor}  wen: 3D extension of the formula of Wen et al. (consistent with the 2D plot);
                        tensor: random-walk mobility tensor
  --cif CIF             Crystal (draws the axes a, b, c and gives the crystallographic directions of the
                        principal axes)
  --plane PLANE         Transport plane of the cross-section (xy / xz / yz)
  --plot PLOT           File name of the 3D figure (an .html of the same name is also written)
  --label LABEL         Label of the curve in the plot
```

### kmc

```text
usage: mobility_tool kmc [-h] --lam LAM [--carrier {hole,electron}] [--rate {marcus,jortner}]
                         [--lam-out LAM_OUT] [--hw HW] [--sigma SIGMA] [--supercell SUPERCELL]
                         [--field FIELD] [--temp TEMP] [--cutoff CUTOFF] [--walkers WALKERS]
                         [--steps STEPS] [--pi]
                         cif

positional arguments:
  cif                   Crystal structure (CIF)

options:
  -h, --help            show this help message and exit
  --lam LAM             Internal reorganization energy / eV
  --carrier {hole,electron}
                        Charge carrier
  --rate {marcus,jortner}
                        Rate expression: marcus or jortner (Marcus–Levich–Jortner)
  --lam-out LAM_OUT     Outer-sphere reorganization energy / eV
  --hw HW               Effective mode energy ħω of the MLJ rate / eV
  --sigma SIGMA         Gaussian energetic disorder σ / eV
  --supercell SUPERCELL
                        Supercell size (cells per axis)
  --field FIELD         V/cm, e.g. '1e4 0 0'
  --temp TEMP           Temperature / K
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
  --walkers WALKERS     Number of KMC walkers
  --steps STEPS         KMC hops per walker
  --pi                  Use the highest occupied / lowest unoccupied π orbital instead of
                        HOMO/LUMO (for aza-aromatics and other molecules with lone-pair σ orbitals
                        near the HOMO)
```

### kmc-validate

```text
usage: mobility_tool kmc-validate [-h] [--quick] [--cif CIF] [--cutoff CUTOFF] [--out OUT]

options:
  -h, --help       show this help message and exit
  --quick          Fewer walkers and cases (about 1/3 of the time)
  --cif CIF        Cell with several molecules: add a comparison of crystal-graph KMC with the homogenized
                   exact solution for this crystal (built-in EHT couplings)
  --cutoff CUTOFF  Neighbour criterion: shortest atom-atom distance / Å
  --out OUT        Output file
```

### diagnose

Transport-regime diagnosis. Example: `python -m mobility_tool diagnose pentacene.cif --lam-h 0.092 --lam-e 0.132`.
With `--uq` the regime is also given as a probability: the largest scaled EHT coupling is drawn from its error against
B3LYP dimer projection in the four-crystal benchmark (holes: ln-error −0.06 ± 0.09, electrons −0.06 ± 0.30), λ from a
log-normal of relative width `--lam-rel` (default 0.10) and σ_J/|J| from a log-normal of width 0.15. The report lists
P(hopping) = P(2|J|/λ < 1), P(transient localization), P(band-like), the 90 % interval of 2|J|/λ and whether the coupling
or λ dominates the uncertainty; with P ≥ 0.9 one model is recommended, otherwise the case is marked borderline.

```text
usage: mobility_tool diagnose [-h] --lam-h LAM_H --lam-e LAM_E [--temp TEMP] [--cutoff CUTOFF]
                              [--supercell SUPERCELL] [--no-phonons] [--uq]
                              [--uq-samples UQ_SAMPLES] [--lam-rel LAM_REL] [--seed SEED]
                              [--json JSON]
                              cif

positional arguments:
  cif                   Crystal structure (CIF)

options:
  -h, --help            show this help message and exit
  --lam-h LAM_H         Hole reorganization energy / eV
  --lam-e LAM_E         Electron reorganization energy / eV
  --temp TEMP           Temperature / K
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å
  --supercell SUPERCELL
                        Supercell of the lattice dynamics, e.g. 3,3,2 (default: >= 16 Å)
  --no-phonons          Skip the lattice dynamics (no σ_J/|J|)
  --uq                  Regime probabilities from the EHT calibration error, λ and σ_J/|J|
                        uncertainties (Monte Carlo)
  --uq-samples UQ_SAMPLES
                        Monte Carlo samples of --uq
  --lam-rel LAM_REL     Relative uncertainty of λ for --uq
  --seed SEED           Random seed of --uq
  --json JSON           Write the full result to this JSON file
```

### tlt

```text
usage: mobility_tool tlt [-h] [--carrier {hole,electron}] [--sigma SIGMA] [--htau HTAU]
                         [--plane PLANE] [--size SIZE] [--realizations REALIZATIONS] [--temp TEMP]
                         [--cutoff CUTOFF] [--pi]
                         cif

positional arguments:
  cif                   Crystal structure (CIF)

options:
  -h, --help            show this help message and exit
  --carrier {hole,electron}
                        Charge carrier
  --sigma SIGMA         σ_J/|J|
  --htau HTAU           ħ/τ_in / meV
  --plane PLANE         Transport plane (e.g. ab)
  --size SIZE           Cells per axis of the TLT Hamiltonian (0 = automatic)
  --realizations REALIZATIONS
                        Number of disorder realizations
  --temp TEMP           Temperature / K
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
  --pi                  Use the highest occupied / lowest unoccupied π orbital instead of
                        HOMO/LUMO (for aza-aromatics and other molecules with lone-pair σ orbitals
                        near the HOMO)
```

### strain-scan

```text
usage: mobility_tool strain-scan [-h] [--axis {a,b,c,all}] [--range RANGE] [--lam-e LAM_E] [--lam-h LAM_H]
                                 [--temp TEMP] [--cutoff CUTOFF]
                                 cif

positional arguments:
  cif                 Crystal structure (CIF)

options:
  -h, --help          show this help message and exit
  --axis {a,b,c,all}  Strain axis (a, b, c or all)
  --range RANGE       Strain range start:stop:step in %
  --lam-e LAM_E       Electron reorganization energy / eV
  --lam-h LAM_H       Hole reorganization energy / eV
  --temp TEMP         Temperature / K
  --cutoff CUTOFF     Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
```

### polar

```text
usage: mobility_tool polar [-h] --cif CIF [--monomer MONOMER] [--frame {all,core}] [--gjf GJF]
                           [--level LEVEL] [--neutral NEUTRAL] [--cation CATION] [--anion ANION]
                           [--polar-log POLAR_LOG] [--alpha-mean ALPHA_MEAN] [--radii RADII]
                           [--ip-gas IP_GAS] [--ea-gas EA_GAS] [--traj TRAJ] [--md-frames MD_FRAMES]
                           [--md-centers MD_CENTERS] [--md-r MD_R] [--out OUT] [--quiet]

options:
  -h, --help            show this help message and exit
  --cif CIF             Crystal structure (CIF)
  --monomer MONOMER     Rigid monomer (xyz / gjf / log or project.json); default: the molecule in the
                        crystal
  --frame {all,core}    Molecular frame: all heavy atoms or the conjugated core
  --gjf GJF             Only write the Gaussian inputs for ESP charges and polarizability (give the file-
                        name prefix)
  --level LEVEL         Method/basis for --gjf (default B3LYP/6-31G(d))
  --neutral NEUTRAL     Gaussian output of the neutral molecule (ESP charges, pop=mk)
  --cation CATION       Gaussian output of the cation (neutral geometry)
  --anion ANION         Gaussian output of the anion (neutral geometry)
  --polar-log POLAR_LOG
                        Gaussian output with polar results: scales the atomic polarizabilities to the DFT
                        molecular polarizability
  --alpha-mean ALPHA_MEAN
                        Target mean molecular polarizability / Å³
  --radii RADII         Cluster radii / Å (with a largest radius ≥ 26 Å the induction term is extrapolated
                        as 1/R, otherwise with a Born tail; 14,18,22 for a quick estimate)
  --ip-gas IP_GAS       Gas-phase ionization energy / eV (gives the solid-state IE)
  --ea-gas EA_GAS       Gas-phase electron affinity / eV
  --traj TRAJ           Full-crystal MD trajectory (npz from ceds md --save-traj): thermal fluctuation σ_E
  --md-frames MD_FRAMES
                        Number of MD frames for σ_E
  --md-centers MD_CENTERS
                        Molecules computed per frame (shared among the sublattices)
  --md-r MD_R           Cluster radius for snapshots / Å
  --out OUT             Results CSV (a .png of the same name is also written)
  --quiet               Less output
```

### morph

```text
usage: mobility_tool morph [-h] --gro GRO [--traj TRAJ] [--frame FRAME] --monomer MONOMER
                           [--resname RESNAME] [--elements ELEMENTS] [--geometry {qm,md}] [--cutoff CUTOFF]
                           [--coupling {eht,zindo,none}] [--zindo-orbitals ZINDO_ORBITALS]
                           [--degeneracy DEGENERACY] [--import-j IMPORT_J] [--export-dimers EXPORT_DIMERS]
                           [--max-pairs MAX_PAIRS] [--collect-dft COLLECT_DFT] [--level LEVEL]
                           [--calib-pairs CALIB_PAIRS] [--calibrate CALIBRATE]
                           [--calibrate-csv CALIBRATE_CSV] [--calib-no-replace] [--scale-h SCALE_H]
                           [--scale-e SCALE_E] [--site-energy {none,static,polar}] [--site-r SITE_R]
                           [--neutral NEUTRAL] [--cation CATION] [--anion ANION] [--lam-h LAM_H]
                           [--lam-e LAM_E] [--temp TEMP] [--field FIELD] [--walkers WALKERS] [--steps STEPS]
                           [--out OUT] [--quiet]

options:
  -h, --help            show this help message and exit
  --gro GRO             Morphology (.gro, may have several frames, or .pdb); the topology when --traj is
                        given
  --traj TRAJ           .xtc / .trr / .dcd trajectory (needs MDAnalysis or mdtraj)
  --frame FRAME         Frame index (from 0)
  --monomer MONOMER     Reference monomer (DFT-optimized structure: xyz / gjf / log / project.json)
  --resname RESNAME     Residue name to map (default: the most frequent residue with the same heavy-atom
                        count as the monomer)
  --elements ELEMENTS   Atom-name to element map, e.g. "CA=C,HA=H"
  --geometry {qm,md}    qm: superposed reference monomer; md: MD atom coordinates
  --cutoff CUTOFF       Neighbour criterion: shortest heavy-atom distance / Å
  --coupling {eht,zindo,none}
                        eht: built-in EHT; zindo: ZINDO/MOO (VOTCA izindo)
  --zindo-orbitals ZINDO_ORBITALS
                        fchk of a Gaussian ZINDO monomer calculation for ZINDO/MOO (default: built-in
                        INDO/S)
  --degeneracy DEGENERACY
                        Near-degeneracy threshold / meV (0 = no merging)
  --import-j IMPORT_J   Read transfer integrals from a CSV (pair, J_hole_meV, J_electron_meV)
  --export-dimers EXPORT_DIMERS
                        Folder for the Gaussian projection inputs of the neighbour pairs
  --max-pairs MAX_PAIRS
                        --export-dimers writes only the first N pairs (0 = all)
  --collect-dft COLLECT_DFT
                        Read back the Gaussian results (fchk) of the --export-dimers folder
  --level LEVEL         Gaussian method/basis (default B3LYP/6-31G(d))
  --calib-pairs CALIB_PAIRS
                        --export-dimers writes only N representative pairs chosen across the |J|
                        distribution (for DFT calibration, typically 15–30)
  --calibrate CALIBRATE
                        Calibrate EHT / ZINDO with the finished Gaussian dimers of this folder: one scale
                        factor k for holes and one for electrons
  --calibrate-csv CALIBRATE_CSV
                        As --calibrate, with reference J from a CSV (pair, J_hole_meV, J_electron_meV)
  --calib-no-replace    After calibration, pairs with a reference value also use k × semiempirical (default:
                        use the reference value)
  --scale-h SCALE_H     Manual hole scale factor k (J ← k·J_semiempirical)
  --scale-e SCALE_E     Manual electron scale factor k
  --site-energy {none,static,polar}
                        Site energies: static = electrostatics of atomic charges; polar = electrostatics +
                        Thole induction (about 0.5 s per molecule)
  --site-r SITE_R       Site-energy cutoff radius / Å (static default 30, polar default 14)
  --neutral NEUTRAL     Gaussian output of the neutral molecule (ESP charges)
  --cation CATION       Gaussian output of the cation (ESP charges)
  --anion ANION         Gaussian output of the anion (ESP charges)
  --lam-h LAM_H         Hole reorganization energy / eV (runs KMC if given)
  --lam-e LAM_E         Electron reorganization energy / eV
  --temp TEMP           Temperature / K
  --field FIELD         Field for an additional field mobility / V cm⁻¹ (averaged along x, y, z)
  --walkers WALKERS     Number of KMC walkers
  --steps STEPS         KMC hops per walker
  --out OUT             Output prefix (writes _sites.csv, _pairs.csv and .png)
  --quiet               Less output
```

### cif2gro

```text
usage: mobility_tool cif2gro [-h] --cif CIF [--monomer MONOMER] [--frame {all,core}] [--supercell SUPERCELL]
                             [--resname RESNAME] --out OUT

options:
  -h, --help            show this help message and exit
  --cif CIF             Crystal structure (CIF)
  --monomer MONOMER     Replace the molecules of the crystal by this monomer (e.g. a DFT-optimized
                        structure)
  --frame {all,core}    Molecular frame: all heavy atoms or the conjugated core
  --supercell SUPERCELL
                        Supercell size (cells per axis)
  --resname RESNAME     Residue name in the .gro file
  --out OUT             Output file
```

### ceds

```text
usage: mobility_tool ceds [-h] --cif CIF [--project PROJECT] [--monomer MONOMER] [--frame {all,core}]
                          [--name NAME] [--level LEVEL] [--source {project,eht}] [--eht-fallback]
                          [--temps TEMPS] [--samples SAMPLES] [--n-local N_LOCAL] [--htau HTAU]
                          [--cutoff CUTOFF] [--scale SCALE] [--ueq UEQ] [--ueq-from-cif] [--g-csv G_CSV]
                          [--fdx-csv FDX_CSV] [--slip-csv SLIP_CSV] [--dy-line DY_LINE] [--model {grid,r2}]
                          [--gphi-csv [GPHI_CSV ...]] [--local-csv [LOCAL_CSV ...]] [--sigma-only]
                          [--pairs PAIRS] [--out OUT] [--supercell SUPERCELL] [--md-charges MD_CHARGES]
                          [--md-equil MD_EQUIL] [--md-prod MD_PROD] [--md-dt MD_DT] [--md-every MD_EVERY]
                          [--md-rc MD_RC] [--htau-list HTAU_LIST] [--raw-geometry] [--save-traj SAVE_TRAJ]
                          [--traj TRAJ] [--n-snap N_SNAP] [--md-half] [--md-k MD_K] [--ph-quantum]
                          [--ph-htau-auto] [--ph-gamma] [--ph-real PH_REAL] [--ph-external PH_EXTERNAL]
                          [--ph-structure PH_STRUCTURE] [--ph-length-unit {auto,angstrom,bohr}]
                          [--ph-phase {positions,cell}] [--al-init AL_INIT] [--al-abs AL_ABS]
                          [--al-seed AL_SEED] [--al-batch AL_BATCH] [--al-target AL_TARGET]
                          [--al-pool AL_POOL] [--al-delta {auto,on,off}] [--al-goal {auto,sigma,mu}]
                          [--al-mu-target AL_MU_TARGET] [--al-budget [N_BOOT]]
                          [--al-budget-sigma-e AL_BUDGET_SIGMA_E] [--al-mu-draws AL_MU_DRAWS]
                          [--al-surfaces AL_SURFACES] [--force]
                          {run,local,md,snapshots,validate,al,phonons}

positional arguments:
  {run,local,md,snapshots,validate,al,phonons}
                        run: local MC for σ_V and μ(T); local: DFT local scans for the crystal dimers; md:
                        full-crystal rigid-body MD for σ_V and μ(T); snapshots: configurations from the MD
                        trajectory, written as ADF batches for direct calculation; validate: snapshot
                        results against the coupling surfaces; al: active-learning coupling surfaces (the
                        first run writes the round-1 ADF batches; after they finish, run again to fit and
                        write the next round until convergence); phonons: rigid-body lattice dynamics,
                        contribution of each phonon branch to σ_V and the mobility (killer phonons)

options:
  -h, --help            show this help message and exit
  --cif CIF             Crystal structure (CIF)
  --project PROJECT     Scan project (source of the coupling surfaces / destination of local scans)
  --monomer MONOMER     Rigid monomer (xyz / gjf / log or project.json); default: the scan-project monomer
                        or the molecule in the crystal
  --frame {all,core}    Molecular-frame definition without a scan project (with a project its setting is
                        used)
  --name NAME           Name of the local scan
  --level LEVEL         Computational level of the scan project
  --source {project,eht}
                        Source of the coupling surfaces
  --eht-fallback        Use built-in EHT surfaces for dimers without DFT surfaces
  --temps TEMPS         Temperatures / K, e.g. 100,150,200,250,300
  --samples SAMPLES     Monte Carlo samples per temperature
  --n-local N_LOCAL     Configurations of a local scan
  --htau HTAU           ħ/τ_in / meV
  --cutoff CUTOFF       Neighbour criterion: shortest atom-atom distance / Å (default 4.0)
  --scale SCALE         Pair-potential scale
  --ueq UEQ             Target U_eq / Å² (calibrates the fluctuation amplitude to the crystallographic
                        displacement parameters)
  --ueq-from-cif        Take U_iso / U_aniso from the CIF as the target
  --g-csv G_CSV         Product model: results table (results.csv) of the G(z, Δy, φ) grid at Δx = 0
  --fdx-csv FDX_CSV     Product model: results table of the Δx scan of a rotated dimer (gives f(Δx))
  --slip-csv SLIP_CSV   Product model: without a Δx scan, take f(Δx) from one row of a parallel-dimer slip
                        map (results.csv or slip_full.csv)
  --dy-line DY_LINE     Δy / Å of the slip-map row used for f(Δx)
  --model {grid,r2}     grid: bicubic G grid × f(Δx); r2: rotation model f·g(Δy,φ)·exp[−β(Δy)(z−4.8)]
  --gphi-csv [GPHI_CSV ...]
                        r2 model: results table of the dense φ scan (fitted with the G grid for g)
  --local-csv [LOCAL_CSV ...]
                        Local-scan results table (results.csv with a group column): fit the local surface
                        directly without ADF outputs
  --sigma-only          Compute σ_V only, no TLT
  --pairs PAIRS         local / snapshots: process only these groups, e.g. "3,7"; "parallel" = all parallel
                        neighbours (snapshots: all neighbours by default)
  --out OUT             Results CSV (md also writes a .png of the same name)

md (full-crystal rigid-body MD):
  --supercell SUPERCELL
                        Supercell na,nb,nc
  --md-charges MD_CHARGES
                        Atomic charges for the MD electrostatics: Gaussian output (ESP charges from pop=mk
                        preferred); recommended for molecules with N/O/metals
  --md-equil MD_EQUIL   Equilibration time per temperature / ps
  --md-prod MD_PROD     Sampling time per temperature / ps
  --md-dt MD_DT         Time step / fs
  --md-every MD_EVERY   Store a frame every this many steps
  --md-rc MD_RC         Non-bonded cutoff / Å (LJ switching starts at rc−1)
  --htau-list HTAU_LIST
                        Also compute μ for these ħ/τ (meV), e.g. "2,10"
  --raw-geometry        Do not shift the mean structure back to the experimental one (use the force-field
                        mean structure)
  --save-traj SAVE_TRAJ
                        Folder for the trajectory of each temperature (npz)
  --traj TRAJ           snapshots: use a saved trajectory (npz) instead of running MD
  --n-snap N_SNAP       snapshots: number of configurations per dimer
  --md-half             local: set the local-scan range from the full-crystal MD fluctuations (±k·σ)
  --md-k MD_K           local --md-half: range ±k·σ_MD

phonons (phonon-resolved dynamic disorder; force field, --supercell, --md-charges, --md-rc as for md):
  --ph-quantum          Quantum (Bose–Einstein) phonon occupation; default classical k_BT/ω²
  --ph-htau-auto        Take ħ/τ of TLT from the phonons (σ²-weighted mean ħω) instead of --htau
  --ph-gamma            Also report the Γ-point-only result for comparison
  --ph-real PH_REAL     TLT realizations (the numerical derivative for each branch uses the same random
                        numbers)
  --ph-external PH_EXTERNAL
                        External (DFT) phonons: phonopy mesh.yaml / qpoints.yaml / band.yaml with
                        eigenvectors (--eigvecs); the mesh must be complete (MESH_SYMMETRY = .FALSE.) and
                        the cell the same as the CIF. CRYSTAL, VASP, QE etc. through the phonopy interface
  --ph-structure PH_STRUCTURE
                        phonopy.yaml, when the --ph-external file has no structure (qpoints.yaml)
  --ph-length-unit {auto,angstrom,bohr}
                        Length unit of the external file (auto: compared with the CIF cell)
  --ph-phase {positions,cell}
                        Phase convention of the eigenvectors: positions = phonopy (with fractional atomic
                        coordinates); cell = cell translations only

al (active-learning coupling surfaces):
  --al-init AL_INIT     Configurations per neighbour pair in round 1
  --al-abs AL_ABS       Absolute convergence floor / meV: a σ_V half-width below it also counts as converged
                        (weakly coupled dimers)
  --al-seed AL_SEED     Finished configurations of existing scans (comma separated) as initial training
                        data, e.g. a previous local scan rub_local; read only in the first run, then
                        included in every round
  --al-batch AL_BATCH   Configurations per unconverged neighbour pair in each later round
  --al-target AL_TARGET
                        Convergence criterion: relative uncertainty of σ_V (95% half-width)
  --al-pool AL_POOL     Candidate pool: MD configurations per neighbour pair
  --al-delta {auto,on,off}
                        Δ-learning: GP mean β₀ + β₁·V_EHT, learning only the DFT − EHT difference (on by
                        default for new tasks; existing tasks keep their setting, switch with on / off)
  --al-goal {auto,sigma,mu}
                        Target of point selection and stopping: sigma = σ_V interval of each pair (default);
                        mu = interval of the TLT mobility, selecting points over all pairs by the posterior
                        variance of μ (v1.21). Existing tasks keep their setting; switch with sigma / mu
  --al-mu-target AL_MU_TARGET
                        Stopping criterion of --al-goal mu: relative 95% half-width of μ (default 0.01)
  --al-budget [N_BOOT]  al: error budget of the mobility (surface posterior, thermal samples, TLT
                        realizations, ħ/τ, σ_E); N_BOOT = number of bootstrap resamples of the thermal
                        samples (default 40)
  --al-budget-sigma-e AL_BUDGET_SIGMA_E
                        Site-energy disorder σ_E / meV for --al-budget, hole and electron (e.g. "31,46",
                        from polar)
  --al-mu-draws AL_MU_DRAWS
                        Posterior draws propagating the surface uncertainty to the TLT mobility after
                        convergence (0 = off)
  --al-surfaces AL_SURFACES
                        run / md: use the GP surfaces fitted by this active-learning task (overrides other
                        sources)
  --force               al: continue even if the previous round has unfinished configurations
```

### scan

```text
usage: mobility_tool scan [-h]
                          {new,level,grid,valid,check,clone,import,status,precheck,collect,cp,ti,report,archive} ...

positional arguments:
  {new,level,grid,valid,check,clone,import,status,precheck,collect,cp,ti,report,archive}
                        Action
    new                 Create a scan project
    level               Add a computational level (writes mono.run)
    grid                Parameter grid (z / Δx / Δy / γ / θ / φ, written start:stop:step or a,b,c)
    valid               Random validation set (Latin hypercube)
    check               Symmetry-equivalence check
    clone               Recompute the same configurations at another level (functional / basis comparison)
    import              Import a scan folder written by make_dimers.py
    status              Progress of all scans
    precheck            Pre-run check / repair of completion marks
    collect             Parse the outputs and write results.csv
    cp                  Gaussian counterpoise-corrected interaction-energy inputs
    ti                  Gaussian transfer-integral inputs (level comparison)
    report              Analysis and report (distance decay, slip maps, separability, V–S, models,
                        validation, level comparison, energies, reorganization)
    archive             Archive (all_structures.xyz, all_results.csv, README, raw outputs)

options:
  -h, --help            show this help message and exit
```

### excited-gen

```text
usage: mobility_tool excited-gen [-h] [-o OUTDIR] [--name NAME] [--method METHOD] [--basis BASIS]
                                 [--sp-basis SP_BASIS] [--solvent SOLVENT] [--nstates NSTATES]
                                 [--charge CHARGE] [--mult MULT] [--nproc NPROC] [--mem MEM]
                                 geom

positional arguments:
  geom                 Structure file (xyz / gjf / log)

options:
  -h, --help           show this help message and exit
  -o, --outdir OUTDIR  Output folder
  --name NAME          Job name (prefix of the generated files)
  --method METHOD      Method (functional)
  --basis BASIS        Basis set
  --sp-basis SP_BASIS  Basis set of the single points
  --solvent SOLVENT    PCM solvent
  --nstates NSTATES    Number of excited states
  --charge CHARGE      Molecular charge (default 0)
  --mult MULT          Spin multiplicity of the neutral molecule (default 1)
  --nproc NPROC        Processors per Gaussian job (%nprocshared)
  --mem MEM            Memory per Gaussian job (%mem), e.g. 16GB
```

### excited-read

```text
usage: mobility_tool excited-read [-h] dir name

positional arguments:
  dir         Folder with the calculations
  name        Job name used by excited-gen

options:
  -h, --help  show this help message and exit
```

### vibronic

```text
usage: mobility_tool vibronic [-h] --ev EV [--temp TEMP] [--fwhm FWHM] log

positional arguments:
  log          Gaussian output (log)

options:
  -h, --help   show this help message and exit
  --ev EV      Vertical emission energy / eV
  --temp TEMP  Temperature / K of the vibronic spectrum
  --fwhm FWHM  Line width (FWHM) / eV
```

### rmsd

```text
usage: mobility_tool rmsd [-h] [--heavy] a b

positional arguments:
  a           First structure
  b           Second structure

options:
  -h, --help  show this help message and exit
  --heavy     Use heavy atoms only
```

### orbcomp

```text
usage: mobility_tool orbcomp [-h] [--frag FRAG] [--orbitals ORBITALS] [--method {scpa,mulliken}] fchk

positional arguments:
  fchk                  Gaussian formatted checkpoint file

options:
  -h, --help            show this help message and exit
  --frag FRAG           e.g. "Ir:1" or "L1:2-30", repeatable; automatic partition if not given
  --orbitals ORBITALS   Orbitals to analyse, e.g. HOMO,LUMO
  --method {scpa,mulliken}
                        Partition method: scpa or mulliken
```

### run

```text
usage: mobility_tool run [-h] [--gaussian GAUSSIAN] [--scratch SCRATCH] [--pattern PATTERN] dir

positional arguments:
  dir                  Folder with the calculations

options:
  -h, --help           show this help message and exit
  --gaussian GAUSSIAN  Path of g16.exe or its folder
  --scratch SCRATCH    Gaussian scratch folder
  --pattern PATTERN    Glob pattern of the input files (default *.gjf)
```

### gui

```text
usage: mobility_tool gui [-h] [--lang {zh,en}]

options:
  -h, --help      show this help message and exit
  --lang {zh,en}  Interface language (default: the last setting)
```
