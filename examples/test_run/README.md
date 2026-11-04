# Test run

A complete run of Mobilis that needs neither Gaussian nor ADF. It reproduces Table 2, Figure 3, Tables 4–5 and
Figures 4, 6 and 7 of the paper and compares every number with `expected_output/`.

```bash
cd examples/test_run
python run_test.py
```

The run takes about 40 s on one core. Output goes to `output/`; the last line reports whether all results agree
with `expected_output/` (relative tolerance 10⁻⁶; the KMC uses fixed random seeds).

| Step | Command | Output |
|---|---|---|
| 1 | `kmc-validate --cif perylene.cif` | KMC against exact solutions in 12 cases, including the four-molecule crystal graph of α-perylene (`kmc_validation.csv`, `.png`) |
| 2 | `mobility --lam 0.2 --path 5.0,30,2` | one-dimensional chain: random-walk mobility and the Deng–Goddard / Wen value, which is half of it (`chain_1d.json`) |
| 3 | `crystal anthracene.cif`, `reorg-gen` | neighbour dimers, symmetry groups and Gaussian inputs for couplings and reorganization energies (`anthracene_inputs/`) |
| 4 | `kmc`, `tlt` | built-in EHT couplings, crystal-graph KMC and transient-localization mobility of anthracene (log) |
| 5 | `benchmark/scripts/*.py` | couplings of 26 dimer groups from stored B3LYP results, built-in ZINDO/MOO and EHT (`couplings.json`); 108 mobility tensors (`models.json`); Figures 4, 6 and 7 |

`benchmark/` contains the input structures and the B3LYP results that Mobilis collected from the Gaussian
dimer-projection and four-point calculations (`*_lambda.json`, `*_mobility_*.json`). The Gaussian inputs themselves
are regenerated in step 3; all inputs and outputs of the paper are in the data repository cited there.
