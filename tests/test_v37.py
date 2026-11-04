"""v1.24: transport-regime diagnosis (diagnose) from a CIF with built-in EHT couplings and harmonic phonons."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.diagnose import diagnose, format_report  # noqa: E402

D = os.path.join(ROOT, "tests", "data")


def test_naphthalene_hopping_and_interlayer():
    d = diagnose(os.path.join(D, "naphthalene_benchmark.cif"), 0.187, 0.255)
    h, e = d["carriers"]["hole"], d["carriers"]["electron"]
    assert h["regime"] == "hopping" and e["regime"] == "hopping"
    assert d["lattice_energy_kcal_mol"] < -10
    # Section 6.5: the interlayer hopping mobility fails for naphthalene electrons, not for holes
    assert any("interlayer" in w for w in e["warnings"]) and not any("interlayer" in w for w in h["warnings"])


def test_pentacene_transient_localization():
    d = diagnose(os.path.join(D, "pentacene_RT.cif"), 0.092, 0.132)
    h = d["carriers"]["hole"]
    assert h["regime"] == "transient localization" and h["two_J_over_lambda"] > 1.5
    assert 0.25 < h["sigma_rel"] < 0.7 and h["interlayer_ratio"] < 0.1
    assert "tlt" in h["command"]


def test_lone_pair_warning_and_cli():
    d = diagnose(os.path.join(D, "TAPy_CF2CF3_938130.cif"), 0.480, 0.373, phonons=False)
    assert any("π character" in w for w in d["carriers"]["hole"]["warnings"])
    assert not any("π character" in w for w in d["carriers"]["electron"]["warnings"])
    assert "regime" in format_report(d)
    r = subprocess.run([sys.executable, "-m", "mobility_tool", "diagnose", os.path.join(D, "TAPy_CF2CF3_938130.cif"),
                        "--lam-h", "0.48", "--lam-e", "0.373", "--no-phonons"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0 and "π character" in r.stdout


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("v1.24 diagnose OK")
