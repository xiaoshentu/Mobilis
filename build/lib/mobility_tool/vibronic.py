"""振动分辨发射光谱（生成函数法）与能隙定律（Englman–Jortner）。

位移谐振子模型 (不含 Duschinsky 转动)，黄昆因子 S_k、频率 ω_k 来自本程序的分解结果或 Gaussian FCHT:
  G(t) = exp{ Σ_k S_k [ (n_k+1) e^{iω_k t} + n_k e^{-iω_k t} − (2n_k+1) ] }    (发射, n_k 为玻色占据数)
  I(E) ∝ E³ ∫ dt e^{i(E−E00)t/ħ} G(t) e^{−Γ²t²/2ħ²}        (高斯展宽 Γ = FWHM / 2.3548)
  垂直发射能 E_v = E00 − λ, λ = Σ S_k ħω_k
能隙定律 (弱耦合): k_nr = C² √(2π) / (ħ √(ħω_M ΔE)) · exp(−γ ΔE/ħω_M),  γ = ln(ΔE / (S_M ħω_M)) − 1
  其中受体模取高频模 (默认 > 1000 cm⁻¹) 的 S 加权平均；C 为电子耦合 (未知时比较 k_nr/C² 的相对大小)。
"""
import numpy as np

from .mobility import HBAR_EVS, KB_EV

CM2EV = 1 / 8065.543937
NM_EV = 1239.841984


def emission_spectrum(freqs_cm, hr, e00=None, e_vert=None, T=300.0, fwhm_ev=0.05, e_range=None, npts=4096,
                      cube=True):
    """返回能量网格 (eV)、归一化强度，以及峰位、FWHM、0-0 能量等。e00 与 e_vert 给一个即可。"""
    f = np.asarray(freqs_cm, float)
    s = np.asarray(hr, float)
    ok = (f > 10) & (s > 0)
    w, s = f[ok] * CM2EV, s[ok]                                      # ħω (eV)
    lam = float((s * w).sum())
    if e00 is None:
        e00 = e_vert + lam
    n = 1 / np.expm1(w / (KB_EV * T)) if T > 0 else np.zeros_like(w)
    sig = fwhm_ev / 2.3548
    # 时间网格 (单位 ħ/eV)：分辨率覆盖最大振动能的多次泛频
    emax = max(8 * (w.max() if len(w) else 0.2) * (1 + s.sum()), 1.0)
    dt = np.pi / emax / 2
    t = (np.arange(npts) - npts // 2) * dt
    g = np.exp((s[None, :] * ((n + 1)[None] * np.exp(1j * w[None] * t[:, None]) + n[None] * np.exp(-1j * w[None] * t[:, None])
                               - (2 * n + 1)[None])).sum(1)) * np.exp(-0.5 * (sig * t) ** 2)
    spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))).real * dt
    de = 2 * np.pi / (npts * dt)
    x = (np.arange(npts) - npts // 2) * de                            # E − E00 (eV)，FFT 约定下 x = −(E−E00)
    E = e00 - x
    order = np.argsort(E)
    E, spec = E[order], spec[order]
    if cube:
        spec = spec * np.clip(E, 0, None) ** 3
    lo, hi = e_range or (e00 - 6 * lam - 4 * fwhm_ev - 0.3, e00 + 4 * fwhm_ev + 0.2)
    sel = (E >= lo) & (E <= hi)
    E, spec = E[sel], np.clip(spec[sel], 0, None)
    spec = spec / spec.max() if spec.max() > 0 else spec
    k = int(np.argmax(spec))
    half = spec >= 0.5
    idx = np.where(half)[0]
    fw = float(E[idx.max()] - E[idx.min()]) if len(idx) else float("nan")
    return {"E_eV": E, "I": spec, "E00_eV": e00, "lambda_eV": lam, "peak_eV": float(E[k]),
            "peak_nm": NM_EV / float(E[k]), "fwhm_eV": fw, "fwhm_cm": fw / CM2EV, "S_total": float(s.sum())}


def energy_gap_law(freqs_cm, hr, gap_ev, c_ev=None, high_cut=1000.0):
    f = np.asarray(freqs_cm, float)
    s = np.asarray(hr, float)
    sel = (f >= high_cut) & (s > 0)
    if not sel.any():
        raise ValueError("没有高频模 (≥ %.0f cm⁻¹) 的黄昆因子" % high_cut)
    sm = float(s[sel].sum())
    wm = float((s[sel] * f[sel]).sum() / sm) * CM2EV
    gamma = np.log(gap_ev / (sm * wm)) - 1
    pref = np.sqrt(2 * np.pi) / (HBAR_EVS * np.sqrt(wm * gap_ev))
    knr_c2 = pref * np.exp(-gamma * gap_ev / wm)                      # s⁻¹ eV⁻²
    out = {"S_M": sm, "hw_M_eV": wm, "hw_M_cm": wm / CM2EV, "gamma": float(gamma), "knr_over_C2": float(knr_c2),
           "gap_eV": gap_ev}
    if c_ev:
        out["knr_s-1"] = float(knr_c2 * c_ev ** 2)
    return out
