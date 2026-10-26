"""Marcus 跳跃速率 + Einstein 关系计算迁移率。

  k_i = (2π/ħ) |J_i|^2 (4π λ k_B T)^(-1/2) exp[-(λ + ΔG)^2 / (4 λ k_B T)]
  标量 (常用文献公式):  D = 1/(2n) Σ_i r_i^2 k_i P_i,   P_i = k_i / Σ_j k_j
  张量 (需给出跳跃矢量): D_ab = 1/2 Σ_i k_i r_ia r_ib   (晶格随机行走的严格长时极限)
  μ = e D / (k_B T)
"""
import json
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5         # eV/K
HBAR_EVS = 6.582119569e-16     # eV·s
ANG2CM = 1e-8


def marcus_rate(j_ev, lam_ev, temperature, dg_ev=0.0):
    kt = KB_EV * temperature
    return (2 * np.pi / HBAR_EVS) * j_ev ** 2 / np.sqrt(4 * np.pi * lam_ev * kt) \
        * np.exp(-(lam_ev + dg_ev) ** 2 / (4 * lam_ev * kt))


def _pathway_j_ev(p):
    if "J_meV" in p:
        return abs(p["J_meV"]) / 1000
    if "J_eV" in p:
        return abs(p["J_eV"])
    raise ValueError(f"路径 {p.get('name', '?')} 缺少 J_meV 或 J_eV")


def compute_mobility(lam_ev, pathways, temperature=298.15, dimension=3, dg_ev=0.0):
    """pathways: [{name, J_meV|J_eV, r (Å) 或 vector [x,y,z] (Å), count (等价近邻数, 默认 1)}]"""
    kt = KB_EV * temperature
    rows, k_list, r_list, m_list, vecs = [], [], [], [], []
    for p in pathways:
        j = _pathway_j_ev(p)
        vec = np.array(p["vector"], float) if "vector" in p else None
        r = float(np.linalg.norm(vec)) if vec is not None else float(p["r"])
        m = int(p.get("count", 1))
        k = marcus_rate(j, lam_ev, temperature, dg_ev)
        rows.append({"name": p.get("name", f"P{len(rows) + 1}"), "r_A": r, "J_meV": j * 1000,
                     "count": m, "k_s-1": k})
        k_list.append(k), r_list.append(r * ANG2CM), m_list.append(m), vecs.append(vec)

    k, r, m = map(np.array, (k_list, r_list, m_list))
    p_single = k / np.sum(m * k)           # 每个近邻的跳跃概率
    for row, pi in zip(rows, p_single):
        row["P"] = float(pi)
    d_scalar = float(np.sum(m * r ** 2 * k * p_single) / (2 * dimension))
    out = {
        "temperature_K": temperature, "lambda_eV": lam_ev, "dimension": dimension,
        "pathways": rows,
        "D_cm2_s": d_scalar, "mobility_cm2_Vs": d_scalar / kt,
    }

    if all(v is not None for v in vecs):
        tensor = np.zeros((3, 3))
        for ki, mi, v in zip(k, m, vecs):
            vc = v * ANG2CM
            tensor += 0.5 * mi * ki * np.outer(vc, vc)
        mu_t = tensor / kt
        w, u = np.linalg.eigh(mu_t)
        out["mobility_tensor_cm2_Vs"] = mu_t.tolist()
        out["mobility_principal_cm2_Vs"] = w[::-1].tolist()
        out["principal_axes"] = u[:, ::-1].T.tolist()
        out["mobility_tensor_average_cm2_Vs"] = float(np.trace(mu_t) / 3)
    return out


def mobility_from_config(path):
    cfg = json.loads(Path(path).read_text())
    temps = cfg.get("temperatures") or [cfg.get("temperature", 298.15)]
    return [compute_mobility(cfg["lambda_eV"], cfg["pathways"], t, cfg.get("dimension", 3),
                             cfg.get("deltaG_eV", 0.0)) for t in temps]


def angular_mobility(tensor, plane="xy", n=361):
    """迁移率张量在给定平面内的角分布 μ(θ) = d̂ᵀ μ d̂。plane 取 'xy' / 'xz' / 'yz' 或两个三维向量。"""
    t = np.asarray(tensor, float)
    if isinstance(plane, str):
        axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        u, v = (np.array(axes[c], float) for c in plane)
    else:
        u = np.array(plane[0], float)          # 复制：传入的可能是晶格矢量，不能原地归一化
        u = u / np.linalg.norm(u)
        v = np.array(plane[1], float)
        v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
    theta = np.linspace(0, 2 * np.pi, n)
    d = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    return theta, np.einsum("ij,jk,ik->i", d, t, d)


# ----------------------------------------------------------------------------------------------
# 二维角分辨迁移率 (Wen, Deng, Han, PCCP 2010, 12, 9267)
#   μ_φ = e/(2 k_B T) Σ_i r_i² W_i P_i cos²γ_i cos²(θ_i − φ)
#   θ_i: 跳跃路径在传输平面内的投影与参考轴的夹角；γ_i: 路径与传输平面的夹角
# ----------------------------------------------------------------------------------------------
def plane_angles(vector, plane="xy"):
    """由三维跳跃矢量求 (r, θ, γ)，θ/γ 以度为单位。plane 为 'xy'/'xz'/'yz' 或 (u, v) 两个向量。"""
    vec = np.asarray(vector, float)
    if isinstance(plane, str):
        axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        u, v = (np.array(axes[c], float) for c in plane)
    else:
        u = np.array(plane[0], float)          # 复制：传入的可能是晶格矢量，不能原地归一化
        u = u / np.linalg.norm(u)
        v = np.array(plane[1], float)
        v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
    r = float(np.linalg.norm(vec))
    pu, pv = vec @ u, vec @ v
    inplane = np.hypot(pu, pv)
    theta = float(np.degrees(np.arctan2(pv, pu)))
    gamma = float(np.degrees(np.arctan2(abs(vec @ np.cross(u, v)), inplane)))
    return r, theta, gamma


def angular_mobility_2d(lam_ev, pathways, temperature=298.15, coupling_key="V_eV", n=361, dg_ev=0.0):
    """pathways: [{name, r (Å), theta (°), gamma (°, 默认 0), <coupling_key>: eV, count}]。
    返回 φ 网格、μ(φ)、各路径速率与概率、最大/最小迁移率等。"""
    kt = KB_EV * temperature
    names, r, th, ga, w, m = [], [], [], [], [], []
    for i, p in enumerate(pathways):
        v = p.get(coupling_key)
        if v is None or v == "":
            continue
        names.append(p.get("name", f"P{i + 1}"))
        r.append(float(p["r"]) * ANG2CM)
        th.append(np.radians(float(p.get("theta", 0.0))))
        ga.append(np.radians(float(p.get("gamma", 0.0))))
        w.append(marcus_rate(abs(float(v)), lam_ev, temperature, dg_ev))
        m.append(int(p.get("count", 1)))
    if not w:
        raise ValueError(f"没有可用的转移积分 ({coupling_key})")
    r, th, ga, w, m = map(np.array, (r, th, ga, w, m))
    prob = w / np.sum(m * w)
    phi = np.linspace(0, 2 * np.pi, n)
    terms = (m * r ** 2 * w * prob * np.cos(ga) ** 2)[None, :] * np.cos(th[None, :] - phi[:, None]) ** 2
    mu = terms.sum(1) / (2 * kt)
    k = int(np.argmax(mu))
    return {
        "phi": phi, "mu": mu, "mu_max": float(mu[k]), "phi_max_deg": float(np.degrees(phi[k]) % 360),
        "mu_min": float(mu.min()), "mu_avg": float(mu.mean()),
        "anisotropy": float(mu.max() / mu.min()) if mu.min() > 0 else float("inf"),
        "pathways": [{"name": nm, "r_A": float(ri / ANG2CM), "theta_deg": float(np.degrees(t)),
                      "gamma_deg": float(np.degrees(g)), "W_s-1": float(wi), "P": float(pi), "count": int(mi)}
                     for nm, ri, t, g, wi, pi, mi in zip(names, r, th, ga, w, prob, m)],
        "lambda_eV": lam_ev, "temperature_K": temperature,
    }


# ----------------------------------------------------------------------------------------------
# 三维各向异性迁移率
#   Wen 公式的三维推广：μ(n̂) = e/(2k_BT) Σ_i m_i r_i² W_i P_i (r̂_i·n̂)² = n̂ᵀ M n̂，
#     M = e/(2k_BT) Σ_i m_i W_i P_i r_i r_iᵀ；n̂ 取在传输平面内时即二维角分辨迁移率 μ_φ。
#   无规行走迁移率张量：μ = e/(k_BT) · ½ Σ_i m_i W_i r_i r_iᵀ (与 KMC 的扩散张量对应)。
# ----------------------------------------------------------------------------------------------
def mobility_3d(lam_ev, pathways, temperature=298.15, coupling_key="V_eV", model="wen", dg_ev=0.0):
    """pathways: [{name, vector [x,y,z] (Å), <coupling_key>: eV, count}]。model: "wen" 或 "tensor"。
    返回二阶张量 M (cm² V⁻¹ s⁻¹) 及其主值、主轴、最大 / 最小方向。"""
    kt = KB_EV * temperature
    vecs, w, m, names = [], [], [], []
    for i, p in enumerate(pathways):
        v = p.get(coupling_key)
        if v is None or v == "":
            continue
        if p.get("vector") is None:
            raise ValueError(f"路径 {p.get('name', i + 1)} 没有三维跳跃矢量 (x, y, z)")
        vecs.append(np.asarray(p["vector"], float) * ANG2CM)
        w.append(marcus_rate(abs(float(v)), lam_ev, temperature, dg_ev))
        m.append(int(p.get("count", 1)))
        names.append(p.get("name", f"P{i + 1}"))
    if not w:
        raise ValueError(f"没有可用的转移积分 ({coupling_key})")
    vecs, w, m = np.array(vecs), np.array(w), np.array(m)
    if model == "wen":
        prob = w / np.sum(m * w)
        M = np.einsum("i,ia,ib->ab", m * w * prob, vecs, vecs) / (2 * kt)
    elif model == "tensor":
        M = np.einsum("i,ia,ib->ab", 0.5 * m * w, vecs, vecs) / kt
    else:
        raise ValueError("model 应为 wen 或 tensor")
    ev, U = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    return {"M": M, "principal": ev, "axes": U.T, "mu_max": float(ev[0]), "dir_max": U[:, 0],
            "mu_min": float(ev[-1]), "dir_min": U[:, -1], "mu_avg": float(np.trace(M) / 3),
            "anisotropy": float(ev[0] / ev[-1]) if ev[-1] > 1e-12 * ev[0] else float("inf"),
            "model": model, "lambda_eV": lam_ev, "temperature_K": temperature, "names": names}


def mobility_on_sphere(M, n_theta=61, n_phi=121):
    """单位球面网格上的 μ(n̂) = n̂ᵀ M n̂。返回 (n̂ (nθ, nφ, 3), μ (nθ, nφ))。"""
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    return n, np.einsum("...a,ab,...b->...", n, np.asarray(M, float), n)


def crystal_direction(vec, lattice):
    """笛卡尔方向 → 晶体学方向 [u v w] (分数坐标，最大分量归一为 1) 与它和 a、b、c 的夹角 (°)。"""
    lat = np.asarray(lattice, float)
    f = np.asarray(vec, float) @ np.linalg.inv(lat)
    f = f / np.abs(f).max()
    if f[np.argmax(np.abs(f))] < 0:
        f = -f
    v = np.asarray(vec, float) / np.linalg.norm(vec)
    ang = [float(np.degrees(np.arccos(min(1.0, abs(v @ (a / np.linalg.norm(a))))))) for a in lat]
    return f, ang
