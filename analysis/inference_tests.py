"""
inference_tests.py — Small-sample inference hardening for the public-data pilot.

(1) Fisher randomization inference: with only 17 treated firms, cluster-robust
    asymptotics are unreliable. We hold the panel fixed, reassign the treated
    label to a random set of 17 of the 74 firms (or permute the continuous PSB
    share across firms), recompute the DiD coefficient, and report the exact
    two-sided permutation p-value  p = (1 + #{|b_perm| >= |b_obs|}) / (1 + B).

(2) Placebo shock date: re-estimate the DiD pretending the shock arrived one
    year earlier, using only true pre-shock data (quarterly: quarters through
    Sep2025 with placebo post = {Dec2024, Mar2025}; annual: FY2023-FY2025 with
    placebo post = FY2025). A real effect should show nothing here.

Outputs: console + ../paper/tables/p_t6_inference.tex
Deterministic (seed=42). numpy only.
"""
import os
import numpy as np

import pilot_analysis as P

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "..", "paper", "tables")
B = 2000
SEED = 42

def fe_projector(firms, times):
    """Return function res(v) residualizing a vector on firm+time FE + const."""
    fset = sorted(set(firms)); tset = sorted(set(times))
    n = len(firms)
    Z = np.zeros((n, len(fset) - 1 + len(tset) - 1 + 1))
    fmap = {f: i for i, f in enumerate(fset)}
    tmap = {t: i for i, t in enumerate(tset)}
    for i, (f, t) in enumerate(zip(firms, times)):
        if fmap[f] > 0:
            Z[i, fmap[f] - 1] = 1.0
        if tmap[t] > 0:
            Z[i, len(fset) - 1 + tmap[t] - 1] = 1.0
    Z[:, -1] = 1.0
    ZtZi = np.linalg.pinv(Z.T @ Z)
    def res(v):
        return v - Z @ (ZtZi @ (Z.T @ v))
    return res

def fw_beta(y_res, x, res):
    xr = res(x)
    denom = xr @ xr
    return (xr @ y_res) / denom if denom > 0 else np.nan

def perm_test(y, firms, times, firm_attr, post, kind, rng, b_obs):
    """kind='binary': reassign treated set of same size; 'share': permute shares."""
    res = perm_test._cache.get(id(firms))
    if res is None:
        res = fe_projector(firms, times)
        perm_test._cache[id(firms)] = res
    y_res = res(y)
    ufirms = sorted(set(firms))
    fidx = np.array([ufirms.index(f) for f in firms])
    vals = np.array([firm_attr[f] for f in ufirms], float)
    hits = 0
    for _ in range(B):
        pv = vals.copy()
        rng.shuffle(pv)
        x = pv[fidx] * post
        b = fw_beta(y_res, x, res)
        if np.isfinite(b) and abs(b) >= abs(b_obs) - 1e-15:
            hits += 1
    return (1 + hits) / (1 + B)

perm_test._cache = {}

def main():
    rng = np.random.default_rng(SEED)
    rows, qrows, psb_rel, psb_share, firms_meta, bankers = P.build()
    # winsorize as in pilot_analysis.run()
    for key in ("cod", "capex"):
        vals = P.winsor([r[key] for r in rows])
        for r, v in zip(rows, vals):
            r[key] = v
    iv = P.winsor([r["irate"] for r in qrows])
    for r, v in zip(qrows, iv):
        r["irate"] = v

    out = {}

    # ---------- quarterly cost-of-debt panel ----------
    rr = [r for r in qrows if np.isfinite(r["irate"])]
    y = np.array([r["irate"] for r in rr])
    firms = [r["t"] for r in rr]
    times = [r["q"] for r in rr]
    post = np.array([r["post"] for r in rr], float)
    res = fe_projector(firms, times)
    y_res = res(y)
    ufirms = sorted(set(firms))
    fidx = np.array([ufirms.index(f) for f in firms])

    psb_attr = {f: float(psb_rel[f]) for f in ufirms}
    shr_attr = {f: float(psb_share[f]) for f in ufirms}
    x_bin = np.array([psb_attr[f] for f in ufirms])[fidx] * post
    x_shr = np.array([shr_attr[f] for f in ufirms])[fidx] * post
    b_bin = fw_beta(y_res, x_bin, res)
    b_shr = fw_beta(y_res, x_shr, res)
    p_bin = perm_test(y, firms, times, psb_attr, post, "binary", rng, b_bin)
    p_shr = perm_test(y, firms, times, shr_attr, post, "share", rng, b_shr)
    out["q_bin"] = (b_bin, p_bin)
    out["q_shr"] = (b_shr, p_shr)
    print(f"quarterly CoD binary: b={b_bin*100:+.2f}pp  permutation p={p_bin:.3f} (B={B})")
    print(f"quarterly CoD share : b={b_shr*100:+.2f}pp  permutation p={p_shr:.3f}")

    # ---------- annual capex (share treatment) ----------
    ra = [r for r in rows if np.isfinite(r["capex"])]
    ya = np.array([r["capex"] for r in ra])
    fa = [r["t"] for r in ra]; ta = [r["fy"] for r in ra]
    pa = np.array([r["post"] for r in ra], float)
    resa = fe_projector(fa, ta)
    ya_res = resa(ya)
    ufa = sorted(set(fa)); fai = np.array([ufa.index(f) for f in fa])
    shr_a = {f: float(psb_share[f]) for f in ufa}
    xa = np.array([shr_a[f] for f in ufa])[fai] * pa
    b_cap = fw_beta(ya_res, xa, resa)
    # permute
    hits = 0
    vals = np.array([shr_a[f] for f in ufa], float)
    for _ in range(B):
        pv = vals.copy(); rng.shuffle(pv)
        b = fw_beta(ya_res, pv[fai] * pa, resa)
        if np.isfinite(b) and abs(b) >= abs(b_cap) - 1e-15:
            hits += 1
    p_cap = (1 + hits) / (1 + B)
    out["a_cap"] = (b_cap, p_cap)
    print(f"annual capex share  : b={b_cap*100:+.2f}pp  permutation p={p_cap:.3f}")

    # ---------- placebo shock: quarterly, post = Dec2024+Mar2025, data <= Sep2025 ----------
    ORDER = {q: i for i, q in enumerate(P.QORDER)}
    cutoff = ORDER["Sep2025"]
    pl = [r for r in qrows if np.isfinite(r["irate"]) and ORDER[r["q"]] <= cutoff]
    ypl = np.array([r["irate"] for r in pl])
    fpl = [r["t"] for r in pl]; tpl = [r["q"] for r in pl]
    postpl = np.array([1.0 if r["q"] in ("Dec2024", "Mar2025") else 0.0 for r in pl])
    xpl_b = np.array([float(psb_rel[r["t"]]) for r in pl]) * postpl
    xpl_s = np.array([float(psb_share[r["t"]]) for r in pl]) * postpl
    bpl_b, sepl_b, npl, Gpl, *_ = P.ols_fe(ypl, xpl_b.reshape(-1, 1), fpl, extra_fe=tpl)
    bpl_s, sepl_s, *_ = P.ols_fe(ypl, xpl_s.reshape(-1, 1), fpl, extra_fe=tpl)[:2]
    out["pl_bin"] = (bpl_b[0], sepl_b[0]); out["pl_shr"] = (bpl_s[0], sepl_s[0])
    print(f"placebo quarterly binary: b={bpl_b[0]*100:+.2f}pp (se {sepl_b[0]*100:.2f}) n={npl}")
    print(f"placebo quarterly share : b={bpl_s[0]*100:+.2f}pp (se {sepl_s[0]*100:.2f})")

    # ---------- placebo annual: drop FY2026, post = FY2025 ----------
    pla = [r for r in rows if np.isfinite(r["cod"]) and r["fy"] != "Mar2026"]
    ypa = np.array([r["cod"] for r in pla])
    fpa = [r["t"] for r in pla]; tpa = [r["fy"] for r in pla]
    postpa = np.array([1.0 if r["fy"] == "Mar2025" else 0.0 for r in pla])
    xpa = np.array([float(psb_rel[r["t"]]) for r in pla]) * postpa
    bpa, sepa, npa, *_ = P.ols_fe(ypa, xpa.reshape(-1, 1), fpa, extra_fe=tpa)
    out["pl_cod_a"] = (bpa[0], sepa[0])
    print(f"placebo annual CoD binary: b={bpa[0]*100:+.2f}pp (se {sepa[0]*100:.2f}) n={npa}")

    # ---------- LaTeX table ----------
    def s(b, p=None, se=None, dec=2):
        v = f"{b*100:.{dec}f}"
        if p is not None:
            return v, f"[{p:.3f}]"
        return v, f"({se*100:.{dec}f})"
    r1 = s(*out["q_bin"]); r2 = s(*out["q_shr"]); r3 = s(*out["a_cap"])
    r4 = s(out["pl_bin"][0], se=out["pl_bin"][1]); r5 = s(out["pl_shr"][0], se=out["pl_shr"][1])
    r6 = s(out["pl_cod_a"][0], se=out["pl_cod_a"][1])
    tex = rf"""\begin{{tabular}}{{lccc}}
\toprule
\multicolumn{{4}}{{l}}{{\emph{{Panel A: randomization inference (exact permutation $p$, {B:,} draws)}}}} \\
 & Quarterly CoD & Quarterly CoD & Annual capex \\
 & (binary) & (share) & (share) \\
\midrule
Treatment $\times$ Post & {r1[0]} & {r2[0]} & {r3[0]} \\
Permutation $p$-value & {r1[1]} & {r2[1]} & {r3[1]} \\
\midrule
\multicolumn{{4}}{{l}}{{\emph{{Panel B: placebo shock one year earlier (pre-shock data only)}}}} \\
 & Quarterly CoD & Quarterly CoD & Annual CoD \\
 & (binary) & (share) & (binary) \\
\midrule
Treatment $\times$ Placebo post & {r4[0]} & {r5[0]} & {r6[0]} \\
 & {r4[1]} & {r5[1]} & {r6[1]} \\
\bottomrule
\end{{tabular}}"""
    with open(os.path.join(TABLES, "p_t6_inference.tex"), "w") as f:
        f.write(tex)
    print("wrote p_t6_inference.tex")

if __name__ == "__main__":
    main()
