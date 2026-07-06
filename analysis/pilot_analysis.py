"""
pilot_analysis.py — Estimate the paper's specifications on the hand-collected
PUBLIC-DATA pilot sample (screener.in financial tables + annual-report banker
lists) and emit LaTeX tables into ../paper/tables/.

Inputs (produced by the collection pipeline in the pilot scratchpad; copy the
data_*.txt and bankers*.txt files into analysis/pilot_raw/ for replication):
  data_<TICKER>.txt   KIND / ANNUAL / QTR pipe-format (see collect_all.py)
  bankers0*.txt       BANKERS|TICKER|Bank1; Bank2; ...|url|note

Design (annual, FY2022-FY2026, post = FY2026):
  cost_of_debt_it = interest_t / [0.5*(borrowings_t + borrowings_{t-1})]
  capex_it        = [Δ(fixed assets + CWIP) + depreciation_t] / assets_{t-1}
  DiD:  y_it = b * PSB_i x Post_t + firm FE + year FE (+ controls), cluster(firm)
  DDD:  + g * PSB_i x Post_t x HiExposure_i + Post x HiExposure
Quarterly event study (Mar2023..Mar2026, ref = Sep2025):
  intrate_iq = interest_q / borrowings_{prev FY} ; coeffs on PSB x quarter.

Only numpy + matplotlib (no external stats packages).
"""
import glob, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("PILOT_RAW", os.path.join(HERE, "pilot_raw"))
TABLES = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(TABLES, exist_ok=True)

# ---------------- PSB classification (12 PSBs + pre-merger names) -----------
PSB_PATTERNS = [
    "state bank of india", "sbi", "bank of baroda", "punjab national bank",
    "canara bank", "union bank", "indian bank", "bank of india",
    "central bank of india", "indian overseas bank", "uco bank",
    "bank of maharashtra", "punjab & sind", "punjab and sind",
    # pre-merger constituents
    "oriental bank", "united bank of india", "syndicate bank", "andhra bank",
    "corporation bank", "allahabad bank", "dena bank", "vijaya bank",
    # other government-owned lenders
    "exim bank", "export-import bank", "sidbi", "nabard",
]
PRIVATE_HINTS = ["idbi"]  # RBI-classified private since Jan 2019

def classify_bank(name):
    n = name.lower().strip()
    if not n:
        return None
    for h in PRIVATE_HINTS:
        if h in n:
            return 0
    for p in PSB_PATTERNS:
        if p in n:
            # guard: "state bank of india" matches before private "sbm" etc.
            return 1
    return 0

# ---------------- sector -> exposure mapping --------------------------------
SECTOR = {  # ticker: (sector, hi_energy_exposure)
 **{t: ("aviation", 1) for t in ["INDIGO", "SPICEJET"]},
 **{t: ("paints", 1) for t in ["ASIANPAINT", "BERGEPAINT", "KANSAINER", "AKZOINDIA"]},
 **{t: ("logistics", 1) for t in ["BLUEDART", "VRLLOG", "TCI", "ALLCARGO"]},
 **{t: ("chemicals", 1) for t in ["SRF", "AARTIIND", "DEEPAKNTR", "TATACHEM", "UPL",
                                   "ATUL", "NAVINFLUOR", "FLUOROCHEM", "JUBLINGREA"]},
 **{t: ("tyres", 1) for t in ["MRF", "APOLLOTYRE", "CEATLTD", "JKTYRE", "BALKRISIND"]},
 **{t: ("plastics", 1) for t in ["SUPREMEIND", "ASTRAL", "FINPIPE", "TIMETECHNO"]},
 **{t: ("cement", 1) for t in ["ULTRACEMCO", "SHREECEM", "DALBHARAT", "JKCEMENT",
                                "RAMCOCEM", "INDIACEM", "BIRLACORPN", "HEIDELBERG"]},
 **{t: ("fertiliser", 1) for t in ["COROMANDEL", "CHAMBLFERT", "DEEPAKFERT", "PARADEEP"]},
 **{t: ("paper", 1) for t in ["JKPAPER", "WSTCSTPAPR"]},
 **{t: ("glass_ceramics", 1) for t in ["KAJARIACER", "ASAHIINDIA", "CERA"]},
 **{t: ("metals", 1) for t in ["TATASTEEL", "JSWSTEEL", "JINDALSTEL", "HINDALCO", "VEDL", "JSL"]},
 **{t: ("textiles", 1) for t in ["ARVIND", "TRIDENT", "VTL", "WELSPUNLIV", "KPRMILL"]},
 **{t: ("pharma", 0) for t in ["SUNPHARMA", "CIPLA", "LUPIN", "AUROPHARMA",
                                "TORNTPHARM", "ZYDUSLIFE", "GLENMARK", "BIOCON"]},
 **{t: ("fmcg", 0) for t in ["MARICO", "DABUR", "EMAMILTD", "GODREJCP", "BRITANNIA"]},
 **{t: ("durables", 0) for t in ["HAVELLS", "VOLTAS", "CROMPTON"]},
 **{t: ("auto", 0) for t in ["TATAMOTORS", "ASHOKLEY", "M&M", "M_M", "TVSMOTOR", "ESCORTS"]},
 **{t: ("engineering", 0) for t in ["CUMMINSIND", "THERMAX", "KEC"]},
 # --- mid-cap expansion (Phase 2), all energy/commodity-intensive => hi exposure ---
 **{t: ("sugar", 1) for t in ["BALRAMCHIN", "EIDPARRY", "TRIVENI", "DHAMPURSUG", "BAJAJHIND", "RENUKA"]},
 **{t: ("epc", 1) for t in ["ASHOKA", "KNRCON", "GRINFRA", "IRB", "NCC", "HCC", "PNCINFRA", "HGINFRA", "PATELENG", "DBL", "JKIL", "SADBHIN"]},
 **{t: ("metals", 1) for t in ["SHYAMMETL", "GPIL", "SARDAEN", "JINDALSAW", "KALYANI", "USHAMART", "RATNAMANI", "APLAPOLLO", "MUKANDLTD", "WELCORP"]},
 **{t: ("cement", 1) for t in ["NCLIND", "SAGCEM", "KESORAMIND"]},
 **{t: ("paper", 1) for t in ["SESHAPAPER"]},
 **{t: ("agri_processing", 1) for t in ["LTFOODS", "GOKULAGRO", "KRBL", "PATANJALI"]},
}

YEARS = ["Mar2022", "Mar2023", "Mar2024", "Mar2025", "Mar2026"]
QORDER = [f"{m}{y}" for y in (2023, 2024, 2025, 2026) for m in ("Mar", "Jun", "Sep", "Dec")
          if not (y == 2026 and m in ("Jun", "Sep", "Dec"))]
QORDER = sorted(set(QORDER), key=lambda q: (int(q[-4:]), {"Mar": 0, "Jun": 1, "Sep": 2, "Dec": 3}[q[:3]]))
POST_Q = {"Dec2025", "Mar2026"}          # first results quarters after 22 Oct 2025
REF_Q = "Sep2025"                        # last fully pre-shock quarter

def num(s):
    s = (s or "").replace(",", "").replace("%", "").strip()
    if s in ("", "NA", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

# ---------------- load ------------------------------------------------------
def load():
    firms, annual, qtr = {}, [], []
    for fp in glob.glob(os.path.join(RAW, "data_*.txt")):
        for line in open(fp, encoding="utf-8"):
            p = [x.strip() for x in line.rstrip("\n").split("|")]
            if p[0] == "KIND" and len(p) >= 4:
                firms[p[1]] = {"name": p[2], "kind": p[3]}
            elif p[0] == "ANNUAL" and len(p) >= 10:
                annual.append(dict(t=p[1], fy=p[2], bor=num(p[3]), intr=num(p[4]),
                                   fa=num(p[5]), cwip=num(p[6]), dep=num(p[7]),
                                   sales=num(p[8]), op=num(p[9]),
                                   ta=num(p[10]) if len(p) > 10 else np.nan))
            elif p[0] == "QTR" and len(p) >= 5:
                qtr.append(dict(t=p[1], q=p[2], sales=num(p[3]), intr=num(p[4])))
    bankers = {}
    for fp in glob.glob(os.path.join(RAW, "bankers*.txt")):
        for line in open(fp, encoding="utf-8"):
            p = [x.strip() for x in line.rstrip("\n").split("|")]
            if p[0] == "BANKERS" and len(p) >= 4 and "UNKNOWN" not in p[2].upper():
                bl = [b.strip() for b in p[2].split(";") if b.strip()]
                bankers[p[1]] = bl
    return firms, annual, qtr, bankers

def load_registry():
    """Read registry_*.txt FACILITY lines -> amount-weighted PSB share per firm.
    Excludes non-bank lenders (NBFCs/HFCs) from both numerator and denominator so
    the share is PSB-bank / all-bank facility value. Returns {ticker: share}."""
    NONBANK = ["finance", "housing development finance", "financial services"]
    psb_amt, tot_amt = {}, {}
    for fp in glob.glob(os.path.join(RAW, "registry_*.txt")):
        for line in open(fp, encoding="utf-8"):
            p = [x.strip() for x in line.rstrip("\n").split("|")]
            if p[0] != "FACILITY" or len(p) < 5:
                continue
            t, bank, amt = p[1], p[2], num(p[4])
            if not np.isfinite(amt) or bank in ("NA", ""):
                continue
            low = bank.lower()
            if any(nb in low for nb in NONBANK):
                continue  # NBFC/HFC, not a bank
            c = classify_bank(bank)
            if c is None:
                continue
            tot_amt[t] = tot_amt.get(t, 0.0) + amt
            if c == 1:
                psb_amt[t] = psb_amt.get(t, 0.0) + amt
    return {t: psb_amt.get(t, 0.0) / v for t, v in tot_amt.items() if v > 0}

# ---------------- regression helpers (numpy) --------------------------------
def ols_fe(y, X, firm_ids, extra_fe=None):
    """OLS of y on X plus firm dummies and extra_fe dummies (list of labels).
    Returns beta for X columns and cluster-robust (by firm) SEs."""
    n = len(y)
    fset = sorted(set(firm_ids))
    fmap = {f: i for i, f in enumerate(fset)}
    D_firm = np.zeros((n, len(fset) - 1))
    for i, f in enumerate(firm_ids):
        j = fmap[f]
        if j > 0:
            D_firm[i, j - 1] = 1.0
    mats = [X, D_firm]
    if extra_fe is not None:
        eset = sorted(set(extra_fe))
        emap = {e: i for i, e in enumerate(eset)}
        D_e = np.zeros((n, len(eset) - 1))
        for i, e in enumerate(extra_fe):
            j = emap[e]
            if j > 0:
                D_e[i, j - 1] = 1.0
        mats.append(D_e)
    mats.append(np.ones((n, 1)))
    Z = np.hstack(mats)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    ZtZinv = np.linalg.pinv(Z.T @ Z)
    meat = np.zeros((Z.shape[1], Z.shape[1]))
    for f in fset:
        m = np.array([ff == f for ff in firm_ids])
        Zg, ug = Z[m], resid[m]
        s = Zg.T @ ug
        meat += np.outer(s, s)
    G = len(fset)
    dfc = (G / (G - 1)) * ((n - 1) / max(n - Z.shape[1], 1))
    V = dfc * ZtZinv @ meat @ ZtZinv
    k = X.shape[1]
    return beta[:k], np.sqrt(np.diag(V)[:k]), n, G, resid, Z, beta

def stars(b, se):
    if not np.isfinite(se) or se == 0:
        return ""
    t = abs(b / se)
    return "$^{***}$" if t > 2.576 else "$^{**}$" if t > 1.96 else "$^{*}$" if t > 1.645 else ""

def fmt(b, se, scale=1.0, dec=2):
    return f"{b*scale:.{dec}f}{stars(b, se)}", f"({se*scale:.{dec}f})"

# ---------------- build panel ------------------------------------------------
def build():
    firms, annual, qtr, bankers = load()
    psb_share, psb_rel, nbank = {}, {}, {}
    for t, bl in bankers.items():
        cl = [classify_bank(b) for b in bl]
        cl = [c for c in cl if c is not None]
        if cl:
            psb_share[t] = sum(cl) / len(cl)
            psb_rel[t] = 1 if psb_share[t] > 0.5 else 0
            nbank[t] = len(cl)
    # amount-weighted registry classification takes precedence where available
    reg = load_registry()
    for t, sh in reg.items():
        psb_share[t] = sh
        psb_rel[t] = 1 if sh > 0.5 else 0
        nbank.setdefault(t, 0)
    A = {}
    for r in annual:
        A.setdefault(r["t"], {})[r["fy"]] = r
    # pre-shock interest coverage (operating profit / interest), avg FY2024-FY2025;
    # used for the H4 "does the cushion reach financially weaker firms?" split.
    icr = {}
    for t, ys in A.items():
        vals = []
        for fy in ("Mar2024", "Mar2025"):
            r = ys.get(fy)
            if r and np.isfinite(r["op"]) and np.isfinite(r["intr"]) and r["intr"] > 0:
                vals.append(r["op"] / r["intr"])
        if vals:
            icr[t] = np.mean(vals)
    rows = []
    for t, ys in A.items():
        if t not in psb_rel or t not in SECTOR:
            continue
        for i, fy in enumerate(YEARS[1:], start=1):
            cur, prev = ys.get(fy), ys.get(YEARS[i - 1])
            if not cur or not prev:
                continue
            avg_bor = np.nanmean([cur["bor"], prev["bor"]])
            if not np.isfinite(avg_bor) or avg_bor < 50:   # need real debt (>= Rs 50 cr)
                continue
            cod = cur["intr"] / avg_bor
            base = prev["fa"] + (prev["cwip"] if np.isfinite(prev["cwip"]) else 0)
            capex = (cur["fa"] + (cur["cwip"] if np.isfinite(cur["cwip"]) else 0)
                     - base + (cur["dep"] if np.isfinite(cur["dep"]) else 0))
            denom = prev["ta"] if np.isfinite(prev.get("ta", np.nan)) and prev["ta"] > 0 else base
            capex_r = capex / denom if denom and np.isfinite(denom) and denom > 0 else np.nan
            rows.append(dict(t=t, fy=fy, post=1 if fy == "Mar2026" else 0,
                             cod=cod, capex=capex_r, psb=psb_rel[t],
                             share=psb_share[t], hi=SECTOR[t][1], sec=SECTOR[t][0],
                             icr=icr.get(t, np.nan),
                             size=np.log(cur["sales"]) if cur["sales"] and np.isfinite(cur["sales"]) and cur["sales"] > 0 else np.nan))
    # weak = below-median pre-shock ICR among sample firms (partial H4)
    icr_vals = [v for v in icr.values() if np.isfinite(v)]
    icr_med = np.median(icr_vals) if icr_vals else np.nan
    for r in rows:
        r["weak"] = 1 if (np.isfinite(r["icr"]) and r["icr"] < icr_med) else 0
    # quarterly
    Q = {}
    for r in qtr:
        Q.setdefault(r["t"], {})[r["q"]] = r
    qrows = []
    for t, qs in Q.items():
        if t not in psb_rel or t not in SECTOR:
            continue
        for q in QORDER:
            r = qs.get(q)
            if not r or not np.isfinite(r["intr"]):
                continue
            fyprev = {"2023": "Mar2022", "2024": "Mar2023", "2025": "Mar2024", "2026": "Mar2025"}
            yr = q[-4:]
            mon = q[:3]
            fy_end = yr if mon == "Mar" else str(int(yr) + 1)
            lagfy = fyprev.get(fy_end)
            bor = A.get(t, {}).get(lagfy, {}).get("bor", np.nan) if lagfy else np.nan
            if not np.isfinite(bor) or bor < 50:
                continue
            qrows.append(dict(t=t, q=q, post=1 if q in POST_Q else 0,
                              irate=r["intr"] / bor, psb=psb_rel[t], hi=SECTOR[t][1]))
    return rows, qrows, psb_rel, psb_share, firms, bankers

def winsor(v, lo=0.01, hi=0.99):
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if m.sum() < 5:
        return v
    a, b = np.nanquantile(v[m], [lo, hi])
    return np.clip(v, a, b)

# ---------------- estimation --------------------------------------------------
def run():
    rows, qrows, psb_rel, psb_share, firms, bankers = build()
    for key in ("cod", "capex"):
        vals = winsor([r[key] for r in rows])
        for r, v in zip(rows, vals):
            r[key] = v
    ivals = winsor([r["irate"] for r in qrows])
    for r, v in zip(qrows, ivals):
        r["irate"] = v

    out = {}
    def did(rows, ykey, ddd=False):
        rr = [r for r in rows if np.isfinite(r[ykey])]
        y = np.array([r[ykey] for r in rr])
        firm = [r["t"] for r in rr]
        yrfe = [r["fy"] for r in rr]
        pp = np.array([r["psb"] * r["post"] for r in rr], float)
        if ddd:
            X = np.column_stack([pp,
                                 [r["psb"] * r["post"] * r["hi"] for r in rr],
                                 [r["post"] * r["hi"] for r in rr]])
        else:
            X = pp.reshape(-1, 1)
        b, se, n, G, *_ = ols_fe(y, X, firm, extra_fe=yrfe)
        return b, se, n, G

    out["cod_did"] = did(rows, "cod")
    out["capex_did"] = did(rows, "capex")
    out["cod_ddd"] = did(rows, "cod", ddd=True)
    out["capex_ddd"] = did(rows, "capex", ddd=True)

    def did_share(rows, ykey):
        rr = [r for r in rows if np.isfinite(r[ykey])]
        y = np.array([r[ykey] for r in rr])
        X = np.array([[r["share"] * r["post"]] for r in rr], float)
        return ols_fe(y, X, [r["t"] for r in rr], extra_fe=[r["fy"] for r in rr])[:4]
    out["cod_share"] = did_share(rows, "cod")
    out["capex_share"] = did_share(rows, "capex")

    # H4 (partial): does the cushion reach financially weaker firms? Split the
    # PSBxPost effect by pre-shock below-median interest coverage.
    def did_h4(rows, ykey):
        rr = [r for r in rows if np.isfinite(r[ykey]) and np.isfinite(r["icr"])]
        if len(rr) < 30:
            return None
        y = np.array([r[ykey] for r in rr])
        X = np.column_stack([[r["psb"] * r["post"] for r in rr],
                             [r["psb"] * r["post"] * r["weak"] for r in rr],
                             [r["post"] * r["weak"] for r in rr]])
        b, se, n, G, *_ = ols_fe(y, X, [r["t"] for r in rr], extra_fe=[r["fy"] for r in rr])
        return b, se, n, G
    out["cod_h4"] = did_h4(rows, "cod")
    out["weak_psb"] = sorted({r["t"] for r in rows if r["psb"] and r["weak"]})

    # quarterly event study on interest rate proxy
    rr = [r for r in qrows if np.isfinite(r["irate"])]
    y = np.array([r["irate"] for r in rr])
    firm = [r["t"] for r in rr]
    qfe = [r["q"] for r in rr]
    ks = [q for q in QORDER if q != REF_Q and any(r["q"] == q for r in rr)]
    X = np.column_stack([[1.0 * (r["psb"] and r["q"] == k) for r in rr] for k in ks])
    bES, seES, nES, GES, *_ = ols_fe(y, X, firm, extra_fe=qfe)
    out["es"] = (ks, bES, seES, nES, GES)
    # quarterly DiD, binary and share treatments
    Xq = np.array([[r["psb"] * r["post"]] for r in rr], float)
    out["irate_did"] = ols_fe(y, Xq, firm, extra_fe=qfe)[:4]
    shr = {t: psb_share[t] for t in psb_share}
    Xs = np.array([[shr[r["t"]] * r["post"]] for r in rr], float)
    out["irate_share"] = ols_fe(y, Xs, firm, extra_fe=qfe)[:4]
    out["treated_firms"] = sorted({r["t"] for r in rows if r["psb"] == 1})

    # descriptives
    desc = {}
    firm_level = {}
    for r in rows:
        firm_level.setdefault(r["t"], []).append(r)
    for grp in (1, 0):
        pre_cod = [r["cod"] for r in rows if r["psb"] == grp and not r["post"] and np.isfinite(r["cod"])]
        pre_cap = [r["capex"] for r in rows if r["psb"] == grp and not r["post"] and np.isfinite(r["capex"])]
        hi = [SECTOR[t][1] for t in firm_level if psb_rel[t] == grp]
        nb = [len(bankers[t]) for t in firm_level if psb_rel[t] == grp and t in bankers]
        sh = [psb_share[t] for t in firm_level if psb_rel[t] == grp]
        desc[grp] = dict(n=len([t for t in firm_level if psb_rel[t] == grp]),
                         cod=(np.mean(pre_cod), np.std(pre_cod)),
                         cap=(np.mean(pre_cap), np.std(pre_cap)),
                         hi=np.mean(hi), nb=np.mean(nb), sh=np.mean(sh))
    out["desc"] = desc
    out["counts"] = dict(firms=len(firm_level), fy_obs=len(rows), q_obs=len(rr),
                         psb=desc[1]["n"], prv=desc[0]["n"])
    return out, rows, qrows

def write_tables(out):
    c = out["counts"]; d = out["desc"]
    def cell(x, dec=2): return f"{x:.{dec}f}"
    # ---- Table 1: descriptives ----
    t1 = rf"""\begin{{tabular}}{{lcc}}
\toprule
 & PSB-attached & Private-bank \\
\midrule
Firms & {d[1]['n']} & {d[0]['n']} \\
Pre-shock cost of debt (\%, mean) & {cell(d[1]['cod'][0]*100)} & {cell(d[0]['cod'][0]*100)} \\
\quad (SD) & ({cell(d[1]['cod'][1]*100)}) & ({cell(d[0]['cod'][1]*100)}) \\
Pre-shock capex / assets (\%, mean) & {cell(d[1]['cap'][0]*100)} & {cell(d[0]['cap'][0]*100)} \\
\quad (SD) & ({cell(d[1]['cap'][1]*100)}) & ({cell(d[0]['cap'][1]*100)}) \\
Pre-shock PSB share of bankers & {cell(d[1]['sh'])} & {cell(d[0]['sh'])} \\
Number of disclosed bankers & {cell(d[1]['nb'],1)} & {cell(d[0]['nb'],1)} \\
High energy-exposure sector (share) & {cell(d[1]['hi'])} & {cell(d[0]['hi'])} \\
\bottomrule
\end{{tabular}}"""
    open(os.path.join(TABLES, "p_t1_desc.tex"), "w").write(t1)
    # ---- Table 2: cost of debt ----
    rows_ = []
    for lbl, key, sc in [("PSB $\\times$ Post", "cod_did", 100),
                         ("PSB share $\\times$ Post", "cod_share", 100),
                         ("PSB $\\times$ Post (quarterly)", "irate_did", 100),
                         ("PSB share $\\times$ Post (quarterly)", "irate_share", 100)]:
        b, se, n, G = out[key]
        v, s = fmt(b[0], se[0], sc)
        rows_.append((lbl, v, s, n, G))
    t2 = r"""\begin{tabular}{lcccc}
\toprule
 & (1) Annual & (2) Annual & (3) Quarterly & (4) Quarterly \\
Dep.\ var. & CoD (\%) & CoD (\%) & Int./borrowings (\%) & Int./borrowings (\%) \\
\midrule
"""
    t2 += f"Treatment $\\times$ Post & {rows_[0][1]} & {rows_[1][1]} & {rows_[2][1]} & {rows_[3][1]} \\\\\n"
    t2 += f" & {rows_[0][2]} & {rows_[1][2]} & {rows_[2][2]} & {rows_[3][2]} \\\\\n"
    t2 += "Treatment & Binary & Share & Binary & Share \\\\\n"
    t2 += "Firm FE, time FE & \\checkmark & \\checkmark & \\checkmark & \\checkmark \\\\\n"
    t2 += f"Observations & {rows_[0][3]} & {rows_[1][3]} & {rows_[2][3]} & {rows_[3][3]} \\\\\n"
    t2 += f"Firms & {rows_[0][4]} & {rows_[1][4]} & {rows_[2][4]} & {rows_[3][4]} \\\\\n"
    t2 += "\\bottomrule\n\\end{tabular}"
    open(os.path.join(TABLES, "p_t2_cod.tex"), "w").write(t2)
    # ---- Table 3: capex ----
    b1, se1, n1, G1 = out["capex_did"]; b2, se2, n2, G2 = out["capex_share"]
    v1, s1 = fmt(b1[0], se1[0], 100); v2, s2 = fmt(b2[0], se2[0], 100)
    t3 = rf"""\begin{{tabular}}{{lcc}}
\toprule
Dep.\ var.: capex / lagged assets (\%) & (1) & (2) \\
\midrule
Treatment $\times$ Post & {v1} & {v2} \\
 & {s1} & {s2} \\
Treatment & Binary & Share \\
Firm FE, year FE & \checkmark & \checkmark \\
Observations & {n1} & {n2} \\
Firms & {G1} & {G2} \\
\bottomrule
\end{{tabular}}"""
    open(os.path.join(TABLES, "p_t3_capex.tex"), "w").write(t3)
    # ---- Table 4: DDD ----
    b, se, n, G = out["cod_ddd"]; bc, sec_, nc, Gc = out["capex_ddd"]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r" & (1) CoD (\%) & (2) Capex (\%) \\", r"\midrule"]
    labs = ["PSB $\\times$ Post", "PSB $\\times$ Post $\\times$ HiExposure", "Post $\\times$ HiExposure"]
    for i, lab in enumerate(labs):
        v1, s1 = fmt(b[i], se[i], 100); v2, s2 = fmt(bc[i], sec_[i], 100)
        lines.append(f"{lab} & {v1} & {v2} \\\\")
        lines.append(f" & {s1} & {s2} \\\\")
    lines += [r"Firm FE, year FE & \checkmark & \checkmark \\",
              f"Observations & {n} & {nc} \\\\", f"Firms & {G} & {Gc} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABLES, "p_t4_ddd.tex"), "w").write("\n".join(lines))
    # ---- event-study figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks, bES, seES, nES, GES = out["es"]
    ref_i = QORDER.index(REF_Q)
    xs, ys_, lo, hi = [], [], [], []
    for q in QORDER:
        if q == REF_Q:
            xs.append(q); ys_.append(0.0); lo.append(0.0); hi.append(0.0); continue
        if q in ks:
            i = ks.index(q)
            xs.append(q); ys_.append(bES[i] * 100)
            lo.append((bES[i] - 1.96 * seES[i]) * 100); hi.append((bES[i] + 1.96 * seES[i]) * 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xi = np.arange(len(xs))
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvspan(xi[-2] - 0.5, xi[-1] + 0.5, color="#E8A33D", alpha=0.15,
               label="post-shock (Oct 2025 →)")
    ax.errorbar(xi, ys_, yerr=[np.array(ys_) - np.array(lo), np.array(hi) - np.array(ys_)],
                fmt="o-", color="#2c3e77", ms=4, lw=1.2, capsize=2.5)
    ax.set_xticks(xi); ax.set_xticklabels([q[:3] + "'" + q[-2:] for q in xs], fontsize=8)
    ax.set_ylabel("PSB $-$ private gap,\nquarterly interest / borrowings (pp)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(TABLES, "p_f1_eventstudy.pdf"))
    print("tables + figure written to", TABLES)

if __name__ == "__main__":
    out, rows, qrows = run()
    c = out["counts"]
    print(f"panel: {c['firms']} firms ({c['psb']} PSB / {c['prv']} private), "
          f"{c['fy_obs']} firm-years, {c['q_obs']} firm-quarters")
    for k in ("cod_did", "capex_did", "irate_did"):
        b, se, n, G = out[k]
        print(f"{k}: b={b[0]:+.4f} se={se[0]:.4f} t={b[0]/se[0]:+.2f} (n={n}, firms={G})")
    b, se, n, G = out["cod_ddd"]
    print(f"cod_ddd: PSBxPost={b[0]:+.4f}({se[0]:.4f}) PSBxPostxHi={b[1]:+.4f}({se[1]:.4f}) "
          f"PostxHi={b[2]:+.4f}({se[2]:.4f})")
    ks, bES, seES, nES, GES = out["es"]
    print("event study (PSB x quarter, ref Sep2025):")
    for k, bb, ss in zip(ks, bES, seES):
        print(f"  {k}: {bb:+.4f} ({ss:.4f})")
    for k in ("cod_share", "capex_share", "irate_share"):
        b, se, n, G = out[k]
        print(f"{k}: b={b[0]:+.4f} se={se[0]:.4f} t={b[0]/se[0]:+.2f} (n={n}, firms={G})")
    print("treated (majority-PSB) firms:", ", ".join(out["treated_firms"]))
    if out["cod_h4"]:
        b, se, n, G = out["cod_h4"]
        print(f"H4 cod: PSBxPost={b[0]:+.4f}({se[0]:.4f}) PSBxPostxWeak={b[1]:+.4f}({se[1]:.4f}) "
              f"PostxWeak={b[2]:+.4f}({se[2]:.4f}) (n={n})")
        print("weak (below-median ICR) PSB firms:", ", ".join(out["weak_psb"]))
    write_tables(out)
