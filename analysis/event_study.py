"""
event_study.py — Phase 3: stock-return event study around the four dated events
of the 2025-26 shock, using free daily prices (derived/prices.json) and the same
PSB classification as the pilot. Zero external stats deps (numpy only).

Events:
  sanctions  2025-10-22  (Rosneft/Lukoil)      -> adverse
  hormuz     2026-02-28  (strait closes)        -> adverse
  blockade   2026-04-13  (US naval blockade)    -> adverse
  accord     2026-06-17  (Islamabad MOU)        -> RELIEF (sign should flip)

Method: market-model abnormal returns. For each firm and event, estimate
alpha,beta by OLS of firm daily return on NIFTY return over an estimation window
[-110,-11] trading days; AR_t = R_t - (alpha+beta*Rm_t); CAR over [-1,+1].
Cross-section: CAR_i = a + b*PSB_i (+ c*PSB_i*HiExposure_i) + e, HC-robust SE.

Prediction: if PSB attachment is crisis insurance, b>0 on adverse events
(cushion) and b<0 on the relief event (insurance unwinds) -- a sign flip that a
mechanical confounder cannot easily produce.
"""
import json, os, glob, datetime as _dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "pilot_raw")
DER = os.path.join(HERE, "derived")
TABLES = os.path.join(HERE, "..", "paper", "tables")

EVENTS = [("sanctions", "2025-10-22", "adverse"),
          ("hormuz",    "2026-02-28", "adverse"),
          ("blockade",  "2026-04-13", "adverse"),
          ("accord",    "2026-06-17", "relief")]
CAR_WIN = (-1, 1)
EST_WIN = (-110, -11)

PSB_PATTERNS = ["state bank of india", "bank of baroda", "punjab national bank",
    "canara bank", "union bank", "indian bank", "bank of india",
    "central bank of india", "indian overseas bank", "uco bank",
    "bank of maharashtra", "punjab & sind", "punjab and sind", "oriental bank",
    "united bank of india", "syndicate bank", "andhra bank", "corporation bank",
    "allahabad bank", "dena bank", "vijaya bank", "exim bank",
    "export-import bank", "export import bank", "sidbi", "nabard"]

def classify(name):
    n = name.lower().strip()
    if not n or "idbi" in n:
        return 0 if n else None
    return 1 if any(p in n for p in PSB_PATTERNS) else 0

def _num(s):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")

def load_psb():
    share = {}
    for fp in glob.glob(os.path.join(RAW, "bankers*.txt")):
        for line in open(fp, encoding="utf-8"):
            p = [x.strip() for x in line.split("|")]
            if p[0] == "BANKERS" and len(p) >= 3 and "UNKNOWN" not in p[2].upper():
                cl = [classify(b) for b in p[2].split(";") if b.strip()]
                cl = [c for c in cl if c is not None]
                if cl:
                    share[p[1]] = sum(cl) / len(cl)
    # amount-weighted registry classification takes precedence
    NONBANK = ["finance", "housing development finance", "financial services"]
    psb_amt, tot_amt = {}, {}
    for fp in glob.glob(os.path.join(RAW, "registry_*.txt")):
        for line in open(fp, encoding="utf-8"):
            p = [x.strip() for x in line.split("|")]
            if p[0] != "FACILITY" or len(p) < 5:
                continue
            bank, amt = p[2], _num(p[4])
            if bank in ("NA", "") or not (amt == amt) or any(nb in bank.lower() for nb in NONBANK):
                continue
            c = classify(bank)
            if c is None:
                continue
            tot_amt[p[1]] = tot_amt.get(p[1], 0.0) + amt
            if c == 1:
                psb_amt[p[1]] = psb_amt.get(p[1], 0.0) + amt
    for t, v in tot_amt.items():
        if v > 0:
            share[t] = psb_amt.get(t, 0.0) / v
    return {t: (1 if s > 0.5 else 0) for t, s in share.items()}, share

# sector exposure (reuse pilot map, condensed)
HI = set(["INDIGO","SPICEJET","ASIANPAINT","BERGEPAINT","KANSAINER","AKZOINDIA",
 "BLUEDART","VRLLOG","TCI","ALLCARGO","SRF","AARTIIND","DEEPAKNTR","TATACHEM","UPL",
 "ATUL","NAVINFLUOR","FLUOROCHEM","JUBLINGREA","MRF","APOLLOTYRE","CEATLTD","JKTYRE",
 "BALKRISIND","SUPREMEIND","ASTRAL","FINPIPE","TIMETECHNO","ULTRACEMCO","SHREECEM",
 "DALBHARAT","JKCEMENT","RAMCOCEM","INDIACEM","BIRLACORPN","HEIDELBERG","COROMANDEL",
 "CHAMBLFERT","DEEPAKFERT","PARADEEP","JKPAPER","WSTCSTPAPR","KAJARIACER","ASAHIINDIA",
 "CERA","TATASTEEL","JSWSTEEL","JINDALSTEL","HINDALCO","VEDL","JSL","ARVIND","TRIDENT",
 "VTL","WELSPUNLIV","KPRMILL",
 # mid-cap expansion (all energy/commodity-intensive)
 "BALRAMCHIN","EIDPARRY","TRIVENI","DHAMPURSUG","BAJAJHIND","RENUKA","ASHOKA","KNRCON",
 "GRINFRA","IRB","NCC","HCC","PNCINFRA","HGINFRA","PATELENG","SHYAMMETL","GPIL","SARDAEN",
 "JINDALSAW","KALYANI","USHAMART","RATNAMANI","APLAPOLLO","NCLIND","SESHAPAPER","CHAMBLFERT",
 "DEEPAKFERT","MUKANDLTD","WELCORP","DBL","LTFOODS","GOKULAGRO"])

def dstr(ts):
    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

def ols_hc(y, X):
    """OLS with HC1 robust SEs. X includes intercept column."""
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    meat = (X * (e**2)[:, None]).T @ X
    V = XtXi @ meat @ XtXi * (n / max(n - k, 1))
    return b, np.sqrt(np.diag(V))

def main():
    prices = json.load(open(os.path.join(DER, "prices.json")))
    nifty = prices["_NIFTY"]
    ndates = [dstr(t) for t, _ in nifty]
    nret = {ndates[i]: (nifty[i][1] / nifty[i-1][1] - 1)
            for i in range(1, len(nifty))}
    psb, share = load_psb()

    # per-firm daily returns keyed by date
    firm_ret = {}
    for t, series in prices.items():
        if t == "_NIFTY":
            continue
        r = {}
        for i in range(1, len(series)):
            d = dstr(series[i][0])
            if series[i-1][1] and series[i][1]:
                r[d] = series[i][1] / series[i-1][1] - 1
        firm_ret[t] = r

    common = ndates  # index trading calendar

    def car_for(t, ev_date):
        r = firm_ret.get(t, {})
        # locate event index on the trading calendar (first date >= ev_date)
        cal = [d for d in common if d in r and d in nret]
        if ev_date not in cal:
            after = [d for d in cal if d >= ev_date]
            if not after:
                return None
            e0 = cal.index(after[0])
        else:
            e0 = cal.index(ev_date)
        est = cal[max(0, e0 + EST_WIN[0]): e0 + EST_WIN[1] + 1]
        if len(est) < 40:
            return None
        ry = np.array([r[d] for d in est])
        rm = np.array([nret[d] for d in est])
        A = np.column_stack([np.ones(len(rm)), rm])
        beta, _ = ols_hc(ry, A)
        car = 0.0
        for k in range(CAR_WIN[0], CAR_WIN[1] + 1):
            j = e0 + k
            if 0 <= j < len(cal):
                d = cal[j]
                car += r[d] - (beta[0] + beta[1] * nret[d])
        return car

    results = {}
    for name, dt_, kind in EVENTS:
        rows = []
        for t in firm_ret:
            if t not in psb:
                continue
            c = car_for(t, dt_)
            if c is None or not np.isfinite(c):
                continue
            rows.append((t, c, psb[t], 1 if t in HI else 0))
        y = np.array([r[1] for r in rows]) * 100  # CAR in %
        p = np.array([r[2] for r in rows], float)
        hi = np.array([r[3] for r in rows], float)
        X1 = np.column_stack([np.ones(len(y)), p])
        b1, se1 = ols_hc(y, X1)
        X2 = np.column_stack([np.ones(len(y)), p, p * hi, hi])
        b2, se2 = ols_hc(y, X2)
        results[name] = dict(kind=kind, n=len(y), nPSB=int(p.sum()),
                             b_psb=b1[1], se_psb=se1[1],
                             b_psbhi=b2[2], se_psbhi=se2[2])
    return results

def stars(b, se):
    if se == 0 or not np.isfinite(se):
        return ""
    t = abs(b / se)
    return "$^{***}$" if t > 2.576 else "$^{**}$" if t > 1.96 else "$^{*}$" if t > 1.645 else ""

def write_table(res):
    order = ["sanctions", "hormuz", "blockade", "accord"]
    lab = {"sanctions": "Sanctions\\newline(22 Oct 25)", "hormuz": "Hormuz\\newline(28 Feb 26)",
           "blockade": "Blockade\\newline(13 Apr 26)", "accord": "Accord\\newline(17 Jun 26)"}
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         " & Sanctions & Hormuz & Blockade & Accord \\\\",
         " & (adverse) & (adverse) & (adverse) & (relief) \\\\",
         r"\midrule"]
    row1 = "PSB $\\times$ event CAR (\\%)"
    row2 = ""
    for k in order:
        r = res[k]
        row1 += f" & {r['b_psb']:.2f}{stars(r['b_psb'], r['se_psb'])}"
        row2 += f" & ({r['se_psb']:.2f})"
    L.append(row1 + r" \\")
    L.append(" " + row2 + r" \\")
    L.append(r"\midrule")
    r3 = "PSB $\\times$ HiExposure"
    r4 = ""
    for k in order:
        r = res[k]
        r3 += f" & {r['b_psbhi']:.2f}{stars(r['b_psbhi'], r['se_psbhi'])}"
        r4 += f" & ({r['se_psbhi']:.2f})"
    L.append(r3 + r" \\")
    L.append(" " + r4 + r" \\")
    L.append(r"\midrule")
    L.append("Firms & " + " & ".join(str(res[k]['n']) for k in order) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(TABLES, "p_t5_eventstudy.tex"), "w").write("\n".join(L))

if __name__ == "__main__":
    res = main()
    for k in ["sanctions", "hormuz", "blockade", "accord"]:
        r = res[k]
        print(f"{k:9s} ({r['kind']:7s}) n={r['n']} nPSB={r['nPSB']}  "
              f"PSB CAR b={r['b_psb']:+.2f} (se {r['se_psb']:.2f}, t={r['b_psb']/r['se_psb']:+.2f})  "
              f"PSBxHi b={r['b_psbhi']:+.2f} (se {r['se_psbhi']:.2f}, t={r['b_psbhi']/r['se_psbhi']:+.2f})")
    write_table(res)
    print("wrote p_t5_eventstudy.tex")
