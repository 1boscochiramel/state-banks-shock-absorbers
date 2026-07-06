"""fetch_prices.py — pull free daily close prices (Yahoo v8 chart API) for the
pilot tickers plus the NIFTY 50 index over the shock window, cache to
prices.json. Public data, no key. NSE symbols are <TICKER>.NS."""
import json, os, glob, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "pilot_raw")
OUT = os.path.join(HERE, "derived")
os.makedirs(OUT, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0"}

# window: 2025-03-01 .. 2026-07-15 (estimation window before 22 Oct 2025 + all four events)
import datetime as _dt
def _epoch(s):
    return int(_dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp())
P1, P2 = _epoch("2025-03-01"), _epoch("2026-07-15")

# Yahoo symbol overrides where <TICKER>.NS is not the ticker
OVERRIDE = {"M&M": "M%26M.NS", "M_M": "M%26M.NS"}

def tickers():
    ts = []
    for fp in sorted(glob.glob(os.path.join(RAW, "data_*.txt"))):
        t = os.path.basename(fp)[5:-4]
        ts.append(t)
    return ts

def ysym(t):
    return OVERRIDE.get(t, t + ".NS")

def fetch(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1}&period2={P2}&interval=1d")
    r = requests.get(url, headers=H, timeout=25)
    j = r.json()
    res = j.get("chart", {}).get("result")
    if not res:
        return None
    ts = res[0]["timestamp"]
    q = res[0]["indicators"]["quote"][0]["close"]
    adj = res[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", q)
    out = [(t, c) for t, c in zip(ts, adj) if c is not None]
    return out

def main():
    data = {}
    fails = []
    # index first
    idx = fetch("%5ENSEI")   # ^NSEI
    if idx:
        data["_NIFTY"] = idx
        print(f"NIFTY: {len(idx)} days")
    else:
        print("NIFTY FETCH FAILED"); return
    for t in tickers():
        try:
            d = fetch(ysym(t))
            if d and len(d) > 100:
                data[t] = d
                print(f"{t}: {len(d)}")
            else:
                fails.append(t); print(f"{t}: FAIL ({len(d) if d else 0})")
        except Exception as e:
            fails.append(t); print(f"{t}: ERR {e}")
        time.sleep(0.6)
    with open(os.path.join(OUT, "prices.json"), "w") as f:
        json.dump(data, f)
    print(f"\nDONE: {len(data)-1} firms + NIFTY cached; fails={fails}")

if __name__ == "__main__":
    main()
