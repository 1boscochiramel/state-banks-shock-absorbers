"""collect_all.py — fetch+parse screener.in pages for the full 80-ticker pilot
sample; emit uniform pipe-format data_<TICKER>.txt files.
Reuses cached html/<TICKER>.html when present; polite 2.5s delay on fetches.
Numbers are transcribed exactly as displayed (commas kept); NA when absent."""
import json, os, re, sys, time
import requests
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "html")
os.makedirs(HTML, exist_ok=True)

TICKERS = [
 # aviation / paints / logistics
 "INDIGO","SPICEJET","ASIANPAINT","BERGEPAINT","KANSAINER","AKZOINDIA","BLUEDART","VRLLOG",
 # chemicals
 "SRF","AARTIIND","DEEPAKNTR","TATACHEM","UPL","ATUL","NAVINFLUOR","FLUOROCHEM",
 # tyres / plastics
 "MRF","APOLLOTYRE","CEATLTD","JKTYRE","BALKRISIND","SUPREMEIND","ASTRAL","FINPIPE",
 # cement
 "ULTRACEMCO","SHREECEM","DALBHARAT","JKCEMENT","RAMCOCEM","INDIACEM","BIRLACORPN","HEIDELBERG",
 # fertiliser / paper / ceramics-glass
 "COROMANDEL","CHAMBLFERT","DEEPAKFERT","PARADEEP","JKPAPER","WSTCSTPAPR","KAJARIACER","ASAHIINDIA",
 # metals / misc industrials
 "TATASTEEL","JSWSTEEL","JINDALSTEL","HINDALCO","VEDL","JSL","TIMETECHNO","JUBLINGREA",
 # textiles / logistics / building
 "ARVIND","TRIDENT","VTL","WELSPUNLIV","KPRMILL","TCI","ALLCARGO","CERA",
 # pharma
 "SUNPHARMA","CIPLA","LUPIN","AUROPHARMA","TORNTPHARM","ZYDUSLIFE","GLENMARK","BIOCON",
 # fmcg / durables
 "MARICO","DABUR","EMAMILTD","GODREJCP","BRITANNIA","HAVELLS","VOLTAS","CROMPTON",
 # auto / engineering
 "TATAMOTORS","ASHOKLEY","M&M","TVSMOTOR","ESCORTS","CUMMINSIND","THERMAX","KEC",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml"}

class SectionTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = {}; self.cur_section = None; self.section_depth = 0
        self.depth = 0; self.in_table = False; self.in_cell = False
        self.cur_table = None; self.cur_row = None; self.cur_cell = []
        self.h1 = None; self._in_h1 = False
    def handle_starttag(self, tag, attrs):
        a = dict(attrs); self.depth += 1
        if tag == "section" and "id" in a:
            self.cur_section = a["id"]; self.section_depth = self.depth
            self.sections.setdefault(self.cur_section, [])
        if tag == "h1": self._in_h1 = True
        if self.cur_section:
            if tag == "table": self.in_table = True; self.cur_table = []
            elif self.in_table and tag == "tr": self.cur_row = []
            elif self.in_table and tag in ("td","th"): self.in_cell = True; self.cur_cell = []
    def handle_endtag(self, tag):
        if self.cur_section:
            if tag in ("td","th") and self.in_cell:
                txt = re.sub(r"\s+"," ", "".join(self.cur_cell)).strip()
                if self.cur_row is not None: self.cur_row.append(txt)
                self.in_cell = False
            elif tag == "tr" and self.cur_row is not None:
                self.cur_table.append(self.cur_row); self.cur_row = None
            elif tag == "table" and self.in_table:
                self.sections[self.cur_section].append(self.cur_table)
                self.in_table = False; self.cur_table = None
        if tag == "section" and self.depth == self.section_depth: self.cur_section = None
        if tag == "h1": self._in_h1 = False
        self.depth -= 1
    def handle_data(self, data):
        if self.in_cell: self.cur_cell.append(data)
        if self._in_h1 and self.h1 is None:
            t = data.strip()
            if t: self.h1 = t

def fetch_page(t):
    cache = os.path.join(HTML, f"{t}.html")
    meta = os.path.join(HTML, f"{t}.meta")
    if os.path.exists(cache):
        kind = "consolidated"
        if os.path.exists(meta):
            kind = open(meta).read().strip()
        return open(cache, encoding="utf-8", errors="replace").read(), kind
    safe = t.replace("&", "%26")
    kind = "consolidated"
    r = requests.get(f"https://www.screener.in/company/{safe}/consolidated/",
                     headers=HEADERS, timeout=30)
    if r.status_code != 200 or "profit-loss" not in r.text:
        r = requests.get(f"https://www.screener.in/company/{safe}/", headers=HEADERS, timeout=30)
        kind = "standalone"
    if "/consolidated" not in r.url:
        kind = "standalone"
    if r.status_code != 200:
        return None, None
    with open(cache, "w", encoding="utf-8") as f: f.write(r.text)
    with open(meta, "w") as f: f.write(kind)
    time.sleep(2.5)
    return r.text, kind

def row_lookup(table, *names):
    """Find a row whose first cell starts with one of names; return list of cells (after first)."""
    for row in table:
        if not row: continue
        head = row[0].replace(" "," ").strip()
        for n in names:
            if head.startswith(n):
                return row[1:]
    return None

def grab(table, years, *names):
    if table is None: return {y: "NA" for y in years}
    hdr = [h.replace(" ","") for h in table[0][1:]] if table and table[0] else []
    vals = row_lookup(table, *names)
    out = {}
    for y in years:
        out[y] = "NA"
        key = y  # e.g. Mar2022
        for i,h in enumerate(hdr):
            if h == key and vals is not None and i < len(vals):
                out[y] = vals[i] if vals[i] != "" else "NA"
    return out

YEARS = ["Mar2022","Mar2023","Mar2024","Mar2025","Mar2026"]

def process(t):
    outp = os.path.join(HERE, f"data_{t.replace('&','_')}.txt")
    try:
        text, kind = fetch_page(t)
    except Exception as e:
        print(t, "FETCH_ERROR", e); return False
    if text is None:
        print(t, "NO_PAGE"); return False
    p = SectionTableParser(); p.feed(text)
    pl = (p.sections.get("profit-loss") or [None])[0]
    bs = (p.sections.get("balance-sheet") or [None])[0]
    q  = (p.sections.get("quarters") or [None])[0]
    if pl is None or bs is None:
        print(t, "MISSING_SECTIONS"); return False
    sales = grab(pl, YEARS, "Sales", "Revenue")
    intr  = grab(pl, YEARS, "Interest")
    dep   = grab(pl, YEARS, "Depreciation")
    op    = grab(pl, YEARS, "Operating Profit", "Financing Profit")
    bor   = grab(bs, YEARS, "Borrowings")
    fa    = grab(bs, YEARS, "Fixed Assets")
    cwip  = grab(bs, YEARS, "CWIP")
    ta    = grab(bs, YEARS, "Total Assets", "Total Liabilities")
    # validity: a real page has at least some numbers; else nuke cache and fail
    if all(sales[y] == "NA" for y in YEARS):
        for ext in (".html", ".meta"):
            fp = os.path.join(HTML, f"{t}{ext}")
            if os.path.exists(fp): os.remove(fp)
        print(t, "EMPTY_TABLES_cache_purged"); return False
    lines = [f"KIND|{t}|{p.h1 or 'NA'}|{kind}"]
    for y in YEARS:
        lines.append(f"ANNUAL|{t}|{y}|{bor[y]}|{intr[y]}|{fa[y]}|{cwip[y]}|{dep[y]}|{sales[y]}|{op[y]}|{ta[y]}")
    if q:
        qhdr = [h.replace(" ","") for h in q[0][1:]]
        qs = row_lookup(q, "Sales", "Revenue"); qi = row_lookup(q, "Interest")
        for i,h in enumerate(qhdr):
            m = re.match(r"(Mar|Jun|Sep|Dec)(20(2[3-9]))", h)
            if not m: continue
            s = qs[i] if qs and i < len(qs) and qs[i] != "" else "NA"
            v = qi[i] if qi and i < len(qi) and qi[i] != "" else "NA"
            lines.append(f"QTR|{t}|{h}|{s}|{v}")
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(t, "OK", kind, f"quarters={sum(1 for l in lines if l.startswith('QTR'))}")
    return True

ok = fail = 0
for t in TICKERS:
    if process(t): ok += 1
    else: fail += 1
print(f"DONE ok={ok} fail={fail}")
