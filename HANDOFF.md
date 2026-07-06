# HANDOFF — project context for continuing this research

*Read this first if you are picking up this project in a fresh session. The repo gives you
the code, data, and paper; this file gives you the reasoning, decisions, state, and the
gotchas that are painful to rediscover. Everything here is self-contained — you do not need
any prior chat history.*

---

## 1. What this is

A working paper: **"State Banks as Shock Absorbers? Relationship Lending, the Cost of Debt,
and Investment during India's 2026 Energy Shock"** by Ann Mary Raphael (MSc Finance
candidate, NUS Business School) and Bosco Chiramel (independent researcher).

It extends **Lin, Srinivasan & Yamada, "The Effect of Government Bank Lending: Evidence from
the Financial Crisis in Japan"** (SSRN 2544446) from crisis-era Japan to India's 2025–26
energy shock. Question: did listed Indian firms whose pre-shock banking relationships are
with **public sector banks (PSBs)** get a cushion — in cost of debt and investment —
relative to private-bank firms, when the energy shock compressed margins?

The paper is `paper/main.pdf` (28 pages). It is a **pre-specified design + public-data pilot**,
not a finished full study — this framing is deliberate and load-bearing (see §4).

## 2. Current status (as of the last commit)

- **SSRN-ready v1.** Compiles clean, internally consistent, honestly framed.
- **Not yet uploaded** to SSRN — waiting on Ann to register an SSRN account, then upload
  with **both authors from the start** (register both; Bosco confirms co-authorship).
- Constraint the authors chose: **public data only** — no CMIE Prowess, no faculty
  data-access collaborator. Work within this unless they say otherwise.
- Companion interactive course (separate repo, on GitHub Pages):
  https://1boscochiramel.github.io/state-banks-in-a-storm/ — teaching tool, no results in it.

## 3. The actual results (state these correctly; do not overclaim)

Public-data pilot: **74 firms, 17 PSB-attached, 289 firm-years, 939 firm-quarters.**
Regenerate the exact numbers by running the scripts (§6); the canonical values are:

- **Cost of debt (H1):** quarterly DiD −0.61pp binary / −1.07pp amount-weighted share.
  Cluster-robust t≈1.9/1.7 (10%). **Under exact Fisher randomization inference** (which the
  17-treated-cluster count requires): permutation **p=0.062 (binary), p=0.047 (share)**.
  Placebo shock dates one year earlier show nothing.
- **Investment (H2):** capex DiD positive throughout (+4–7pp); cluster t≈1.5 (ns) but
  randomization **p=0.035**. Read as "supported by exact inference, less settled than H1."
- **Equity event study:** raw PSB effect insignificant at all four events; but PSB×HiExposure
  at the sanctions announcement **+2.85pp, t≈3.4** — cushion concentrated among energy-exposed
  firms. The relief-day sign-flip prediction did NOT hold (reported honestly).
- **Partial H4 (allocation):** cushion accrues to financially *stronger* PSB firms
  (PSB×Post −3.2pp; PSB×Post×Weak +5.8pp, t≈2.3) → weak firms get ~no cushion. Echoes
  LSY's "no zombie tilt / bright side." Fragile (7-vs-10 firm split; below-median-ICR is
  NOT the true ICR<1 zombie margin). Reported heavily hedged.

**Sign is robust; magnitude/significance are not.** As the treated group grew 7→12→15→17,
the cost-of-debt coefficient stayed negative but drifted −0.88→−0.61 and 5%→10%. See §4.

## 4. Key decisions & framing — DO NOT BREAK THESE

The paper's credibility rests on intellectual honesty. If you edit it, preserve:

1. **It is a design + pilot, not a finished result.** The full CMIE Prowess study is
   pre-specified (Appendix B table shells, all cells "·") but NOT estimated. Never present
   the pilot as resolving the question — it is underpowered by design.
2. **Report the FULL 17-firm sample, never a cherry-picked subsample.** We deliberately did
   NOT stop at the 12-firm sample that gave 5% significance, because selecting the sample
   that maximizes significance is exactly the specification-search the paper disavows. The
   dilution-as-you-add-firms pattern is reported openly as evidence of a modest, real,
   sector-heterogeneous effect.
3. **Three contributions** (intro + conclusion): (i) *method* — measuring amount-weighted
   bank relationships from public CRISIL rating-rationale annexures; (ii) *empirical* — the
   pilot; (iii) *design* — the pre-specified Prowess study. The method contribution stands
   even if the causal result is thin.
4. **Inference is by exact randomization test, not asymptotics** — because 17 treated
   clusters make cluster-robust SEs unreliable. This is the answer to the paper's single
   biggest technical objection. Keep it front and center.
5. **Honesty markers throughout** — placebo tests, the equity falsification test that didn't
   confirm, the H4 hedges, "indicative not decisive." These are features, not bugs.

## 5. Data & method

- **Treatment (PSB attachment):** a firm is PSB-attached if the majority of its banking
  relationships are public-sector, **frozen pre-shock** (so crisis switching can't
  contaminate). Two public sources: (a) annual-report "Corporate Information" banker lists
  (unweighted count); (b) — the sharper source — **CRISIL rating-rationale "Details of Bank
  Lenders & Facilities" annexures**, which give named banks + sanctioned amounts →
  **amount-weighted** PSB share. Amount-weighting matters (e.g. Balrampur Chini is private by
  bank-count but PSB by rupees). Registry classification takes precedence over AR count.
  `pilot_analysis.load_registry()` computes it; NBFCs/HFCs excluded from the ratio.
- **12 PSBs (post-2020 mergers)** + pre-merger constituent names are the classification list;
  **IDBI is classified PRIVATE** (RBI reclassified it in 2019). See `paper/SOURCES.md` Data
  Appendix for the full string-matching table.
- **Outcomes:** cost of debt = interest expense / avg borrowings (quarterly: interest /
  prior-FY borrowings, since quarterly balance sheets aren't public). Capex = Δ(fixed
  assets + CWIP) + depreciation, over lagged assets.
- **Exposure:** sector-level high/low indicator (firm-level power&fuel not in public
  financials — this is a real limitation; BRSR could fix it, see §8).
- **Shock:** post = FY2026Q3 (Oct 2025 Rosneft/Lukoil sanctions); Hormuz crisis Feb–Jun 2026;
  Islamabad accord 17 Jun 2026 (partial reversal → design is quarterly). Full verified
  chronology with sources in `paper/SOURCES.md`.
- **Every factual claim** in the paper maps to a public URL in `paper/SOURCES.md` — it was
  adversarially fact-checked. Consult it before changing any institutional/shock claim.

## 6. How to reproduce

```bash
pip install -r requirements.txt      # numpy, matplotlib, requests
PILOT_RAW=analysis/pilot_raw python analysis/pilot_analysis.py   # panel, DiD, H4, tables
python analysis/event_study.py                                   # equity CARs
PILOT_RAW=analysis/pilot_raw python analysis/inference_tests.py  # randomization + placebo
```
Scripts are deterministic (`seed=42`). They print estimates and write LaTeX tables into
`paper/tables/`. To refresh prices: `python analysis/fetch_prices.py` first.

**Compile the PDF:** needs **bibtex + THREE pdflatex passes** or cross-references break and
the page count is wrong (learned the hard way):
`pdflatex main; bibtex main; pdflatex main; pdflatex main; pdflatex main`.
On the original author machine pdflatex was NOT on PATH (MiKTeX at a custom location) — on a
fresh machine just use a normal TeX install.

## 7. Gotchas & lessons learned (save yourself the pain)

- **Do NOT mass-fan-out Sonnet subagents for data collection.** 13 concurrent agents →
  API connection errors (ConnectionRefused / closed mid-response); the agents also
  self-delegate (spawn their own sub-agents) and download full PDFs instead of writing
  files. Net yield from that approach was ~zero.
- **The reliable collection method is WebFetch driven in the main thread**, targeting CRISIL
  rationale URLs (found via WebSearch "<company> CRISIL rating rationale bank facilities").
  ~40% of firms are CRISIL-rated with named-bank annexures; CARE/ICRA/Ind-Ra rationales list
  facility amounts but NOT bank names (they hide lenders behind a "click here" link).
- **Financials scrape reliably and deterministically** from screener.in via
  `analysis/pilot_raw/collect_all.py` / `collect_midcap.py` (zero model quota). Yahoo v8
  chart API gives free daily prices (`<TICKER>.NS`, `^NSEI`).
- **Adding more treated firms past ~15 DILUTES the headline** (distressed/atypical PSB firms
  like Dilip Buildcon, Gokul Agro don't show the cushion). Do not chase N to boost
  significance — it backfires and risks cherry-picking. The honest ceiling is ~here.
- Prices window in `fetch_prices.py`: the epoch timestamps must cover 2025-03 to 2026-07
  (an early bug fetched the wrong year — all-pre-shock — and gave n=0 event-study firms).

## 8. What's done / not done / next steps

**Done:** verified shock chronology; screener financials (142 firms scraped, 74 usable);
CRISIL registry (20 firms classified, in `registry_*.txt`); pilot estimation; exact
inference; equity event study; partial H4; paper written + fact-checked + restructured;
replication repo.

**NOT done / open:**
- **Route 1 (the real unlock, declined by authors): CMIE Prowess** — firm-level exposure,
  the true ICR<1 zombie margin, thousands of firms. The pipeline is pre-written
  (`analysis/build_panel.py`, `run_regressions.py`) and runs the moment Prowess data exists.
  This is what turns the pilot into a real paper. Gated on data access (NUS subscription
  unconfirmed — Ann can check via NUS Libraries / askalib@nus.edu.sg).
- **Optional public upgrade: BRSR firm-level energy intensity** (mandatory disclosure for
  top-1000 listed firms) → would upgrade exposure from sector- to firm-level, sharpening the
  triple-difference and event study. Not done: agent-collection is unreliable and coverage is
  partial for mid-caps; marginal at N=17. Attempt via main-thread WebFetch if wanted.
- **v2 (~August 2026): the reversal quarter.** Q1 FY2027 results (the post-accord quarter)
  publish late July–Aug 2026. Re-running `collect_*.py` then adds the built-in falsification
  test the paper already advertises (cushion should fade as the shock resolves) — essentially
  free. Good reason to follow up with Prof. Srinivasan a month after the first email.
- **SSRN upload** pending Ann's account; both authors from start.

**Do NOT fabricate** any Appendix B (full-study) table entries — they must come from a real
Prowess run. Do not overstate the pilot's significance.

## 9. File map

```
paper/main.pdf, main.tex        the paper (28pp)
paper/references.bib            bibliography (all 20 refs verified)
paper/SOURCES.md               claim-by-claim public-source register + fact-check log
paper/tables/                  auto-generated pilot tables + event-study figure
analysis/pilot_analysis.py     panel build, DiD/triple-diff/H4, writes pilot tables
analysis/event_study.py        market-model CARs around the 4 dated events
analysis/inference_tests.py    exact Fisher randomization + placebo shock dates
analysis/fetch_prices.py       Yahoo daily prices -> derived/prices.json
analysis/build_panel.py        FULL-STUDY pipeline (Prowess) — pre-specified, not yet run
analysis/run_regressions.py    FULL-STUDY specs — pre-specified
analysis/pilot_raw/
  data_<TICKER>.txt            financials transcribed from public exchange filings (screener)
  bankers0*.txt                bankers from annual-report Corporate Information pages
  registry_*.txt               amount-weighted bank facilities from CRISIL rationales
  collect_all.py, collect_midcap.py   the screener scrapers
analysis/derived/prices.json   daily-price snapshot (regenerable)
README.md                      how to verify + data provenance/licensing
```

## 10. If you (a new session) are asked to extend it

- **More treated firms:** WebSearch "<company> CRISIL rating rationale bank facilities" →
  WebFetch the rationale HTML → transcribe the "Details of Bank Lenders & Facilities" annexure
  into `registry_midcap.txt` (pipe format: `SOURCE|TICKER|agency|date|url` then
  `FACILITY|TICKER|bank|type|amount`) → add the ticker's sector to the SECTOR map in
  `pilot_analysis.py` and to the HI set in `event_study.py` → re-run. Remember §7: past ~15
  treated firms this dilutes, so weigh whether it helps.
- **Any empirical change** → re-run all three scripts, recompile (bibtex + 3 passes),
  re-verify prose numbers match tables, and keep the README results prose in step.
- **Preserve the honest framing (§4) above all.** The paper's value is its rigor and honesty,
  not a splashy result.
