<!-- Continuing this project (human or AI)? Read HANDOFF.md first — it has the full
     project context, decisions, results, gotchas, and next steps, self-contained. -->

# State Banks as Shock Absorbers? — replication package

**Relationship lending, the cost of debt, and investment during India's 2026 energy shock.**

Ann Mary Raphael (NUS Business School) · Bosco Chiramel · working paper, 2026.

This repository is the full, public replication package for the paper. Everything in it
— every financial figure, every bank relationship, every price — comes from public
disclosures, and every number in the paper's pilot results can be regenerated from the
code and data here. The paper PDF is in [`paper/main.pdf`](paper/main.pdf).

---

## What the paper does (one paragraph)

We extend the government-bank-lending framework of Lin, Srinivasan & Yamada
(*The Effect of Government Bank Lending: Evidence from the Financial Crisis in Japan*,
SSRN 2544446) from crisis-era Japan to India's 2025–26 energy shock (the October 2025
Rosneft/Lukoil sanctions and the February–June 2026 Strait of Hormuz crisis). We ask whether
listed Indian firms whose pre-shock banking relationships are with public sector banks (PSBs)
were cushioned — in their cost of debt and their investment — relative to observably similar
private-bank firms. The paper contributes (1) a **method** for measuring amount-weighted
corporate–bank relationships from public CRISIL rating-rationale annexures; (2) a **public-data
pilot** on 74 firms (17 PSB-attached) with a modest cost-of-debt cushion significant under exact
randomization inference (permutation *p* = 0.047 for the amount-weighted treatment), a positive
investment cushion (*p* = 0.035), a significant equity-market cushion among energy-exposed firms,
and a partial allocation test suggesting the cushion reaches financially *stronger* firms; and
(3) a **pre-specified design** for the full CMIE Prowess study.

The pilot is deliberately underpowered relative to its question (17 treated firms, sector-level
exposure) and the paper says so throughout — it is a design + public-data pilot, not a settled
empirical finding.

---

## How to verify the results

```bash
pip install -r requirements.txt        # numpy, matplotlib, requests

# 1. Main pilot estimates + tables + event-study figure  (uses committed data)
PILOT_RAW=analysis/pilot_raw python analysis/pilot_analysis.py

# 2. Equity event study around the four dated shock events
python analysis/event_study.py

# 3. Exact randomization inference + placebo shock-date tests
PILOT_RAW=analysis/pilot_raw python analysis/inference_tests.py
```

Each script prints its estimates to the console and writes LaTeX tables into `paper/tables/`.
The scripts are deterministic (randomization inference uses `seed=42`). To re-fetch the daily
prices from scratch instead of using the committed snapshot, run
`python analysis/fetch_prices.py` first.

To rebuild the PDF: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

---

## Layout

```
paper/
  main.tex, references.bib, main.pdf   the paper
  SOURCES.md          claim-by-claim source register (every fact -> public URL)
  tables/             auto-generated result tables + event-study figure
analysis/
  pilot_analysis.py   builds the firm panel, runs DiD / triple-diff / H4, writes tables
  event_study.py      market-model CARs around the 4 dated events
  inference_tests.py  exact Fisher randomization + placebo shock dates
  fetch_prices.py     pulls daily prices (Yahoo) -> derived/prices.json
  build_panel.py      full-study pipeline for CMIE Prowess (pre-specified, not yet run)
  run_regressions.py  full-study specifications (pre-specified)
  pilot_raw/
    data_<TICKER>.txt   firm financials transcribed from public exchange filings
    bankers0*.txt       bankers from annual-report "Corporate Information" pages
    registry_*.txt      amount-weighted bank facilities from CRISIL rating rationales
    collect_all.py, collect_midcap.py   the screener.in scrapers
  derived/
    prices.json         daily-price snapshot (regenerable via fetch_prices.py)
```

---

## Data provenance and licensing

All data is derived from **public** sources, documented claim-by-claim in
[`paper/SOURCES.md`](paper/SOURCES.md):

- **Firm financials** (`data_*.txt`): transcribed from screener.in, which reproduces
  NSE/BSE exchange-filed statements. These are numeric figures from statutory filings.
- **Bank relationships** (`bankers*.txt`, `registry_*.txt`): transcribed verbatim from
  company annual reports and from CRISIL credit-rating-rationale "Details of Bank Lenders &
  Facilities" annexures — both public documents; source URLs are recorded in each file.
- **Daily prices** (`derived/prices.json`): Yahoo Finance, used under fair-use for
  non-commercial academic replication; regenerable and not redistributed as a product.

The **code** is released under the MIT License (see `LICENSE`). The transcribed data is
provided for academic replication only; underlying figures remain the property of the
original filers/exchanges. This is a screening-grade pilot, **not** investment advice.

## Companion

An interactive course explaining the identification design:
https://1boscochiramel.github.io/state-banks-in-a-storm/
