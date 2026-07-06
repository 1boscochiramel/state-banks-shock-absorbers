"""
build_panel.py — Construct the firm-year/firm-quarter panel for the state-bank
relationship paper from CMIE Prowess exports.

INPUTS (place raw Prowess exports in analysis/raw/):
  1. identity.csv       — Prowess "Identity" block: prowess_code, company_name,
                          nic_code (industry), incorporation_year, ownership_group,
                          listing status (NSE/BSE).
  2. bankers.csv        — Prowess "Bankers" field: prowess_code, year, banker_name.
                          One row per (firm, year, bank). This is the key
                          relationship variable.
  3. financials.csv     — Annual (and, where available, quarterly) financials:
                          prowess_code, year, total_assets, gross_fixed_assets,
                          net_fixed_assets, capex (cash-flow stmt "purchase of
                          fixed assets"), total_borrowings, secured_borrowings,
                          bank_borrowings, interest_expense, pbdita, sales,
                          power_and_fuel_expense, raw_material_expense,
                          import_of_raw_materials, export_earnings,
                          cash_and_bank_balance, current_ratio.
  4. psb_list.csv       — Hand-built mapping: banker_name_raw -> bank_std_name,
                          is_psb (1 = public sector bank incl. SBI + 11 other
                          PSBs post-2020 mergers; 0 = private/foreign/coop/NBFC).
                          Build this once; Prowess banker strings are messy
                          (e.g., "State Bank Of India", "S.B.I.", "SBI").

OUTPUT:
  analysis/derived/panel.parquet — one row per firm-year with:
    psb_share        : fraction of the firm's listed bankers that are PSBs
    psb_main         : 1 if a PSB is the first-listed (main) banker
    psb_rel          : 1 if psb_share (pre-shock, e.g. FY2024) > 0.5  [treatment]
    cost_of_debt     : interest_expense / avg(total_borrowings_t, t-1)
    capex_ratio      : capex / lagged total_assets
    energy_intensity : power_and_fuel_expense / sales, pre-shock average
    import_intensity : import_of_raw_materials / raw_material_expense, pre-shock
    size, leverage, tangibility, profitability, cash, age  [controls]
    exposure_hi      : 1 if pre-shock energy_intensity above sample median

Treatment definition is fixed PRE-SHOCK (FY2024 or earlier) so that the
relationship variable is not itself an outcome of the shock.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "derived"
OUT.mkdir(exist_ok=True)

# ---- knobs -----------------------------------------------------------------
PRE_SHOCK_YEAR = 2024      # last fiscal year fully unaffected by the shock
SHOCK_YEAR = 2026          # first fiscal year treated as "post" (FY2025-26)
WINSOR = (0.01, 0.99)      # winsorize ratios at 1/99 pct


def winsorize(s: pd.Series, lo=WINSOR[0], hi=WINSOR[1]) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def load_bankers() -> pd.DataFrame:
    bankers = pd.read_csv(RAW / "bankers.csv")
    psb = pd.read_csv(RAW / "psb_list.csv")
    b = bankers.merge(psb, on="banker_name_raw", how="left", validate="m:1")
    unmatched = b[b["is_psb"].isna()]["banker_name_raw"].unique()
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} banker strings unmapped -> update psb_list.csv:")
        for u in unmatched[:40]:
            print("   ", u)
    # firm-year PSB share and main-banker flag (Prowess lists main banker first)
    b["rank"] = b.groupby(["prowess_code", "year"]).cumcount()
    agg = b.groupby(["prowess_code", "year"]).agg(
        n_banks=("banker_name_raw", "size"),
        psb_share=("is_psb", "mean"),
        psb_main=("is_psb", "first"),
    ).reset_index()
    return agg


def build() -> pd.DataFrame:
    fin = pd.read_csv(RAW / "financials.csv")
    ident = pd.read_csv(RAW / "identity.csv")
    rel = load_bankers()

    df = fin.merge(ident, on="prowess_code", how="left", validate="m:1")
    df = df.merge(rel, on=["prowess_code", "year"], how="left")
    df = df.sort_values(["prowess_code", "year"])
    g = df.groupby("prowess_code")

    # ---- outcomes ----
    avg_borrow = (df["total_borrowings"] + g["total_borrowings"].shift(1)) / 2
    df["cost_of_debt"] = df["interest_expense"] / avg_borrow
    df["capex_ratio"] = df["capex"] / g["total_assets"].shift(1)

    # ---- treatment: PSB relationship, frozen pre-shock ----
    pre = df[df["year"] <= PRE_SHOCK_YEAR]
    pre_rel = pre.groupby("prowess_code")["psb_share"].mean().rename("psb_share_pre")
    df = df.merge(pre_rel, on="prowess_code", how="left")
    df["psb_rel"] = (df["psb_share_pre"] > 0.5).astype(int)
    df["post"] = (df["year"] >= SHOCK_YEAR).astype(int)

    # ---- shock exposure, frozen pre-shock ----
    df["energy_int_raw"] = df["power_and_fuel_expense"] / df["sales"]
    df["import_int_raw"] = df["import_of_raw_materials"] / df["raw_material_expense"]
    for v in ["energy_int_raw", "import_int_raw"]:
        pre_v = df[df["year"] <= PRE_SHOCK_YEAR].groupby("prowess_code")[v].mean()
        df = df.merge(pre_v.rename(v.replace("_raw", "_pre")), on="prowess_code", how="left")
    df["exposure_hi"] = (
        df["energy_int_pre"] > df["energy_int_pre"].median()
    ).astype(int)

    # ---- controls ----
    df["size"] = np.log(df["total_assets"])
    df["leverage"] = df["total_borrowings"] / df["total_assets"]
    df["tangibility"] = df["net_fixed_assets"] / df["total_assets"]
    df["profitability"] = df["pbdita"] / df["total_assets"]
    df["cash"] = df["cash_and_bank_balance"] / df["total_assets"]
    df["age"] = np.log(1 + df["year"] - df["incorporation_year"])
    df["nic2"] = df["nic_code"].astype(str).str[:2]

    # ---- cleaning ----
    for v in ["cost_of_debt", "capex_ratio", "leverage", "profitability",
              "tangibility", "cash", "energy_int_pre", "import_int_pre"]:
        df[v] = winsorize(df[v])
    df = df[(df["total_assets"] > 0) & df["psb_share_pre"].notna()]
    # drop financials (NIC 64-66) and firms with < 3 obs
    df = df[~df["nic2"].isin(["64", "65", "66"])]
    df = df[df.groupby("prowess_code")["year"].transform("size") >= 3]

    df.to_parquet(OUT / "panel.parquet")
    print(f"panel: {df['prowess_code'].nunique()} firms, {len(df)} firm-years")
    return df


if __name__ == "__main__":
    build()
