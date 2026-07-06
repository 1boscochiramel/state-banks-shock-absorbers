"""
run_regressions.py — Estimate the paper's main specifications and emit LaTeX
tables into ../paper/tables/ that drop straight into main.tex.

Requires: pip install pyfixest pandas pyarrow   (pyfixest gives reghdfe-style
high-dimensional FE + clustered SEs; syntax below follows pyfixest >= 0.24)

SPECIFICATIONS (mirror the DiD in the text):

  T2  Baseline DiD, cost of debt:
      cod_it = b1 PSB_i x Post_t + G'X_it + a_i + d_{industry x year} + e_it
  T3  Baseline DiD, capex:
      capex_it = b1 PSB_i x Post_t + G'X_it + a_i + d_{industry x year} + e_it
  T4  Triple-diff with energy exposure:
      + b2 PSB_i x Post_t x HiExposure_i (+ lower-order terms not absorbed)
  T5  Dynamic event-study coefficients PSB_i x year dummies (pre-trends)
  T6  Heterogeneity: size terciles, financial constraints (SA index), group
      affiliation
  T7  Misallocation check: interest coverage < 1 subsample ("zombie" margin)

SEs clustered by firm; robustness: cluster by main bank, two-way firm+year.
"""

import pandas as pd
import pyfixest as pf
from pathlib import Path

HERE = Path(__file__).parent
TABLES = HERE.parent / "paper" / "tables"
TABLES.mkdir(exist_ok=True)

CONTROLS = "size + leverage + tangibility + profitability + cash + age"


def main():
    df = pd.read_parquet(HERE / "derived" / "panel.parquet")
    df["psb_post"] = df["psb_rel"] * df["post"]
    df["psb_post_hi"] = df["psb_post"] * df["exposure_hi"]
    df["post_hi"] = df["post"] * df["exposure_hi"]

    # ---- T2: cost of debt ----
    m_cod = [
        pf.feols("cost_of_debt ~ psb_post | prowess_code + year",
                 data=df, vcov={"CRV1": "prowess_code"}),
        pf.feols(f"cost_of_debt ~ psb_post + {CONTROLS} | prowess_code + year",
                 data=df, vcov={"CRV1": "prowess_code"}),
        pf.feols(f"cost_of_debt ~ psb_post + {CONTROLS} | prowess_code + nic2^year",
                 data=df, vcov={"CRV1": "prowess_code"}),
    ]
    pf.etable(m_cod, type="tex", file_name=str(TABLES / "t2_cost_of_debt.tex"))

    # ---- T3: capex ----
    m_cap = [
        pf.feols("capex_ratio ~ psb_post | prowess_code + year",
                 data=df, vcov={"CRV1": "prowess_code"}),
        pf.feols(f"capex_ratio ~ psb_post + {CONTROLS} | prowess_code + year",
                 data=df, vcov={"CRV1": "prowess_code"}),
        pf.feols(f"capex_ratio ~ psb_post + {CONTROLS} | prowess_code + nic2^year",
                 data=df, vcov={"CRV1": "prowess_code"}),
    ]
    pf.etable(m_cap, type="tex", file_name=str(TABLES / "t3_capex.tex"))

    # ---- T4: triple diff ----
    m_ddd = [
        pf.feols(f"cost_of_debt ~ psb_post + psb_post_hi + post_hi + {CONTROLS}"
                 " | prowess_code + nic2^year", data=df,
                 vcov={"CRV1": "prowess_code"}),
        pf.feols(f"capex_ratio ~ psb_post + psb_post_hi + post_hi + {CONTROLS}"
                 " | prowess_code + nic2^year", data=df,
                 vcov={"CRV1": "prowess_code"}),
    ]
    pf.etable(m_ddd, type="tex", file_name=str(TABLES / "t4_tripledd.tex"))

    # ---- T5: event study (pre-trends) ----
    m_es = pf.feols(
        f"cost_of_debt ~ i(year, psb_rel, ref=2024) + {CONTROLS}"
        " | prowess_code + year", data=df, vcov={"CRV1": "prowess_code"})
    m_es.iplot().savefig(TABLES / "f_eventstudy_cod.pdf")
    pf.etable([m_es], type="tex", file_name=str(TABLES / "t5_eventstudy.tex"))

    # ---- T1: descriptives ----
    desc_vars = ["cost_of_debt", "capex_ratio", "size", "leverage",
                 "tangibility", "profitability", "cash", "energy_int_pre",
                 "psb_share_pre"]
    t1 = (df.groupby("psb_rel")[desc_vars].agg(["mean", "std", "count"]).T)
    t1.to_latex(TABLES / "t1_descriptives.tex", float_format="%.3f")

    print("Tables written to", TABLES)


if __name__ == "__main__":
    main()
