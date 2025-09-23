"""Reverse engineering Rate My Team projections using provided feature tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TableSpec:
    raw: str
    columns: list[str]

    def to_frame(self, *, index_col: str | None = None) -> pd.DataFrame:
        records: list[list[str]] = []
        for line in self.raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if not parts:
                continue
            records.append(parts)
        df = pd.DataFrame(records, columns=self.columns)
        if index_col:
            df = df.set_index(index_col)
        return df


EXPECTED = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t0\t0.00\t+0.00\t1.00\t0.84\t+0.16\t1.00\t0.84\t+0.16\t1.00\t0.84\t+0.16\t1.00\t0.84\t+0.16\t0.42\t0.84
Bruno Fernandes\tMUN\t9.0\t0\t0.08\t-0.09\t1.07\t0.82\t+0.19\t1.07\t0.96\t+0.11\t1.07\t0.88\t+0.19\t1.07\t0.96\t+0.11\t0.27\t0.82
Minteh\tBHA\t5.9\t0\t0.00\t+0.00\t1.20\t0.74\t+0.31\t1.20\t0.89\t+0.31\t1.20\t0.89\t+0.31\t1.20\t0.89\t+0.31\t0.74\t0.74
Ladislav Krejci II\tWOL\t4.5\t0\t0.04\t-0.04\t1.00\t0.68\t+0.32\t1.00\t0.72\t+0.28\t1.00\t0.68\t+0.32\t1.00\t0.72\t+0.28\t0.68\t0.68
Isidor\tSUN\t5.5\t0\t0.00\t+0.00\t1.12\t0.62\t+0.43\t1.12\t0.70\t+0.43\t1.12\t0.70\t+0.43\t1.12\t0.70\t+0.43\t0.10\t0.62
Haaland\tMCI\t14.2\t0\t0.01\t-0.01\t1.18\t0.52\t+0.57\t1.18\t0.63\t+0.56\t1.18\t0.62\t+0.57\t1.18\t0.63\t+0.56\t0.26\t0.52
Wood\tNFO\t7.6\t0\t0.00\t+0.00\t0.00\t0.41\t-0.50\t0.00\t0.50\t-0.50\t0.00\t0.50\t-0.50\t0.00\t0.50\t-0.50\t0.21\t0.00
Munetsi\tWOL\t5.5\t0\t0.00\t+0.00\t0.00\t0.38\t-0.76\t0.00\t0.76\t-0.76\t0.00\t0.76\t-0.76\t0.00\t0.76\t-0.76\t0.19\t0.00
Amad Diallo\tMUN\t6.3\t0\t0.05\t-0.05\t0.00\t0.38\t-0.38\t0.00\t0.43\t-0.43\t0.00\t0.38\t-0.38\t0.00\t0.43\t-0.43\t0.19\t0.00
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "assists_actual",
        "assists_x",
        "assists_delta",
        "goals_actual",
        "goals_x",
        "goals_delta",
        "gi_actual",
        "gi_x",
        "gi_delta",
        "npg_actual",
        "npg_x",
        "npg_delta",
        "npi_actual",
        "npi_x",
        "npi_delta",
        "shot_xg_per",
        "goal_xg_per",
    ],
)

INVOLVEMENT = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t5\t436\t25.80\t16.51\t10.11\t3.5\t12.18\t8.88\t4.75\t7.4\t0.83\t0.21\t25\t2.48\t7\t7
Bruno Fernandes\tMUN\t9.0\t5\t444\t81.89\t45.00\t26.76\t1.1\t53.11\t29.59\t17.03\t1.7\t1.22\t0.81\t66.7\t1.22\t7\t7
Minteh\tBHA\t5.9\t5\t443\t49.60\t36.80\t26.40\t1.8\t24.80\t19.00\t12.60\t3.6\t4.40\t2.20\t50\t4.20\t17\t17
Ladislav Krejci II\tWOL\t4.5\t5\t366\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0
Isidor\tSUN\t5.5\t5\t436\t25.80\t16.51\t10.11\t3.9\t12.18\t8.88\t4.75\t7.4\t0.83\t0.21\t25\t2.48\t7\t7
Haaland\tMCI\t14.2\t5\t415\t24.72\t16.27\t10.84\t3.6\t14.96\t10.63\t6.72\t6.0\t1.08\t0.65\t60\t1.08\t16\t16
Wood\tNFO\t7.6\t5\t365\t19.73\t15.04\t8.88\t4.6\t14.55\t12.33\t6.66\t6.2\t0.25\t0.25\t100\t0.25\t10\t10
Munetsi\tWOL\t5.5\t4\t360\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0
Amad Diallo\tMUN\t6.3\t5\t236\t39.33\t29.12\t21.55\t2.3\t21.18\t16.64\t12.86\t4.3\t3.40\t1.89\t55.6\t3.03\t0\t0
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "App",
        "Mins",
        "Touches_total",
        "Touches_opp_half",
        "Touches_final_third",
        "Mins_per_touch",
        "Passes_total",
        "Passes_opp_half",
        "Passes_final_third",
        "Mins_per_pass_received",
        "Take_ons_total",
        "Take_ons_success",
        "Take_on_pct",
        "Tackled",
        "GI_pct",
        "FGI_pct",
    ],
)

SET_PIECES = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t5\t436\t0.21\t0.21\t0.41\t0.41\t0\t0\t0\t0
Bruno Fernandes\tMUN\t9.0\t5\t444\t0.20\t0\t0.81\t0\t2.23\t0.41\t1.42\t0.20
Minteh\tBHA\t5.9\t5\t450\t0.20\t0\t0.20\t0\t0\t0\t0\t0
Ladislav Krejci II\tWOL\t4.5\t5\t366\t0\t0\t0\t0\t0\t0\t0\t0
Isidor\tSUN\t5.5\t5\t436\t0.21\t0.21\t0.41\t0.41\t0\t0\t0\t0
Haaland\tMCI\t14.2\t5\t415\t0\t0\t1.08\t1.08\t0\t0\t0\t0
Wood\tNFO\t7.6\t5\t365\t0\t0.25\t0.25\t0\t0\t0\t0\t0
Munetsi\tWOL\t5.5\t4\t360\t0\t0\t0\t0\t0\t0\t0\t0
Amad Diallo\tMUN\t6.3\t5\t236\t0.05\t0\t0\t0\t0\t0\t0\t0
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "App",
        "Mins",
        "Pen",
        "SP_goals_total",
        "SP_att",
        "SP_heads",
        "Corners_taken",
        "Corners_success",
        "FK_cross",
        "FK_success",
    ],
)

BPS_PLUS = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t441\t24\t0\t0\t12\t0.62\t6.19\t0\t0.62\t1.24\t0\t2\t0\t0\t0\t0.21\t0.83\t14\t20.44\t-6.40\t14.04\t5.99\t15
Bruno Fernandes\tMUN\t9.0\t446\t18\t0\t0\t12\t0.61\t6.08\t3.65\t2.03\t0.81\t0\t14\t0.61\t3.04\t0.81\t0.81\t1.82\t8\t30.81\t-5.27\t25.54\t20.07\t4.5
Minteh\tBHA\t5.9\t450\t0\t0\t0\t0\t0\t6.00\t1.20\t1.40\t0.20\t0\t14\t0\t1.60\t0\t2.20\t0.60\t10\t31.80\t-10.80\t21.00\t7.20\t12.5
Ladislav Krejci II\tWOL\t4.5\t366\t0\t0\t0\t0\t0\t6.10\t0\t0\t0\t0\t0\t0\t0\t0\t0.68\t0.68\t10\t18.30\t-5.10\t13.20\t13.20\t6.8
Isidor\tSUN\t5.5\t436\t24\t0\t12\t0.62\t6.19\t0\t0.62\t1.24\t0\t2\t0\t0\t0\t0.21\t0.83\t14\t20.44\t-6.40\t14.04\t5.99\t15
Haaland\tMCI\t14.2\t413\t144\t0\t0\t0\t0.65\t6.51\t0\t0.43\t0.87\t0\t0\t1.30\t0.43\t0\t0.65\t0.43\t20\t46.84\t-8.02\t38.82\t6.94\t13
Wood\tNFO\t7.6\t370\t48\t0\t0\t0\t0\t6.66\t0\t0.00\t0.25\t0\t0\t0\t0.74\t0\t0.25\t0.74\t10\t22.93\t-4.19\t18.74\t6.90\t13
Munetsi\tWOL\t5.5\t360\t0\t0\t0\t0\t0\t6.00\t0\t0\t0\t0\t0\t0\t0\t0\t0.76\t0.76\t6\t15.36\t-4.56\t10.80\t10.80\t8.0
Amad Diallo\tMUN\t6.3\t236\t0\t0\t0\t0\t0\t6.81\t0\t0.38\t0\t0\t4\t0\t1.13\t0.76\t1.89\t2.27\t10\t18.53\t-5.29\t13.24\t13.24\t6.8
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "Mins",
        "G",
        "A",
        "CS",
        "P",
        "WG",
        "Mins_component",
        "PC",
        "Rec",
        "CBI",
        "GLC",
        "Tkl",
        "BCC",
        "KP",
        "Cr",
        "Dri",
        "FW",
        "On",
        "BPS_plus",
        "BPS_minus",
        "BPS",
        "Base_BPS",
        "Mins_per_base_BPS",
    ],
)

BPS_MINUS = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t441\t0\t0\t0\t0\t-0.21\t-0.41\t-1.24\t-2.48\t0\t0\t-0.83\t0\t-1.24\t20.44\t-6.40\t14.04\t5.99\t15
Bruno Fernandes\tMUN\t9.0\t446\t0\t0\t-1.22\t0\t0\t-0.61\t-0.61\t-1.22\t0\t0\t-1.01\t0\t-0.61\t30.81\t-5.27\t25.54\t20.07\t4.5
Minteh\tBHA\t5.9\t450\t0\t0\t0\t0\t-0.60\t-1.20\t-2.40\t-4.20\t0\t0\t-1.80\t0\t-0.60\t31.80\t-10.80\t21.00\t7.20\t12.5
Ladislav Krejci II\tWOL\t4.5\t366\t0\t0\t0\t0\t0\t-0.72\t-0.72\t-1.44\t0\t0\t-0.72\t0\t0\t18.30\t-5.10\t13.20\t13.20\t6.8
Isidor\tSUN\t5.5\t436\t0\t0\t0\t0\t-0.21\t-0.41\t-1.24\t-2.48\t0\t0\t-0.83\t0\t-1.24\t20.44\t-6.40\t14.04\t5.99\t15
Haaland\tMCI\t14.2\t413\t0\t0\t0\t0\t0\t-2.39\t-3.25\t-1.08\t0\t0\t-1.30\t0\t0\t46.84\t-8.02\t38.82\t6.94\t13
Wood\tNFO\t7.6\t370\t0\t0\t0\t0\t-0.49\t-1.23\t-1.48\t-0.25\t0\t0\t-0.74\t0\t0\t22.93\t-4.19\t18.74\t6.90\t13
Munetsi\tWOL\t5.5\t360\t0\t0\t0\t0\t0\t-0.76\t-0.76\t-1.52\t0\t0\t-0.76\t0\t0\t15.36\t-4.56\t10.80\t10.80\t8.0
Amad Diallo\tMUN\t6.3\t236\t0\t0\t0\t0\t0\t-0.76\t-1.13\t-3.03\t0\t0\t-0.38\t0\t0\t18.53\t-5.29\t13.24\t13.24\t6.8
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "Mins",
        "Off",
        "BCM",
        "Tkld",
        "ELC",
        "ELG",
        "FC",
        "PC",
        "YC",
        "BPS_plus",
        "BPS_minus",
        "BPS",
        "Base_BPS",
        "Mins_per_base_BPS",
    ],
)

KPI_ATTACK = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t436\t21\t2\t31\t5.4\t218.0\t43.6\t62.3\t62.3\t43.6\t0\t436\t1282.4\t163.3\t144.9
Bruno Fernandes\tMUN\t9.0\t444\t34\t45\t19\t9.8\t222.0\t37.0\t88.8\t111.0\t44.4\t30.0\t17.8\t448.5\t161.5\t118.7
Minteh\tBHA\t5.9\t450\t34\t20\t28\t8.2\t150.0\t34.6\t37.5\t90.0\t17.3\t56.0\t112.5\t1666.7\t164.8\t150.0
Ladislav Krejci II\tWOL\t4.5\t366\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0
Isidor\tSUN\t5.5\t436\t21\t2\t31\t5.4\t218.0\t43.6\t62.3\t62.3\t43.6\t0\t436\t1282.4\t163.3\t144.9
Haaland\tMCI\t14.2\t415\t44\t7\t62\t11.4\t69.2\t19.8\t20.8\t41.5\t14.3\t208.0\t0\t883.0\t66.3\t61.7
Wood\tNFO\t7.6\t365\t17\t11\t33\t6.2\t182.5\t36.5\t36.5\t73.0\t28.1\t122.0\t0\t3318.2\t183.4\t173.8
Munetsi\tWOL\t5.5\t360\t14\t18\t31\t6.3\t0\t29.8\t47.6\t47.6\t26.4\t79.0\t59.5\t952.0\t290.2\t222.4
Amad Diallo\tMUN\t6.3\t236\t39\t29\t22\t6.8\t0\t36.3\t25.9\t15.1\t2.5\t3.21\t0.86\t2000.0\t280.0\t245.6
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "Mins",
        "I",
        "C",
        "T",
        "II",
        "Mins_per_goal",
        "Mins_per_shot",
        "Mins_per_involvement",
        "Mins_per_shot_on",
        "Mins_per_touch",
        "Mins_per_CC",
        "Mins_per_cross",
        "Mins_per_xA",
        "Mins_per_xG",
        "Mins_per_xGI",
    ],
)

KPI_DEF = TableSpec(
    raw="""
Mateta\tCRY\t7.6\t5\t5\t436\t14.1\t0\t33.5\t109.0\t33.5\t109.0\t436.0
Bruno Fernandes\tMUN\t9.0\t5\t5\t444\t8.1\t444.0\t55.5\t0\t13.5\t21.1\t34.2
Minteh\tBHA\t5.9\t5\t5\t450\t10.7\t0\t112.5\t450.0\t16.1\t32.1\t50.0
Ladislav Krejci II\tWOL\t4.5\t5\t5\t366\t26.5\t0\t79.6\t0\t44.2\t199.0\t366.0
Isidor\tSUN\t5.5\t5\t5\t436\t14.1\t0\t33.5\t109.0\t33.5\t109.0\t436.0
Haaland\tMCI\t14.2\t5\t5\t415\t20.8\t0\t46.1\t0\t41.5\t207.5\t415.0
Wood\tNFO\t7.6\t5\t5\t365\t45.6\t0\t121.7\t0\t73.0\t121.7\t0.0
Munetsi\tWOL\t5.5\t4\t4\t360\t19.8\t360.0\t0\t360.0\t29.8\t59.5\t119.0
Amad Diallo\tMUN\t6.3\t5\t3\t238\t19.8\t238.0\t0\t238.0\t29.8\t59.5\t119.0
""",
    columns=[
        "Name",
        "Team",
        "Cost",
        "App",
        "Starts",
        "Mins",
        "Def_contrib",
        "Blocks",
        "Clearances",
        "Interceptions",
        "Recoveries",
        "Tackles",
        "Tackles_won",
    ],
)

POINTS = TableSpec(
    raw="""
Mateta\tCRY\tFWD\t7.6\t3.1\t3.1\t3.7\t2.7\t4.1\t3.8\t20.6\t2.7
Bruno Fernandes\tMUN\tMID\t9.0\t5.0\t6.0\t4.2\t5.2\t4.8\t4.4\t30.1\t3.3
Minteh\tBHA\tMID\t5.9\t3.5\t4.3\t4.2\t3.8\t4.7\t3.9\t24.9\t4.2
Ladislav Krejci II\tWOL\tDEF\t4.5\t1.5\t1.8\t2.1\t2.4\t1.7\t1.4\t11.9\t2.6
Isidor\tSUN\tFWD\t5.5\t2.4\t2.2\t2.4\t1.7\t2.0\t1.7\t13.8\t2.5
Haaland\tMCI\tFWD\t14.2\t7.6\t5.8\t5.8\t5.2\t5.9\t5.3\t34.6\t2.4
Wood\tNFO\tFWD\t7.6\t4.0\t2.9\t3.1\t3.1\t3.5\t3.8\t20.2\t2.7
Munetsi\tWOL\tMID\t5.5\t1.4\t1.6\t1.7\t2.0\t1.6\t1.5\t10.2\t1.8
Amad Diallo\tMUN\tMID\t6.3\t3.6\t4.3\t3.0\t3.8\t3.5\t3.2\t21.8\t3.5
""",
    columns=[
        "Name",
        "Team",
        "Pos",
        "Price",
        "GW6",
        "GW7",
        "GW8",
        "GW9",
        "GW10",
        "GW11",
        "GW6_11_Pts",
        "GW6_11_Value",
    ],
)


NUMERIC_EXCLUDE = {"Name", "Team", "Pos"}


def make_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in NUMERIC_EXCLUDE:
            continue
        series = df[col].astype(str).str.replace("+", "", regex=False)
        series = series.where(series != "-", pd.NA)
        df[col] = pd.to_numeric(series, errors="coerce")
    return df


def load_tables() -> Dict[str, pd.DataFrame]:
    return {
        "expected": make_numeric(EXPECTED.to_frame(index_col="Name")),
        "involvement": make_numeric(INVOLVEMENT.to_frame(index_col="Name")),
        "set_pieces": make_numeric(SET_PIECES.to_frame(index_col="Name")),
        "bps_plus": make_numeric(BPS_PLUS.to_frame(index_col="Name")),
        "kpi_attack": make_numeric(KPI_ATTACK.to_frame(index_col="Name")),
        "kpi_def": make_numeric(KPI_DEF.to_frame(index_col="Name")),
        "points": make_numeric(POINTS.to_frame(index_col="Name")),
    }


def derive_dataset(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = tables["points"].copy()
    for key in ("expected", "involvement", "set_pieces", "bps_plus"):
        df = df.join(tables[key], how="inner", rsuffix=f"_{key}")
    df = df.loc[:, ~df.columns.duplicated()]
    df["avg_minutes"] = df["Mins"] / df["App"]
    df["minutes_share"] = (df["avg_minutes"] / 90).clip(upper=1.0)
    goal_points = df["Pos"].map({"FWD": 4, "MID": 5, "DEF": 6, "GK": 6})
    df["xG_pts"] = df["goals_x"] * goal_points
    df["xA_pts"] = df["assists_x"] * 3
    df["appearance_pts"] = df["minutes_share"] * 2
    df["base_pts"] = df["xG_pts"] + df["xA_pts"] + df["appearance_pts"]
    df["residual_GW6"] = df["GW6"] - df["base_pts"]
    df["BPS_net"] = df["BPS"]
    return df


def fit_bonus_factor(df: pd.DataFrame) -> float:
    mask = df["BPS_net"].notna()
    x = df.loc[mask, "BPS_net"].values
    y = df.loc[mask, "residual_GW6"].values
    if x.size < 2:
        return 0.0
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def main() -> None:
    tables = load_tables()
    dataset = derive_dataset(tables)
    bonus_factor = fit_bonus_factor(dataset)
    dataset["bonus_component"] = dataset["BPS_net"].fillna(0) * bonus_factor
    dataset["reconstructed_GW6"] = dataset["base_pts"] + dataset["bonus_component"]

    # Fit a simple linear model to back out per-event weights directly from the sample
    features = dataset[["minutes_share", "goals_x", "assists_x"]].fillna(0)
    y = dataset["GW6"].values
    X = np.column_stack([np.ones(len(features)), features.values])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    dataset["regression_estimate"] = X @ coef
    dataset["reg_minutes"] = features["minutes_share"] * coef[1]
    dataset["reg_goals"] = features["goals_x"] * coef[2]
    dataset["reg_assists"] = features["assists_x"] * coef[3]

    summary_cols = [
        "Team",
        "Pos",
        "GW6",
        "base_pts",
        "bonus_component",
        "reconstructed_GW6",
        "residual_GW6",
        "BPS_net",
        "regression_estimate",
    ]
    summary = dataset[summary_cols].sort_values("GW6", ascending=False)
    out_path = Path("ml/analysis/rmt_reconstruction_summary.csv")
    summary.to_csv(out_path)
    print("Saved summary to", out_path)
    print("Regression coefficients (intercept, minutes_share, goals_x, assists_x):", coef.tolist())
    print(summary.to_string(float_format=lambda v: f"{v:0.2f}"))


if __name__ == "__main__":
    main()
