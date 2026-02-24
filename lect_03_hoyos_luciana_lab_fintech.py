# =============================================================
# LAB FINTECH (SINTÉTICO 2025)
# Pipeline modularizado de preprocesamiento y EDA
# =============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import plotly.express as px
from plotly.io import write_html
import umap.umap_ as umap

warnings.filterwarnings("ignore")

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
DATA_CSV = "fintech_top_sintetico_2025.csv"
DATA_DICT = "fintech_top_sintetico_dictionary.json"
OUTPUT_DIR = Path("./data_output_finanzas_sintetico")
SPLIT_DATE = "2025-09-01"

DATE_COL = "Month"
ID_COLS = ["Company"]
CAT_COLS = ["Country", "Region", "Segment", "Subsegment", "IsPublic", "Ticker"]
NUM_COLS = [
    "Users_M","NewUsers_K","TPV_USD_B","TakeRate_pct","Revenue_USD_M",
    "ARPU_USD","Churn_pct","Marketing_Spend_USD_M","CAC_USD",
    "CAC_Total_USD_M","Close_USD","Private_Valuation_USD_B"
]
PRICE_COLS = ["Close_USD"]


# -------------------------------------------------
# UTILIDADES
# -------------------------------------------------

def load_dictionary(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Descripción:", metadata.get("description"))
    print("Periodo:", metadata.get("period"))
    return metadata


def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    df = pd.read_csv(path)

    if DATE_COL not in df.columns:
        raise KeyError(f"No existe la columna {DATE_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)
    print("Dataset cargado:", df.shape)
    return df


def quick_eda(df: pd.DataFrame):
    print("\n=== ESTRUCTURA ===")
    print(df.info())
    print("\n=== NULOS (TOP 15) ===")
    print(df.isna().sum().sort_values(ascending=False).head(15))


# -------------------------------------------------
# VISUALIZACIONES
# -------------------------------------------------

def generate_interactive_plots(df: pd.DataFrame, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    eda_cols = [
        "Users_M","Revenue_USD_M","TPV_USD_B",
        "ARPU_USD","Churn_pct","CAC_USD","Close_USD"
    ]
    eda_cols = [c for c in eda_cols if c in df.columns]

    # Scatter Matrix
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=eda_cols,
        color="Segment" if "Segment" in df.columns else None,
        height=900
    )
    write_html(fig_matrix, output_dir / "interactive_scatter_matrix.html", include_plotlyjs="cdn")

    # Scatter 3D
    fig_3d = px.scatter_3d(
        df,
        x="Users_M",
        y="Revenue_USD_M",
        z="Close_USD",
        color="Region" if "Region" in df.columns else None,
        size="TPV_USD_B" if "TPV_USD_B" in df.columns else None,
        height=700
    )
    write_html(fig_3d, output_dir / "interactive_scatter_3d.html", include_plotlyjs="cdn")

    # UMAP 2D
    X = df[eda_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    umap_2d = umap.UMAP(n_components=2, random_state=42)
    emb_2d = umap_2d.fit_transform(X_scaled)

    df_umap = pd.DataFrame(emb_2d, columns=["UMAP_1","UMAP_2"])
    if "Segment" in df.columns:
        df_umap["Segment"] = df["Segment"]

    fig_umap = px.scatter(
        df_umap,
        x="UMAP_1",
        y="UMAP_2",
        color="Segment" if "Segment" in df_umap.columns else None,
        height=700
    )
    write_html(fig_umap, output_dir / "interactive_umap_2d.html", include_plotlyjs="cdn")


# -------------------------------------------------
# LIMPIEZA Y FEATURE ENGINEERING
# -------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__")

    return df


def create_returns(df: pd.DataFrame) -> pd.DataFrame:

    for price in PRICE_COLS:
        if price in df.columns:
            df[price+"_ret"] = (
                df.sort_values([ID_COLS[0], DATE_COL])
                  .groupby(ID_COLS)[price]
                  .pct_change()
                  .fillna(0.0)
            )
            df[price+"_logret"] = np.log1p(df[price+"_ret"]).fillna(0.0)

    return df


# -------------------------------------------------
# PREPARACIÓN PARA ML
# -------------------------------------------------

def prepare_ml_data(df: pd.DataFrame):

    X = df.drop(columns=[DATE_COL] + ID_COLS, errors="ignore")

    X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=True)

    cutoff = pd.to_datetime(SPLIT_DATE)
    idx_train = df[DATE_COL] < cutoff

    X_train = X[idx_train].copy()
    X_test = X[~idx_train].copy()

    numeric_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test


# -------------------------------------------------
# EXPORTACIÓN
# -------------------------------------------------

def export_outputs(X_train, X_test, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "fintech_train.parquet"
    test_path = output_dir / "fintech_test.parquet"

    X_train.to_parquet(train_path, index=False)
    X_test.to_parquet(test_path, index=False)

    with open(output_dir / "features_columns.txt","w",encoding="utf-8") as f:
        f.write("\n".join(X_train.columns))

    print("Archivos exportados correctamente.")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("=== INICIO PIPELINE FINTECH SINTÉTICO ===")

    load_dictionary(DATA_DICT)
    df = load_dataset(DATA_CSV)

    quick_eda(df)
    generate_interactive_plots(df, OUTPUT_DIR)

    df = clean_data(df)
    df = create_returns(df)

    X_train, X_test = prepare_ml_data(df)

    export_outputs(X_train, X_test, OUTPUT_DIR)

    print("✔ Proceso finalizado con éxito.")


if __name__ == "__main__":
    main()