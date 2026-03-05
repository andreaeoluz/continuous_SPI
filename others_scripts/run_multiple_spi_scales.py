# =====================================================================
# run_multiple_spi_scales.py
# =====================================================================

import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils_data import load_grid_data, calculate_spi
from data_preparation import create_datasets, prepare_classic_data
from model_convlstm3d import ConvLSTM3D
from model_classic import run_classic
from train_model import train_model, evaluate_model

# =====================================================================
# SCIENTIFIC STYLE
# =====================================================================

def set_journal_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.3,
    })

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIR = "EXPERIMENTS"
os.makedirs(BASE_DIR, exist_ok=True)

DATA_PATH = "data/pr_Area1.xlsx"
SPLIT_DATE = "2018-01-01"

SPI_SCALES = [1, 3, 6, 9, 12]
P_VALUES = [3, 6, 9, 12, 15, 18, 21, 24]
Q = 1

DL_PARAMS = dict(
    batch_size=4,
    epochs=100,
    lr=1e-3,
    hidden=(32, 16, 8),
)

CLASSIC_PARAMS = dict(
    sampling_rate=0.1,
    max_samples=50000,
)

EVAL_MODE = "last"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_journal_style()

# =====================================================================
# MAIN FUNCTION FOR ONE SPI SCALE
# =====================================================================

def run_spi_scale(scale, df_pr, verbose=True):
    """Runs experiments for a specific SPI scale."""
    
    scale_dir = os.path.join(BASE_DIR, f"SPI-{scale}")
    metrics_dir = os.path.join(scale_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENTS FOR SPI-{scale}")
        print("=" * 70)
    
    df_spi = calculate_spi(
        df_pr,
        scale=scale,
        split_date=SPLIT_DATE,
    )
    
    results_scale = []
    
    for P in P_VALUES:
        
        if verbose:
            print(f"\n[SPI-{scale}] P={P}")
        
        combo_dir = os.path.join(scale_dir, f"P{P}")
        os.makedirs(combo_dir, exist_ok=True)
        
        try:
            ds_train, ds_val = create_datasets(
                df_pr, df_spi, P, Q, SPLIT_DATE
            )
            
            # ConvLSTM3D
            model_dl = ConvLSTM3D(hidden=DL_PARAMS["hidden"]).to(DEVICE)
            
            model_dl = train_model(
                model_dl, ds_train, ds_val, P, Q,
                DL_PARAMS["epochs"], DL_PARAMS["lr"], 
                DL_PARAMS["batch_size"], DEVICE,
                eval_mode=EVAL_MODE,
            )
            
            metrics_dl = evaluate_model(model_dl, ds_val, Q, DEVICE)
            
            results_scale.append({
                "spi_scale": scale,
                "model": "ConvLSTM3D",
                "P": P,
                "wi": metrics_dl["wi"],
                "rmse": metrics_dl["rmse"],
                "mae": metrics_dl["mae"],
                "nse": metrics_dl["nse"],
                "bias": metrics_dl["bias"],
            })
            
            torch.save({
                "model_state_dict": model_dl.state_dict(),
                "P": P,
                "best_wi": model_dl.best_wi,
            }, os.path.join(combo_dir, "ConvLSTM3D_best.pt"))
            
            # Classic models (RF and XGBoost)
            X_train, Y_train_seq, X_val, Y_val_seq, H, W = (
                prepare_classic_data(
                    df_pr, df_spi, P, Q, SPLIT_DATE,
                    sampling_rate=CLASSIC_PARAMS["sampling_rate"],
                    max_samples=CLASSIC_PARAMS["max_samples"],
                    random_seed=123
                )
            )
            
            from joblib import dump
            
            for model_name in ["RF", "XGBoost"]:
                
                res = run_classic(
                    model_name, X_train, Y_train_seq,
                    X_val, Y_val_seq, P, Q
                )
                
                if res["model"] is None:
                    continue
                
                metrics = res["metrics"]
                
                results_scale.append({
                    "spi_scale": scale,
                    "model": model_name,
                    "P": P,
                    "wi": metrics["wi"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "nse": metrics["nse"],
                    "bias": metrics["bias"],
                })
                
                dump(res["model"], os.path.join(combo_dir, f"{model_name}_best.joblib"))
            
            # Save best model info
            df_combo = pd.DataFrame([
                r for r in results_scale 
                if r["spi_scale"] == scale and r["P"] == P
            ])
            
            if not df_combo.empty and np.isfinite(df_combo["wi"]).any():
                best_row = df_combo.loc[df_combo["wi"].idxmax()]
                
                with open(os.path.join(combo_dir, "best_model.txt"), "w") as f:
                    f.write(f"SPI-{scale} | P={P} | Q={Q}\n")
                    f.write(f"Best model: {best_row['model']}\n")
                    f.write(f"WI: {best_row['wi']:.6f}\n")
                    f.write(f"NSE: {best_row['nse']:.6f}\n")
                    f.write(f"RMSE: {best_row['rmse']:.6f}\n")
                    f.write(f"MAE: {best_row['mae']:.6f}\n")
                    f.write(f"Bias: {best_row['bias']:.6f}\n")
        
        except Exception as e:
            if verbose:
                print(f"  [ERROR] P={P}: {str(e)}")
            continue
    
    df_scale = pd.DataFrame(results_scale)
    if not df_scale.empty:
        df_scale.to_csv(os.path.join(metrics_dir, "results.csv"), index=False)
        
        best_global = df_scale.loc[df_scale["wi"].idxmax()]
        
        with open(os.path.join(scale_dir, "best_config.txt"), "w") as f:
            f.write(f"SPI-{scale} - GLOBAL BEST CONFIGURATION\n")
            f.write(f"Model: {best_global['model']}\n")
            f.write(f"P: {best_global['P']}\n")
            f.write(f"WI: {best_global['wi']:.6f}\n")
            f.write(f"NSE: {best_global['nse']:.6f}\n")
    
    return df_scale


# =====================================================================
# RESULTS CONSOLIDATION
# =====================================================================

def consolidate_results(all_results):
    """Consolidates results from all scales into organized tables."""
    if not all_results:
        return {}
    
    df_all = pd.concat(all_results, ignore_index=True)
    df_all = df_all[np.isfinite(df_all["wi"])]
    
    agg_by_scale = df_all.groupby("spi_scale").agg({
        "wi": ["mean", "std", "max"],
        "nse": ["mean", "std", "max"],
        "rmse": ["mean", "std", "min"],
        "mae": ["mean", "std", "min"],
    }).round(4)
    
    best_by_scale = []
    for scale in df_all["spi_scale"].unique():
        df_scale = df_all[df_all["spi_scale"] == scale]
        best = df_scale.loc[df_scale["wi"].idxmax()]
        best_by_scale.append({
            "spi_scale": scale,
            "model": best["model"],
            "P": best["P"],
            "wi": best["wi"],
            "nse": best["nse"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "bias": best["bias"],
        })
    
    df_best = pd.DataFrame(best_by_scale).round(4)
    
    model_comparison = df_all.pivot_table(
        index="spi_scale",
        columns="model",
        values="wi",
        aggfunc="mean"
    ).round(4)
    
    evolution = []
    for scale in df_all["spi_scale"].unique():
        df_scale = df_all[df_all["spi_scale"] == scale]
        for P in sorted(df_scale["P"].unique()):
            df_p = df_scale[df_scale["P"] == P]
            if not df_p.empty:
                best_p = df_p.loc[df_p["wi"].idxmax()]
                evolution.append({
                    "spi_scale": scale,
                    "P": P,
                    "best_model": best_p["model"],
                    "wi": best_p["wi"],
                    "nse": best_p["nse"],
                })
    
    df_evolution = pd.DataFrame(evolution).round(4)
    
    return {
        "all_results": df_all,
        "agg_by_scale": agg_by_scale,
        "best_by_scale": df_best,
        "model_comparison": model_comparison,
        "evolution": df_evolution,
    }


# =====================================================================
# EXPORT TO EXCEL
# =====================================================================

def export_to_excel(consolidated, output_path):
    """Exports all tables to an Excel file."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        
        consolidated["all_results"].to_excel(
            writer, sheet_name="All_Results", index=False
        )
        
        consolidated["agg_by_scale"].to_excel(
            writer, sheet_name="Agg_by_Scale"
        )
        
        consolidated["best_by_scale"].to_excel(
            writer, sheet_name="Best_by_Scale", index=False
        )
        
        consolidated["model_comparison"].to_excel(
            writer, sheet_name="Model_Comparison"
        )
        
        consolidated["evolution"].to_excel(
            writer, sheet_name="Evolution_P", index=False
        )
        
        stats = consolidated["all_results"].describe().round(4)
        stats.to_excel(writer, sheet_name="Descriptive_Stats")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "=" * 70)
    print("MULTI-SCALE SPI EXPERIMENT (Q=1)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"SPI scales: {SPI_SCALES}")
    print(f"P values: {P_VALUES}")
    print(f"Split date: {SPLIT_DATE}")
    print("=" * 70)
    
    print("\nLoading precipitation data...")
    df_pr = load_grid_data(DATA_PATH)
    
    all_results = []
    start_time = time.time()
    
    for scale in SPI_SCALES:
        
        scale_start = time.time()
        
        df_scale = run_spi_scale(scale, df_pr, verbose=True)
        
        if df_scale is not None and not df_scale.empty:
            all_results.append(df_scale)
            
            scale_time = (time.time() - scale_start) / 60
            print(f"\n[SPI-{scale}] Completed in {scale_time:.1f} min")
            print(f"  Valid samples: {len(df_scale)}")
            print(f"  Best WI: {df_scale['wi'].max():.4f}")
    
    if all_results:
        print("\n" + "=" * 70)
        print("CONSOLIDATING RESULTS...")
        print("=" * 70)
        
        consolidated = consolidate_results(all_results)
        
        excel_path = os.path.join(BASE_DIR, "spi_scales_comparison.xlsx")
        export_to_excel(consolidated, excel_path)
        
        consolidated["all_results"].to_csv(
            os.path.join(BASE_DIR, "all_results.csv"), 
            index=False
        )
        
        print("\n" + "=" * 70)
        print("SUMMARY - BEST CONFIGURATION BY SCALE")
        print("=" * 70)
        print(consolidated["best_by_scale"].to_string(index=False))
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (mean WI)")
        print("=" * 70)
        print(consolidated["model_comparison"].to_string())
        
        print(f"\n✅ Results exported to: {excel_path}")
    
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETED IN {total_time:.1f} MINUTES")
    print("=" * 70)


# =====================================================================
# EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    torch.manual_seed(123)
    np.random.seed(123)
    torch.backends.cudnn.deterministic = True
    
    main()