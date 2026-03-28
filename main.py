"""
Experimental framework for multi-horizon SPI forecasting.
"""

import os
import time
import torch
import pandas as pd
import numpy as np

from utils_data import load_grid_data, calculate_spi, check_leakage
from visualization_spi_classes import generate_visualizations

from data_preparation import (
    create_datasets,
    prepare_classic_data,
    create_datasets_unified, 
    prepare_classic_data_unified
)

from model_convlstm3d import ConvLSTM3D
from model_classic import run_classic

from train_model import (
    train_model,
    evaluate_model,
)

# Configuration
BASE_DIR = "EXPERIMENTS"
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

DATA_PATH = "data/pr_Area2.xlsx"
SPLIT_DATE = "2018-01-01"
SPI_SCALE = 3

DL_PARAMS = dict(
    batch_size=4,
    epochs=100,
    lr=1e-3,
    hidden=(32, 16, 8),
)

Ps = [3, 6, 9, 12, 15, 18, 21, 24]
Q_VALUES = [1, 3, 6, 9, 12]

EVAL_MODE = "last"


if __name__ == "__main__":

    torch.manual_seed(123)
    np.random.seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GENERATE_VIS = True

    print("=" * 70)
    print("SPI FORECASTING FRAMEWORK - MULTI-HORIZON")
    print("=" * 70)
    print(f"Device: {device}")

    # Load data
    df_pr = load_grid_data(DATA_PATH)
    df_spi = calculate_spi(df_pr, scale=SPI_SCALE, split_date=SPLIT_DATE)
    check_leakage(df_pr, df_spi, SPLIT_DATE, scale=SPI_SCALE)

    results_global = []
    start_time = time.time()

    combos = [(P, Q) for P in Ps for Q in Q_VALUES]
    print(f"\nTotal combinations: {len(combos)}")

    pd.DataFrame(combos, columns=["P", "Q"]).to_csv(
        os.path.join(BASE_DIR, "grid_structure.csv"), index=False
    )

    for idx, (P, Q) in enumerate(combos, start=1):

        print("\n" + "=" * 70)
        print(f"[EXPERIMENT {idx}/{len(combos)}] P={P} | Q={Q}")
        print("=" * 70)

        combo_results = []

        try:
            combo_dir = os.path.join(BASE_DIR, f"P{P}_Q{Q}")
            os.makedirs(combo_dir, exist_ok=True)

            # Datasets
            ds_train, ds_val = create_datasets_unified(
                df_pr, df_spi, P, Q, SPLIT_DATE, mode="full"
            )

            # ConvLSTM3D
            model_dl = ConvLSTM3D(hidden=DL_PARAMS["hidden"]).to(device)

            model_dl = train_model(
                model_dl, ds_train, ds_val, P, Q,
                DL_PARAMS["epochs"], DL_PARAMS["lr"], 
                DL_PARAMS["batch_size"], device, eval_mode=EVAL_MODE,
            )

            metrics_dl = evaluate_model(model_dl, ds_val, Q, device)

            combo_results.append({
                "model": "ConvLSTM3D",
                "P": P, "Q": Q,
                "wi": metrics_dl["wi"],
                "rmse": metrics_dl["rmse"],
                "mae": metrics_dl["mae"],
                "nse": metrics_dl["nse"],
                "bias": metrics_dl["bias"],
                "wi_by_h": metrics_dl["wi_by_h"],
            })

            # Save DL model
            dl_dir = os.path.join(combo_dir, "ConvLSTM3D")
            os.makedirs(dl_dir, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model_dl.state_dict(),
                    "P": P, "Q": Q,
                    "hidden": DL_PARAMS["hidden"],
                    "best_wi": model_dl.best_wi,
                    "best_wi_by_h": model_dl.best_wi_by_h,
                    "eval_mode": EVAL_MODE,
                },
                os.path.join(dl_dir, "best_model.pt"),
            )

            # Save WI by horizon
            pd.DataFrame({
                "horizon": range(1, Q + 1),
                "wi": model_dl.best_wi_by_h,
            }).to_csv(os.path.join(dl_dir, "WI_by_h.csv"), index=False)

            # Visualizations
            if GENERATE_VIS:
                model_dl.eval()
                generate_visualizations(
                    model=model_dl, df_pr=df_pr, df_spi=df_spi,
                    P=P, Q=Q, split_date=SPLIT_DATE, device=device,
                    model_name="ConvLSTM3D", out_dir=dl_dir,
                )

            # Classical models
            X_train, Y_train_seq, X_val, Y_val_seq, H, W = (
                prepare_classic_data_unified(
                    df_pr, df_spi, P, Q, SPLIT_DATE
                )
            )

            from joblib import dump

            for model_name in ["RF", "XGBoost"]:

                res = run_classic(
                    model_name, X_train, Y_train_seq, X_val, Y_val_seq, P, Q,
                )

                if res["model"] is None:
                    continue

                metrics = res["metrics"]

                combo_results.append({
                    "model": model_name,
                    "P": P, "Q": Q,
                    "wi": metrics["wi"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "nse": metrics["nse"],
                    "bias": metrics["bias"],
                    "wi_by_h": metrics["wi_by_h"],
                })

                classic_dir = os.path.join(combo_dir, model_name)
                os.makedirs(classic_dir, exist_ok=True)

                dump(res["model"], os.path.join(classic_dir, "best_model.joblib"))

            # Best model for this combo
            df_combo = pd.DataFrame(combo_results)
            df_combo = df_combo[np.isfinite(df_combo["wi"])]

            if not df_combo.empty:
                best_row = df_combo.sort_values(by="wi", ascending=False).iloc[0]
                
                with open(os.path.join(combo_dir, "best_model.txt"), "w") as f:
                    f.write(f"Best model for P={P}, Q={Q}\n")
                    f.write(f"Model: {best_row['model']}\n")
                    f.write(f"WI: {best_row['wi']:.6f}\n")
                    f.write(f"RMSE: {best_row['rmse']:.6f}\n")
                    f.write(f"MAE: {best_row['mae']:.6f}\n")
                    f.write(f"NSE: {best_row['nse']:.6f}\n")
                    f.write(f"Bias: {best_row['bias']:.6f}\n")

                best_row.to_frame().T.to_json(
                    os.path.join(combo_dir, "best_model_metrics.json"),
                    orient="records", indent=4
                )

            results_global.extend(combo_results)

        except Exception as e:
            print(f"[ERROR] P={P}, Q={Q}: {e}")
            continue

    # Global results
    df_results = pd.DataFrame(results_global)
    
    excel_path = None  

    if not df_results.empty:
        
        df_global = df_results[
            ["model", "P", "Q", "wi", "rmse", "mae", "nse", "bias"]
        ].sort_values(["model", "P", "Q"])

        rows_wi_h = []
        for _, row in df_results.iterrows():
            if "wi_by_h" in row and row["wi_by_h"] is not None:
                for h, wi_val in enumerate(row["wi_by_h"], start=1):
                    rows_wi_h.append({
                        "model": row["model"], "P": row["P"], "Q": row["Q"],
                        "horizon": h, "wi": wi_val,
                    })

        df_wi_long = pd.DataFrame(rows_wi_h)

        excel_path = os.path.join(METRICS_DIR, "all_models_metrics.xlsx")  # <-- MOVER PARA DENTRO DO IF
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_global.to_excel(writer, sheet_name="Global_Metrics", index=False)
            if not df_wi_long.empty:
                df_wi_long.to_excel(writer, sheet_name="WI_by_Horizon", index=False)

    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} min")
    if excel_path:  # <-- AGORA excel_path ESTÁ DEFINIDA
        print(f"Results saved to: {excel_path}")