# =====================================================================
# visualization_spi_classes.py
# Scientific visualization – MULTI-OUTPUT Framework (P,Q)
# Journal-ready layout
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
import torch

from dataset import SPIDataset, spi_to_class


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
# COLORMAPS
# =====================================================================

CMAP_SPI = "RdBu"

CMAP_ERROR = "viridis"

CMAP_ACC = "RdYlGn"

CMAP_CATEGORICAL = ListedColormap(
    ["#4daf4a", "#fee08b", "#fdae61", "#d73027"]
)


# =====================================================================
# GEO TIFF
# =====================================================================

def save_geotiff(array, lats, lons, out_path):

    H, W = array.shape
    res_x = abs(lons[1] - lons[0])
    res_y = abs(lats[1] - lats[0])

    transform = from_origin(
        lons.min(),
        lats.max(),
        res_x,
        res_y
    )

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        transform=transform,
        crs=CRS.from_epsg(4326),
    ) as dst:
        dst.write(array.astype(np.float32), 1)


# =====================================================================
# BASE MAP FUNCTION (fixed side layout)
# =====================================================================

def plot_map(data, lats, lons, title=None, cmap="viridis",
             vmin=None, vmax=None,
             cbar_label=None, save_path=None):

    set_journal_style()

    fig, (ax, cax) = plt.subplots(
        1, 2,
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05},
    )

    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        aspect="auto",
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close(fig)


# =====================================================================
# CLASSIFICATION METRICS
# =====================================================================

def calculate_classification_metrics(real_cls, pred_cls):

    real = real_cls.ravel()
    pred = pred_cls.ravel()

    mask = ~np.isnan(real) & ~np.isnan(pred)
    real, pred = real[mask], pred[mask]

    if len(real) == 0:
        return None

    return {
        "accuracy": accuracy_score(real, pred),
        "precision_macro": precision_score(real, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(real, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(real, pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(real, pred),
        "confusion_matrix": confusion_matrix(real, pred, labels=range(7)),
    }


def save_metrics_excel(metrics, out_path):

    df_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)",
                   "F1-Score (Macro)", "Cohen's Kappa"],
        "Value": [
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["kappa"],
        ],
    })

    df_cm = pd.DataFrame(metrics["confusion_matrix"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        df_cm.to_excel(writer, sheet_name="Confusion_Matrix")


# =====================================================================
# MULTI-HORIZON VISUALIZATION (h=1 and h=Q) — MEMORY SAFE
# =====================================================================

def generate_visualizations(
    model,
    df_pr,
    df_spi,
    P,
    Q,
    split_date,
    device,
    model_name,
    out_dir,
):

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[VIS] {model_name} | P={P} | Q={Q}")

    ds_val = SPIDataset(
        df_pr,
        df_spi,
        P,
        Q,
        train=False,
        split_date=split_date,
    )

    if len(ds_val) == 0:
        print("No validation data.")
        return

    lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
    lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))

    model.eval()

    # ==============================================================
    # Incremental initialization
    # ==============================================================

    def init_acc(shape):
        return {
            "sum_real": np.zeros(shape, dtype=np.float64),
            "sum_pred": np.zeros(shape, dtype=np.float64),
            "sum_abs": np.zeros(shape, dtype=np.float64),
            "sum_acc": np.zeros(shape, dtype=np.float64),
            "count": 0,
        }

    acc_h1 = None
    acc_hQ = None

    # For global metrics (classification)
    all_real_cls = []
    all_pred_cls = []

    # ==============================================================
    # Incremental loop (without storing everything)
    # ==============================================================

    with torch.no_grad():

        for i in range(len(ds_val)):

            x, y_seq = ds_val[i]
            x = x.unsqueeze(0).to(device)

            pred_seq = model.forecast(x, Q).squeeze(0).cpu().numpy()

            # =========================
            # h = 1
            # =========================

            real_h1 = y_seq[0].numpy()
            pred_h1 = pred_seq[0]

            if acc_h1 is None:
                acc_h1 = init_acc(real_h1.shape)

            acc_h1["sum_real"] += real_h1
            acc_h1["sum_pred"] += pred_h1
            acc_h1["sum_abs"] += np.abs(pred_h1 - real_h1)

            real_cls_h1 = spi_to_class(real_h1)
            pred_cls_h1 = spi_to_class(pred_h1)

            acc_h1["sum_acc"] += (real_cls_h1 == pred_cls_h1)
            acc_h1["count"] += 1

            # =========================
            # h = Q
            # =========================

            real_hQ = y_seq[Q - 1].numpy()
            pred_hQ = pred_seq[Q - 1]

            if acc_hQ is None:
                acc_hQ = init_acc(real_hQ.shape)

            acc_hQ["sum_real"] += real_hQ
            acc_hQ["sum_pred"] += pred_hQ
            acc_hQ["sum_abs"] += np.abs(pred_hQ - real_hQ)

            real_cls_hQ = spi_to_class(real_hQ)
            pred_cls_hQ = spi_to_class(pred_hQ)

            acc_hQ["sum_acc"] += (real_cls_hQ == pred_cls_hQ)
            acc_hQ["count"] += 1

            # Global metrics (classification)
            all_real_cls.append(spi_to_class(y_seq.numpy()))
            all_pred_cls.append(spi_to_class(pred_seq))

    # ==============================================================
    # Function to save maps
    # ==============================================================

    def finalize(acc, horizon_label):

        if acc is None or acc["count"] == 0:
            return

        mean_real = acc["sum_real"] / acc["count"]
        mean_pred = acc["sum_pred"] / acc["count"]
        mae_map = acc["sum_abs"] / acc["count"]
        acc_map = (acc["sum_acc"] / acc["count"]) * 100

        h_dir = os.path.join(out_dir, f"horizon_{horizon_label}")
        os.makedirs(h_dir, exist_ok=True)

        plot_map(
            mean_real,
            lats,
            lons,
            title=None,
            cmap=CMAP_SPI,
            vmin=-3,
            vmax=3,
            cbar_label="SPI",
            save_path=os.path.join(h_dir, "spi_observed.pdf"),
        )

        plot_map(
            mean_pred,
            lats,
            lons,
            title=None,
            cmap=CMAP_SPI,
            vmin=-3,
            vmax=3,
            cbar_label="SPI",
            save_path=os.path.join(h_dir, "spi_predicted.pdf"),
        )

        plot_map(
            mae_map,
            lats,
            lons,
            title=None,
            cmap=CMAP_ERROR,
            cbar_label="|SPI error|",
            save_path=os.path.join(h_dir, "mae_spatial.pdf"),
        )

        plot_map(
            acc_map,
            lats,
            lons,
            title=None,
            cmap=CMAP_ACC,
            vmin=0,
            vmax=100,
            cbar_label="Accuracy (%)",
            save_path=os.path.join(h_dir, "accuracy_spatial.pdf"),
        )

        save_geotiff(mean_real, lats, lons, os.path.join(h_dir, "spi_observed.tif"))
        save_geotiff(mean_pred, lats, lons, os.path.join(h_dir, "spi_predicted.tif"))
        save_geotiff(mae_map, lats, lons, os.path.join(h_dir, "mae_spatial.tif"))
        save_geotiff(acc_map, lats, lons, os.path.join(h_dir, "accuracy_spatial.tif"))

        print(f"[VIS] h={horizon_label} saved to {h_dir}")

    # Generate maps
    finalize(acc_h1, 1)

    if Q > 1:
        finalize(acc_hQ, Q)

    # ==============================================================
    # Global classification metrics
    # ==============================================================

    if len(all_real_cls) > 0:

        real_cls_all = np.concatenate(all_real_cls, axis=0)
        pred_cls_all = np.concatenate(all_pred_cls, axis=0)

        metrics = calculate_classification_metrics(
            real_cls_all,
            pred_cls_all
        )

        if metrics is not None:

            save_metrics_excel(
                metrics,
                os.path.join(out_dir, "classification_metrics.xlsx"),
            )

            print(
                f"[GLOBAL] Acc={metrics['accuracy']:.3f} | "
                f"F1={metrics['f1_macro']:.3f} | "
                f"Kappa={metrics['kappa']:.3f}"
            )

    print(f"[VIS] Completed → {out_dir}")