# generate_monthly_maps.py - Generate monthly SPI prediction maps for all models

"""
Generate monthly SPI prediction maps for ConvLSTM3D, RF and XGBoost models
using test period (2025 only - 12 months). Also exports each map as GeoTIFF.
"""

import os
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from matplotlib.ticker import FuncFormatter

from dataset import SPIDataset
from utils_data import load_grid_data, load_or_calculate_spi
from model_convlstm3d import ConvLSTM3D
from plots import set_journal_style
from model_classic import predict_multioutput
from config import DATA_PATH, TRAIN_END_YEAR, REF_DATE, SPI_SCALE_FIXED, BASE_DIR
from metrics import compute_all_metrics

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS_BASE = BASE_DIR

# Test period: 2025 (12 months)
TEST_START_YEAR = 2025
TEST_END_YEAR = 2025

HORIZON = 1  # Lead time for monthly maps (1-step ahead)

MODELS_TO_PROCESS = ["ConvLSTM3D", "RF", "XGBoost"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# GEOTIFF UTILITIES
# ============================================================================

def save_as_tiff(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 output_path: str, description: str = "", nodata: float = -9999.0) -> None:
    """
    Save a 2D numpy array as GeoTIFF file.

    Args:
        data: 2D array [H, W]
        lats: Array of latitudes (sorted descending for origin='upper')
        lons: Array of longitudes (sorted ascending)
        output_path: Path to save the TIFF file
        description: Description to add to TIFF metadata
        nodata: Value to use for NoData pixels
    """
    H, W = data.shape

    # Calculate geotransform (assuming regular grid)
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    lon_res = (lon_max - lon_min) / (W - 1) if W > 1 else 1.0
    lat_res = (lat_max - lat_min) / (H - 1) if H > 1 else 1.0

    transform = from_origin(lon_min, lat_max, lon_res, lat_res)
    crs = CRS.from_epsg(4326)  # WGS84

    # Handle NaN values
    data_with_nodata = np.where(np.isnan(data), nodata, data)

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=H, width=W, count=1, dtype=data_with_nodata.dtype,
        crs=crs, transform=transform, nodata=nodata,
        compress='lzw', tiled=True, blockxsize=256, blockysize=256
    ) as dst:
        dst.write(data_with_nodata, 1)
        dst.update_tags(description=description, monthly_prediction="true", horizon=str(HORIZON))

    print(f"      TIFF saved: {output_path}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_convlstm_model(exp_dir: str, P: int, Q: int, device) -> ConvLSTM3D:
    """Load trained ConvLSTM3D model with correct hyperparameters."""
    model_path = os.path.join(exp_dir, "ConvLSTM3D", "best_model.pt")

    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract hyperparameters from checkpoint
    hidden = checkpoint.get('hidden', (64, 32, 16))
    dropout_p = checkpoint.get('dropout', 0.3)

    # Try loading from training_config.json if exists
    config_path = os.path.join(exp_dir, "ConvLSTM3D", "training_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'hyperparams' in config:
                    hidden = config['hyperparams'].get('hidden', hidden)
                    dropout_p = config['hyperparams'].get('dropout', dropout_p)
        except Exception as e:
            print(f"  ⚠ Could not read config: {e}")

    print(f"  ConvLSTM3D hyperparameters: hidden={hidden}, dropout={dropout_p}")

    model = ConvLSTM3D(hidden=hidden, dropout_p=dropout_p).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_classic_model(exp_dir: str, model_name: str):
    """Load trained classical model (RF or XGBoost)."""
    model_path = os.path.join(exp_dir, model_name, "best_model.joblib")

    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None

    return load(model_path)


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_classic(model, x_tensor: torch.Tensor, P: int, Q: int,
                    lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Generate prediction for classical model (RF or XGBoost)."""
    H, W = len(lats), len(lons)
    x_np = x_tensor.detach().cpu().numpy()

    # Remove batch dimension if present
    while x_np.ndim > 4:
        x_np = x_np.squeeze(0)

    if x_np.ndim == 3:
        x_np = x_np.reshape(1, *x_np.shape)

    X_all = []
    pixel_positions = []

    for i in range(H):
        for j in range(W):
            window = x_np[:, :, i, j]
            features = window.reshape(-1)

            if np.isnan(features).sum() > 0.5 * len(features):
                continue

            X_all.append(features)
            pixel_positions.append((i, j))

    if len(X_all) == 0:
        print(f"  ⚠ No valid pixels found for prediction")
        return np.full((H, W), np.nan)

    X_all = np.asarray(X_all, dtype=np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Predict all horizons at once
    preds = predict_multioutput(model, X_all, Q)  # [n_pixels, Q]

    # Return first horizon (h=0)
    spi_pred = np.full((H, W), np.nan)
    for pred, (i, j) in zip(preds[:, 0], pixel_positions):
        spi_pred[i, j] = pred

    return spi_pred


def get_prediction_convlstm(model, x_tensor: torch.Tensor, horizon: int = 1) -> np.ndarray:
    """Return ConvLSTM3D prediction for given horizon."""
    with torch.no_grad():
        if horizon == 1:
            spi_pred = model.forward_one_step(x_tensor)
        else:
            pred_seq = model.forecast(x_tensor, horizon)
            spi_pred = pred_seq[:, horizon - 1, :, :]

        return spi_pred.detach().cpu().numpy().squeeze()


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def get_input_target(target_date, df_pr, df_spi, dates, lats, lons, P: int):
    """
    For a target date, returns:
        x_tensor: [1, P, 3, H, W] (model input)
        spi_real: [H, W] (observed SPI at target date)
    """
    H, W = len(lats), len(lons)
    idx_target = list(dates).index(target_date)
    idx_start = idx_target - P

    if idx_start < 0:
        raise ValueError(f"Insufficient data before {target_date} (P={P})")

    x_pr = np.full((P, H, W), np.nan, dtype=np.float32)
    x_spi = np.full((P, H, W), np.nan, dtype=np.float32)

    for t, idx_t in enumerate(range(idx_start, idx_target)):
        date_t = dates[idx_t]

        grid_pr = (df_pr[date_t].unstack(level=1).reindex(index=lats, columns=lons))
        x_pr[t] = grid_pr.values.astype(np.float32)

        grid_spi = (df_spi[date_t].unstack(level=1).reindex(index=lats, columns=lons))
        x_spi[t] = grid_spi.values.astype(np.float32)

    # Compute delta SPI
    x_dspi = np.zeros_like(x_spi)
    if P > 1:
        x_dspi[1:] = x_spi[1:] - x_spi[:-1]

    x = np.stack([x_pr, x_spi, x_dspi], axis=1)
    x = np.nan_to_num(x, nan=0.0)

    # Get observed SPI at target date
    spi_real = (df_spi[target_date].unstack(level=1).reindex(index=lats, columns=lons).values)

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), spi_real


def find_best_config_for_model(model_name: str, experiments_base: str,
                               test_length: int = 12) -> tuple:
    """Find best P/Q configuration for a model based on test results."""
    metrics_path = os.path.join(experiments_base, "metrics", "test_results_all_models.xlsx")

    if not os.path.exists(metrics_path):
        print(f"⚠ Metrics file not found: {metrics_path}")
        return None, None

    try:
        df_metrics = pd.read_excel(metrics_path, sheet_name="Test_Metrics")
        df_model = df_metrics[df_metrics["model"] == model_name].copy()

        if df_model.empty:
            print(f"⚠ Model {model_name} not found in metrics")
            return None, None

        # Filter by test_source = "test" (real test data)
        if "test_source" in df_model.columns:
            df_real = df_model[df_model["test_source"] == "test"]
        else:
            df_real = df_model

        if not df_real.empty and not df_real["wi"].isna().all():
            best_idx = df_real["wi"].idxmax()
            best_row = df_real.loc[best_idx]
            print(f"  Best config with REAL test: P={int(best_row['P'])}, Q={int(best_row['Q'])}, WI={best_row['wi']:.4f}")
            return int(best_row["P"]), int(best_row["Q"])
        else:
            # Fallback to any configuration
            best_idx = df_model["wi"].idxmax()
            best_row = df_model.loc[best_idx]
            print(f"  Best config (any): P={int(best_row['P'])}, Q={int(best_row['Q'])}, WI={best_row['wi']:.4f}")
            return int(best_row["P"]), int(best_row["Q"])

    except Exception as e:
        print(f"⚠ Error reading metrics for {model_name}: {e}")
        return None, None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_figure_journal(observed: np.ndarray, predicted: np.ndarray, date,
                          lats: np.ndarray, lons: np.ndarray,
                          horizon: int, model_name: str) -> plt.Figure:
    """Create individual model map figure."""
    set_journal_style()

    def smart_format(x, pos):
        if abs(x - int(x)) < 1e-6:
            return f"{int(x)}"
        return f"{x:.2f}".rstrip('0').rstrip('.')

    formatter = FuncFormatter(smart_format)

    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    aspect_ratio = lon_range / lat_range

    fig_w = 12
    fig_h = fig_w / aspect_ratio if aspect_ratio > 0 else 8
    fig_h = max(6, min(fig_h, 10))

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    vmin, vmax = -3, 3
    cmap = 'RdBu'

    xticks = np.linspace(lons.min(), lons.max(), 4)
    yticks = np.linspace(lats.min(), lats.max(), 4)

    # Observed
    im1 = axes[0].imshow(observed, cmap=cmap, vmin=vmin, vmax=vmax,
                         extent=extent, origin='upper', aspect='equal',
                         interpolation='bilinear')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].tick_params(labelsize=9)
    axes[0].set_xticks(xticks)
    axes[0].set_yticks(yticks)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_title(f"Observed SPI - {date.strftime('%Y-%m')}", fontsize=11)

    # Predicted
    axes[1].imshow(predicted, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin='upper', aspect='equal',
                   interpolation='bilinear')
    axes[1].set_xlabel("Longitude")
    axes[1].tick_params(labelsize=9)
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    axes[1].set_xticks(xticks)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].set_title(f"{model_name} Predicted SPI - {date.strftime('%Y-%m')}", fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("SPI", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    return fig


def create_comparison_figure(observed: np.ndarray, predictions: dict, date,
                             lats: np.ndarray, lons: np.ndarray,
                             horizon: int) -> plt.Figure:
    """Create comparison figure with all models in 2x2 grid."""
    set_journal_style()

    def smart_format(x, pos):
        if abs(x - int(x)) < 1e-6:
            return f"{int(x)}"
        return f"{x:.2f}".rstrip('0').rstrip('.')

    formatter = FuncFormatter(smart_format)

    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    aspect_ratio = lon_range / lat_range

    fig_w = 12
    fig_h = 12 / aspect_ratio if aspect_ratio > 0 else 10
    fig_h = max(8, min(fig_h, 12))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    vmin, vmax = -3, 3
    cmap = 'RdBu'

    xticks = np.linspace(lons.min(), lons.max(), 4)
    yticks = np.linspace(lats.min(), lats.max(), 4)

    # Plot 1: Observed
    im1 = axes[0].imshow(observed, cmap=cmap, vmin=vmin, vmax=vmax,
                         extent=extent, origin='upper', aspect='equal',
                         interpolation='bilinear')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].tick_params(labelsize=8)
    axes[0].set_xticks(xticks)
    axes[0].set_yticks(yticks)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_title(f"Observed SPI\n{date.strftime('%Y-%m')}", fontsize=10)

    # Plot 2: ConvLSTM3D
    if "ConvLSTM3D" in predictions:
        axes[1].imshow(predictions["ConvLSTM3D"], cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin='upper', aspect='equal',
                       interpolation='bilinear')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].tick_params(labelsize=8)
    axes[1].set_xticks(xticks)
    axes[1].set_yticks(yticks)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_title(f"ConvLSTM3D\nPredicted SPI", fontsize=10)

    # Plot 3: Random Forest
    if "RF" in predictions:
        axes[2].imshow(predictions["RF"], cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin='upper', aspect='equal',
                       interpolation='bilinear')
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].tick_params(labelsize=8)
    axes[2].set_xticks(xticks)
    axes[2].set_yticks(yticks)
    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].yaxis.set_major_formatter(formatter)
    axes[2].set_title(f"Random Forest\nPredicted SPI", fontsize=10)

    # Plot 4: XGBoost
    if "XGBoost" in predictions:
        axes[3].imshow(predictions["XGBoost"], cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin='upper', aspect='equal',
                       interpolation='bilinear')
    axes[3].set_xlabel("Longitude")
    axes[3].set_ylabel("Latitude")
    axes[3].tick_params(labelsize=8)
    axes[3].set_xticks(xticks)
    axes[3].set_yticks(yticks)
    axes[3].xaxis.set_major_formatter(formatter)
    axes[3].yaxis.set_major_formatter(formatter)
    axes[3].set_title(f"XGBoost\nPredicted SPI", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("SPI", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 70)
    print("GENERATING MONTHLY MAPS - TEST PERIOD (2025)")
    print("WITH TIFF EXPORT")
    print("=" * 70)

    print("\n1. LOADING DATA")
    print("-" * 50)

    df_pr = load_grid_data(DATA_PATH)

    df_spi, indices = load_or_calculate_spi(
        df_pr,
        scale=SPI_SCALE_FIXED,
        train_end_year=TRAIN_END_YEAR,
        ref_date=REF_DATE,
        cache_dir=EXPERIMENTS_BASE,
        force_recompute=False
    )

    train_idx, val_idx, test_idx = indices
    dates = pd.to_datetime(df_pr.columns)

    print(f"\nFull period: {dates[0].date()} to {dates[-1].date()}")
    print(f"Training: {dates[train_idx[0]].date()} to {dates[train_idx[-1]].date()}")
    print(f"Validation: {dates[val_idx[0]].date()} to {dates[val_idx[-1]].date()}")
    print(f"Test: {dates[test_idx[0]].date()} to {dates[test_idx[-1]].date()}")

    lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
    lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))
    H, W = len(lats), len(lons)
    print(f"\nGrid: {H}×{W} pixels")

    print("\n2. SELECTING TEST PERIOD DATES (2025)")
    print("-" * 50)

    test_dates = dates[test_idx]

    if len(test_dates) == 0:
        raise ValueError("No test data available! Check dates.")

    # Filter only 2025
    test_dates_filtered = [d for d in test_dates if d.year == TEST_START_YEAR]

    print(f"Total months in test period: {len(test_dates)}")
    print(f"Months in {TEST_START_YEAR}: {len(test_dates_filtered)}")

    if len(test_dates_filtered) == 0:
        print(f"⚠ No data for {TEST_START_YEAR}, using all test period")
        selected_dates = test_dates
    else:
        selected_dates = test_dates_filtered

    print(f"Selected period: {selected_dates[0].date()} to {selected_dates[-1].date()}")

    print("\n3. FINDING BEST CONFIGURATIONS")
    print("-" * 50)

    test_length = len(selected_dates)

    model_configs = {}
    for model_name in MODELS_TO_PROCESS:
        P, Q = find_best_config_for_model(model_name, EXPERIMENTS_BASE, test_length=test_length)
        if P is not None:
            model_configs[model_name] = {"P": P, "Q": Q}

    print(f"\nSelected configurations:")
    for model_name, cfg in model_configs.items():
        print(f"  {model_name}: P={cfg['P']}, Q={cfg['Q']}")

    print("\n4. LOADING MODELS")
    print("-" * 50)

    loaded_models = {}

    for model_name in MODELS_TO_PROCESS:
        if model_name not in model_configs:
            print(f"  ⚠ {model_name}: no configuration, skipping...")
            continue

        cfg = model_configs[model_name]
        exp_dir = os.path.join(EXPERIMENTS_BASE, f"P{cfg['P']}_Q{cfg['Q']}")

        if not os.path.exists(exp_dir):
            print(f"  ⚠ {model_name}: experiment directory not found: {exp_dir}")
            continue

        if model_name == "ConvLSTM3D":
            model = load_convlstm_model(exp_dir, cfg['P'], cfg['Q'], DEVICE)
        else:
            model = load_classic_model(exp_dir, model_name)

        if model is not None:
            loaded_models[model_name] = {
                "model": model,
                "P": cfg['P'],
                "Q": cfg['Q']
            }
            print(f"  ✅ {model_name} loaded")
        else:
            print(f"  ❌ {model_name} failed to load")

    if len(loaded_models) == 0:
        print("\n❌ No models were loaded. Check paths.")
        return

    print("\n5. GENERATING MAPS AND TIFFS")
    print("-" * 50)

    out_dir = Path("monthly_maps_test_2025")
    out_dir.mkdir(exist_ok=True)

    individual_dir = out_dir / "individual_maps"
    individual_dir.mkdir(exist_ok=True)

    panel_dir = out_dir / "panel_maps"
    panel_dir.mkdir(exist_ok=True)

    # Create TIFF directories
    tiff_dir = out_dir / "tiff_exports"
    tiff_dir.mkdir(exist_ok=True)

    observed_tiff_dir = tiff_dir / "observed"
    observed_tiff_dir.mkdir(exist_ok=True)

    predictions_tiff_dir = tiff_dir / "predictions"
    predictions_tiff_dir.mkdir(exist_ok=True)

    all_predictions = {}
    monthly_metrics = {model_name: [] for model_name in loaded_models.keys()}

    with torch.no_grad():
        for i, date in enumerate(selected_dates):
            print(f"\nProcessing {i+1}/{len(selected_dates)}: {date.date()}...")

            # Use first model's P for input extraction (all models use same P)
            first_model = list(loaded_models.keys())[0]
            P_first = loaded_models[first_model]["P"]

            try:
                x_tensor, spi_real = get_input_target(
                    date, df_pr, df_spi, dates, lats, lons, P_first
                )

                # Move to device
                x_tensor = x_tensor.to(DEVICE)

                all_predictions[date] = {"real": spi_real, "models": {}}

                # Save observed SPI as TIFF
                observed_tiff_path = observed_tiff_dir / f"observed_SPI_{date.strftime('%Y%m')}.tif"
                save_as_tiff(
                    spi_real, lats, lons, observed_tiff_path,
                    description=f"Observed SPI for {date.strftime('%Y-%m')}"
                )

                for model_name, model_info in loaded_models.items():
                    model = model_info["model"]
                    P_model = model_info["P"]
                    Q_model = model_info["Q"]

                    # Use appropriate input window size for each model
                    if P_model != P_first:
                        x_tensor_model, _ = get_input_target(
                            date, df_pr, df_spi, dates, lats, lons, P_model
                        )
                        x_tensor_model = x_tensor_model.to(DEVICE)
                    else:
                        x_tensor_model = x_tensor

                    # Generate prediction
                    if model_name == "ConvLSTM3D":
                        spi_pred = get_prediction_convlstm(model, x_tensor_model, HORIZON)
                    else:
                        spi_pred = predict_classic(
                            model, x_tensor_model, P_model, Q_model, lats, lons
                        )

                    all_predictions[date]["models"][model_name] = spi_pred

                    # Save prediction as TIFF
                    pred_tiff_path = predictions_tiff_dir / f"{model_name}_prediction_SPI_{date.strftime('%Y%m')}.tif"
                    save_as_tiff(
                        spi_pred, lats, lons, pred_tiff_path,
                        description=f"{model_name} prediction for {date.strftime('%Y-%m')}"
                    )

                    # Calculate metrics
                    mask = ~np.isnan(spi_real) & ~np.isnan(spi_pred)
                    
                    if mask.sum() > 0:
                        obs_tensor = torch.tensor(spi_real[mask], dtype=torch.float32)
                        pred_tensor = torch.tensor(spi_pred[mask], dtype=torch.float32)
                        
                        # Calcula WI, RMSE e MAE em um único passo
                        all_metrics = compute_all_metrics(obs_tensor, pred_tensor)
                        
                        monthly_metrics[model_name].append({
                            'date': date.date(),
                            'rmse': all_metrics['rmse'],
                            'mae': all_metrics['mae'],
                            'wi': all_metrics['wi']
                        })

                    # Individual figure (PDF)
                    fig = create_figure_journal(
                        spi_real, spi_pred, date, lats, lons,
                        HORIZON, model_name
                    )

                    fname = individual_dir / f"{model_name}_{date.strftime('%Y%m')}_h{HORIZON}.pdf"
                    plt.savefig(fname, dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    print(f"    ✅ {model_name}: PDF saved, TIFF saved")

                # Comparison figure (if at least 2 models)
                preds_for_fig = {}
                for model_name in loaded_models.keys():
                    if model_name in all_predictions[date]["models"]:
                        preds_for_fig[model_name] = all_predictions[date]["models"][model_name]

                if len(preds_for_fig) >= 2:
                    fig_comp = create_comparison_figure(
                        spi_real, preds_for_fig, date, lats, lons, HORIZON
                    )
                    comp_fname = panel_dir / f"comparison_{date.strftime('%Y%m')}_h{HORIZON}.pdf"
                    plt.savefig(comp_fname, dpi=300, bbox_inches='tight')
                    plt.close(fig_comp)
                    print(f"    ✅ Comparison figure saved")

            except Exception as e:
                print(f"  ❌ Error on {date.date()}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n6. SAVING METRICS")
    print("-" * 50)

    for model_name, metrics_list in monthly_metrics.items():
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.set_index('date', inplace=True)
            metrics_df.to_excel(out_dir / f"monthly_metrics_{model_name}_h{HORIZON}.xlsx")
            print(f"  ✅ {model_name}: metrics saved")

            print(f"     Mean WI:   {metrics_df['wi'].mean():.4f} ± {metrics_df['wi'].std():.4f}")
            print(f"     Mean RMSE: {metrics_df['rmse'].mean():.4f} ± {metrics_df['rmse'].std():.4f}")
            print(f"     Mean MAE:  {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}")

    # Summary table
    summary_data = []
    for model_name, metrics_list in monthly_metrics.items():
        if metrics_list:
            df_metrics = pd.DataFrame(metrics_list)
            summary_data.append({
                "model": model_name,
                "horizon": HORIZON,
                "wi_mean": df_metrics["wi"].mean(),
                "wi_std": df_metrics["wi"].std(),
                "rmse_mean": df_metrics["rmse"].mean(),
                "rmse_std": df_metrics["rmse"].std(),
                "mae_mean": df_metrics["mae"].mean(),
                "mae_std": df_metrics["mae"].std(),
                "n_samples": len(metrics_list)
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(out_dir / "monthly_metrics_summary.xlsx", index=False)
        print(f"\n  ✅ Summary saved to {out_dir / 'monthly_metrics_summary.xlsx'}")

        print("\n" + "-" * 50)
        print("MONTHLY METRICS SUMMARY")
        print("-" * 50)
        print(df_summary.to_string(index=False))

    # Create metadata file with TIFF references
    print("\n7. SAVING METADATA")
    print("-" * 50)

    metadata = {
        'config': {
            'test_year': TEST_START_YEAR,
            'horizon': HORIZON,
            'models': list(loaded_models.keys()),
            'grid_shape': [H, W],
            'latitudes': lats.tolist(),
            'longitudes': lons.tolist(),
            'crs': 'EPSG:4326'
        },
        'dates': [d.strftime('%Y-%m-%d') for d in selected_dates],
        'tiff_files': {
            'observed': [str(observed_tiff_dir / f"observed_SPI_{d.strftime('%Y%m')}.tif") for d in selected_dates],
            'predictions': {
                model_name: [str(predictions_tiff_dir / f"{model_name}_prediction_SPI_{d.strftime('%Y%m')}.tif") for d in selected_dates]
                for model_name in loaded_models.keys()
            }
        }
    }

    metadata_path = tiff_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Metadata saved to {metadata_path}")

    print(f"\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print(f"Files saved in: {out_dir.resolve()}")
    print(f"TIFF files saved in: {tiff_dir.resolve()}")
    print("=" * 70)

    # Final summary of exported TIFFs
    print("\n" + "=" * 70)
    print("TIFF EXPORT SUMMARY")
    print("=" * 70)
    print(f"Observed maps: {len(selected_dates)} TIFFs")
    print(f"Prediction maps: {len(selected_dates) * len(loaded_models)} TIFFs")
    print(f"Total TIFFs exported: {len(selected_dates) * (1 + len(loaded_models))}")
    print("\nDirectory structure:")
    print(f"  {tiff_dir}/")
    print(f"    observed/")
    for date in selected_dates:
        print(f"      observed_SPI_{date.strftime('%Y%m')}.tif")
    print(f"    predictions/")
    for model_name in loaded_models.keys():
        print(f"      {model_name}_prediction_SPI_*.tif")
    print("=" * 70)


if __name__ == "__main__":
    main()