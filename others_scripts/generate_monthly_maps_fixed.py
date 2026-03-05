# =====================================================================
# generate_monthly_maps_fixed.py
# =====================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import SPIDataset
from utils_data import load_grid_data, calculate_spi, check_leakage
from model_convlstm3d import ConvLSTM3D
from visualization_spi_classes import set_journal_style

# ==============================================================
# CONFIGURATION
# ==============================================================
DATA_PATH = "data/pr_Area1.xlsx"
MODEL_PATH = "EXPERIMENTS/P3_Q1/ConvLSTM3D/best_model.pt"
SPI_SCALE = 3
SPLIT_DATE = "2018-01-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_MONTHS = 12
HORIZON = 1

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

# ==============================================================
# LOAD DATA
# ==============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df_pr = load_grid_data(DATA_PATH)

spi_cache_path = Path("EXPERIMENTS/df_spi_calculado.pkl")

if spi_cache_path.exists():
    print(f"Loading SPI from cache: {spi_cache_path}")
    df_spi = pd.read_pickle(spi_cache_path)
else:
    print("Calculating SPI with correct temporal split...")
    df_spi = calculate_spi(
        df_pr,
        scale=SPI_SCALE,
        split_date=SPLIT_DATE,
    )
    check_leakage(df_pr, df_spi, SPLIT_DATE, scale=SPI_SCALE)
    df_spi.to_pickle(spi_cache_path)
    print(f"SPI saved to {spi_cache_path}")

dates = pd.to_datetime(df_pr.columns)
sd = pd.to_datetime(SPLIT_DATE)
idx_split = next((i for i, d in enumerate(dates) if d >= sd), len(dates))

print(f"\nFull period: {dates[0].date()} to {dates[-1].date()}")
print(f"Split date: {SPLIT_DATE}")
print(f"  Training: {dates[0].date()} to {dates[idx_split-1].date()}")
print(f"  Validation: {dates[idx_split].date()} to {dates[-1].date()}")

# ==============================================================
# LOAD MODEL
# ==============================================================
print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
P = checkpoint['P']
Q = checkpoint['Q']
hidden = checkpoint['hidden']

print(f"Model: P={P}, Q={Q}, hidden={hidden}")
print(f"Best WI: {checkpoint.get('best_wi', 'N/A')}")

model = ConvLSTM3D(hidden=hidden).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==============================================================
# SELECT DATES OF INTEREST
# ==============================================================
print("\n" + "=" * 60)
print("SELECTING DATES FOR VISUALIZATION")
print("=" * 60)

validation_dates = dates[idx_split:]

if len(validation_dates) == 0:
    raise ValueError("No validation data! Check SPLIT_DATE.")

target_dates = validation_dates[-min(N_MONTHS, len(validation_dates)):]

print(f"Total months in validation: {len(validation_dates)}")
print(f"Selected period: {target_dates[0].date()} to {target_dates[-1].date()}")

# ==============================================================
# EXTRACT LAT/LON
# ==============================================================
lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))
H, W = len(lats), len(lons)

print(f"Grid: {H}×{W} pixels")

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def get_input_target(target_date):
    """Returns input tensor and observed SPI for a given date."""
    idx_target = list(dates).index(target_date)
    idx_start = idx_target - P
    
    if idx_start < 0:
        raise ValueError(f"Insufficient data before {target_date} (P={P})")
    
    x_pr = np.full((P, H, W), np.nan, dtype=np.float32)
    x_spi = np.full((P, H, W), np.nan, dtype=np.float32)
    
    for t, idx_t in enumerate(range(idx_start, idx_target)):
        date_t = dates[idx_t]
        x_pr[t] = df_pr[date_t].values.reshape(H, W)
        x_spi[t] = df_spi[date_t].values.reshape(H, W)
    
    x_dspi = np.zeros_like(x_spi)
    if P > 1:
        x_dspi[1:] = x_spi[1:] - x_spi[:-1]
    
    x = np.stack([x_pr, x_spi, x_dspi], axis=1)
    x = np.nan_to_num(x, nan=0.0)
    
    spi_observed = df_spi[target_date].values.reshape(H, W)
    
    return (
        torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        spi_observed
    )

def get_prediction(x_tensor):
    """Returns prediction as numpy array."""
    with torch.no_grad():
        if HORIZON == 1:
            spi_pred = model.forward_one_step(x_tensor)
        else:
            pred_seq = model.forecast(x_tensor, Q)
            spi_pred = pred_seq[:, HORIZON - 1, :, :]
        
        spi_pred_np = spi_pred.detach().cpu().numpy().squeeze()
    
    return spi_pred_np

def create_figure(observed, predicted, lats, lons):
    """Creates figure with observed | predicted | colorbar."""
    img_height = len(lats)
    img_width = len(lons)
    aspect_ratio = img_width / img_height
    
    base_height = 4
    base_width = base_height * aspect_ratio
    
    fig = plt.figure(figsize=(base_width * 2.2, base_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(observed, cmap='RdBu', vmin=-3, vmax=3,
                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                     aspect='auto')
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.tick_params(labelsize=8)
    
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(predicted, cmap='RdBu', vmin=-3, vmax=3,
                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                     aspect='auto')
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.tick_params(axis='y', left=False, labelleft=False)
    
    cax = fig.add_subplot(gs[2])
    cbar = plt.colorbar(im1, cax=cax, label='SPI')
    cbar.ax.tick_params(labelsize=8)
    
    return fig

# ==============================================================
# GENERATE INDIVIDUAL MAPS
# ==============================================================
print("\n" + "=" * 60)
print("GENERATING INDIVIDUAL MAPS")
print("=" * 60)

out_dir = Path("monthly_maps")
out_dir.mkdir(exist_ok=True)

set_journal_style()

monthly_metrics = {}
all_predictions = {}

with torch.no_grad():
    for i, date in enumerate(target_dates):
        print(f"Processing {i+1}/{len(target_dates)}: {date.date()}...")
        
        try:
            x_tensor, spi_observed = get_input_target(date)
            spi_predicted = get_prediction(x_tensor)
            
            all_predictions[date] = {
                'observed': spi_observed,
                'predicted': spi_predicted
            }
            
            if np.all(np.isnan(spi_observed)) or np.all(np.isnan(spi_predicted)):
                print(f"  ⚠ Invalid data for {date.date()}, skipping...")
                continue
            
            mask = ~np.isnan(spi_observed) & ~np.isnan(spi_predicted)
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((spi_observed[mask] - spi_predicted[mask])**2))
                mae = np.mean(np.abs(spi_observed[mask] - spi_predicted[mask]))
                monthly_metrics[date.date()] = {
                    'rmse': float(rmse),
                    'mae': float(mae)
                }
            
            fig = create_figure(spi_observed, spi_predicted, lats, lons)
            
            fname = out_dir / f"SPI{SPI_SCALE}_{date.date()}_h{HORIZON}.pdf"
            plt.savefig(fname, dpi=300, bbox_inches='tight', format='pdf')
            plt.close(fig)
            
            print(f"  ✅ Saved: {fname.name}")
            
        except Exception as e:
            print(f"  ❌ Error on {date.date()}: {e}")
            continue

# ==============================================================
# SAVE METRICS
# ==============================================================
if monthly_metrics:
    metrics_df = pd.DataFrame.from_dict(monthly_metrics, orient='index')
    metrics_df.index.name = 'date'
    metrics_df.to_csv(out_dir / f"monthly_metrics_h{HORIZON}.csv")
    print(f"\n✅ Monthly metrics saved")
    print(metrics_df.round(4))

# ==============================================================
# GENERATE SUMMARY FIGURE
# ==============================================================
print("\n" + "=" * 60)
print("GENERATING SUMMARY FIGURE")
print("=" * 60)

if len(all_predictions) == 0:
    print("❌ No valid predictions for summary")
else:
    n_months = len(all_predictions)
    n_cols = 4
    n_rows = (n_months + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten() if n_months > 1 else [axes]
    
    last_im = None
    
    for idx, (date, data) in enumerate(all_predictions.items()):
        try:
            spi_observed = data['observed']
            spi_predicted = data['predicted']
            
            ax_obs = axes[idx * 2]
            im_obs = ax_obs.imshow(spi_observed, cmap='RdBu', vmin=-3, vmax=3,
                                   extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                                   aspect='auto')
            ax_obs.set_title(f"{date.date()}", fontsize=10)
            ax_obs.set_xticks([])
            ax_obs.set_yticks([])
            
            ax_pred = axes[idx * 2 + 1]
            im_pred = ax_pred.imshow(spi_predicted, cmap='RdBu', vmin=-3, vmax=3,
                                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                                     aspect='auto')
            ax_pred.set_title(f"{date.date()}", fontsize=10)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            
            last_im = im_obs
            
        except Exception as e:
            print(f"Error in summary for {date.date()}: {e}")
            continue
    
    for idx in range(len(all_predictions) * 2, len(axes)):
        axes[idx].set_visible(False)
    
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax, label='SPI')
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        summary_path = out_dir / f"summary_{target_dates[0].date()}_to_{target_dates[-1].date()}_h{HORIZON}.pdf"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ Summary figure saved: {summary_path}")
    else:
        print("❌ Could not generate summary figure")
    
    plt.close(fig)

print(f"\nAll files saved to: {out_dir.resolve()}")

# ==============================================================
# SUMMARY STATISTICS
# ==============================================================
if monthly_metrics:
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    metrics_array = np.array([(v['rmse'], v['mae']) for v in monthly_metrics.values()])
    
    print(f"Mean RMSE: {np.mean(metrics_array[:, 0]):.4f} ± {np.std(metrics_array[:, 0]):.4f}")
    print(f"Mean MAE:  {np.mean(metrics_array[:, 1]):.4f} ± {np.std(metrics_array[:, 1]):.4f}")
    print(f"Best month (RMSE): {min(monthly_metrics.items(), key=lambda x: x[1]['rmse'])[0]}")
    print(f"Worst month (RMSE): {max(monthly_metrics.items(), key=lambda x: x[1]['rmse'])[0]}")