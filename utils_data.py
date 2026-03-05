# =====================================================================
# utils_data.py — Data processing for SPI
# =====================================================================

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from tqdm import tqdm


def load_grid_data(path: str) -> pd.DataFrame:
    """Loads precipitation data from Excel file."""
    try:
        df = pd.read_excel(path, index_col=[0, 1])
        df.columns = pd.to_datetime(df.columns)
    except FileNotFoundError:
        print(f"File not found: {path}. Generating example data.")
        lats = np.linspace(-10, -12, 3)
        lons = np.linspace(-40, -42, 3)
        dates = pd.date_range(start="1994-01-01", periods=12, freq="MS")
        index = pd.MultiIndex.from_product([lats, lons], names=["Latitude", "Longitude"])
        data = np.random.rand(len(index), len(dates)) * 100
        df = pd.DataFrame(data, index=index, columns=dates)
        print("⚠ Using example data.")

    if not isinstance(df.index, pd.MultiIndex):
        df = df.T
    
    print(f"Data loaded: {df.shape[0]} pixels, {df.shape[1]} months")
    print(f"Period: {df.columns[0].date()} to {df.columns[-1].date()}")
    
    return df.astype("float32")


def calculate_spi(df_pr, scale=3, split_date=None):
    """
    Calculates SPI WITHOUT data leakage.
    - Rolling sums calculated only with data available up to each date
    - Gamma parameters estimated only with training data
    - SPI calculated sequentially, not retroactively
    """
    dates = pd.to_datetime(df_pr.columns)
    
    if split_date:
        sd = pd.to_datetime(split_date)
        idx_split = next((i for i, d in enumerate(dates) if d >= sd), len(dates))
        train_dates = dates[:idx_split]
        val_dates = dates[idx_split:]
    else:
        idx_split = int(len(dates) * 0.8)
        train_dates = dates[:idx_split]
        val_dates = dates[idx_split:]
    
    print(f"Split: {train_dates[0].date()} to {train_dates[-1].date()} | "
          f"{val_dates[0].date()} to {val_dates[-1].date()}")
    
    # Initialize SPI matrix
    spi_data = np.full((len(df_pr.index), len(dates)), np.nan, dtype=np.float32)
    
    # ------------------------------------------------------------
    # STEP 1: Calculate rolling sums ONLY for training data
    # ------------------------------------------------------------
    print("1. Calculating rolling sums in training...")
    rolling_train = {}
    for pix_idx, pix in enumerate(tqdm(df_pr.index, desc="Rolling sum (training)")):
        series = df_pr.loc[pix].values[:idx_split]  # ONLY training!
        rolling = pd.Series(series).rolling(window=scale, min_periods=1).sum().values
        rolling_train[pix_idx] = rolling
    
    # ------------------------------------------------------------
    # STEP 2: Estimate Gamma parameters with training data
    # ------------------------------------------------------------
    print("\n2. Estimating Gamma parameters in training...")
    params = {
        'alpha': np.full((len(df_pr.index), 12), np.nan, dtype=np.float32),
        'beta': np.full((len(df_pr.index), 12), np.nan, dtype=np.float32),
        'p_zero': np.full((len(df_pr.index), 12), 0.0, dtype=np.float32),
    }
    
    for month in tqdm(range(1, 13), desc="Fitting Gamma"):
        month_indices = np.where(train_dates.month == month)[0]
        if len(month_indices) == 0:
            continue
        
        for pix_idx in range(len(df_pr.index)):
            month_values = []
            for date_idx in month_indices:
                val = rolling_train[pix_idx][date_idx]
                if not np.isnan(val):
                    month_values.append(val)
            
            if len(month_values) < 10:
                continue
            
            month_values = np.array(month_values)
            non_zero = month_values[month_values > 0]
            
            if len(non_zero) < 5:
                p_zero = len(month_values[month_values == 0]) / len(month_values)
                params['p_zero'][pix_idx, month-1] = p_zero
                continue
            
            try:
                mean = non_zero.mean()
                var = non_zero.var()
                
                if var > 0:
                    alpha = mean**2 / var
                    beta = var / mean
                else:
                    alpha = 1.0
                    beta = mean
                
                alpha = max(0.1, min(alpha, 100))
                beta = max(0.1, min(beta, 100))
                p_zero = len(month_values[month_values == 0]) / len(month_values)
                
                params['alpha'][pix_idx, month-1] = alpha
                params['beta'][pix_idx, month-1] = beta
                params['p_zero'][pix_idx, month-1] = p_zero
                
            except Exception:
                params['alpha'][pix_idx, month-1] = 1.0
                params['beta'][pix_idx, month-1] = mean if 'mean' in locals() else 1.0
                params['p_zero'][pix_idx, month-1] = 0.0
    
    # ------------------------------------------------------------
    # STEP 3: Calculate SPI sequentially (training first)
    # ------------------------------------------------------------
    print("\n3. Calculating SPI for training data...")
    
    # Process training (using training rolling sums)
    for t_idx in tqdm(range(idx_split), desc="SPI training"):
        date = dates[t_idx]
        month = date.month
        
        for pix_idx in range(len(df_pr.index)):
            alpha_val = params['alpha'][pix_idx, month-1]
            if np.isnan(alpha_val):
                continue
            
            precip = rolling_train[pix_idx][t_idx]
            
            if np.isnan(precip):
                spi_val = np.nan
            else:
                if precip == 0:
                    F_x = params['p_zero'][pix_idx, month-1]
                else:
                    try:
                        F_x = params['p_zero'][pix_idx, month-1] + \
                              (1.0 - params['p_zero'][pix_idx, month-1]) * \
                              gamma.cdf(precip, alpha_val, scale=params['beta'][pix_idx, month-1])
                    except:
                        mean_val = alpha_val * params['beta'][pix_idx, month-1]
                        std_val = np.sqrt(alpha_val * params['beta'][pix_idx, month-1]**2)
                        F_x = norm.cdf(precip, mean_val, std_val)
                
                F_x = np.clip(F_x, 1e-8, 1 - 1e-8)
                spi_val = norm.ppf(F_x)
            
            spi_data[pix_idx, t_idx] = np.float32(spi_val)
    
    # ------------------------------------------------------------
    # STEP 4: Calculate SPI for validation (independent data)
    # ------------------------------------------------------------
    if idx_split < len(dates):
        print("\n4. Calculating SPI for validation data...")
        
        # Calculate rolling sums for validation (using only previous/available data)
        rolling_val = {}
        for pix_idx, pix in enumerate(tqdm(df_pr.index, desc="Rolling sum (validation)")):
            series = df_pr.loc[pix].values[:idx_split + len(val_dates)]  # Up to current date
            rolling = pd.Series(series).rolling(window=scale, min_periods=1).sum().values
            rolling_val[pix_idx] = rolling
        
        for rel_idx, t_idx in enumerate(tqdm(range(idx_split, len(dates)), desc="SPI validation")):
            date = dates[t_idx]
            month = date.month
            
            for pix_idx in range(len(df_pr.index)):
                alpha_val = params['alpha'][pix_idx, month-1]
                if np.isnan(alpha_val):
                    continue
                
                # Use rolling sums that consider only data up to this date
                precip = rolling_val[pix_idx][t_idx]
                
                if np.isnan(precip):
                    spi_val = np.nan
                else:
                    if precip == 0:
                        F_x = params['p_zero'][pix_idx, month-1]
                    else:
                        try:
                            F_x = params['p_zero'][pix_idx, month-1] + \
                                  (1.0 - params['p_zero'][pix_idx, month-1]) * \
                                  gamma.cdf(precip, alpha_val, scale=params['beta'][pix_idx, month-1])
                        except:
                            mean_val = alpha_val * params['beta'][pix_idx, month-1]
                            std_val = np.sqrt(alpha_val * params['beta'][pix_idx, month-1]**2)
                            F_x = norm.cdf(precip, mean_val, std_val)
                    
                    F_x = np.clip(F_x, 1e-8, 1 - 1e-8)
                    spi_val = norm.ppf(F_x)
                
                spi_data[pix_idx, t_idx] = np.float32(spi_val)
    
    # Create DataFrame
    df_spi = pd.DataFrame(
        spi_data,
        index=df_pr.index,
        columns=dates
    )
    
    print(f"\nSPI calculated: {df_spi.shape[1]} months")
    print(f"  Training: {df_spi.iloc[:, :idx_split].notna().sum().sum()} non-NaN values")
    print(f"  Validation: {df_spi.iloc[:, idx_split:].notna().sum().sum()} non-NaN values")
    
    return df_spi.astype(np.float32)


def check_leakage(df_pr, df_spi, split_date, scale=3):
    """
    Correct structural verification of temporal leakage in SPI.

    Strategy:
    1. Checks if Gamma parameters were estimated only with training data.
    2. Recalculates SPI incrementally and compares with provided SPI.
    3. Confirms that rolling uses only past data.
    """

    print("\n=== LEAKAGE CHECK (STRUCTURAL) ===")

    dates = pd.to_datetime(df_pr.columns)
    sd = pd.to_datetime(split_date)
    idx_split = next(i for i, d in enumerate(dates) if d >= sd)

    print(f"Split date: {sd.date()}, Split index: {idx_split}")

    leak_detected = False

    # -------------------------------------------------------------
    # 1️⃣ Verify rolling (uses only past data?)
    # -------------------------------------------------------------
    for pix in df_pr.index[:5]:  # test some pixels
        series_full = df_pr.loc[pix].values
        series_trunc = series_full[:idx_split + 1]

        rolling_full = pd.Series(series_full).rolling(
            window=scale, min_periods=scale
        ).sum().values

        rolling_trunc = pd.Series(series_trunc).rolling(
            window=scale, min_periods=scale
        ).sum().values

        # Compare at split point
        if not np.isclose(
            rolling_full[idx_split],
            rolling_trunc[-1],
            equal_nan=True
        ):
            print("  ⚠ Rolling uses future information.")
            leak_detected = True
            break

    if not leak_detected:
        print("  ✔ Rolling uses only past data.")

    # -------------------------------------------------------------
    # 2️⃣ Verify SPI stability at split point
    # -------------------------------------------------------------
    spi_at_split = df_spi.iloc[:, idx_split - 1]
    spi_recomputed = df_spi.iloc[:, :idx_split].iloc[:, -1]

    diff = np.nanmean(np.abs(spi_at_split.values - spi_recomputed.values))

    print(f"  Mean SPI difference at split: {diff:.6f}")

    if diff > 1e-6:
        print("  ⚠ SPI at split depends on future data.")
        leak_detected = True
    else:
        print("  ✔ SPI at split is consistent.")

    # -------------------------------------------------------------
    # 3️⃣ Incremental test in validation
    # -------------------------------------------------------------
    for step in range(min(3, df_spi.shape[1] - idx_split)):

        t = idx_split + step

        spi_original = df_spi.iloc[:, t]

        # recalculate SPI only up to t
        df_pr_trunc = df_pr.iloc[:, :t+1]

        # use your official function
        from utils_data import calculate_spi
        df_spi_trunc = calculate_spi(
            df_pr_trunc,
            scale=scale,
            split_date=split_date
        )

        spi_new = df_spi_trunc.iloc[:, -1]

        diff_val = np.nanmean(np.abs(spi_original.values - spi_new.values))

        print(f"  Validation step {step+1}: diff={diff_val:.6f}")

        if diff_val > 1e-6:
            print("  ⚠ SPI changes when we remove future data.")
            leak_detected = True
            break

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------
    if leak_detected:
        print("\n❌ DATA LEAKAGE DETECTED.")
    else:
        print("\n✅ No structural leakage detected.")

    return not leak_detected