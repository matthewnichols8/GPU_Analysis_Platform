import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
BASE_DIR = Path(__file__)

@dataclass
class ValidationReport:
    missing_cols          : int
    rows_out_of_timestamp : int
    unknown_gpu_models    : int
    numeric_out_of_bounds : int

    def __str__(self):
        return f"Validation Report --> Missing Columns: {self.missing_cols}, Rows Outside of Timestamp: {self.rows_out_of_timestamp}, Unknown Gpu Models: {self.unknown_gpu_models}, Out of Bounds Numerics: {self.numeric_out_of_bounds}"


@dataclass
class CleaningReport:
    replaced_fps   : int
    num_outliers   : int
    forward_filled : int
    rows_dropped   : int
    log            : list

    def __str__(self):
        return f"Cleaning Report --> Replaced FPS: {self.replaced_fps}, Outliers: {self.num_outliers}, Forward Filled: {self.forward_filled}, Rows Dropped: {self.rows_dropped}"

def validate(df: pd.DataFrame) -> ValidationReport:
    """Checks for the following conditions:
       1) All required columns are present
       2) No rows exist outside timestamp
       3) gpu_model values are valid
       4) numeric columns are physically plausible
    """
    missing_cols = rows_out_of_timestamp = unknown_gpu_models = numeric_out_of_bounds = 0

    # Check Columns
    valid_cols = {'timestamp', 'gpu_model', 'driver_version',
                  'workload', 'fps', 'power_w', 'temp_c', 'latency_ms',
                  'vram_used_gb', 'gpu_util_pct'}
    missing  = valid_cols - set(df.columns)
    missing_cols = len(missing)
    if missing_cols > 0:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check rows
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp("2024-01-31")
    mask = (df["timestamp"] < start) | (df["timestamp"] >= end)
    rows_out_of_timestamp = mask.sum()
    if (rows_out_of_timestamp > 0):
        print(f"Warning: {rows_out_of_timestamp} rows outside of expected timestamp")
    
    # Check gpu_model
    valid_gpus = {"RTX 4090", "RTX 4080", "RTX 4070"}
    mask = ~df["gpu_model"].isin(valid_gpus)
    unknown_gpu_models = sum(mask)
    if unknown_gpu_models > 0:
        print(f"Warning: {unknown_gpu_models} unknown gpu_models found")

    # numeric columns are valid
    bounds = {
        "temp_c"       : (20, 115),
        "power_w"      : (0, 600),
        "fps"          : (0, 500),
        "latency_ms"   : (0, 100),
        "vram_used_gb" : (0, 24),
        "gpu_util_pct" : (0, 100),
    }
    for key, value in bounds.items():
        mask = (df[key] < value[0]) | (df[key] >= value[1])
        numeric_out_of_bounds += sum(mask)
    if (numeric_out_of_bounds > 0):
        print(f"Warning: {numeric_out_of_bounds} numerics out of bounds (includes throttling window temp spikes)")

    # Assign ValidationReport Object
    report = ValidationReport(
        missing_cols=missing_cols,
        rows_out_of_timestamp=rows_out_of_timestamp,
        unknown_gpu_models=unknown_gpu_models,
        numeric_out_of_bounds=numeric_out_of_bounds
    )
    return report

def iqr_clip(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.clip(q1 - 3 * iqr, q3 + 3 * iqr)

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    replaced_fps = num_outliers = forward_filled = rows_dropped = 0
    log = []

    # Replace fps == 0 w/ NaN
    non_compute = df["workload"] != "compute"
    replaced_fps = ((df["fps"] == 0) & non_compute).sum()
    df.loc[non_compute, "fps"] = df.loc[non_compute, "fps"].replace(0, float('nan'))
    log.append(f"Replaced {replaced_fps} fps zeros with NaN")
    if replaced_fps > 0:
        print(f"Warning: {replaced_fps} fps replaced")

    # Cap outliers
    numerics = ["temp_c", "power_w", "fps", "latency_ms", "vram_used_gb", 'gpu_util_pct']
    before = df[list(numerics)].copy()
    for metric in numerics:
        df[metric] = df.groupby(['gpu_model', 'workload'])[metric].transform(iqr_clip)
    num_outliers = (df[list(numerics)] != before).any(axis=1).sum()
    log.append(f"Capped {num_outliers} outliers using IQR method")
    if num_outliers > 0:
        print(f"Warning: {num_outliers} outliers capped")

    # Forward fill NaN values within each gpu_model and workload group
    before = df[list(numerics)].copy()
    for metric in numerics:
        df[metric] = df.groupby(['gpu_model', 'workload'])[metric].transform(
            lambda x: x.ffill()
        )
    forward_filled = (df[list(numerics)] != before).any(axis=1).sum()
    log.append(f"Forward filled {forward_filled} NaN values")
    if (forward_filled > 0):
        print(f"Warning: {forward_filled} forward filled")

    # Drop rows that still contain NaN after forward-filling
    rows_dropped = df.isna().any(axis=1).sum()
    df = df.dropna(axis=0)
    log.append(f"Dropped {rows_dropped} rows with remaining NaN")
    if rows_dropped > 0:
        print(f"Warning: {rows_dropped} rows dropped")

    report = CleaningReport(
        replaced_fps=replaced_fps,
        num_outliers=num_outliers,
        forward_filled=forward_filled,
        rows_dropped=rows_dropped,
        log=log
    )

    # for entry in report.log:
        # print(entry)

    return df, report



def engineer_features(df : pd.DataFrame) -> pd.DataFrame:
    """Adds More Informative Columns to the DF"""
    # Efficiency
    df["efficiency"] = df["fps"] / df["power_w"].replace(0, float('nan'))

    # Temp
    df["temp_category"] = np.select(
        [df["temp_c"] < 70, df["temp_c"] <= 85],
        ["cool", "warm"],
        default="hot"
    )

    # Rolling Mean
    df["fps_rolling_mean"] = (
        df.groupby(['gpu_model', 'workload'])['fps'].transform(
            lambda x: x.rolling(60, min_periods=1).mean()
        )
    )

    # Z-Score
    df["fps_zscore"] = (
        df.groupby(['gpu_model', 'workload'])['fps'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    )

    # Throttling
    group_mean = df.groupby(['gpu_model', 'workload'])['fps'].transform('mean')
    group_std  = df.groupby(['gpu_model', 'workload'])['fps'].transform('std')
    df["is_throttling"] = df["fps"] < (group_mean - 2 * group_std)

    return df

def run(df: pd.DataFrame):
    validation_report = validate(df)
    df, CleaningReport = clean(df)
    df = engineer_features(df)
    return df, validation_report, CleaningReport

from data_generator import generate

def main():
    df = generate()
    df, val_report, clean_report = run(df)
    print(val_report)
    print(clean_report)

if __name__ == "__main__":
    main()