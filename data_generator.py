import pandas as pd
import numpy as np


def generate_base(seed: int = 42) -> pd.DataFrame:
    """Generate base DataFrame with realistic GPU performance data.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with timestamp, categorical, and numeric columns.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=30*24*60*60, freq="s")

    gpu_models      = ["RTX 4090", "RTX 4080", "RTX 4070"]
    driver_versions = ["545.84", "546.01", "546.17", "547.01"]
    workloads       = ["gaming", "raytracing", "compute"]

    n = len(timestamps)
    df = pd.DataFrame({
        "timestamp"      : timestamps,
        "gpu_model"      : rng.choice(gpu_models, n),
        "workload"       : rng.choice(workloads, n),
    })
    # Assign driver version in weekly blocks
    days_per_driver = 7
    df["driver_version"] = pd.cut(
        df["timestamp"].dt.day,
        bins=[0, 7, 14, 21, 31],
        labels=driver_versions
)

    is_gaming = df["workload"] == "gaming"
    is_rt     = df["workload"] == "raytracing"

    df["fps"] = np.select(
        [is_gaming, is_rt],
        [rng.normal(144, 8, n), rng.normal(85, 6, n)],
        default=0.0
    )
    df["power_w"] = np.select(
        [is_gaming, is_rt],
        [rng.normal(310, 15, n), rng.normal(380, 12, n)],
        default=rng.normal(350, 10, n)
    )
    df["temp_c"]       = rng.normal(75, 5, n)
    df["latency_ms"]   = rng.normal(7, 1.5, n)
    df["vram_used_gb"] = rng.normal(10, 2, n)
    df["gpu_util_pct"] = rng.normal(93, 4, n)

    df["gpu_util_pct"] = df["gpu_util_pct"].clip(0, 100)
    df["vram_used_gb"] = df["vram_used_gb"].clip(0, 24)
    df["fps"]          = df["fps"].clip(0, None)
    df["latency_ms"]   = df["latency_ms"].clip(0, None)
    df["power_w"]      = df["power_w"].clip(0, None)
    df["temp_c"]       = df["temp_c"].clip(20, 110)

    # print(df["gpu_util_pct"].max())
    # print(df["fps"].min())
    # print(df["temp_c"].max())

    return df


def inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject zeros, NaNs, and outlier spikes into the DataFrame.

    Args:
        df: Base DataFrame to inject noise into.
        rng: Random generator instance for reproducibility.

    Returns:
        DataFrame with injected noise.
    """
    df = df.copy()

    # 2% fps zeros on non-compute rows
    non_compute = df[df["workload"] != "compute"].index
    zero_idx = rng.choice(non_compute, size=int(0.02 * len(non_compute)), replace=False)
    df.loc[zero_idx, "fps"] = 0.0

    # 1% NaN values across numeric columns
    numeric_cols = ["power_w", "temp_c", "latency_ms", "vram_used_gb", "gpu_util_pct"]
    nan_idx  = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
    nan_cols = rng.choice(numeric_cols, size=len(nan_idx))
    for col in numeric_cols:
        col_mask = nan_cols == col
        df.loc[nan_idx[col_mask], col] = np.nan

    # 20 outlier spikes
    spike_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[spike_idx[:10], "power_w"] = rng.uniform(450, 500, 10)
    df.loc[spike_idx[10:], "temp_c"]  = rng.uniform(95, 105, 10)

    return df


def inject_throttling(df: pd.DataFrame) -> pd.DataFrame:
    """Inject a 4-hour thermal throttling event on Jan 15.

    Args:
        df: DataFrame to inject throttling into.

    Returns:
        DataFrame with fps reduced 25% and temp raised 10C in the window.
    """
    df = df.copy()
    start = pd.Timestamp("2024-01-15 14:00:00")
    end   = pd.Timestamp("2024-01-15 18:00:00")
    mask  = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df.loc[mask, "fps"]    *= 0.75
    df.loc[mask, "temp_c"] += 10
    return df


def inject_driver_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Inject an 8% fps regression for driver version 546.17.

    Args:
        df: DataFrame to inject regression into.

    Returns:
        DataFrame with 546.17 fps values reduced by 8%.
    """
    df = df.copy()
    mask = (df["driver_version"] == "546.17") & (df["fps"] > 0)
    df.loc[mask, "fps"] *= 0.92
    return df


def generate(seed: int = 42) -> pd.DataFrame:
    """Run the full generation pipeline with all injections.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Complete DataFrame ready for pipeline processing.
    """
    rng = np.random.default_rng(seed)
    df  = generate_base(seed)
    df  = inject_noise(df, rng)
    df  = inject_throttling(df)
    df  = inject_driver_regression(df)
    return df


def save(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV without row index.

    Args:
        df: DataFrame to save.
        path: Output file path.
    """
    df.to_csv(path, index=False)


def main() -> None:
    df = generate()
    save(df, "gpu_data.csv")
    print(f"Saved {len(df)} rows to gpu_data.csv")


if __name__ == "__main__":
    main()