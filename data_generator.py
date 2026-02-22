import pandas as pd
import numpy as np

def generate_base(seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=30*24*60*60, freq="s")
    
    gpu_models      = ["RTX 4090", "RTX 4080", "RTX 4070"]
    driver_versions = ["545.84", "546.01", "546.17", "547.01"]
    workloads       = ["gaming", "raytracing", "compute"]

    n = len(timestamps)
    df = pd.DataFrame({
        "timestamp"      : timestamps,
        "gpu_model"      : rng.choice(gpu_models, n),
        "driver_version" : rng.choice(driver_versions, n),
        "workload"       : rng.choice(workloads, n)
    })

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

    return df

def inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    non_compute = df[df["workload"] != "compute"].index
    zero_idx = rng.choice(non_compute, size=int(0.02 * len(non_compute)), replace=False)
    df.loc[zero_idx, "fps"] = 0.0
    numeric_cols = ["power_w", "temp_c", "latency_ms", "vram_used_gb", "gpu_util_pct"]
    nan_idx = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
    nan_cols = rng.choice(numeric_cols, size=len(nan_idx))
    for col in numeric_cols:
        col_mask = nan_cols == col
        df.loc[nan_idx[col_mask], col] = np.nan
    spike_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[spike_idx[:10], "power_w"] = rng.uniform(450, 500, 10)
    df.loc[spike_idx[10:], "temp_c"] = rng.uniform(95, 105, 10)

    return df

def inject_throttling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = pd.Timestamp("2024-01-15 14:00:00")
    end = pd.Timestamp("2024-01-15 18:00:00")
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df.loc[mask, "fps"] *= 0.75
    df.loc[mask, "temp_c"] += 10
    return df

def inject_driver_regression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = (df["driver_version"] == "546.17") & (df["fps"] > 0)
    df.loc[mask, "fps"] *= 0.92
    return df

def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = generate_base(seed)
    df = inject_noise(df, rng)
    df = inject_throttling(df)
    df = inject_driver_regression(df)
    return df

def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def main():
    df = generate()
    save(df, "gpu_data.csv")
    print(f"Saved {len(df)} rows to gpu_data.csv")

if __name__ == '__main__':
    main()


