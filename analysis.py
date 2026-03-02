import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import scipy.stats

@dataclass
class WorkloadStats:
    mean   : float
    median : float
    std    : float
    p5     : float
    p95    : float

@dataclass
class MetricProfile:
    fps         : WorkloadStats
    power_w     : WorkloadStats
    temp_c      : WorkloadStats
    efficiency : WorkloadStats

@dataclass
class GpuProfile:
    gpu_model              : str
    gaming                 : MetricProfile
    raytracing             : MetricProfile
    compute                : MetricProfile
    throttling_rate        : float
    peak_efficiency_window : any # type: ignore

    def __str__(self) -> str:
        return f"""
    ----------------------------------------------------------------------------------------
    GPU Profile: {self.gpu_model}
    Throttling Rate: {self.throttling_rate:.2f}%
    Peak Efficiency Window: {self.peak_efficiency_window}
    
    Gaming:
        FPS:        mean={self.gaming.fps.mean:.2f}, std={self.gaming.fps.std:.2f}, p5={self.gaming.fps.p5:.2f}, p95={self.gaming.fps.p95:.2f}
        Power (W):  mean={self.gaming.power_w.mean:.2f}, std={self.gaming.power_w.std:.2f}
        Efficiency: mean={self.gaming.efficiency.mean:.4f}

    Raytracing:
        FPS:        mean={self.raytracing.fps.mean:.2f}, std={self.raytracing.fps.std:.2f}, p5={self.raytracing.fps.p5:.2f}, p95={self.raytracing.fps.p95:.2f}
        Power (W):  mean={self.raytracing.power_w.mean:.2f}, std={self.raytracing.power_w.std:.2f}
        Efficiency: mean={self.raytracing.efficiency.mean:.4f}
    
    Compute:
        FPS:        mean={self.compute.fps.mean:.2f}, std={self.compute.fps.std:.2f}, p5={self.compute.fps.p5:.2f}, p95={self.compute.fps.p95:.2f}
        Power (W):  mean={self.compute.power_w.mean:.2f}, std={self.compute.power_w.std:.2f}
        Efficiency: mean={self.compute.efficiency.mean:.4f}
    ----------------------------------------------------------------------------------------
    """

@dataclass
class AnomalyStats:
    total_anomalies       : int
    anomaly_rate_GPU      : dict
    anomaly_rate_workload : dict
    extreme_anomalies     : list

@dataclass
class AnomalyReport:
    z_score : AnomalyStats
    iqr     : AnomalyStats 
    overlap : int

    def __str__(self):
        def fmt_dict(d: dict) -> str:
            return "\n      ".join([f"{k}: {v:.2%}" for k, v in d.items()])

        return (
            f"  Anomaly Report\n"
            f"  ══════════════════════════════\n"
            f"  Z-Score Method:\n"
            f"      Total Anomalies : {self.z_score.total_anomalies}\n"
            f"      Anomaly Rate by GPU:\n"
            f"          {fmt_dict(self.z_score.anomaly_rate_GPU)}\n"
            f"      Anomaly Rate by Workload:\n"
            f"          {fmt_dict(self.z_score.anomaly_rate_workload)}\n\n"
            f"  IQR Method:\n"
            f"      Total Anomalies : {self.iqr.total_anomalies}\n"
            f"      Anomaly Rate by GPU:\n"
            f"          {fmt_dict(self.iqr.anomaly_rate_GPU)}\n"
            f"      Anomaly Rate by Workload:\n"
            f"          {fmt_dict(self.iqr.anomaly_rate_workload)}\n\n"
            f"  Overlap (flagged by both) : {self.overlap}\n"
            f"  ══════════════════════════════"
        )
    
@dataclass
class RegressionReport:
    p_value        : float
    driver_version : Optional[str] # Can be Optional if no regression found
    effect_size    : float

    def __str__(self) -> str:
        return f"""
    ----------------------------------------------------------------------------------------
    Regression Report
    ══════════════════════════════
    P-value:        {self.p_value:.2e}
    Driver Version: {self.driver_version}
    Effect Size:    {self.effect_size:.4f}
    ══════════════════════════════
    ----------------------------------------------------------------------------------------   
"""

def get_workload_stats(stats : pd.DataFrame, workload_type : str, metric: str) -> WorkloadStats:
    """Calculate the workload stats for a specific workload and metric"""
    result = WorkloadStats(
        mean = stats.loc[workload_type, (metric, "mean")], # type: ignore
        median = stats.loc[workload_type, (metric, "median")], # type: ignore
        std = stats.loc[workload_type, (metric, "std")], # type: ignore
        p5 = stats.loc[workload_type, (metric, "p5")], # type: ignore
        p95 = stats.loc[workload_type, (metric, "p95")], # type: ignore
    )
    return result

def get_metric_profile(stats : pd.DataFrame, workload_type : str) -> MetricProfile:
    result = MetricProfile(
        fps = get_workload_stats(stats, workload_type, "fps"),
        power_w = get_workload_stats(stats, workload_type, "power_w"),
        temp_c = get_workload_stats(stats, workload_type, "temp_c"),
        efficiency = get_workload_stats(stats, workload_type, "efficiency")  
    )
    return result

def profile_gpu(df : pd.DataFrame, gpu_model : str) -> GpuProfile:
    """Creates a GpuProfile class instance"""
    # Per-workload stats
    stats = df.groupby("workload")[["fps", "power_w", "temp_c", "efficiency"]].agg({
    "fps"       : ["mean", "median", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "power_w"   : ["mean", "median", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "temp_c"    : ["mean", "median", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "efficiency": ["mean", "median", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
})
    stats.columns = pd.MultiIndex.from_tuples([
        (col, stat) for col in ["fps", "power_w", "temp_c", "efficiency"]
        for stat in ["mean", "median", "std", "p5", "p95"]
    ])
    # Thermal Throttling
    throttling_rate = df["is_throttling"].mean() * 100
    # Peak Efficiency Window
    hourly = df.resample("1h", on="timestamp")["efficiency"].mean()
    peak_efficiency_window = hourly.idxmax()
    # mean_efficiency = df["efficiency"].mean()
    
    # Get results
    gaming = get_metric_profile(stats, "gaming")
    raytracing = get_metric_profile(stats, "raytracing")
    compute = get_metric_profile(stats, "compute")

    result = GpuProfile(
        gpu_model=gpu_model,
        gaming=gaming,
        raytracing=raytracing,
        compute=compute,
        throttling_rate=throttling_rate,
        peak_efficiency_window=peak_efficiency_window
    )

    return result

def iqr_flag(series : pd.Series) -> pd.Series:
    """Used to detect anomalies in IQR"""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

def detect_anomalies(df : pd.DataFrame) -> AnomalyReport:
    """Detects Anomalies from DataFrame"""
    numeric_cols = ["power_w", "temp_c", "latency_ms", "vram_used_gb", "gpu_util_pct"]
    # Z-Score Methods --> AnomalyStats Object
    z_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        z = (df[col] - df[col].mean()) / df[col].std()
        z_mask = z_mask | (z.abs() > 3)
    z_total_anomalies = z_mask.sum()
            
    # IQR Methods --> AnomalyStats Object
    iqr_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        iqr_mask = iqr_mask | iqr_flag(df[col])
    i_total_anomalies = iqr_mask.sum()

    # Calculate overlap
    overlap = (z_mask & iqr_mask).sum()

    # Calculate rates
    df["z_flagged"] = z_mask
    df["iqr_flagged"] = iqr_mask
    z_anomaly_rate_gpu      = df.groupby("gpu_model")["z_flagged"].mean().to_dict()
    z_anomaly_rate_workload = df.groupby("workload")["z_flagged"].mean().to_dict()
    i_anomaly_rate_gpu      = df.groupby("gpu_model")["iqr_flagged"].mean().to_dict()
    i_anomaly_rate_workload = df.groupby("workload")["iqr_flagged"].mean().to_dict()

    # Calculate Z Extremes
    z_scores = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        z_scores[col] = (df[col] - df[col].mean()) / df[col].std()
    z_scores["combined"] = z_scores.abs().sum(axis=1)
    z_sorted = z_scores.sort_values("combined", ascending=False)
    z_extremes = df.loc[z_sorted.head(10).index]

    # IQR extremes
    iqr_scores = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_scores[col] = (df[col] - df[col].median()).abs() / (iqr + 1e-9)
    iqr_scores["combined"] = iqr_scores.abs().sum(axis=1)
    iqr_extremes = df.loc[iqr_scores.sort_values("combined", ascending=False).head(10).index]

    # Make Anomaly Stats for Z_Score and IQR
    z_stats = AnomalyStats(
        total_anomalies=z_total_anomalies,
        anomaly_rate_GPU=z_anomaly_rate_gpu,
        anomaly_rate_workload=z_anomaly_rate_workload,
        extreme_anomalies=z_extremes.to_dict("records")
    )

    iqr_stats = AnomalyStats(
        total_anomalies=i_total_anomalies,
        anomaly_rate_GPU=i_anomaly_rate_gpu,
        anomaly_rate_workload=i_anomaly_rate_workload,
        extreme_anomalies=iqr_extremes.to_dict("records")
    )

    # Aggregate result into an Anomaly Report
    result = AnomalyReport(
        z_score=z_stats,
        iqr=iqr_stats,
        overlap=overlap
    )

    return result

def detect_driver_regression(df : pd.DataFrame) -> RegressionReport:
    # ANOVA
    groups = [group.dropna().values for _, group in df.groupby("driver_version")["fps"]]
    f_stat, p_value = scipy.stats.f_oneway(*groups)

    # P-Values
    if p_value < 0.05: # Regression Detected
        # Effect Size
        overall_mean = df["fps"].mean()
        group_means  = df.groupby("driver_version")["fps"].mean()
        group_counts = df.groupby("driver_version")["fps"].count()

        ss_between = (group_counts * (group_means - overall_mean) ** 2).sum()
        ss_total   = ((df["fps"] - overall_mean) ** 2).sum()
        eta_squared = ss_between / ss_total
        # Assign Driver Version (minimum)
        driver_version = group_means.idxmin()
    else:
        driver_version = None
        eta_squared = 0.0

    result = RegressionReport(
        p_value=p_value,
        driver_version=driver_version, # type: ignore
        effect_size=eta_squared
    )

    return result

from data_generator import generate
from pipeline import run

def main():
    df = generate()
    df, val_report, clean_report = run(df) 
    profile = profile_gpu(df, "RTX 4080")
    anomaly_report = detect_anomalies(df)
    regression_report = detect_driver_regression(df)
    print(profile) 
    print(anomaly_report)
    print(regression_report)

if __name__ == "__main__":
    main()