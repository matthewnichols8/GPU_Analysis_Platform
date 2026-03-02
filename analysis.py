import pandas as pd
import numpy as np
from dataclasses import dataclass

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

from data_generator import generate
from pipeline import run

def main():
    df = generate()
    df, val_report, clean_report = run(df) 
    profile = profile_gpu(df, "RTX 4080")
    print(profile) 

if __name__ == "__main__":
    main()