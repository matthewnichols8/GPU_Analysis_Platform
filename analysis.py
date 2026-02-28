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
    effieicency : WorkloadStats

@dataclass
class GpuProfile:
    gpu_model              : str
    gaming                 : MetricProfile
    raytracing             : MetricProfile
    compute                : MetricProfile
    throttling_rate        : float
    peak_efficiency_window : pd.Timestamp



def profile_gpu(df : pd.DataFrame, gpu_model : str) -> GpuProfile:
    """Creates a GpuProfile class instance"""
    result = GpuProfile()
    return result