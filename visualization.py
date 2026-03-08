import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_annotated_timeseries(df : pd.DataFrame, gpu_model : str, metric : str, figsize=(14, 6)) -> None:
    """"Plots an Annotated Timeseries based off the DataFrame, GPU Model, and Metric"""
    df = df[df["gpu_model"] == gpu_model]
    df = df[df["workload"] == "gaming"]

    # Save throttling column before resample
    throttling = df.set_index("timestamp")["is_throttling"].resample("1min").max().reset_index()
    # Save Driver Version Column before resample
    driver = df.set_index("timestamp")["driver_version"].resample("1min").first().reset_index()
    # Save Z-Score column before resample
    zscore = df.set_index("timestamp")["fps_zscore"].resample("1min").mean().reset_index()
    # Resample for every min instead to drop number of samples
    df = df.set_index("timestamp").resample("1min")[metric].mean().reset_index()
    # Add Throttling Back After Resample
    df["is_throttling"] = throttling["is_throttling"]
    # Add Driver Version Back
    df["driver_version"] = driver["driver_version"]
    # Add Z-Score Back
    df["fps_zscore"] = zscore["fps_zscore"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["timestamp"], df[metric], label=metric, alpha=0.7)
    ax.set_title(f"{metric} over time - {gpu_model}")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    # Set new y_lim because the y-axis starts at 0
    ax.set_ylim(df[metric].min() * 0.9, df[metric].max() * 1.1)
    ax.legend()
    ax.grid(True)

    #--------------
    # Add Overlays
    #--------------

    # Rolling Mean Line
    rolling  = df[metric].rolling(60, min_periods=1).mean()
    ax.plot(df["timestamp"], rolling, label="Rolling Mean", color="orange", linewidth=2)

    # Throttling Shading
    rolling = df[metric].rolling(60, min_periods=1).mean()
    overall_mean = df[metric].mean()
    throttling_mask = rolling < (overall_mean * 0.85)

    y_min = df[metric].min() * 0.95
    y_max = df[metric].max() * 1.05
    ax.set_ylim(y_min, y_max)

    ax.fill_between(df["timestamp"], y_min, y_max,
                    where=throttling_mask, #type: ignore
                    color="red", alpha=0.2, label="Throttling")

    # Driver Version Changes
    driver_changes = [
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-22"),
    ]
    # driver_changes = df[df["driver_version"] != df["driver_version"].shift()]["timestamp"]
    for i, ts in enumerate(driver_changes):
        ax.axvline(x=ts, color="green", linestyle="--", alpha=0.7, linewidth=1, # type: ignore
                   label="Driver Change" if i == 0 else "")

    # Add Anomaly Points
    anomalies = df[df["fps_zscore"].abs() > 3]
    ax.scatter(anomalies["timestamp"], anomalies[metric], 
           color="red", marker="x", s=50, label="Anomalies", zorder=5)
    
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    return None

def plot_efficiency_heatmap(df : pd.DataFrame, figsize=(14, 6)) -> None:
    """"Plots an Efficiency and Throttling Heatmap"""
    pivot_efficiency = df.pivot_table(index="gpu_model", columns="workload", values="efficiency", aggfunc="mean")
    pivot_throttling = df.pivot_table(index="gpu_model", columns="workload", values="is_throttling", aggfunc="mean")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    sns.heatmap(pivot_efficiency, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax1)
    ax1.set_title("Mean Efficiency (FPS/W)")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    sns.heatmap(pivot_throttling, annot=True, fmt=".2%", cmap="Reds", ax=ax2)
    ax2.set_title("Throttling Rate")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()
    return None

from data_generator import generate
from pipeline import run
from analysis import analyze

def main():
    df = generate()
    df, val_report, clean_report = run(df)
    df, gpu_profile, anomaly_report, regression_report, thermal_report = analyze(df, "RTX 4080")
    print(val_report)
    print(clean_report)
    print(gpu_profile) 
    print(anomaly_report)
    print(regression_report)
    print(thermal_report)
    # plot_annotated_timeseries(df, "RTX 4080", "fps")
    # plot_efficiency_heatmap(df)
    return None

if __name__ == '__main__':
    main()