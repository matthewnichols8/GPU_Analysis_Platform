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

def plot_distribution_comparison(df: pd.DataFrame, metric: str, figsize=(18, 6)) -> None:
    """Plots a subplot for each workload"""
    workloads = ["compute", "gaming", "raytracing"]
    gpu_models = df["gpu_model"].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"{metric} Distribution by Workload", fontsize=14)
    
    # Get consistent x range across all subplots
    x_min = df[metric].min()
    x_max = df[metric].max()
    
    for ax, workload in zip(axes, workloads):
        workload_df = df[df["workload"] == workload]
        for gpu in gpu_models:
            gpu_df = workload_df[workload_df["gpu_model"] == gpu]
            sns.kdeplot(gpu_df[metric], ax=ax, label=gpu)
            ax.axvline(gpu_df[metric].median(), linestyle="--", alpha=0.7,
                       color=sns.color_palette()[list(gpu_models).index(gpu)])
        ax.set_title(workload)
        ax.set_xlim(x_min, x_max)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_regression_dashboard(df: pd.DataFrame, figsize=(16, 12)) -> None:
    """Produces a 2x2 grid of scatter plots with regression lines and R²"""
    
    def add_regression(ax, x, y, gpu, color):
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = np.linspace(x.min(), x.max(), 100)
        predicted = p(x)
        ss_residual = ((y - predicted) ** 2).sum()
        ss_total    = ((y - y.mean()) ** 2).sum()
        r_squared   = 1 - (ss_residual / ss_total)
        ax.plot(x_sorted, p(x_sorted), color=color, linewidth=2)
        return r_squared
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors   = {"RTX 4090": "blue", "RTX 4080": "green", "RTX 4070": "orange"}
    subplots = [
        (axes[0,0], "gaming",     "fps",    "Gaming: FPS vs Power"),
        (axes[0,1], "gaming",     "temp_c", "Gaming: Temp vs Power"),
        (axes[1,0], "raytracing", "fps",    "Raytracing: FPS vs Power"),
        (axes[1,1], "raytracing", "temp_c", "Raytracing: Temp vs Power"),
    ]

    for ax, workload, y_metric, title in subplots:
        r2_values = []
        for gpu, color in colors.items():
            gpu_df = df[(df["workload"] == workload) & (df["gpu_model"] == gpu)]
            ax.scatter(gpu_df["power_w"], gpu_df[y_metric],
                       color=color, alpha=0.1, s=1, label=gpu)
            r2 = add_regression(ax, gpu_df["power_w"], gpu_df[y_metric], gpu, color)
            r2_values.append(r2)
        ax.set_title(f"{title} | R²={sum(r2_values)/len(r2_values):.3f}")
        ax.set_xlabel("power_w")
        ax.set_ylabel(y_metric)
        ax.legend(loc="upper right", fontsize=8, markerscale=3)

    fig.suptitle("Regression Dashboard", fontsize=14)
    plt.tight_layout()
    plt.show()

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
    # plot_distribution_comparison(df, "power_w")
    plot_regression_dashboard(df)
    return None

if __name__ == '__main__':
    main()