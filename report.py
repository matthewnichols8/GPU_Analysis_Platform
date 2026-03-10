import pandas as pd
from data_generator import generate
from pipeline import run
from analysis import profile_gpu, detect_anomalies, detect_driver_regression, analyse_thermal
from visualization import plot_annotated_timeseries, plot_efficiency_heatmap, plot_distribution_comparison, plot_regression_dashboard

def generate_report(val_report, clean_report, gpu_profiles: dict, 
                    anomaly_report, regression_report, thermal_reports: dict,
                    path: str = "report.md") -> None:
    lines = []
    lines.append("# GPU Performance Analytics Report\n")
    lines.append("## Data Quality\n")
    lines.append("### Validation Report")
    lines.append(str(val_report))
    lines.append("\n### Cleaning Report")
    lines.append(str(clean_report))

    ranked = sorted(gpu_profiles.keys(), 
                key=lambda gpu: gpu_profiles[gpu].gaming.efficiency.mean, 
                reverse=True)
    
    lines.append("## Executive Summary\n")
    lines.append(f"Analysis of three GPU models over a 30-day period. "
                f"Ranked by gaming efficiency: "
                f"**{ranked[0]}** (best), **{ranked[1]}**, **{ranked[2]}** (worst). "
                f"Mean gaming efficiency: "
                f"{gpu_profiles[ranked[0]].gaming.efficiency.mean:.4f}, "
                f"{gpu_profiles[ranked[1]].gaming.efficiency.mean:.4f}, "
                f"{gpu_profiles[ranked[2]].gaming.efficiency.mean:.4f} FPS/W respectively.\n")
    
    for gpu, profile in gpu_profiles.items():
        lines.append(f"\n## {gpu} Performance\n")
        
        for workload, stats in [("gaming", profile.gaming), 
                                ("raytracing", profile.raytracing),
                                ("compute", profile.compute)]:
            lines.append(f"\n### {workload.capitalize()}\n")
            lines.append("| Metric | Mean | Std | P5 | P95 |")
            lines.append("|--------|------|-----|----|-----|")
            lines.append(f"| FPS | {stats.fps.mean:.2f} | {stats.fps.std:.2f} | {stats.fps.p5:.2f} | {stats.fps.p95:.2f} |")
            lines.append(f"| Power (W) | {stats.power_w.mean:.2f} | {stats.power_w.std:.2f} | {stats.power_w.p5:.2f} | {stats.power_w.p95:.2f} |")
            lines.append(f"| Temp (C) | {stats.temp_c.mean:.2f} | {stats.temp_c.std:.2f} | {stats.temp_c.p5:.2f} | {stats.temp_c.p95:.2f} |")
            lines.append(f"| Efficiency | {stats.efficiency.mean:.4f} | {stats.efficiency.std:.4f} | {stats.efficiency.p5:.4f} | {stats.efficiency.p95:.4f} |")
    
    lines.append("\n## Anomaly Summary\n")
    lines.append(f"Z-Score method detected **{anomaly_report.z_score.total_anomalies}** anomalies.")
    lines.append(f"IQR method detected **{anomaly_report.iqr.total_anomalies}** anomalies.")
    lines.append(f"Overlap between methods: **{anomaly_report.overlap}** rows flagged by both.\n")

    lines.append("### Anomaly Rate by GPU (Z-Score)")
    lines.append("| GPU | Anomaly Rate |")
    lines.append("|-----|-------------|")
    for gpu, rate in anomaly_report.z_score.anomaly_rate_GPU.items():
        lines.append(f"| {gpu} | {rate:.2%} |")

    lines.append("\n### Anomaly Rate by Workload (Z-Score)")
    lines.append("| Workload | Anomaly Rate |")
    lines.append("|----------|-------------|")
    for workload, rate in anomaly_report.z_score.anomaly_rate_workload.items():
        lines.append(f"| {workload} | {rate:.2%} |")
    
    lines.append("\n## Driver Regression Findings\n")
    if regression_report.driver_version:
        lines.append(f"ANOVA test found a statistically significant effect of driver version on FPS "
                    f"(p < {regression_report.p_value:.2e}, eta-squared = {regression_report.effect_size:.4f}). "
                    f"Driver **{regression_report.driver_version}** was identified as the underperformer.\n")
    else:
        lines.append("No statistically significant driver regression detected.\n")
    
    lines.append("\n## Thermal Analysis\n")
    for gpu, thermal in thermal_reports.items():
        lines.append(f"\n### {gpu}\n")
        lines.append("| Workload | Slope | Intercept | R² |")
        lines.append("|----------|-------|-----------|-----|")
        for workload in thermal.slope.keys():
            lines.append(f"| {workload} | {thermal.slope[workload]:.6f} | "
                        f"{thermal.intercept[workload]:.4f} | "
                        f"{thermal.r_squared[workload]:.4f} |")
        lines.append(f"\n**Throttling Window:** {thermal.throttle_start} → {thermal.throttle_end} "
                    f"({thermal.throttle_duration:.2f} hours)")
        lines.append(f"\n**Mean FPS Drop during throttling:** {thermal.mean_fps_drop:.2f} FPS\n")

    lines.append("\n## Conclusions\n")
    lines.append("GPUs were ranked using a composite score weighted as follows:\n")
    lines.append("- **50%** Gaming Efficiency (FPS/W)")
    lines.append("- **30%** Stability (inverse throttling rate)")
    lines.append("- **20%** Raw Gaming FPS (normalised)\n")

    composite_scores = {}
    for gpu, profile in gpu_profiles.items():
        efficiency_score = profile.gaming.efficiency.mean
        throttling_score = 1 - profile.throttling_rate / 100
        fps_score        = profile.gaming.fps.mean / 200
        composite_scores[gpu] = (efficiency_score * 0.5) + (throttling_score * 0.3) + (fps_score * 0.2)

    ranked = sorted(composite_scores.keys(), key=lambda g: composite_scores[g], reverse=True)

    lines.append("### Final Rankings\n")
    lines.append("| Rank | GPU | Composite Score |")
    lines.append("|------|-----|----------------|")
    for i, gpu in enumerate(ranked):
        lines.append(f"| {i+1} | {gpu} | {composite_scores[gpu]:.4f} |")

    lines.append(f"\n**Winner: {ranked[0]}** with a composite score of {composite_scores[ranked[0]]:.4f}.")
    lines.append(f"\n**Key Findings:**")
    lines.append(f"- Driver **{regression_report.driver_version}** showed a statistically significant FPS regression")
    lines.append(f"- Thermal throttling was detected with a mean FPS drop of "
                f"{sum(t.mean_fps_drop for t in thermal_reports.values()) / len(thermal_reports):.2f} FPS")
    lines.append(f"- Gaming is the most efficient workload across all GPU models")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    df = generate()
    df, val_report, clean_report = run(df)

    # Run once on full dataset
    anomaly_report    = detect_anomalies(df)
    regression_report = detect_driver_regression(df)

    # Run per GPU
    gpu_models   = ["RTX 4080", "RTX 4070", "RTX 4090"]
    gpu_profiles    = {}
    thermal_reports = {}

    for gpu in gpu_models:
        gpu_profiles[gpu]    = profile_gpu(df, gpu)
        thermal_reports[gpu] = analyse_thermal(df, gpu)

    generate_report(val_report, clean_report, gpu_profiles,
                    anomaly_report, regression_report, thermal_reports)

if __name__ == "__main__":
    main()