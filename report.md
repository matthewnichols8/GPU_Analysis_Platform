# GPU Performance Analytics Report

## Data Quality

### Validation Report
Validation Report --> Missing Columns: 0, Rows Outside of Timestamp: 0, Unknown Gpu Models: 0, Out of Bounds Numerics: 103867

### Cleaning Report
Cleaning Report --> Replaced FPS: 34548, Outliers: 66291, Forward Filled: 60099, Rows Dropped: 1
## Executive Summary

Analysis of three GPU models over a 30-day period. Ranked by gaming efficiency: **RTX 4080** (best), **RTX 4070**, **RTX 4090** (worst). Mean gaming efficiency: 0.4682, 0.4682, 0.4682 FPS/W respectively.


## RTX 4080 Performance


### Gaming

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 144.73 | 11.06 | 126.67 | 162.15 |
| Power (W) | 310.00 | 14.99 | 285.37 | 334.67 |
| Temp (C) | 73.23 | 5.11 | 64.87 | 81.63 |
| Efficiency | 0.4682 | 0.0449 | 0.3967 | 0.5427 |

### Raytracing

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 80.03 | 8.41 | 66.25 | 93.71 |
| Power (W) | 379.99 | 12.01 | 360.22 | 399.74 |
| Temp (C) | 76.72 | 5.09 | 68.40 | 85.12 |
| Efficiency | 0.2109 | 0.0240 | 0.1719 | 0.2507 |

### Compute

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 1.82 | 2.77 | 0.00 | 7.91 |
| Power (W) | 349.98 | 10.01 | 333.50 | 366.43 |
| Temp (C) | 75.22 | 5.07 | 66.93 | 83.58 |
| Efficiency | 0.0053 | 0.0080 | 0.0000 | 0.0228 |

## RTX 4070 Performance


### Gaming

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 144.73 | 11.06 | 126.67 | 162.15 |
| Power (W) | 310.00 | 14.99 | 285.37 | 334.67 |
| Temp (C) | 73.23 | 5.11 | 64.87 | 81.63 |
| Efficiency | 0.4682 | 0.0449 | 0.3967 | 0.5427 |

### Raytracing

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 80.03 | 8.41 | 66.25 | 93.71 |
| Power (W) | 379.99 | 12.01 | 360.22 | 399.74 |
| Temp (C) | 76.72 | 5.09 | 68.40 | 85.12 |
| Efficiency | 0.2109 | 0.0240 | 0.1719 | 0.2507 |

### Compute

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 1.82 | 2.77 | 0.00 | 7.91 |
| Power (W) | 349.98 | 10.01 | 333.50 | 366.43 |
| Temp (C) | 75.22 | 5.07 | 66.93 | 83.58 |
| Efficiency | 0.0053 | 0.0080 | 0.0000 | 0.0228 |

## RTX 4090 Performance


### Gaming

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 144.73 | 11.06 | 126.67 | 162.15 |
| Power (W) | 310.00 | 14.99 | 285.37 | 334.67 |
| Temp (C) | 73.23 | 5.11 | 64.87 | 81.63 |
| Efficiency | 0.4682 | 0.0449 | 0.3967 | 0.5427 |

### Raytracing

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 80.03 | 8.41 | 66.25 | 93.71 |
| Power (W) | 379.99 | 12.01 | 360.22 | 399.74 |
| Temp (C) | 76.72 | 5.09 | 68.40 | 85.12 |
| Efficiency | 0.2109 | 0.0240 | 0.1719 | 0.2507 |

### Compute

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|-----|
| FPS | 1.82 | 2.77 | 0.00 | 7.91 |
| Power (W) | 349.98 | 10.01 | 333.50 | 366.43 |
| Temp (C) | 75.22 | 5.07 | 66.93 | 83.58 |
| Efficiency | 0.0053 | 0.0080 | 0.0000 | 0.0228 |

## Anomaly Summary

Z-Score method detected **26499** anomalies.
IQR method detected **64312** anomalies.
Overlap between methods: **26448** rows flagged by both.

### Anomaly Rate by GPU (Z-Score)
| GPU | Anomaly Rate |
|-----|-------------|
| RTX 4070 | 1.00% |
| RTX 4080 | 1.03% |
| RTX 4090 | 1.03% |

### Anomaly Rate by Workload (Z-Score)
| Workload | Anomaly Rate |
|----------|-------------|
| compute | 0.93% |
| gaming | 1.06% |
| raytracing | 1.08% |

## Driver Regression Findings

ANOVA test found a statistically significant effect of driver version on FPS (p < 0.00e+00, eta-squared = 0.0023). Driver **546.17** was identified as the underperformer.


## Thermal Analysis


### RTX 4080

| Workload | Slope | Intercept | R² |
|----------|-------|-----------|-----|
| compute | 0.050247 | 57.6313 | 0.0098 |
| raytracing | 0.049989 | 57.7243 | 0.0138 |
| gaming | 0.049235 | 57.9677 | 0.0209 |

**Throttling Window:** 2024-01-20 04:29:08 → 2024-01-20 04:31:11 (0.03 hours)

**Mean FPS Drop during throttling:** 30.97 FPS


### RTX 4070

| Workload | Slope | Intercept | R² |
|----------|-------|-----------|-----|
| gaming | 0.048840 | 58.0812 | 0.0205 |
| raytracing | 0.050197 | 57.6381 | 0.0140 |
| compute | 0.049831 | 57.7809 | 0.0096 |

**Throttling Window:** 2024-01-28 11:53:59 → 2024-01-28 11:55:22 (0.02 hours)

**Mean FPS Drop during throttling:** 45.54 FPS


### RTX 4090

| Workload | Slope | Intercept | R² |
|----------|-------|-----------|-----|
| compute | 0.050993 | 57.3797 | 0.0102 |
| gaming | 0.050731 | 57.4982 | 0.0221 |
| raytracing | 0.049504 | 57.9168 | 0.0137 |

**Throttling Window:** 2024-01-11 17:31:01 → 2024-01-11 17:32:31 (0.03 hours)

**Mean FPS Drop during throttling:** 28.88 FPS


## Conclusions

GPUs were ranked using a composite score weighted as follows:

- **50%** Gaming Efficiency (FPS/W)
- **30%** Stability (inverse throttling rate)
- **20%** Raw Gaming FPS (normalised)

### Final Rankings

| Rank | GPU | Composite Score |
|------|-----|----------------|
| 1 | RTX 4080 | 0.6740 |
| 2 | RTX 4070 | 0.6740 |
| 3 | RTX 4090 | 0.6740 |

**Winner: RTX 4080** with a composite score of 0.6740.

**Key Findings:**
- Driver **546.17** showed a statistically significant FPS regression
- Thermal throttling was detected with a mean FPS drop of 35.13 FPS
- Gaming is the most efficient workload across all GPU models