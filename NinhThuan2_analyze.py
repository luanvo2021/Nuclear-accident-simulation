import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm

# Đọc dữ liệu từ file CSV
data = pd.read_csv("NinhThuan_results_time.csv")

# Biểu đồ 1: Hexbin plot (Tốc độ gió vs Mức phóng xạ)
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

ax1 = fig.add_subplot(gs[0])
hb = ax1.hexbin(
    data["wind_speed"],
    data["radiation_level"],
    gridsize=50,
    cmap="Spectral_r",
    norm=LogNorm(vmin=1),
    mincnt=1
)
cb = fig.colorbar(hb, ax=ax1, label="Log10(Counts)")
ax1.set_xlabel("Wind Speed (m/s)")
ax1.set_ylabel("Radiation Level (mSv/day)")
ax1.set_title("Spectral Scatter Plot: Radiation vs Wind Speed")
ax1.axhline(y=0.05, color="red", linestyle="--", label="Safety Threshold (0.05 mSv/day)")
ax1.legend()

# Biểu đồ 2: Radiation by Distance
ax2 = fig.add_subplot(gs[1])
distances = data["distance"].unique()
colors = sns.color_palette("hls", len(distances))
for i, distance in enumerate(distances):
    subset = data[data["distance"] == distance]
    mean_radiation = subset.groupby("wind_speed")["radiation_level"].mean()
    ax2.plot(mean_radiation.index, mean_radiation.values, label=f"{int(distance)} km", color=colors[i])

ax2.set_xlabel("Wind Speed (m/s)")
ax2.set_ylabel("Radiation Level (mSv/day)")
ax2.set_title("Radiation by Distance")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Biểu đồ 3: Line plot theo khoảng cách (với errorbar)
plt.figure(figsize=(10, 6))
sns.lineplot(x="distance", y="radiation_level", data=data, errorbar="sd", marker="o")
plt.xlabel("Distance from Fukushima Daiichi (km)")
plt.ylabel("Radiation Level (mSv/day)")
plt.title("Average Radiation Spread at Different Distances")
plt.grid(True)
plt.show()

# Biểu đồ 4: Scatter plot (Tốc độ gió vs Mức phóng xạ)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    data["wind_speed"],
    data["radiation_level"],
    c=data["distance"],
    cmap="viridis",
    s=data["rainfall"]*10,
    alpha=0.6
)
plt.colorbar(scatter, label="Distance (km)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Radiation Level (mSv/day)")
plt.title("Radiation Spread vs Wind Speed at Different Distances")
plt.show()

# Biểu đồ 5: Box plot (Phân bố mức phóng xạ theo khoảng cách)
plt.figure(figsize=(10, 6))
sns.boxplot(x="distance", y="radiation_level", data=data)
plt.xlabel("Distance from Ninh Thuan 2 (km)")
plt.ylabel("Radiation Level (mSv/day)")
plt.title("Radiation Level Distribution at Different Distances")
plt.show()

# Biểu đồ 6: Radiation theo thời gian (Cải tiến với errorbar)
plt.figure(figsize=(12, 6))
for distance in distances:
    subset = data[data["distance"] == distance]
    mean_radiation = subset.groupby("time_day")["radiation_level"].mean()
    std_radiation = subset.groupby("time_day")["radiation_level"].std()
    time_days = np.arange(subset["time_day"].min(), subset["time_day"].max() + 1)
    plt.plot(time_days, mean_radiation.reindex(time_days, fill_value=0), label=f"{int(distance)} km", marker="o")
    plt.fill_between(time_days, (mean_radiation - std_radiation).reindex(time_days, fill_value=0), 
                     (mean_radiation + std_radiation).reindex(time_days, fill_value=0), alpha=0.2)

plt.xlabel("Time (days)")
plt.ylabel("Radiation Level (mSv/day)")
plt.title("Radiation Spread Over Time at Different Distances")
plt.grid(True)
plt.legend(title="Distance (km)")
plt.axhline(y=0.05, color="red", linestyle="--", label="Safety Threshold (0.05 mSv/day)")
plt.legend()
plt.show()