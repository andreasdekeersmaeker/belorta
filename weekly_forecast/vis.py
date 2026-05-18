import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# -----------------------
# Data
# -----------------------
years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
train_weeks = np.array([48, 96, 149, 198, 247, 299, 347, 395, 448])

mae = np.array([142575, 72204, 112059, 69077, 77701, 98859, 90009, 66277, 101793])
mape = np.array([34.2, 31.4, 46.8, 28.7, 22.9, 32.0, 24.5, 22.3, 23.5])

# -----------------------
# Style
# -----------------------
plt.style.use("default")

fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
ax.set_facecolor("#FAFAFA")

# Remove clutter
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

# Light grid
ax.grid(axis="y", alpha=0.12, linewidth=1.5)
ax.set_axisbelow(True)

# -----------------------
# MAIN STORY: MAE
# -----------------------
ax.plot(
    years,
    mae,
    linewidth=4,
    marker="o",
    markersize=10,
)

# soft fill underneath
ax.fill_between(
    years,
    mae,
    alpha=0.12
)

# Highlight last point
ax.scatter(years[-1], mae[-1], s=250, zorder=5)

# -----------------------
# Secondary metric: MAPE
# -----------------------
ax2 = ax.twinx()

for spine in ["top", "right", "left", "bottom"]:
    ax2.spines[spine].set_visible(False)

ax2.plot(
    years,
    mape,
    linewidth=2.5,
    linestyle="--",
    marker="o",
    alpha=0.7
)

# -----------------------
# Labels / annotations
# -----------------------
for x, y in zip(years, mae):
    ax.text(
        x,
        y + 3500,
        f"{round(y/1000):.0f}k",
        ha="center",
        fontsize=11,
        fontweight="bold",
        alpha=0.85
    )

# elegant x-axis labels
xticks = [
    f"{y}\n{w} weeks"
    for y, w in zip(years, train_weeks)
]

ax.set_xticks(years)
ax.set_xticklabels(xticks, fontsize=11)

# y-axis formatting
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
)

ax2.yaxis.set_major_formatter(
    mticker.PercentFormatter()
)

# Remove tick marks
ax.tick_params(axis='both', length=0)
ax2.tick_params(axis='both', length=0)

# -----------------------
# Titles
# -----------------------
fig.suptitle(
    "Model Learning Evolution",
    fontsize=24,
    fontweight="bold",
    y=0.96
)

ax.set_title(
    "Walk-forward validation performance as training history increases",
    fontsize=13,
    pad=20,
    alpha=0.75
)

# Metric labels directly on chart
ax.text(
    years[-1] + 0.15,
    mae[-1],
    "MAE (kg)",
    fontsize=12,
    fontweight="bold"
)

ax2.text(
    years[-1] + 0.15,
    mape[-1],
    "MAPE active",
    fontsize=11,
    alpha=0.7
)

# clean layout
plt.tight_layout()
plt.show()