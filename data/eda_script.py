"""
PawsIQ — EDA Notebook Script
==============================
This is the script version of eda.ipynb.
To convert to a Jupyter notebook:
    pip install jupytext
    jupytext --to notebook eda_script.py -o eda.ipynb

Or just run directly:
    python eda_script.py
"""

# %% [markdown]
# # PawsIQ — Exploratory Data Analysis
# **Dataset:** Synthetic · 2 years (Jan 2023 – Dec 2024) · NJ-based pet services
#
# **Goals of this notebook:**
# 1. Understand booking demand patterns by time (hour, day, month)
# 2. Validate that surge pricing signal is learnable
# 3. Profile service types and revenue distribution
# 4. Understand review/sentiment baseline before NLP modeling
# 5. Surface key numbers for the resume + README
#
# **Next step after this:** `ml/demand_forecast/train.ipynb`

# %% — Imports & config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# Style constants
CREAM = "#F7F5F0"; INK = "#141810"; SAGE = "#3D6B4F"
BARK = "#C4A882"; WARN = "#D4622A"; STONE = "#6B7063"; RULE = "#E4E1D8"

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor": RULE, "axes.labelcolor": STONE,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "xtick.color": STONE, "ytick.color": STONE,
    "grid.color": RULE, "grid.linewidth": 0.6,
    "font.family": "sans-serif",
})

# %% — Load data
df       = pd.read_csv("data/synthetic/bookings.csv",  parse_dates=["scheduled_at"])
reviews  = pd.read_csv("data/synthetic/reviews.csv")
walkers  = pd.read_csv("data/synthetic/walkers.csv")
pets     = pd.read_csv("data/synthetic/pets.csv")
clients  = pd.read_csv("data/synthetic/clients.csv")

completed = df[df["status"] == "completed"].copy()

print(f"Bookings:  {len(df):,}  |  Completed: {len(completed):,}  ({len(completed)/len(df)*100:.1f}%)")
print(f"Reviews:   {len(reviews):,}")
print(f"Clients:   {len(clients):,}  |  Walkers: {len(walkers):,}  |  Pets: {len(pets):,}")

# %% [markdown]
# ## 1. Dataset Overview

# %% — Summary stats
print("=== Booking Status ===")
print(df["status"].value_counts())
print("\n=== Service Type ===")
print(df["service_type"].value_counts())
print("\n=== Revenue Summary ===")
print(completed["final_price"].describe().round(2))

# %% [markdown]
# ## 2. Demand Patterns
# Key question: **can we predict booking volume from time features?**
# If hour-of-day and day-of-week are strong signals, XGBoost can learn them.

# %% — Chart 1: Demand by hour
hour_demand = completed.groupby("hour_of_day").size().reset_index(name="bookings")
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [WARN if h in (7,8,9,17,18,19) else SAGE for h in hour_demand["hour_of_day"]]
ax.bar(hour_demand["hour_of_day"], hour_demand["bookings"], color=colors, width=0.7, zorder=2)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Completed Bookings")
ax.set_title("Booking Demand by Hour of Day")
ax.set_xticks(range(6, 21))
ax.set_xticklabels([f"{h}:00" for h in range(6,21)], rotation=45, ha="right")
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()

# Insight
peak_pct = completed["is_peak_hour"].mean() * 100
print(f"\n>> {peak_pct:.1f}% of bookings fall in peak hours (7-9am, 5-7pm)")
print(f">> Peak avg price: ${completed[completed['is_peak_hour']==1]['final_price'].mean():.2f}")
print(f">> Off-peak avg price: ${completed[completed['is_peak_hour']==0]['final_price'].mean():.2f}")

# %% — Chart 2: Demand heatmap
heat = completed.groupby(["day_of_week","hour_of_day"]).size().unstack(fill_value=0)
heat = heat.reindex(columns=range(6,21), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 4))
cmap = sns.light_palette(SAGE, as_cmap=True)
sns.heatmap(heat, ax=ax, cmap=cmap, linewidths=0.3, linecolor=CREAM,
            yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            xticklabels=[f"{h}:00" for h in range(6,21)])
ax.set_title("Demand Heatmap — Hour × Day of Week")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# %% — Chart 3: Day of week
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_demand = completed.groupby("day_of_week").size().reindex(range(7), fill_value=0)
fig, ax = plt.subplots(figsize=(7, 4))
colors_dow = [SAGE if i < 5 else BARK for i in range(7)]
ax.bar(dow_labels, dow_demand.values, color=colors_dow, width=0.6, zorder=2)
ax.set_title("Booking Demand by Day of Week")
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
print(f"\n>> Weekday avg: {dow_demand[:5].mean():.0f} bookings/day")
print(f">> Weekend avg: {dow_demand[5:].mean():.0f} bookings/day")

# %% [markdown]
# ## 3. Revenue & Monthly Trends

# %% — Chart 4: Monthly revenue
completed["year_month"] = completed["scheduled_at"].dt.to_period("M")
monthly = completed.groupby("year_month").agg(
    revenue=("final_price","sum"), bookings=("booking_id","count")).reset_index()
monthly["ym_str"] = monthly["year_month"].astype(str)
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax2 = ax1.twinx()
ax1.fill_between(range(len(monthly)), monthly["revenue"], alpha=0.25, color=SAGE, zorder=2)
ax1.plot(range(len(monthly)), monthly["revenue"], color=SAGE, linewidth=2, zorder=3)
ax2.plot(range(len(monthly)), monthly["bookings"], color=BARK, linewidth=1.5, linestyle="--", zorder=3)
ax1.set_title("Monthly Revenue & Booking Volume (2023–2024)")
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly["ym_str"], rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Revenue ($)", color=SAGE); ax2.set_ylabel("Bookings", color=BARK)
plt.tight_layout(); plt.show()
print(f"\n>> Total 2-year revenue: ${completed['final_price'].sum():,.2f}")
print(f">> Avg monthly revenue: ${monthly['revenue'].mean():,.2f}")

# %% [markdown]
# ## 4. Surge Pricing Signal
# **This is the core feature for the dynamic pricing model.**
# We want to confirm surge multiplier varies meaningfully with time features.

# %% — Chart 5: Surge distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(completed["surge_multiplier"], bins=30, color=SAGE, edgecolor=CREAM, zorder=2)
axes[0].axvline(completed["surge_multiplier"].mean(), color=WARN, linewidth=1.5, linestyle="--",
                label=f'Mean ×{completed["surge_multiplier"].mean():.3f}')
axes[0].set_title("Surge Multiplier Distribution"); axes[0].legend(framealpha=0.0)
surge_by_hour = completed.groupby("hour_of_day")["surge_multiplier"].mean()
colors_s = [WARN if h in (7,8,9,17,18,19) else SAGE for h in surge_by_hour.index]
axes[1].bar(surge_by_hour.index, surge_by_hour.values, color=colors_s, width=0.7, zorder=2)
axes[1].axhline(1.0, color=STONE, linewidth=1, linestyle=":")
axes[1].set_title("Avg Surge Multiplier by Hour")
axes[1].set_xticks(range(6,21)); axes[1].set_xticklabels([str(h) for h in range(6,21)], rotation=45)
[ax.yaxis.grid(True, zorder=0) or ax.set_axisbelow(True) for ax in axes]
plt.tight_layout(); plt.show()
print(f"\n>> Surge range: x{completed['surge_multiplier'].min():.3f} — x{completed['surge_multiplier'].max():.3f}")
print(f">> Surge std dev: {completed['surge_multiplier'].std():.4f} (good spread for regression)")

# %% [markdown]
# ## 5. Service Type Analysis

# %% — Chart 6: Service breakdown
svc_labels = {"walk_30":"30-min Walk","walk_60":"60-min Walk","drop_in":"Drop-in","overnight":"Overnight"}
svc_counts = completed["service_type"].value_counts()
svc_rev    = completed.groupby("service_type")["final_price"].sum().reindex(svc_counts.index)
labels     = [svc_labels[s] for s in svc_counts.index]
palette    = [SAGE, "#5C8A6F", BARK, WARN]
fig, axes  = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].pie(svc_counts.values, labels=labels, autopct="%1.1f%%", colors=palette,
            startangle=140, wedgeprops={"edgecolor":CREAM,"linewidth":1.5},
            textprops={"fontsize":9})
axes[0].set_title("Bookings by Service Type")
axes[1].barh([svc_labels[s] for s in svc_rev.index], svc_rev.values, color=palette, zorder=2)
axes[1].set_title("Revenue by Service Type"); axes[1].xaxis.grid(True, zorder=0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Reviews & Sentiment Baseline

# %% — Chart 7: Ratings + sentiment
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
rating_counts = reviews["rating"].value_counts().sort_index()
bar_colors = [WARN if r <= 2 else (BARK if r == 3 else SAGE) for r in rating_counts.index]
axes[0].bar([f"{r}★" for r in rating_counts.index], rating_counts.values, color=bar_colors, width=0.6, zorder=2)
axes[0].set_title("Review Rating Distribution"); axes[0].yaxis.grid(True, zorder=0)
sent_counts  = reviews["sentiment_label"].value_counts()
sent_colors  = {"positive":SAGE,"neutral":BARK,"negative":WARN}
axes[1].pie(sent_counts.values, labels=[s.capitalize() for s in sent_counts.index],
            autopct="%1.1f%%", colors=[sent_colors[s] for s in sent_counts.index],
            startangle=90, wedgeprops={"edgecolor":CREAM,"linewidth":1.5},
            textprops={"fontsize":9})
axes[1].set_title("Review Sentiment Distribution")
plt.tight_layout(); plt.show()
print(f"\n>> Avg rating: {reviews['rating'].mean():.2f} / 5.0")
print(f">> Positive reviews: {(reviews['sentiment_label']=='positive').mean()*100:.1f}%")
print(f">> Negative reviews: {(reviews['sentiment_label']=='negative').mean()*100:.1f}%")

# %% [markdown]
# ## 7. Walker Performance

# %% — Chart 8: Walker stats
walker_stats = completed.groupby("walker_id").agg(
    walks=("booking_id","count"), revenue=("final_price","sum")).reset_index()
walker_stats = walker_stats.merge(walkers[["user_id","name","rating"]], left_on="walker_id", right_on="user_id")
walker_stats = walker_stats.sort_values("walks", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(walker_stats["name"], walker_stats["walks"], color=SAGE, zorder=2)
axes[0].set_title("Walker Volume"); axes[0].xaxis.grid(True, zorder=0)
bar_colors_r = [WARN if r < 4.5 else SAGE for r in walker_stats["rating"]]
axes[1].barh(walker_stats["name"], walker_stats["rating"], color=bar_colors_r, zorder=2)
axes[1].set_xlim(4.0, 5.1); axes[1].set_title("Walker Ratings"); axes[1].xaxis.grid(True, zorder=0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Feature Correlation (ML Readiness Check)
# Confirming our key features correlate with demand before training.

# %% — Correlation check
ml_features = completed[["hour_of_day","day_of_week","month","is_peak_hour","is_weekend","surge_multiplier","final_price"]].copy()
corr = ml_features.corr()
fig, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(10, 145, as_cmap=True)
sns.heatmap(corr, ax=ax, cmap=cmap, center=0, annot=True, fmt=".2f",
            square=True, linewidths=0.5, linecolor=CREAM,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Summary: Key Numbers for Resume & README
#
# | Metric | Value |
# |--------|-------|
# | Total bookings (2 yrs) | 9,935 |
# | Completion rate | 93.4% |
# | Peak-hour bookings | 58.4% of total |
# | Peak vs off-peak price | $27.94 vs $23.21 (+20.4%) |
# | Total simulated revenue | $240,963 |
# | Avg surge multiplier | ×1.187 |
# | Total reviews | 6,672 |
# | Avg rating | 4.46 / 5.0 |
# | Positive sentiment | 87.7% |
#
# **ML readiness:** `is_peak_hour`, `hour_of_day`, and `month` all show
# meaningful correlation with `final_price` and booking volume.
# XGBoost demand model and Ridge pricing model are ready to train.
#
# **Next:** `ml/demand_forecast/train.ipynb`

# %% — Save features for ML
ml_ready = completed[[
    "booking_id","hour_of_day","day_of_week","month",
    "is_peak_hour","is_weekend","zip","service_type",
    "surge_multiplier","final_price","base_price"
]].copy()
os.makedirs("data/processed", exist_ok=True)
ml_ready.to_csv("data/processed/ml_features.csv", index=False)
print(f"\nSaved ml_features.csv — {len(ml_ready):,} rows, {len(ml_ready.columns)} columns")
print("Ready for ml/demand_forecast/train.ipynb")
