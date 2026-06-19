import pandas as pd
import numpy as np

np.random.seed(42)

ZONES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
FUEL_CATEGORIES = ["Gas", "Hydro", "Wind", "Steam", "Nuclear", "Solar"]
ASSET_CATEGORIES = {
    "Gas": "Peaker", "Steam": "Baseload", "Nuclear": "Baseload",
    "Hydro": "Baseload", "Wind": "Renewable", "Solar": "Renewable"
}

NUM_ASSETS = 50
assets = []
for i in range(1, NUM_ASSETS + 1):
    fuel = np.random.choice(FUEL_CATEGORIES, p=[0.35, 0.15, 0.15, 0.15, 0.05, 0.15])
    base_capacity = {
        "Gas": np.random.uniform(50, 400), "Steam": np.random.uniform(100, 600),
        "Nuclear": np.random.uniform(500, 1200), "Hydro": np.random.uniform(20, 300),
        "Wind": np.random.uniform(10, 150), "Solar": np.random.uniform(5, 80)
    }[fuel]

    if fuel in ["Gas", "Steam"]:
        summer_cap = base_capacity * np.random.uniform(0.88, 0.95)
        winter_cap = base_capacity * np.random.uniform(1.0, 1.05)
    elif fuel == "Solar":
        summer_cap = base_capacity * np.random.uniform(1.0, 1.08)
        winter_cap = base_capacity * np.random.uniform(0.75, 0.9)
    elif fuel == "Wind":
        summer_cap = base_capacity * np.random.uniform(0.85, 0.95)
        winter_cap = base_capacity * np.random.uniform(1.0, 1.1)
    else:
        summer_cap = base_capacity * np.random.uniform(0.95, 1.0)
        winter_cap = base_capacity * np.random.uniform(0.98, 1.02)

    if ASSET_CATEGORIES[fuel] == "Peaker":
        planned = np.random.randint(2, 7)
        unplanned = np.random.randint(1, 5)
    elif ASSET_CATEGORIES[fuel] == "Baseload":
        planned = np.random.randint(1, 4)
        unplanned = np.random.randint(0, 3)
    else:
        planned = np.random.randint(0, 3)
        unplanned = np.random.randint(0, 4)

    # NEW: distance to mapped weather station (km)
    distance_to_station_km = round(np.random.uniform(0.5, 45), 1)

    # NEW: a per-asset baseline "tendency" toward higher/lower impact
    # This creates a unit-level effect, some generators are just generally
    # more prone to impact than others, regardless of weather
    asset_baseline = np.random.beta(2, 2)  # spread 0-1, centered near 0.5

    assets.append({
        "asset_id": f"AST-{i:03d}",
        "summer_capacity_mw": round(summer_cap, 1),
        "winter_capacity_mw": round(winter_cap, 1),
        "fuel_category": fuel,
        "broad_asset_category": ASSET_CATEGORIES[fuel],
        "operating_region": np.random.choice(ZONES),
        "planned_event_count_ytd": planned,
        "unplanned_event_count_ytd": unplanned,
        "distance_to_station_km": distance_to_station_km,
        "asset_baseline": asset_baseline
    })

asset_df = pd.DataFrame(assets)

# ─────────────────────────────────────────────
# DAILY RECORDS
# ─────────────────────────────────────────────
NUM_DAYS = 730
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=NUM_DAYS)

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    if month in [3, 4, 5]: return "Spring"
    if month in [6, 7, 8]: return "Summer"
    return "Fall"

def get_capacity_for_season(asset, season):
    if season == "Summer":
        return asset["summer_capacity_mw"]
    elif season == "Winter":
        return asset["winter_capacity_mw"]
    else:
        return (asset["summer_capacity_mw"] + asset["winter_capacity_mw"]) / 2

COLD_SENSITIVITY = {"Gas": 0.15, "Steam": 0.10, "Nuclear": 0.05, "Hydro": 0.08, "Wind": 0.06, "Solar": 0.02}
HEAT_SENSITIVITY = {"Gas": 0.12, "Steam": 0.15, "Nuclear": 0.08, "Hydro": 0.05, "Wind": 0.04, "Solar": 0.10}

records = []

for _, asset in asset_df.iterrows():
    # Start prev_impact near the asset's baseline tendency
    prev_impact = asset["asset_baseline"]
    recent_impacts = [asset["asset_baseline"]] * 7
    days_since_event = 30

    for day_idx, date in enumerate(dates):
        month = date.month
        season = get_season(month)
        dependable_capacity = get_capacity_for_season(asset, season)

        seasonal_base_temp = {"Winter": 25, "Spring": 50, "Summer": 78, "Fall": 52}[season]
        temp_avg = seasonal_base_temp + np.random.normal(0, 8)
        temp_min = temp_avg - np.random.uniform(5, 12)
        temp_max = temp_avg + np.random.uniform(5, 12)
        temp_range = temp_max - temp_min
        humidity = np.clip(np.random.normal(60, 15), 10, 100)
        wind_speed = np.clip(np.random.normal(10, 5), 0, 40)
        pressure = np.random.normal(1013, 8)

        cold_day_flag = int(temp_avg < 20)
        hot_day_flag = int(temp_avg > 85)
        high_wind_flag = int(wind_speed > 20)
        high_humidity_flag = int(humidity > 85)

        recent_avg_impact = np.mean(recent_impacts)
        recent_max_impact = np.max(recent_impacts)
        recent_event_frequency = sum(1 for r in recent_impacts if r > 0.6)
        prior_high_impact_flag = int(prev_impact > 0.6)
        prior_zero_impact_flag = int(prev_impact < 0.02)

        # ── NEW impact_ratio logic ──
        # Base centers around the asset's own baseline tendency (creates spread across 0-1)
        base = asset["asset_baseline"] + np.random.normal(0, 0.12)

        # Weather effects (smaller relative to base now, since base dominates)
        cold_effect = cold_day_flag * COLD_SENSITIVITY[asset["fuel_category"]] * np.random.uniform(0.5, 2.0)
        heat_effect = hot_day_flag * HEAT_SENSITIVITY[asset["fuel_category"]] * np.random.uniform(0.5, 2.0)
        wind_effect = high_wind_flag * (0.10 if asset["fuel_category"] == "Wind" else 0.03) * np.random.uniform(0.5, 1.5)
        humidity_effect = high_humidity_flag * (0.06 if asset["fuel_category"] == "Solar" else 0.02)

        # Distance to station: farther = less reliable weather match = more noise/error
        distance_noise = np.random.normal(0, asset["distance_to_station_km"] / 300)

        # Recency persistence
        history_effect = (recent_avg_impact - asset["asset_baseline"]) * 0.25

        impact_ratio = base + cold_effect + heat_effect + wind_effect + humidity_effect + history_effect + distance_noise
        impact_ratio = np.clip(impact_ratio, 0, 1)

        impacted_mw = round(impact_ratio * dependable_capacity, 2)
        days_since_event = 0 if impact_ratio > 0.6 else days_since_event + 1

        records.append({
            "asset_id": asset["asset_id"],
            "date": date.strftime("%Y-%m-%d"),
            "season": season,
            "month": month,
            "day_of_year": date.dayofyear,
            "day_of_week": date.dayofweek,
            "dependable_capacity_mw": round(dependable_capacity, 1),
            "temp_min": round(temp_min, 1),
            "temp_max": round(temp_max, 1),
            "temp_avg": round(temp_avg, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "pressure": round(pressure, 1),
            "temp_range": round(temp_range, 1),
            "cold_day_flag": cold_day_flag,
            "hot_day_flag": hot_day_flag,
            "high_wind_flag": high_wind_flag,
            "high_humidity_flag": high_humidity_flag,
            "prev_impact_ratio": round(prev_impact, 4),
            "recent_avg_impact": round(recent_avg_impact, 4),
            "recent_max_impact": round(recent_max_impact, 4),
            "recent_event_frequency": recent_event_frequency,
            "days_since_last_event": days_since_event,
            "prior_high_impact_flag": prior_high_impact_flag,
            "prior_zero_impact_flag": prior_zero_impact_flag,
            "impact_ratio": round(impact_ratio, 4),
            "impacted_mw": impacted_mw
        })

        recent_impacts.append(impact_ratio)
        recent_impacts = recent_impacts[-7:]
        prev_impact = impact_ratio

daily_df = pd.DataFrame(records)

print("Impact ratio distribution:")
print(daily_df["impact_ratio"].describe())
print(f"\n% exactly 0: {(daily_df['impact_ratio'] == 0).mean()*100:.2f}%")
print(f"% above 0.5: {(daily_df['impact_ratio'] > 0.5).mean()*100:.2f}%")

asset_df.to_csv("assets.csv", index=False)
daily_df.to_csv("daily_records.csv", index=False)
print("\nSaved assets.csv and daily_records.csv")