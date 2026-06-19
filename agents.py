import os
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    groq_key = input("Enter your Groq API key: ")
os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Load model and data ──
def load_assets():
    assets = pd.read_csv("assets.csv")
    daily = pd.read_csv("daily_records.csv")
    df = daily.merge(assets, on="asset_id", how="left")
    model = XGBRegressor()
    model.load_model("model.json")
    return df, model

df, model = load_assets()

categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]
model_features = model.get_booster().feature_names

latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()
latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
for col in model_features:
    if col not in latest_encoded.columns:
        latest_encoded[col] = 0
latest_df["predicted_impact_ratio"] = model.predict(latest_encoded[model_features])
latest_df["predicted_impacted_mw"] = latest_df["predicted_impact_ratio"] * latest_df["dependable_capacity_mw"]

zone_summary = latest_df.groupby("operating_region").agg(
    total_capacity_mw=("dependable_capacity_mw", "sum"),
    predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
    num_generators=("asset_id", "count")
).reset_index()
zone_summary["risk_pct"] = (zone_summary["predicted_mw_at_risk"] / zone_summary["total_capacity_mw"]) * 100
zone_summary["risk_level"] = zone_summary["risk_pct"].apply(
    lambda x: "RED" if x > 65 else ("YELLOW" if x > 45 else "GREEN")
)

# ── State ──
class ReliabilityState(TypedDict):
    messages: Annotated[list, add_messages]
    zone: str
    risk_pct: float
    top_generators: str
    diagnosis: str
    brief: str
    approved: bool

# ── Agent 1: Monitor ──
def monitor_agent(state: ReliabilityState):
    zone = state["zone"]
    zone_data = zone_summary[zone_summary["operating_region"] == zone].iloc[0]
    risk_pct = zone_data["risk_pct"]
    mw_at_risk = zone_data["predicted_mw_at_risk"]
    num_gens = zone_data["num_generators"]
    risk_level = zone_data["risk_level"]

    zone_generators = latest_df[latest_df["operating_region"] == zone].sort_values(
        "predicted_impact_ratio", ascending=False
    ).head(3)

    top_gen_str = "\n".join([
        f"  - {row['asset_id']} ({row['fuel_category']}): impact ratio {row['predicted_impact_ratio']:.3f}, {row['predicted_impacted_mw']:.1f} MW at risk"
        for _, row in zone_generators.iterrows()
    ])

    response = llm.invoke(
        f"You are a Grid Monitoring Agent. Zone {zone} is at {risk_level} risk.\n"
        f"Risk: {risk_pct:.1f}% of capacity predicted at risk. Total MW at risk: {mw_at_risk:.1f} MW.\n"
        f"Number of generators in zone: {num_gens}.\n"
        f"Top at-risk generators:\n{top_gen_str}\n\n"
        f"In 2-3 sentences, summarize this alert for an operations team."
    )

    return {
        "messages": [response],
        "risk_pct": risk_pct,
        "top_generators": top_gen_str
    }

# ── Agent 2: Diagnosis ──
def diagnosis_agent(state: ReliabilityState):
    zone = state["zone"]
    monitor_summary = state["messages"][-1].content

    zone_generators = latest_df[latest_df["operating_region"] == zone].sort_values(
        "predicted_impact_ratio", ascending=False
    ).head(1)

    if len(zone_generators) == 0:
        return {"messages": [llm.invoke("No generators found.")], "diagnosis": "No data"}

    top_asset = zone_generators.iloc[0]
    asset_encoded = pd.get_dummies(
        latest_df[latest_df["asset_id"] == top_asset["asset_id"]],
        columns=categorical_cols
    )
    for col in model_features:
        if col not in asset_encoded.columns:
            asset_encoded[col] = 0

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(asset_encoded[model_features])
    shap_series = pd.Series(shap_values[0], index=model_features).abs().sort_values(ascending=False)
    top_features = shap_series.head(5)
    feature_str = ", ".join([f"{k.replace('_',' ')} ({v:.3f})" for k, v in top_features.items()])

    response = llm.invoke(
        f"You are a Diagnosis Agent for grid reliability.\n"
        f"Monitor reported: {monitor_summary}\n\n"
        f"The highest-risk generator is {top_asset['asset_id']} ({top_asset['fuel_category']}, "
        f"{top_asset['dependable_capacity_mw']:.1f} MW capacity).\n"
        f"Top SHAP-based risk drivers: {feature_str}.\n\n"
        f"In 2-3 sentences, explain the likely root cause of this zone's elevated risk."
    )

    return {"messages": [response], "diagnosis": response.content}

# ── Agent 3: Reporter ──
def reporter_agent(state: ReliabilityState):
    zone = state["zone"]
    monitor_summary = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
    diagnosis = state["diagnosis"]

    response = llm.invoke(
        f"You are a Reporting Agent. Draft a concise operator brief (3-4 sentences) for Zone {zone}.\n"
        f"Monitor summary: {monitor_summary}\n"
        f"Diagnosis: {diagnosis}\n\n"
        f"The brief should: state the risk level and MW at risk, explain the main cause, "
        f"and suggest one actionable next step. Write in a professional, factual tone."
    )

    return {"messages": [response], "brief": response.content}

# ── Build graph ──
graph = StateGraph(ReliabilityState)
graph.add_node("monitor", monitor_agent)
graph.add_node("diagnosis", diagnosis_agent)
graph.add_node("reporter", reporter_agent)

graph.set_entry_point("monitor")
graph.add_edge("monitor", "diagnosis")
graph.add_edge("diagnosis", "reporter")
graph.add_edge("reporter", END)

app = graph.compile()

# ── Test run ──
red_zones = zone_summary[zone_summary["risk_level"] == "RED"]["operating_region"].tolist()
test_zone = red_zones[0] if red_zones else zone_summary.iloc[0]["operating_region"]

print(f"\nRunning agent pipeline for Zone {test_zone}...\n")

result = app.invoke({
    "messages": [],
    "zone": test_zone,
    "risk_pct": 0.0,
    "top_generators": "",
    "diagnosis": "",
    "brief": "",
    "approved": False
})

print("MONITOR:")
print(result["messages"][0].content)
print("\nDIAGNOSIS:")
print(result["messages"][1].content)
print("\nOPERATOR BRIEF:")
print(result["messages"][2].content)