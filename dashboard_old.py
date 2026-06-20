import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go

st.set_page_config(
    page_title="Grid Reliability Dashboard",
    layout="wide",
    page_icon="💡"
)

def load_css(filepath):
    with open(filepath) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


@st.cache_data
def load_data():
    assets = pd.read_csv("assets.csv")
    daily = pd.read_csv("daily_records.csv")
    df = daily.merge(assets, on="asset_id", how="left")
    return assets, daily, df

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("model.json")
    return model

assets, daily, df = load_data()
model = load_model()

categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]
model_features = model.get_booster().feature_names

st.sidebar.markdown("### 💡 Grid Reliability")
st.sidebar.markdown("**Date:** " + df["date"].max())
st.sidebar.markdown("**Generators:** 50 synthetic assets")
st.sidebar.markdown("**Zones:** 11 NY load zones")
st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Thresholds**")
st.sidebar.markdown("🔴 RED: > 65% capacity at risk")
st.sidebar.markdown("🟡 YELLOW: 45–65% at risk")
st.sidebar.markdown("🟢 GREEN: < 45% at risk")
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic data for demo purposes. No real operational data used.")

st.title("💡 Grid Reliability Dashboard")
st.caption("XGBoost-powered capacity-impact predictions across NY's 11 load zones · Synthetic data")
st.markdown("---")

latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()
latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
for col in model_features:
    if col not in latest_encoded.columns:
        latest_encoded[col] = 0
X_latest = latest_encoded[model_features]
predicted_ratio = model.predict(X_latest)
latest_df["predicted_impact_ratio"] = predicted_ratio
latest_df["predicted_impacted_mw"] = predicted_ratio * latest_df["dependable_capacity_mw"]

zone_summary = latest_df.groupby("operating_region").agg(
    total_capacity_mw=("dependable_capacity_mw", "sum"),
    predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
    num_generators=("asset_id", "count")
).reset_index()
zone_summary["risk_pct"] = (zone_summary["predicted_mw_at_risk"] / zone_summary["total_capacity_mw"]) * 100

def get_risk_color(pct):
    if pct > 65: return "🔴 RED"
    elif pct > 45: return "🟡 YELLOW"
    else: return "🟢 GREEN"

zone_summary["risk_level"] = zone_summary["risk_pct"].apply(get_risk_color)
color_map = {"🔴 RED": "#DC2626", "🟡 YELLOW": "#D97706", "🟢 GREEN": "#16A34A"}

zone_coords = {
    "A": (42.9, -78.9), "B": (43.15, -77.6), "C": (43.05, -76.15),
    "D": (44.0, -75.5), "E": (43.1, -75.2), "F": (42.65, -73.75),
    "G": (41.7, -73.9), "H": (41.2, -73.8), "I": (40.95, -73.85),
    "J": (40.71, -74.0), "K": (40.8, -73.1)
}

zone_names = {
    "A": "West", "B": "Genesee", "C": "Central", "D": "North",
    "E": "Mohawk Valley", "F": "Capital", "G": "Hudson Valley",
    "H": "Millwood", "I": "Dunwoodie", "J": "NYC", "K": "Long Island"
}

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Capacity", f"{zone_summary['total_capacity_mw'].sum():.0f} MW")
m2.metric("Total MW at Risk", f"{zone_summary['predicted_mw_at_risk'].sum():.0f} MW")
m3.metric("Zones at RED", f"{(zone_summary['risk_level'] == '🔴 RED').sum()} / 11")
m4.metric("Avg Impact Ratio", f"{latest_df['predicted_impact_ratio'].mean():.3f}")

st.markdown("---")
st.subheader("NY Zone Risk Map")

def build_map(summary, title):
    fig = go.Figure()
    for _, row in summary.iterrows():
        zone = row["operating_region"]
        if zone not in zone_coords:
            continue
        lat, lon = zone_coords[zone]
        color = color_map[row["risk_level"]]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode="markers+text",
            marker=dict(size=35, color=color, line=dict(width=2, color="white")),
            text=[zone],
            textfont=dict(size=14, color="white", family="Arial Black"),
            hovertext=f"Zone {zone} ({zone_names.get(zone,'')})<br>Risk Level: {row['risk_level']}<br>Risk %: {row['risk_pct']:.1f}%<br>MW at Risk: {row['predicted_mw_at_risk']:.1f}<br>Generators: {row['num_generators']}",
            hoverinfo="text",
            showlegend=False
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#1A2332", size=16)),
        geo=dict(
    scope="usa",
    projection=dict(type="albers usa"),
    center=dict(lat=42.8, lon=-75.5),
    projection_scale=12,
    showland=True,
    landcolor="#EDEFF3",
    showsubunits=True,
    subunitcolor="#C5CAD4",
    bgcolor="#F7F8FA"
),
height=500,
paper_bgcolor="#F7F8FA",
font=dict(color="#1A2332", family="IBM Plex Sans"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

st.plotly_chart(build_map(zone_summary, f"Zone Risk · {latest_date}"), use_container_width=True)

st.markdown("---")
st.subheader("🌨️ Cold Snap Simulator")
st.caption("Adjust forecast temperature to see how predicted zone risk changes in real time")

temp_override = st.slider("Forecast Temperature (°F)", min_value=-10, max_value=100, value=45, step=1)

sim_df = latest_df.copy()
sim_df["temp_avg"] = temp_override
sim_df["temp_min"] = temp_override - 8
sim_df["temp_max"] = temp_override + 8
sim_df["temp_range"] = 16
sim_df["cold_day_flag"] = int(temp_override < 20)
sim_df["hot_day_flag"] = int(temp_override > 85)

sim_encoded = pd.get_dummies(sim_df, columns=categorical_cols)
for col in model_features:
    if col not in sim_encoded.columns:
        sim_encoded[col] = 0
X_sim = sim_encoded[model_features]
sim_df["predicted_impact_ratio"] = model.predict(X_sim)
sim_df["predicted_impacted_mw"] = sim_df["predicted_impact_ratio"] * sim_df["dependable_capacity_mw"]

sim_zone_summary = sim_df.groupby("operating_region").agg(
    total_capacity_mw=("dependable_capacity_mw", "sum"),
    predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
    num_generators=("asset_id", "count")
).reset_index()
sim_zone_summary["risk_pct"] = (sim_zone_summary["predicted_mw_at_risk"] / sim_zone_summary["total_capacity_mw"]) * 100
sim_zone_summary["risk_level"] = sim_zone_summary["risk_pct"].apply(get_risk_color)

s1, s2, s3 = st.columns(3)
s1.metric("Avg Impact Ratio", f"{sim_df['predicted_impact_ratio'].mean():.3f}",
          delta=f"{sim_df['predicted_impact_ratio'].mean() - latest_df['predicted_impact_ratio'].mean():.3f}")
s2.metric("Total MW at Risk", f"{sim_df['predicted_impacted_mw'].sum():.0f} MW",
          delta=f"{sim_df['predicted_impacted_mw'].sum() - latest_df['predicted_impacted_mw'].sum():.0f} MW")
s3.metric("Zones at RED", f"{(sim_zone_summary['risk_level'] == '🔴 RED').sum()} / 11",
          delta=f"{(sim_zone_summary['risk_level'] == '🔴 RED').sum() - (zone_summary['risk_level'] == '🔴 RED').sum()}")

st.plotly_chart(build_map(sim_zone_summary, f"Simulated Zone Risk at {temp_override}°F"), use_container_width=True)

st.markdown("---")
st.subheader("🔍 Zone Drill-Down")

selected_zone = st.selectbox(
    "Select a zone to inspect:",
    options=sorted(zone_summary["operating_region"].tolist()),
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')} · {zone_summary[zone_summary['operating_region']==z]['risk_level'].values[0]}"
)

zone_generators = latest_df[latest_df["operating_region"] == selected_zone].copy()
zone_gen_encoded = pd.get_dummies(zone_generators, columns=categorical_cols)
for col in model_features:
    if col not in zone_gen_encoded.columns:
        zone_gen_encoded[col] = 0
zone_generators["predicted_impact_ratio"] = model.predict(zone_gen_encoded[model_features])
zone_generators["predicted_impacted_mw"] = zone_generators["predicted_impact_ratio"] * zone_generators["dependable_capacity_mw"]
zone_generators["risk_level"] = zone_generators["predicted_impact_ratio"].apply(
    lambda x: "🔴 HIGH" if x > 0.65 else ("🟡 MODERATE" if x > 0.45 else "🟢 LOW")
)

zs = zone_summary[zone_summary["operating_region"] == selected_zone].iloc[0]
zc1, zc2, zc3 = st.columns(3)
zc1.metric("Zone Risk Level", zs["risk_level"])
zc2.metric("Zone MW at Risk", f"{zs['predicted_mw_at_risk']:.1f} MW")
zc3.metric("Generators in Zone", str(int(zs["num_generators"])))

display_cols = ["asset_id", "fuel_category", "broad_asset_category",
                "dependable_capacity_mw", "predicted_impact_ratio",
                "predicted_impacted_mw", "risk_level", "recent_avg_impact"]

st.dataframe(
    zone_generators[display_cols].sort_values("predicted_impact_ratio", ascending=False).round(3),
    use_container_width=True
)
st.markdown("---")
st.subheader("🔬 Generator Risk Explanation")
st.caption("Select a generator to understand why the model predicted its risk level")

selected_asset = st.selectbox(
    "Select a generator:",
    options=zone_generators["asset_id"].tolist(),
    format_func=lambda a: f"{a} · {zone_generators[zone_generators['asset_id']==a]['fuel_category'].values[0]} · {zone_generators[zone_generators['asset_id']==a]['risk_level'].values[0]}"
)

asset_row = zone_generators[zone_generators["asset_id"] == selected_asset].iloc[0]
asset_encoded = zone_gen_encoded[zone_generators["asset_id"] == selected_asset][model_features]

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(asset_encoded)
shap_series = pd.Series(shap_values[0], index=model_features).abs().sort_values(ascending=False)
top_features = shap_series.head(5)

# Plain English explanation
def explain_feature(fname, fval_series):
    val = asset_encoded[fname].values[0] if fname in asset_encoded.columns else 0
    explanations = {
        "recent_avg_impact": f"recent average impact ratio is high ({val:.3f})" if val > 0.4 else f"recent average impact is moderate ({val:.3f})",
        "prev_impact_ratio": f"previous day impact was high ({val:.3f})" if val > 0.4 else f"previous day impact was low ({val:.3f})",
        "cold_day_flag": "today is a cold stress day (temp < 20°F)" if val == 1 else "no cold stress today",
        "hot_day_flag": "today is a heat stress day (temp > 85°F)" if val == 1 else "no heat stress today",
        "high_wind_flag": "high wind conditions detected" if val == 1 else "wind within normal range",
        "temp_avg": f"average temperature is {val:.1f}°F",
        "dependable_capacity_mw": f"generator capacity is {val:.1f} MW",
        "days_since_last_event": f"{int(val)} days since last high-impact event",
        "recent_max_impact": f"recent peak impact ratio was {val:.3f}",
        "prior_high_impact_flag": "prior high-impact event on record" if val == 1 else "no prior high-impact flag",
    }
    return explanations.get(fname, f"{fname.replace('_', ' ')} = {val:.3f}")

explanation_lines = []
for fname in top_features.index:
    explanation_lines.append(f"• {explain_feature(fname, asset_encoded)}")

risk_emoji = "🔴" if asset_row["risk_level"] == "🔴 HIGH" else ("🟡" if asset_row["risk_level"] == "🟡 MODERATE" else "🟢")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0A1628,#050510);border:1px solid #00F5FF44;
border-radius:10px;padding:20px;margin-top:10px;">
<h4 style="color:#00F5FF;font-family:Orbitron,monospace;margin:0 0 10px 0;">
{risk_emoji} {selected_asset} · {asset_row['fuel_category']} · {asset_row['broad_asset_category']}
</h4>
<p style="color:#8ab0c8;font-size:0.85rem;margin:0 0 12px 0;">
Predicted Impact Ratio: <span style="color:#00F5FF;font-weight:bold;">{asset_row['predicted_impact_ratio']:.3f}</span> · 
Predicted MW at Risk: <span style="color:#FFE600;font-weight:bold;">{asset_row['predicted_impacted_mw']:.1f} MW</span>
</p>
<p style="color:#aaa;font-size:0.8rem;margin:0 0 8px 0;font-style:italic;">Top 5 risk drivers:</p>
{"".join(f'<p style="color:#c8d8e8;font-size:0.85rem;margin:4px 0;">'+line+'</p>' for line in explanation_lines)}
</div>
""", unsafe_allow_html=True)
# ============================================================
# MULTI-AGENT RISK ANALYSIS SECTION
# LangGraph pipeline with conditional routing:
#   GREEN  -> Monitor only, stop
#   YELLOW -> Monitor + Diagnosis, stop (no report, not urgent yet)
#   RED    -> Monitor + Diagnosis + Reporter (full escalation)
# ============================================================

st.markdown("---")
st.subheader("🤖 Multi-Agent Risk Analysis")
st.caption("LangGraph pipeline: Monitor → Diagnosis → Reporter agents, with conditional routing based on zone risk level")

groq_key = st.text_input("Enter Groq API Key to activate agents:", type="password")

if st.button("⚡ Run Agent Pipeline for Zone " + selected_zone):
    if not groq_key:
        st.warning("Please enter your Groq API key above.")
    else:
        import os
        os.environ["GROQ_API_KEY"] = groq_key

        from langchain_groq import ChatGroq
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages
        from typing import TypedDict, Annotated
        import shap

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        # --- STATE SCHEMA ---
        # This defines what data flows between agents in the graph.
        # risk_level was added so the routing functions below can read it.
        class ReliabilityState(TypedDict):
            messages: Annotated[list, add_messages]
            zone: str
            risk_level: str
            risk_pct: float
            top_generators: str
            diagnosis: str
            brief: str

        # --- AGENT NODES ---
        # Each node function: reads state, does work (SHAP / LLM call), returns updates to state.

        def monitor_agent(state):
            zone = state["zone"]
            zone_data = zone_summary[zone_summary["operating_region"] == zone].iloc[0]
            risk_pct = zone_data["risk_pct"]
            mw_at_risk = zone_data["predicted_mw_at_risk"]
            risk_level = zone_data["risk_level"]
            top_gens = latest_df[latest_df["operating_region"] == zone].sort_values(
                "predicted_impact_ratio", ascending=False).head(3)
            top_gen_str = "\n".join([
                f"  - {r['asset_id']} ({r['fuel_category']}): impact {r['predicted_impact_ratio']:.3f}, {r['predicted_impacted_mw']:.1f} MW"
                for _, r in top_gens.iterrows()
            ])
            response = llm.invoke(
                f"You are a Grid Monitoring Agent. Zone {zone} is at {risk_level} risk.\n"
                f"Risk: {risk_pct:.1f}% of capacity at risk. MW at risk: {mw_at_risk:.1f}.\n"
                f"Top generators:\n{top_gen_str}\n\n"
                f"Summarize this alert in 2-3 sentences for an operations team."
            )
            return {"messages": [response], "risk_pct": risk_pct, "top_generators": top_gen_str}

        def diagnosis_agent(state):
            zone = state["zone"]
            monitor_summary = state["messages"][-1].content
            top_asset = latest_df[latest_df["operating_region"] == zone].sort_values(
                "predicted_impact_ratio", ascending=False).iloc[0]
            asset_enc = pd.get_dummies(
                latest_df[latest_df["asset_id"] == top_asset["asset_id"]],
                columns=categorical_cols)
            for col in model_features:
                if col not in asset_enc.columns:
                    asset_enc[col] = 0
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(asset_enc[model_features])
            shap_series = pd.Series(shap_vals[0], index=model_features).abs().sort_values(ascending=False)
            feature_str = ", ".join([f"{k.replace('_',' ')} ({v:.3f})" for k, v in shap_series.head(5).items()])
            response = llm.invoke(
                f"You are a Diagnosis Agent.\n"
                f"Monitor reported: {monitor_summary}\n"
                f"Highest-risk generator: {top_asset['asset_id']} ({top_asset['fuel_category']}, {top_asset['dependable_capacity_mw']:.1f} MW).\n"
                f"Top SHAP risk drivers: {feature_str}.\n\n"
                f"In 2-3 sentences, explain the likely root cause."
            )
            return {"messages": [response], "diagnosis": response.content}

        def reporter_agent(state):
            zone = state["zone"]
            monitor_summary = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
            diagnosis = state["diagnosis"]
            response = llm.invoke(
                f"You are a Reporting Agent. Draft a concise operator brief for Zone {zone}.\n"
                f"Monitor: {monitor_summary}\nDiagnosis: {diagnosis}\n\n"
                f"State risk level and MW at risk, explain main cause, suggest one next step. Professional tone, 3-4 sentences."
            )
            return {"messages": [response], "brief": response.content}

        # --- ROUTING FUNCTIONS ---
        # These decide which node runs next, based on risk_level in state.
        # This is the actual "agentic" decision logic, not a fixed chain.

        def route_after_monitor(state):
            risk_level = state["risk_level"]
            if "GREEN" in risk_level:
                return "stop"            # healthy zone, no need to investigate further
            elif "YELLOW" in risk_level:
                return "diagnose_only"   # elevated, find root cause, but not urgent enough to report
            else:
                return "full"            # RED, full escalation path

        def route_after_diagnosis(state):
            if "RED" in state["risk_level"]:
                return "full"            # RED continues to Reporter
            else:
                return "stop"            # YELLOW stops here, diagnosis was enough

        # --- BUILD THE GRAPH ---
        graph = StateGraph(ReliabilityState)
        graph.add_node("monitor", monitor_agent)
        graph.add_node("diagnosis", diagnosis_agent)
        graph.add_node("reporter", reporter_agent)

        graph.set_entry_point("monitor")

        # Branch 1: after Monitor, decide whether to stop, diagnose only, or go full
        graph.add_conditional_edges(
            "monitor",
            route_after_monitor,
            {
                "stop": END,
                "diagnose_only": "diagnosis",
                "full": "diagnosis"
            }
        )

        # Branch 2: after Diagnosis, decide whether to continue to Reporter or stop
        graph.add_conditional_edges(
            "diagnosis",
            route_after_diagnosis,
            {
                "full": "reporter",
                "stop": END
            }
        )

        graph.add_edge("reporter", END)

        pipeline = graph.compile()

        # --- RUN THE PIPELINE ---
        zone_risk_level = zone_summary[zone_summary["operating_region"] == selected_zone]["risk_level"].values[0]

        with st.spinner("Running agent pipeline..."):
            result = pipeline.invoke({
                "messages": [], "zone": selected_zone,
                "risk_level": zone_risk_level,
                "risk_pct": 0.0, "top_generators": "",
                "diagnosis": "", "brief": ""
            })

        # --- EXTRACT RESULTS SAFELY ---
        # Message count now varies by path: GREEN=1, YELLOW=2, RED=3.
        # len() checks prevent IndexError on shorter paths.
        messages = result["messages"]
        monitor_out = messages[0].content if len(messages) > 0 else None
        diagnosis_out = messages[1].content if len(messages) > 1 else None
        brief_out = messages[2].content if len(messages) > 2 else None

        # --- DISPLAY: tell the user which path was taken ---
        if "GREEN" in zone_risk_level:
            st.info("🟢 Zone is GREEN. Monitor ran, no further investigation needed. Pipeline stopped here.")
        elif "YELLOW" in zone_risk_level:
            st.info("🟡 Zone is YELLOW. Monitor + Diagnosis ran. No formal report generated, not yet urgent.")
        else:
            st.info("🔴 Zone is RED. Full pipeline ran: Monitor + Diagnosis + Reporter.")

        # --- DISPLAY: Monitor output (always present) ---
        if monitor_out:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0A1628,#050510);border:1px solid #00F5FF44;
            border-radius:10px;padding:20px;margin:10px 0;">
            <p style="color:#00F5FF88;font-size:0.7rem;letter-spacing:0.1em;margin:0 0 6px 0;">MONITOR AGENT</p>
            <p style="color:#c8d8e8;font-size:0.9rem;margin:0;">{monitor_out}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- DISPLAY: Diagnosis output (YELLOW and RED only) ---
        if diagnosis_out:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0A1628,#050510);border:1px solid #FFE60044;
            border-radius:10px;padding:20px;margin:10px 0;">
            <p style="color:#FFE60088;font-size:0.7rem;letter-spacing:0.1em;margin:0 0 6px 0;">DIAGNOSIS AGENT</p>
            <p style="color:#c8d8e8;font-size:0.9rem;margin:0;">{diagnosis_out}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- DISPLAY: Reporter output + approval (RED only) ---
        if brief_out:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0A1628,#050510);border:1px solid #FF2D5544;
            border-radius:10px;padding:20px;margin:10px 0;">
            <p style="color:#FF2D5588;font-size:0.7rem;letter-spacing:0.1em;margin:0 0 6px 0;">OPERATOR BRIEF (PENDING APPROVAL)</p>
            <p style="color:#c8d8e8;font-size:0.9rem;margin:0;">{brief_out}</p>
            </div>
            """, unsafe_allow_html=True)

            approved = st.checkbox("✅ Approve and publish this brief")
            if approved:
                st.success(f"Brief approved for Zone {selected_zone}. Ready to distribute to operations team.")