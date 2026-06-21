# ============================================================
# PAGE: EXECUTIVE SUMMARY
# Runs the LangGraph multi-agent pipeline (Monitor -> Diagnosis ->
# Reporter) for a selected zone, with conditional routing based on
# risk level:
#   GREEN  -> Monitor only, stop
#   YELLOW -> Monitor + Diagnosis, stop (not urgent enough to report)
#   RED    -> Monitor + Diagnosis + Reporter (full escalation)
#
# Groq API key is read from a plain OS environment variable.
# Locally: set GROQ_API_KEY in Windows environment variables.
# On Render: set GROQ_API_KEY in the service's Environment tab.
# No st.secrets / secrets.toml dependency at all, removes that
# entire failure category.
# ============================================================

import streamlit as st
import pandas as pd
import os
from shared import (
    load_css, get_latest_predictions, zone_names, categorical_cols
)

# --- LOAD CSS ---
load_css("styles.css")

# --- LOAD DATA / MODEL / PREDICTIONS ---
assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_latest_predictions()

# --- SIDEBAR: FOOTER ---
st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

# --- PAGE HEADER ---
st.title("📋 Executive Summary")
st.caption("Multi-agent analysis: Monitor, Diagnosis, and Reporter agents, with conditional routing by risk level")
st.markdown("---")

# --- ZONE SELECTOR ---
zone_options = sorted(zone_summary["operating_region"].tolist())
default_zone = st.session_state.get("selected_zone", zone_options[0])
default_index = zone_options.index(default_zone) if default_zone in zone_options else 0

selected_zone = st.selectbox(
    "Select a zone to analyze:",
    options=zone_options,
    index=default_index,
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')} · {zone_summary[zone_summary['operating_region']==z]['risk_level'].values[0]}"
)
st.session_state["selected_zone"] = selected_zone

st.markdown("---")

# --- CHECK FOR API KEY ---
groq_key = os.environ.get("GROQ_API_KEY")

if not groq_key:
    st.warning(
        "No Groq API key found. Set GROQ_API_KEY as an environment variable "
        "(locally in Windows env vars, or in Render's Environment tab)."
    )
else:
    if st.button("⚡ Run Agent Pipeline for Zone " + selected_zone):
        from langchain_groq import ChatGroq
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages
        from typing import TypedDict, Annotated
        import shap
        from knowledge_base import retrieve_relevant_knowledge

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_key)

        class ReliabilityState(TypedDict):
            messages: Annotated[list, add_messages]
            zone: str
            risk_level: str
            risk_pct: float
            top_generators: str
            diagnosis: str
            brief: str

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

            rag_query = f"{top_asset['fuel_category']} generator risk factors: {feature_str}"
            retrieved_knowledge = retrieve_relevant_knowledge(rag_query, n_results=2)

            response = llm.invoke(
                f"You are a Diagnosis Agent.\n"
                f"Monitor reported: {monitor_summary}\n"
                f"Highest-risk generator: {top_asset['asset_id']} ({top_asset['fuel_category']}, {top_asset['dependable_capacity_mw']:.1f} MW).\n"
                f"Top SHAP risk drivers: {feature_str}.\n\n"
                f"Relevant operational guidance:\n{retrieved_knowledge}\n\n"
                f"Using the SHAP drivers AND the operational guidance above, explain the likely "
                f"root cause in 2-3 sentences. Ground your explanation in the retrieved guidance "
                f"where it applies, rather than general assumptions."
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

        def route_after_monitor(state):
            risk_level = state["risk_level"]
            if "GREEN" in risk_level:
                return "stop"
            elif "YELLOW" in risk_level:
                return "diagnose_only"
            else:
                return "full"

        def route_after_diagnosis(state):
            if "RED" in state["risk_level"]:
                return "full"
            else:
                return "stop"

        graph = StateGraph(ReliabilityState)
        graph.add_node("monitor", monitor_agent)
        graph.add_node("diagnosis", diagnosis_agent)
        graph.add_node("reporter", reporter_agent)

        graph.set_entry_point("monitor")
        graph.add_conditional_edges(
            "monitor", route_after_monitor,
            {"stop": END, "diagnose_only": "diagnosis", "full": "diagnosis"}
        )
        graph.add_conditional_edges(
            "diagnosis", route_after_diagnosis,
            {"full": "reporter", "stop": END}
        )
        graph.add_edge("reporter", END)

        pipeline = graph.compile()

        zone_risk_level = zone_summary[zone_summary["operating_region"] == selected_zone]["risk_level"].values[0]

        with st.spinner("Running agent pipeline..."):
            result = pipeline.invoke({
                "messages": [], "zone": selected_zone,
                "risk_level": zone_risk_level,
                "risk_pct": 0.0, "top_generators": "",
                "diagnosis": "", "brief": ""
            })

        messages = result["messages"]
        monitor_out = messages[0].content if len(messages) > 0 else None
        diagnosis_out = messages[1].content if len(messages) > 1 else None
        brief_out = messages[2].content if len(messages) > 2 else None

        if "GREEN" in zone_risk_level:
            st.info("🟢 Zone is GREEN. Monitor ran, no further investigation needed. Pipeline stopped here.")
        elif "YELLOW" in zone_risk_level:
            st.info("🟡 Zone is YELLOW. Monitor + Diagnosis ran. No formal report generated, not yet urgent.")
        else:
            st.info("🔴 Zone is RED. Full pipeline ran: Monitor + Diagnosis + Reporter.")

        if monitor_out:
            st.markdown(f"""
            <div class="app-card">
            <p class="app-card-label monitor">MONITOR AGENT</p>
            <p class="app-card-body">{monitor_out}</p>
            </div>
            """, unsafe_allow_html=True)

        if diagnosis_out:
            st.markdown(f"""
            <div class="app-card">
            <p class="app-card-label diagnosis">DIAGNOSIS AGENT</p>
            <p class="app-card-body">{diagnosis_out}</p>
            </div>
            """, unsafe_allow_html=True)

        if brief_out:
            st.markdown(f"""
            <div class="app-card">
            <p class="app-card-label reporter">OPERATOR BRIEF (PENDING APPROVAL)</p>
            <p class="app-card-body">{brief_out}</p>
            </div>
            """, unsafe_allow_html=True)

            approved = st.checkbox("✅ Approve and publish this brief")
            if approved:
                st.success(f"Brief approved for Zone {selected_zone}. Ready to distribute to operations team.")