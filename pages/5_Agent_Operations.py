# ============================================================
# PAGE: AGENT OPERATIONS
#
# Runs a LangGraph multi-agent pipeline for a selected zone:
#   GREEN  -> Monitor only, stop
#   YELLOW -> Monitor + Diagnosis, stop
#   RED    -> Monitor + Diagnosis + Reporter (full escalation)
#
# Pre-pipeline agents (pure Python, no LLM):
#   1. Anomaly Detection Agent - flags statistical anomalies
#   2. Confidence Scorer - computes prediction reliability score
#
# LLM agents with guardrails:
#   Monitor - tool-calling, validates SUMMARY+FLAG+zone names
#   Diagnosis - RAG + feature importance, validates length+zones
#   Reporter - operator brief, validates action word+zone names
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import asyncio
from shared import (
    load_css, get_live_weather_predictions, zone_names, categorical_cols
)
from agents_utils import (
    call_with_retry,
    validate_monitor_output,
    validate_diagnosis_output,
    validate_reporter_output,
    validate_zone_names,
    run_anomaly_detection
)
from langchain_core.tools import tool as lc_tool

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

# --- MCP TOOLS ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server import get_zone_risk, get_generator_predictions, list_all_zones

@lc_tool
def tool_get_zone_risk(zone: str) -> str:
    """Get the current risk level and predicted MW at risk for a zone."""
    return get_zone_risk(zone)

@lc_tool
def tool_get_generator_predictions(zone: str) -> str:
    """Get top generators by predicted risk in a zone."""
    return get_generator_predictions(zone)

@lc_tool
def tool_list_all_zones() -> str:
    """List all 11 NY load zones with their current risk levels."""
    return list_all_zones()

mcp_tools = [tool_get_zone_risk, tool_get_generator_predictions, tool_list_all_zones]

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1B2A,#1A3A5C);padding:2rem 2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;">
    <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.15em;color:#64B5F6;text-transform:uppercase;margin-bottom:0.4rem;">Grid Reliability Intelligence Platform</div>
    <div style="font-size:1.8rem;font-weight:700;color:#FFFFFF;margin-bottom:0.4rem;">Agentic AI Operations</div>
    <div style="font-size:0.9rem;color:#90A4AE;">Anomaly Detection · Confidence Scoring · Tool-calling Monitor · RAG Diagnosis · Human Approval</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

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

groq_key = os.environ.get("GROQ_API_KEY")

if not groq_key:
    st.warning("No Groq API key found. Set GROQ_API_KEY as an environment variable.")
else:
    if st.button("Run Agent Pipeline for Zone " + selected_zone):
        from langchain_groq import ChatGroq
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import create_react_agent
        from typing import TypedDict, Annotated
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
            needs_investigation: bool
            anomaly_context: str
            confidence_score: float
            confidence_level: str

        # ── Pre-pipeline Agent 1: Anomaly Detection ──
        anomaly_result = run_anomaly_detection(selected_zone, zone_summary, latest_df)
        anomaly_context = anomaly_result["context"]

        # ── Pre-pipeline Agent 2: Confidence Scoring ──
        zone_row = zone_summary[zone_summary["operating_region"] == selected_zone].iloc[0]
        risk_pct = float(zone_row["risk_pct"])
        all_risk = zone_summary["risk_pct"].values
        mean_risk = float(np.mean(all_risk))
        std_risk = float(np.std(all_risk))
        z_score = abs((risk_pct - mean_risk) / std_risk) if std_risk > 0 else 0
        confidence_score = round(
            max(0.0, 100.0 - (z_score * 15.0) - (len(anomaly_result["anomalies"]) * 15.0)), 1
        )
        confidence_level = (
            "HIGH" if confidence_score >= 80
            else "MEDIUM" if confidence_score >= 60
            else "LOW"
        )

        # ── Show pre-pipeline results ──
        st.markdown("### Pre-Pipeline Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Anomalies Detected", len(anomaly_result["anomalies"]))
        with c2:
            st.metric("Prediction Confidence", f"{confidence_score}%")
        with c3:
            color = "🟢" if confidence_level == "HIGH" else ("🟡" if confidence_level == "MEDIUM" else "🔴")
            st.metric("Confidence Level", f"{color} {confidence_level}")

        if anomaly_result["anomalies"]:
            for flag in anomaly_result["anomalies"]:
                st.warning(f"🔍 {flag}")
        else:
            st.success(f"No statistical anomalies detected for Zone {selected_zone}. Predictions are reliable.")

        st.markdown("---")

        monitor_react_agent = create_react_agent(llm, mcp_tools)

        async def monitor_agent(state):
            zone = state["zone"]
            context = state.get("anomaly_context", "")
            conf_score = state.get("confidence_score", 100.0)
            conf_level = state.get("confidence_level", "HIGH")
            try:
                result = await monitor_react_agent.ainvoke({
                    "messages": [(
                        "user",
                        f"You are a Grid Monitoring Agent. Use the available tools "
                        f"to look up the risk level and top generators for Zone {zone}.\n\n"
                        f"Context from anomaly detection system:\n{context}\n\n"
                        f"Prediction confidence for this zone: {conf_score}% ({conf_level}). "
                        f"If confidence is LOW or MEDIUM, flag this in your summary.\n\n"
                        f"Then write two things, clearly labeled:\n"
                        f"SUMMARY: a 2-3 sentence summary of the situation for an operations team.\n"
                        f"FLAG: write YES if anything looks inconsistent, unusual, or "
                        f"warrants deeper investigation, otherwise write NO."
                    )]
                })
                final_text = result["messages"][-1].content
            except Exception:
                final_text = (
                    f"SUMMARY: Monitor agent failed to retrieve data for Zone {zone} "
                    f"after multiple attempts. Manual review recommended.\n"
                    f"FLAG: YES"
                )

            is_valid, reason = validate_monitor_output(final_text)
            if not is_valid:
                final_text = (
                    f"SUMMARY: Monitor output was malformed ({reason}). "
                    f"Manual review recommended for Zone {zone}.\n"
                    f"FLAG: YES"
                )

            is_valid_zones, zone_reason = validate_zone_names(final_text)
            if not is_valid_zones:
                final_text += f"\n\n[GUARDRAIL WARNING: {zone_reason}]"

            needs_investigation = "FLAG:YES" in final_text.upper().replace(" ", "")

            return {
                "messages": [final_text],
                "needs_investigation": needs_investigation
            }

        def diagnosis_agent(state):
            zone = state["zone"]
            context = state.get("anomaly_context", "")
            conf_score = state.get("confidence_score", 100.0)
            last_msg = state["messages"][-1]
            monitor_summary = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

            top_asset = latest_df[latest_df["operating_region"] == zone].sort_values(
                "predicted_impact_ratio", ascending=False).iloc[0]
            asset_enc = pd.get_dummies(
                latest_df[latest_df["asset_id"] == top_asset["asset_id"]],
                columns=categorical_cols)
            for col in model_features:
                if col not in asset_enc.columns:
                    asset_enc[col] = 0

            importance_scores = model.get_booster().get_score(importance_type="gain")
            shap_series = pd.Series(importance_scores).reindex(model_features).fillna(0).abs().sort_values(ascending=False)
            feature_str = ", ".join([
                f"{k.replace('_', ' ')} ({v:.3f})"
                for k, v in shap_series.head(5).items()
            ])

            rag_query = f"{top_asset['fuel_category']} generator risk factors: {feature_str}"
            retrieved_knowledge = retrieve_relevant_knowledge(rag_query, n_results=2)

            try:
                response = call_with_retry(lambda: llm.invoke(
                    f"You are a Diagnosis Agent.\n"
                    f"Monitor reported: {monitor_summary}\n"
                    f"Anomaly detection context:\n{context}\n"
                    f"Prediction confidence: {conf_score}% — factor this into your certainty level.\n"
                    f"Highest-risk generator: {top_asset['asset_id']} "
                    f"({top_asset['fuel_category']}, {top_asset['dependable_capacity_mw']:.1f} MW).\n"
                    f"Top risk drivers: {feature_str}.\n\n"
                    f"Relevant operational guidance:\n{retrieved_knowledge}\n\n"
                    f"Using the risk drivers AND the operational guidance above, explain the likely "
                    f"root cause in 2-3 sentences. Ground your explanation in the retrieved guidance "
                    f"where it applies, rather than general assumptions."
                ))
                diagnosis_text = response.content
            except Exception:
                diagnosis_text = (
                    f"Diagnosis unavailable after multiple attempts. "
                    f"Top risk drivers for {top_asset['asset_id']}: {feature_str}. "
                    f"Manual review recommended."
                )

            is_valid, reason = validate_diagnosis_output(diagnosis_text)
            if not is_valid:
                diagnosis_text = (
                    f"Diagnosis output was insufficient ({reason}). "
                    f"Raw risk drivers: {feature_str}."
                )

            is_valid_zones, zone_reason = validate_zone_names(diagnosis_text)
            if not is_valid_zones:
                diagnosis_text += f"\n\n[GUARDRAIL WARNING: {zone_reason}]"

            return {"messages": [diagnosis_text], "diagnosis": diagnosis_text}

        def reporter_agent(state):
            zone = state["zone"]
            context = state.get("anomaly_context", "")
            conf_score = state.get("confidence_score", 100.0)
            conf_level = state.get("confidence_level", "HIGH")
            messages = state["messages"]
            last_two = messages[-2] if len(messages) >= 2 else ""
            monitor_summary = last_two.content if hasattr(last_two, "content") else str(last_two)
            diagnosis = state["diagnosis"]

            try:
                response = call_with_retry(lambda: llm.invoke(
                    f"You are a Reporting Agent. Draft a concise operator brief for Zone {zone}.\n"
                    f"Monitor: {monitor_summary}\n"
                    f"Diagnosis: {diagnosis}\n"
                    f"Anomaly detection context:\n{context}\n"
                    f"Prediction confidence: {conf_score}% ({conf_level}). "
                    f"If confidence is not HIGH, note this caveat in the brief.\n\n"
                    f"State the risk level and MW at risk, explain the main cause, and commit to ONE "
                    f"specific, actionable next step. Professional tone, 3-4 sentences."
                ))
                brief_text = response.content
            except Exception:
                brief_text = (
                    f"Reporter unavailable after multiple attempts. "
                    f"Zone {zone} requires manual operator review based on Diagnosis findings."
                )

            is_valid, reason = validate_reporter_output(brief_text)
            if not is_valid:
                brief_text += (
                    f"\n\n[Note: {reason} "
                    f"Please verify this brief contains a concrete next step before approving.]"
                )

            is_valid_zones, zone_reason = validate_zone_names(brief_text)
            if not is_valid_zones:
                brief_text += f"\n\n[GUARDRAIL WARNING: {zone_reason}]"

            return {"messages": [brief_text], "brief": brief_text}

        def route_after_monitor(state):
            risk_level = state["risk_level"]
            flagged = state.get("needs_investigation", False)
            if "GREEN" in risk_level and not flagged:
                return "stop"
            elif "RED" in risk_level:
                return "full"
            else:
                return "diagnose_only"

        def route_after_diagnosis(state):
            if "RED" in state["risk_level"] or state.get("needs_investigation", False):
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

        zone_risk_level = zone_summary[
            zone_summary["operating_region"] == selected_zone
        ]["risk_level"].values[0]

        async def run_pipeline():
            return await pipeline.ainvoke({
                "messages": [],
                "zone": selected_zone,
                "risk_level": zone_risk_level,
                "risk_pct": risk_pct,
                "top_generators": "",
                "diagnosis": "",
                "brief": "",
                "needs_investigation": False,
                "anomaly_context": anomaly_context,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level
            })

        with st.spinner("Running agent pipeline..."):
            result = asyncio.run(run_pipeline())

        def extract_text(msg):
            if msg is None:
                return None
            if hasattr(msg, "content"):
                return msg.content
            return str(msg)

        messages = result["messages"]
        monitor_out = extract_text(messages[0]) if len(messages) > 0 else None
        diagnosis_out = extract_text(messages[1]) if len(messages) > 1 else None
        brief_out = extract_text(messages[2]) if len(messages) > 2 else None

        flagged = result.get("needs_investigation", False)
        if "GREEN" in zone_risk_level and not flagged:
            st.info("Zone is GREEN. Monitor found nothing unusual. Pipeline stopped here.")
        elif "GREEN" in zone_risk_level and flagged:
            st.warning("Zone is GREEN by the numbers but Monitor flagged something worth investigating.")
        elif "YELLOW" in zone_risk_level:
            st.info("Zone is YELLOW. Monitor + Diagnosis ran. No formal report generated.")
        else:
            st.info("Zone is RED. Full pipeline ran: Monitor + Diagnosis + Reporter.")

        if monitor_out:
            st.markdown(f"""
            <div class="app-card">
            <p class="app-card-label monitor">MONITOR AGENT (tool-calling)</p>
            <p class="app-card-body">{monitor_out}</p>
            </div>
            """, unsafe_allow_html=True)

        if diagnosis_out:
            st.markdown(f"""
            <div class="app-card">
            <p class="app-card-label diagnosis">DIAGNOSIS AGENT (RAG + Feature Importance)</p>
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

            approved = st.checkbox("Approve and publish this brief")
            if approved:
                st.success(
                    f"Brief approved for Zone {selected_zone}. "
                    f"Ready to distribute to operations team."
                )