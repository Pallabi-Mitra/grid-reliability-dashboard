# ============================================================
# PAGE: AGENT OPERATIONS
#
# Runs a LangGraph multi-agent pipeline for a selected zone:
#   GREEN  -> Monitor only, stop
#   YELLOW -> Monitor + Diagnosis, stop
#   RED    -> Monitor + Diagnosis + Reporter (full escalation)
#
# Monitor's FLAG output can override routing, escalating a
# YELLOW or GREEN zone if the LLM detects something unusual.
# All three agents have retry logic and output guardrails.
# ============================================================

import streamlit as st
import pandas as pd
import os
import shap
import sys
import asyncio
from shared import (
    load_css, get_latest_predictions, zone_names, categorical_cols
)
from agents_utils import (
    call_with_retry,
    validate_monitor_output,
    validate_diagnosis_output,
    validate_reporter_output
)
from langchain_mcp_adapters.client import MultiServerMCPClient

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_latest_predictions()


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


MCP_CONFIG = {
    "grid_reliability": {
        "command": sys.executable,
        "args": ["mcp_server.py"],
        "transport": "stdio",
        "cwd": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    }
}


@st.cache_resource
def get_mcp_tools():
    async def _load():
        client = MultiServerMCPClient(MCP_CONFIG)
        return await client.get_tools()
    return asyncio.run(_load())


st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.title("Agent Operations")
st.caption("Tool-calling Monitor agent, SHAP+RAG Diagnosis agent, and Reporter agent with conditional routing")
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
        from langchain.agents import create_react_agent
        from typing import TypedDict, Annotated
        from knowledge_base import retrieve_relevant_knowledge

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_key)
        mcp_tools = get_mcp_tools()

        class ReliabilityState(TypedDict):
            messages: Annotated[list, add_messages]
            zone: str
            risk_level: str
            risk_pct: float
            top_generators: str
            diagnosis: str
            brief: str
            needs_investigation: bool

        monitor_react_agent = create_react_agent(llm, mcp_tools)

        async def monitor_agent(state):
            zone = state["zone"]
            try:
                result = await monitor_react_agent.ainvoke({
                    "messages": [(
                        "user",
                        f"You are a Grid Monitoring Agent. Use the available tools "
                        f"to look up the risk level and top generators for Zone {zone}.\n\n"
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

            needs_investigation = "FLAG:YES" in final_text.upper().replace(" ", "")

            return {
                "messages": [final_text],
                "needs_investigation": needs_investigation
            }

        def diagnosis_agent(state):
            zone = state["zone"]
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

            explainer = get_shap_explainer(model)
            shap_vals = explainer.shap_values(asset_enc[model_features])
            shap_series = pd.Series(shap_vals[0], index=model_features).abs().sort_values(ascending=False)
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
                    f"Highest-risk generator: {top_asset['asset_id']} "
                    f"({top_asset['fuel_category']}, {top_asset['dependable_capacity_mw']:.1f} MW).\n"
                    f"Top SHAP risk drivers: {feature_str}.\n\n"
                    f"Relevant operational guidance:\n{retrieved_knowledge}\n\n"
                    f"Using the SHAP drivers AND the operational guidance above, explain the likely "
                    f"root cause in 2-3 sentences. Ground your explanation in the retrieved guidance "
                    f"where it applies, rather than general assumptions."
                ))
                diagnosis_text = response.content
            except Exception:
                diagnosis_text = (
                    f"Diagnosis unavailable after multiple attempts. "
                    f"Top SHAP drivers for {top_asset['asset_id']}: {feature_str}. "
                    f"Manual review recommended."
                )

            is_valid, reason = validate_diagnosis_output(diagnosis_text)
            if not is_valid:
                diagnosis_text = (
                    f"Diagnosis output was insufficient ({reason}). "
                    f"Raw SHAP drivers: {feature_str}."
                )

            return {"messages": [diagnosis_text], "diagnosis": diagnosis_text}

        def reporter_agent(state):
            zone = state["zone"]
            messages = state["messages"]
            last_two = messages[-2] if len(messages) >= 2 else ""
            monitor_summary = last_two.content if hasattr(last_two, "content") else str(last_two)
            diagnosis = state["diagnosis"]

            try:
                response = call_with_retry(lambda: llm.invoke(
                    f"You are a Reporting Agent. Draft a concise operator brief for Zone {zone}.\n"
                    f"Monitor: {monitor_summary}\nDiagnosis: {diagnosis}\n\n"
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

        zone_risk_level = zone_summary[zone_summary["operating_region"] == selected_zone]["risk_level"].values[0]

        async def run_pipeline():
            return await pipeline.ainvoke({
                "messages": [], "zone": selected_zone,
                "risk_level": zone_risk_level,
                "risk_pct": 0.0, "top_generators": "",
                "diagnosis": "", "brief": "",
                "needs_investigation": False
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
            <p class="app-card-label diagnosis">DIAGNOSIS AGENT (SHAP + RAG)</p>
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
                st.success(f"Brief approved for Zone {selected_zone}. Ready to distribute to operations team.")