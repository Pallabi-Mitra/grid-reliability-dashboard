# ============================================================
# PAGE: AGENT OPERATIONS
# (renamed from "Executive Summary" — this page runs the actual
# multi-agent reasoning, not a static report)
#
# Runs a LangGraph multi-agent pipeline for a selected zone:
#
#   MONITOR agent:
#     A real tool-calling agent (LangGraph's create_react_agent).
#     It is given a goal and a list of MCP tools, and DECIDES for
#     itself which tools to call and when, the same pattern a
#     production agent would use to fetch data it doesn't already
#     have. It can also flag the situation as needing deeper
#     investigation, that flag can change what happens next.
#
#   DIAGNOSIS agent:
#     Computes SHAP locally (fast, cached), retrieves relevant
#     operational guidance via RAG, and asks the LLM to synthesize
#     both into a root-cause explanation. This is the step where an
#     LLM is genuinely earning its place, reconciling two different
#     kinds of evidence into one explanation is hard to template.
#
#   REPORTER agent:
#     Synthesizes Monitor + Diagnosis into an operator brief with a
#     concrete recommendation and a self-assessed urgency level.
#
# Routing (which agents run) is based on zone risk level:
#   GREEN  -> Monitor only, stop
#   YELLOW -> Monitor + Diagnosis, stop (not urgent enough to report)
#   RED    -> Monitor + Diagnosis + Reporter (full escalation)
#
# Monitor's own "needs_investigation" flag can ALSO force the
# pipeline into the full path even on a YELLOW zone, this is the
# part where the LLM's own judgment affects control flow, not
# just the text it produces.
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

# --- LOAD CSS ---
load_css("styles.css")

# --- LOAD DATA / MODEL / PREDICTIONS ---
assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_latest_predictions()


# --- CACHED SHAP EXPLAINER ---
# Building a TreeExplainer is expensive; computing shap_values for a
# single row is fast. Caching means it's built ONCE per app session,
# not rebuilt on every Diagnosis call.
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


# ============================================================
# MCP CONNECTION
# mcp_server.py exposes 4 tools (get_zone_risk, get_generator_predictions,
# get_shap_explanation, list_all_zones). We launch it as a subprocess
# (stdio transport, the simplest MCP transport, no network setup needed)
# and load its tools so a LangGraph agent can call them. This is what
# makes Monitor a REAL tool-calling agent instead of a function that
# reads dataframes directly.
# ============================================================
from langchain_mcp_adapters.client import MultiServerMCPClient



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
    """
    Connects to mcp_server.py as a subprocess and returns its tools
    as LangChain-compatible tool objects. Cached so the connection
    is made once per app session, not on every button click.
    """
    async def _load():
        client = MultiServerMCPClient(MCP_CONFIG)
        return await client.get_tools()
    return asyncio.run(_load())


# --- SIDEBAR: FOOTER ---
st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

# --- PAGE HEADER ---
st.title("🤖 Agent Operations")
st.caption("Tool-calling Monitor agent, SHAP+RAG Diagnosis agent, and Reporter agent, with conditional routing by risk level")
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
        from langgraph.prebuilt import create_react_agent
        from typing import TypedDict, Annotated
        from knowledge_base import retrieve_relevant_knowledge

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_key)
        mcp_tools = get_mcp_tools()

        # --- STATE SCHEMA ---
        # needs_investigation is NEW: Monitor's own LLM call sets this,
        # and route_after_monitor reads it, this is the mechanism by
        # which the LLM's judgment can change the pipeline's path,
        # not just produce text.
        class ReliabilityState(TypedDict):
            messages: Annotated[list, add_messages]
            zone: str
            risk_level: str
            risk_pct: float
            top_generators: str
            diagnosis: str
            brief: str
            needs_investigation: bool

        # ============================================================
        # MONITOR AGENT — real tool-calling agent
        # create_react_agent builds an agent that loops: think, decide
        # whether to call a tool, call it, look at the result, repeat,
        # until it has enough to answer. We don't tell it WHICH tool to
        # call or in what order, it decides that itself based on the
        # goal in the prompt. This is the actual mechanism of "agentic."
        # ============================================================
        monitor_react_agent = create_react_agent(llm, mcp_tools)

        async def monitor_agent(state):
            zone = state["zone"]

            result = await monitor_react_agent.ainvoke({
                "messages": [(
                    "user",
                    f"You are a Grid Monitoring Agent. Use the available tools to look up "
                    f"the risk level and top generators for Zone {zone}.\n\n"
                    f"Then write two things, clearly labeled:\n"
                    f"SUMMARY: a 2-3 sentence summary of the situation for an operations team.\n"
                    f"FLAG: write YES if anything in the data looks inconsistent, unusual, or "
                    f"warrants deeper investigation beyond the routine risk level, otherwise write NO."
                )]
            })

            final_text = result["messages"][-1].content

            # Parse the FLAG line out of the agent's own response. This is
            # a simple, honest way to let the LLM's judgment (not just our
            # fixed risk_pct thresholds) influence what happens next.
            needs_investigation = "FLAG: YES" in final_text.upper().replace(" ", "") \
                or "FLAG:YES" in final_text.upper().replace(" ", "")

            return {
                "messages": [final_text],
                "needs_investigation": needs_investigation
            }

        # ============================================================
        # DIAGNOSIS AGENT — SHAP + RAG synthesis
        # This is the strongest LLM use case in the pipeline: it has to
        # reconcile two different kinds of evidence (SHAP's numeric
        # feature importances, and retrieved text-based operational
        # guidance) into one coherent causal explanation. That synthesis
        # is genuinely hard to template, which is why this step keeps
        # the LLM even where Monitor's summarization alone wouldn't.
        # ============================================================
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

            explainer = get_shap_explainer(model)
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

        # ============================================================
        # REPORTER AGENT — recommendation, not just description
        # Prompted to commit to an actual recommendation and an urgency
        # level, not just restate Monitor/Diagnosis. A real decision,
        # not narration.
        # ============================================================
        def reporter_agent(state):
            zone = state["zone"]
            monitor_summary = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
            diagnosis = state["diagnosis"]
            response = llm.invoke(
                f"You are a Reporting Agent. Draft a concise operator brief for Zone {zone}.\n"
                f"Monitor: {monitor_summary}\nDiagnosis: {diagnosis}\n\n"
                f"State the risk level and MW at risk, explain the main cause, and commit to ONE "
                f"specific, actionable next step (not a vague suggestion). Professional tone, 3-4 sentences."
            )
            return {"messages": [response], "brief": response.content}

        # ============================================================
        # ROUTING — now reads Monitor's own judgment, not just the
        # fixed risk_pct threshold. A YELLOW zone that Monitor flagged
        # as needing investigation gets escalated to the full path,
        # this is the LLM's output changing control flow.
        # ============================================================
        def route_after_monitor(state):
            risk_level = state["risk_level"]
            flagged = state.get("needs_investigation", False)

            if "GREEN" in risk_level and not flagged:
                return "stop"
            elif "RED" in risk_level:
                return "full"
            else:
                # YELLOW, or GREEN-but-flagged: investigate either way
                return "diagnose_only"

        def route_after_diagnosis(state):
            if "RED" in state["risk_level"] or state.get("needs_investigation", False):
                return "full"
            else:
                return "stop"

        # --- BUILD GRAPH ---
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

        with st.spinner("Running agent pipeline... (Monitor is calling tools, this takes a bit longer than before)"):
            result = asyncio.run(run_pipeline())

        messages = result["messages"]
        monitor_out = messages[0].content if len(messages) > 0 else None
        diagnosis_out = messages[1].content if len(messages) > 1 else None
        brief_out = messages[2].content if len(messages) > 2 else None

        # --- DISPLAY: which path was taken, now mentions the flag too ---
        flagged = result.get("needs_investigation", False)
        if "GREEN" in zone_risk_level and not flagged:
            st.info("🟢 Zone is GREEN, Monitor found nothing unusual. Pipeline stopped here.")
        elif "GREEN" in zone_risk_level and flagged:
            st.warning("🟡 Zone is GREEN by the numbers, but Monitor flagged something worth investigating. Running Diagnosis.")
        elif "YELLOW" in zone_risk_level:
            st.info("🟡 Zone is YELLOW. Monitor + Diagnosis ran. No formal report generated, not yet urgent.")
        else:
            st.info("🔴 Zone is RED. Full pipeline ran: Monitor + Diagnosis + Reporter.")

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

            approved = st.checkbox("✅ Approve and publish this brief")
            if approved:
                st.success(f"Brief approved for Zone {selected_zone}. Ready to distribute to operations team.")