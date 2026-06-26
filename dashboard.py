# ============================================================
# ENTRY POINT
# Pages declared explicitly with st.Page. Default nav widget
# hidden via position="hidden" so we render our own ordered
# sidebar: identity header, then nav links, in chosen order.
# ============================================================

import streamlit as st

st.set_page_config(page_title="Grid Reliability Intelligence Platform", layout="wide", page_icon="💡")

overview = st.Page("pages/1_Overview.py", title="Dashboard", icon=":material/dashboard:")
simulator = st.Page("pages/2_Scenario_Simulator.py", title="Scenario Simulator", icon=":material/thermostat:")
risk_drivers = st.Page("pages/3_Risk_Drivers.py", title="Risk Drivers", icon=":material/troubleshoot:")
analytics = st.Page("pages/4_Performance_Analytics.py", title="Performance Analytics", icon=":material/bar_chart:")
agent_ops = st.Page("pages/5_Agent_Operations.py", title="Agent Operations", icon=":material/smart_toy:")
ml_pipeline = st.Page("pages/6_ML_Pipeline.py", title="ML Pipeline", icon=":material/science:")
nav = st.navigation(
    [overview, simulator, risk_drivers, agent_ops, analytics,ml_pipeline],
    position="hidden"
)

st.sidebar.markdown("""
<div class="sidebar-header">
    <div class="sidebar-platform-name">Grid Reliability Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: CUSTOM NAV LINKS, in chosen order ---
st.sidebar.page_link(overview, label="Dashboard")
st.sidebar.page_link(simulator, label="Scenario Simulator")
st.sidebar.page_link(risk_drivers, label="Risk Drivers")
st.sidebar.page_link(analytics, label="Performance Analytics")
st.sidebar.page_link(agent_ops, label="Agent Operations")
st.sidebar.page_link(ml_pipeline, label="ML Pipeline")


nav.run()