# ============================================================
# KNOWLEDGE BASE / RAG LAYER
# Stores operational runbook entries in ChromaDB, embedded locally
# via sentence-transformers (no external embedding API needed).
# The Diagnosis agent queries this with SHAP findings to retrieve
# the most relevant domain knowledge before generating its
# root-cause explanation, grounding the LLM in real operational
# context instead of letting it guess from general knowledge.
# ============================================================

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

RUNBOOK_ENTRIES = [
    {
        "id": "gas_cold_stress",
        "text": "Gas peaker units operating in sub-20°F conditions for multiple consecutive days "
                "show elevated forced-outage risk due to fuel-line viscosity changes and ignition "
                "system sensitivity. Recommend pre-heating protocol and increased monitoring "
                "frequency during sustained cold snaps."
    },
    {
        "id": "gas_heat_stress",
        "text": "Gas turbine output and reliability degrade above 85°F ambient temperature due to "
                "reduced air density affecting combustion efficiency. Units operating near rated "
                "capacity during heat events show higher derate likelihood. Recommend output "
                "de-rating or supplemental cooling during extreme heat."
    },
    {
        "id": "steam_recency",
        "text": "Steam units with extended intervals since their last maintenance event show "
                "compounding risk, as minor inefficiencies in boiler and turbine components tend "
                "to accumulate gradually rather than fail abruptly. Days-since-last-event is a "
                "strong leading indicator for this asset class specifically."
    },
    {
        "id": "wind_variability",
        "text": "Wind generation risk is driven primarily by output variability rather than "
                "mechanical degradation. High recent impact ratios for wind assets typically "
                "reflect short-term wind resource shortfall rather than an equipment issue, "
                "and usually self-resolve without intervention."
    },
    {
        "id": "hydro_baseline",
        "text": "Hydro units generally show the most stable and predictable output profile of any "
                "fuel type, since output is governed by reservoir levels and dispatch schedules "
                "rather than weather extremes. Elevated risk scores for hydro assets warrant "
                "closer review, as they deviate from this fuel type's typical stability."
    },
    {
        "id": "nuclear_baseline",
        "text": "Nuclear units operate at consistently high capacity factors with minimal "
                "weather sensitivity. Any meaningful predicted risk increase for a nuclear asset "
                "is atypical for the fuel type and should be treated as a higher-confidence "
                "signal warranting prompt review, even if the absolute risk value seems moderate."
    },
    {
        "id": "recent_impact_trend",
        "text": "A high recent-average-impact value combined with a high recent-max-impact value "
                "suggests a sustained degradation pattern rather than a single anomalous event. "
                "This combination typically warrants root-cause investigation rather than routine "
                "monitoring."
    },
    {
        "id": "prior_event_flag",
        "text": "Assets with a prior high-impact event on record show elevated recurrence risk for "
                "60 to 90 days following the initial event, even after apparent stabilization. "
                "Maintenance teams should treat recent high-impact history as an active risk "
                "factor, not a resolved one."
    },
    {
        "id": "solar_weather_coupling",
        "text": "Solar asset risk correlates closely with cloud cover and seasonal daylight "
                "variation rather than mechanical condition. Risk spikes for solar assets are "
                "usually weather-driven and transient, distinct from the gradual degradation "
                "patterns seen in thermal generation."
    },
    {
        "id": "capacity_distance_interaction",
        "text": "Larger-capacity generators located farther from their interconnection station "
                "show modestly higher transmission-related risk exposure. This factor compounds "
                "with weather and maintenance-recency signals rather than acting independently."
    },
]


@st.cache_resource
def get_runbook_collection():
    """
    Builds the ChromaDB collection ONCE per app process and caches it
    via st.cache_resource, so the embedding model is loaded a single
    time, not reloaded on every agent pipeline run. This was the
    actual cause of the slowness and Render crashes, every click was
    reloading sentence-transformers from scratch.
    """
    client = chromadb.EphemeralClient()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="grid_runbook",
        embedding_function=embedding_fn
    )

    if collection.count() == 0:
        collection.add(
            ids=[entry["id"] for entry in RUNBOOK_ENTRIES],
            documents=[entry["text"] for entry in RUNBOOK_ENTRIES]
        )

    return collection


def retrieve_relevant_knowledge(query_text, n_results=2):
    """
    Queries the cached runbook collection with query_text and returns
    the top n_results most relevant entries as a formatted string.
    """
    collection = get_runbook_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)

    retrieved_docs = results["documents"][0] if results["documents"] else []
    if not retrieved_docs:
        return "No specific operational guidance found for this profile."

    return "\n".join(f"- {doc}" for doc in retrieved_docs)