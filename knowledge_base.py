# ============================================================
# KNOWLEDGE BASE / RAG LAYER
# Stores operational runbook entries and retrieves the most
# relevant ones for a given query before the Diagnosis agent
# generates its explanation. Retrieval is built behind a swappable
# interface (Retriever base class) so the embedding method is an
# implementation detail, not a fixed dependency. Today's corpus
# is 10 documents, so a lightweight TF-IDF retriever is the right
# sized choice: no heavy model download, no memory risk, same
# retrieve-then-ground mechanism as any RAG system. If the corpus
# grew to thousands of entries, a DenseEmbeddingRetriever could be
# swapped in here without touching the agent code at all.
# ============================================================

import streamlit as st
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# ============================================================
# RETRIEVER INTERFACE
# Any retriever implementation must support retrieve(query, n_results)
# and return a list of document text strings, ranked most relevant first.
# ============================================================
class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, n_results: int = 2) -> list[str]:
        ...


# ============================================================
# TF-IDF RETRIEVER
# Sparse, statistical retrieval: documents and the query are turned
# into word-frequency vectors, ranked by cosine similarity. No model
# download, no GPU/torch dependency, near-zero memory footprint.
# Well-suited to small, distinct-vocabulary corpora like this one.
# ============================================================
class TFIDFRetriever(Retriever):
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        texts = [doc["text"] for doc in documents]
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, n_results: int = 2) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_matrix)[0]
        top_indices = similarities.argsort()[::-1][:n_results]
        return [self.documents[i]["text"] for i in top_indices]


# ============================================================
# RETRIEVER FACTORY (cached)
# Builds the retriever once per app session. Swapping retrieval
# strategy later means changing this one line, nothing in the
# agent code needs to change since both implement the same interface.
# ============================================================
@st.cache_resource
def get_retriever() -> Retriever:
    return TFIDFRetriever(RUNBOOK_ENTRIES)


def retrieve_relevant_knowledge(query_text: str, n_results: int = 2) -> str:
    """
    Queries the active retriever and returns the top n_results most
    relevant runbook entries as a single formatted string, ready to
    drop into an LLM prompt. Same signature as before, so the
    Diagnosis agent's call site doesn't need to change.
    """
    retriever = get_retriever()
    retrieved_docs = retriever.retrieve(query_text, n_results)

    if not retrieved_docs:
        return "No specific operational guidance found for this profile."

    return "\n".join(f"- {doc}" for doc in retrieved_docs)