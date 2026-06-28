# ============================================================
# AGENT UTILITIES: RETRY LOGIC AND GUARDRAILS
#
# Retry: wraps any LLM call and retries up to 3 times on failure.
#   - Waits 2s after first failure, 4s after second
#   - Only raises the error after all 3 attempts fail
#   - Usage: response = call_with_retry(lambda: llm.invoke(prompt))
#
# Guardrails: validates LLM output before it enters state or
#   gets displayed. Each agent has its own rule based on what
#   a well-formed response must contain. If validation fails,
#   returns (False, reason) so the caller can show a clear
#   message instead of silently passing bad output downstream.
#
# Anomaly Detection: pure Python, no LLM. Runs before any
#   LLM agent to flag statistical anomalies in ML predictions.
# ============================================================

import time
import re
import numpy as np
import pandas as pd


def call_with_retry(fn, max_attempts=3, base_wait=2):
    """
    Calls fn() up to max_attempts times on any exception.
    Waits base_wait seconds after first failure, base_wait*2 after second.
    Returns the result if any attempt succeeds.
    Raises the last exception if all attempts fail.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(base_wait * (attempt + 1))
    raise last_error


def validate_monitor_output(text: str) -> tuple:
    """
    Monitor must produce both a SUMMARY section and a FLAG line.
    Without FLAG, the routing logic that reads it breaks.
    """
    text_upper = text.upper().replace(" ", "")
    if "SUMMARY" not in text_upper:
        return False, "Monitor response is missing a SUMMARY section."
    if "FLAG" not in text_upper:
        return False, "Monitor response is missing a FLAG line."
    return True, ""


def validate_diagnosis_output(text: str) -> tuple:
    """
    Diagnosis must be a substantive explanation, not empty
    or one-line. 80 characters is the minimum for 2-3 sentences.
    """
    if not text or len(text.strip()) < 80:
        return False, "Diagnosis response is too short to be a valid explanation."
    return True, ""


def validate_reporter_output(text: str) -> tuple:
    """
    Reporter must commit to a concrete action, not just describe
    the situation. Checks for action-oriented language.
    """
    action_words = [
        "recommend", "inspect", "contact", "check", "schedule",
        "initiate", "dispatch", "review", "prioritize", "conduct",
        "monitor", "investigate", "escalate", "notify", "assess"
    ]
    if not any(word in text.lower() for word in action_words):
        return False, "Reporter response does not contain a concrete recommended action."
    return True, ""


VALID_ZONES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}


def validate_zone_names(text: str) -> tuple:
    """
    Checks that any zone letters mentioned in LLM output
    are real NY load zones (A through K).
    Catches hallucinated zone names like Zone X, Zone L etc.
    """
    mentioned = set(re.findall(r'\bZone\s+([A-Z])\b', text))
    invalid = mentioned - VALID_ZONES
    if invalid:
        return False, f"LLM mentioned invalid zone(s): {invalid}. Valid zones are A through K only."
    return True, ""


def run_anomaly_detection(
    zone: str,
    zone_summary: pd.DataFrame,
    latest_df: pd.DataFrame
) -> dict:
    """
    Pure Python anomaly detection. Runs BEFORE any LLM agent.
    Checks actual ML predictions for statistical anomalies.
    Returns a dict of flags and context for downstream agents.

    Checks:
    - Zone risk is more than 2 std devs above system mean
    - More than 50% of generators in zone have impact ratio > 0.80
    - Zone risk contradicts temperature signal (cold day but LOW risk)
    """
    result = {
        "zone": zone,
        "anomalies": [],
        "context": "",
        "severity": "NORMAL"
    }

    zone_data = zone_summary[zone_summary["operating_region"] == zone]
    if zone_data.empty:
        result["context"] = f"No data found for Zone {zone}."
        return result

    zone_row = zone_data.iloc[0]
    risk_pct = zone_row["risk_pct"]
    all_risk = zone_summary["risk_pct"].values
    mean_risk = np.mean(all_risk)
    std_risk = np.std(all_risk)

    # Check 1: statistical outlier across all zones
    if std_risk > 0:
        z_score = (risk_pct - mean_risk) / std_risk
        if z_score > 2.0:
            result["anomalies"].append(
                f"Zone {zone} risk {risk_pct:.1f}% is {z_score:.1f} std devs "
                f"above system mean {mean_risk:.1f}% — statistical outlier"
            )
            result["severity"] = "HIGH"

    # Check 2: majority of generators in high risk territory
    zone_gens = latest_df[latest_df["operating_region"] == zone]
    if "predicted_impact_ratio" in zone_gens.columns and len(zone_gens) > 0:
        high_risk_count = (zone_gens["predicted_impact_ratio"] > 0.80).sum()
        total = len(zone_gens)
        if high_risk_count > total * 0.5:
            result["anomalies"].append(
                f"{high_risk_count}/{total} generators in Zone {zone} "
                f"have impact ratio above 0.80 — widespread stress"
            )
            result["severity"] = "HIGH"

    # Check 3: cold stress day with unexpectedly low risk
    if "cold_day_flag" in zone_gens.columns and len(zone_gens) > 0:
        cold_flag = zone_gens["cold_day_flag"].mean()
        if cold_flag > 0.5 and risk_pct < 45:
            result["anomalies"].append(
                f"Zone {zone} shows cold stress conditions but only {risk_pct:.1f}% risk — "
                f"verify temperature inputs are correct"
            )

    # Check 4: hot stress day with unexpectedly low risk
    if "hot_day_flag" in zone_gens.columns and len(zone_gens) > 0:
        hot_flag = zone_gens["hot_day_flag"].mean()
        if hot_flag > 0.5 and risk_pct < 45:
            result["anomalies"].append(
                f"Zone {zone} shows heat stress conditions but only {risk_pct:.1f}% risk — "
                f"verify temperature inputs are correct"
            )

    if result["anomalies"]:
        result["context"] = (
            f"ANOMALY DETECTION FLAGS for Zone {zone}:\n"
            + "\n".join(f"- {a}" for a in result["anomalies"])
            + "\n\nPlease factor these anomalies into your analysis."
        )
    else:
        result["context"] = (
            f"No statistical anomalies detected for Zone {zone}. "
            f"Current risk {risk_pct:.1f}% is within normal system range."
        )

    return result