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
# ============================================================

import time


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