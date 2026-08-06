import re
from enum import Enum

MAX_REPROMPT_ATTEMPTS = 1  # one LLM fix-args pass, then replan
MAX_RETRY_ATTEMPTS = 3  # transient backoff retries


class ErrorClass(Enum):
    TRANSIENT = "transient"
    VALIDATION = "validation"
    UNKNOWN_ENTITY = "unknown_entity"
    POLICY = "policy"


class RecoveryAction(Enum):
    RETRY = "retry"
    REPROMPT = "reprompt"
    REPLAN = "replan"
    HALT = "halt"


_RECOVERY: dict[ErrorClass, RecoveryAction] = {
    ErrorClass.TRANSIENT: RecoveryAction.RETRY,
    ErrorClass.VALIDATION: RecoveryAction.REPROMPT,
    ErrorClass.UNKNOWN_ENTITY: RecoveryAction.REPLAN,
    ErrorClass.POLICY: RecoveryAction.HALT,
}

_TRANSIENT = re.compile(
    r"rate.?limit|429|timeout|timed?\s*out|503|temporarily|unavailable|connection\s*reset",
    re.IGNORECASE,
)
_VALIDATION = re.compile(
    r"validation|invalid\s*arg|pydantic|type\s*error|missing\s*required|schema",
    re.IGNORECASE,
)
_UNKNOWN = re.compile(
    r"unknown\s*(city|entity|key)|key\s*error|not\s*found|no\s*such",
    re.IGNORECASE,
)
_POLICY = re.compile(
    r"policy|forbidden|unauthorized|blocked|disallowed|content\s*filter",
    re.IGNORECASE,
)


def classify_error(error: str | BaseException) -> ErrorClass:
    """maps error strings into four classes of errors:
    1. rate limit or timeout
    2. validation error
    3. missing information
    4. policy violation

    and a recovery policy follows from that class
    """
    text = str(error)
    # order matters: policy before unknown
    if _POLICY.search(text):
        return ErrorClass.POLICY
    if _TRANSIENT.search(text):
        return ErrorClass.TRANSIENT
    if _VALIDATION.search(text):
        return ErrorClass.VALIDATION
    if _UNKNOWN.search(text):
        return ErrorClass.UNKNOWN_ENTITY
    # default treat unknown as validatin-shaped to LLM can try once
    return ErrorClass.VALIDATION


def recovery_for(error_class: ErrorClass, attempt: int = 0) -> RecoveryAction:
    action = _RECOVERY[error_class]

    if action is RecoveryAction.REPROMPT and attempt >= MAX_REPROMPT_ATTEMPTS:
        return RecoveryAction.REPLAN

    if action is RecoveryAction.RETRY and attempt >= MAX_RETRY_ATTEMPTS:
        return RecoveryAction.HALT  # fail-closed

    return action


def classify_and_recover(
    error: str | BaseException,
    attempt: int = 0,
) -> tuple[ErrorClass, RecoveryAction]:
    cls = classify_error(error)
    return cls, recovery_for(cls, attempt)
