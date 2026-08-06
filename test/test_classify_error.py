from classify_error import (
    ErrorClass,
    RecoveryAction,
    classify_and_recover,
    recovery_for,
)


def test_recovery_follows_class():
    assert recovery_for(ErrorClass.TRANSIENT) is RecoveryAction.RETRY
    assert recovery_for(ErrorClass.VALIDATION) is RecoveryAction.REPROMPT
    assert recovery_for(ErrorClass.UNKNOWN_ENTITY) is RecoveryAction.REPLAN
    assert recovery_for(ErrorClass.POLICY) is RecoveryAction.HALT


def test_classify_and_recover():
    assert classify_and_recover("429 rate limit") == (
        ErrorClass.TRANSIENT,
        RecoveryAction.RETRY,
    )
    assert classify_and_recover("KeyError: 'Atlantis'") == (
        ErrorClass.UNKNOWN_ENTITY,
        RecoveryAction.REPLAN,
    )


def test_reprompt_escalates_to_replan():
    assert recovery_for(ErrorClass.VALIDATION, attempt=0) is RecoveryAction.REPROMPT
    assert recovery_for(ErrorClass.VALIDATION, attempt=1) is RecoveryAction.REPLAN


def test_retry_escalates_after_cap():
    assert recovery_for(ErrorClass.TRANSIENT, attempt=2) is RecoveryAction.RETRY
    assert recovery_for(ErrorClass.TRANSIENT, attempt=3) is RecoveryAction.HALT
