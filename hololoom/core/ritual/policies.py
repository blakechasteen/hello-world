"""
Failure policies for rituals and liturgies.

FailurePolicy — what happens when a single ritual fails.
LiturgyFailurePolicy — what happens when a multi-agent liturgy can't meet quorum.
"""

from enum import Enum


class FailurePolicy(Enum):
    SKIP = "skip"
    RETRY_ONCE = "retry_once"
    HALT = "halt"
    ESCALATE = "escalate"
    LOG_AND_CONTINUE = "log_and_continue"
    COMPENSATE = "compensate"


class LiturgyFailurePolicy(Enum):
    PARTIAL_QUORUM = "partial_quorum"
    DEFER = "defer"
    ABORT = "abort"
