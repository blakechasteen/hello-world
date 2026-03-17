"""
Weaving Constants
=================

Constants for the weaving pipeline, including the pre_satisfied
field set used by the resolver.
"""

# Fields available before any stage runs.
# The resolver needs these to know what's satisfied at Level 0.
#
# Why this set is small: only fields that stages actually declare in
# `reads` need to be here. Infrastructure fields accessed via helper
# methods (add_error, record_timing, is_blocked) don't participate in
# dependency resolution — they're in ReadWriteTracker's infra exclusion set.
#
# If a future stage needs to read safety_blocked or conscience_blocked,
# just add it here — it's already initialized on the context.
WEAVING_PRE_SATISFIED = frozenset({
    # Primary input (set by create_weaving_context)
    'query',
    'pattern_override',
    'complexity_override',
    'auto_enhance',

    # Infrastructure (initialized by factory, accumulated by runner)
    'start_time',
    'stage_timings',  # accumulated by on_stage_complete, read by production_metrics
})
