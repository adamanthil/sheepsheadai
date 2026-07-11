"""Measurement instruments: rigorous_eval, probes, panel tooling.

Layering rule: modules here must never import trainers, the league, or
orchestrators (``sheepshead.training.train_*``, ``league``, ``exploiter``,
``pfsp_runtime``, ``run_*``) — orchestrators sit on top of instruments,
not the reverse. Importing ``sheepshead.training.training_utils`` (shared
reward/eval utilities) is allowed.
"""
