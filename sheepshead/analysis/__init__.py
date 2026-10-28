"""Measurement instruments: rigorous_eval, probes, panel tooling.

Layering rule: instruments sit below the training stack. They may import
the training library (reward math, configs, corpus formats, the episode
runtime, supervised-fit helpers) but never ``train_ppo``, ``league``, or
any training module that imports those or ``sheepshead.analysis`` -- the
orchestrators sit on top of instruments, not the reverse. Study
definitions an orchestrator and its report share (``panels``,
``recall_ablation_study``) live here for that reason.
sheepshead/tests/test_import_layering.py enforces this.
"""
