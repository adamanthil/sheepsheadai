"""Reinforcement-learning building blocks the trainers are assembled from:
the population episode runtime and committee-search targets
(pfsp_runtime), distillation-corpus row plumbing (corpus_rows), the
search-advantage model (search_advantage), the target-entropy controller,
the leaster watchdog, and seeding/CSV/plot helpers (training_utils).

Layering: imports only core, agent and search. Trainer configuration
(config) and the program's decision rules (stop_rules) stay in
``sheepshead.training`` beside the trainers they control.
"""
