"""Training: the PPO trainer and its league machinery (roster, worker
streams), oracle pretraining, distillation-corpus generation, policy
iteration, the training-program orchestrator and ablation drivers, and
the configuration (config) and decision rules (stop_rules) they run on.

The top ranked layer: built from ``sheepshead.rl`` and gated by
``sheepshead.eval``; only the research and devtools leaves import it.
"""
