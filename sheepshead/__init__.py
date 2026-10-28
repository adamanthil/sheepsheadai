"""Sheepshead RL core package.

Layers, bottom to top (a package imports only from layers below it;
sheepshead/tests/test_import_layering.py enforces this):

  game, rewards, scripted_agent   the engine, scoring and reward math
  agent       networks, PPO, observation, deploy sampling
  search      the ISMCTS teacher
  rl          training building blocks (episode runtime, corpus plumbing,
              search-advantage model, entropy controller, watchdog)
  eval        maintained measurement instruments and model artifacts
  training    trainers, league, corpus/policy-iteration drivers, the
              program orchestrator, and their configs

The app (app/) imports only the engine and agent layers.

and two leaves that may import any of those but are imported by nothing:
``research`` (journal experiments, incl. the ISMCTS-era ``validation``)
and ``devtools`` (golden gate, benchmarks).
"""
