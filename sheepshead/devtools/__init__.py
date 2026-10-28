"""Developer tooling: the bit-exact golden gate, model-artifact fetching,
and performance benchmarks/audits.

Layering: devtools sits below ``sheepshead.analysis`` (instruments import
``fetch_models``), so it never imports ``sheepshead.analysis``, nor the
trainers or orchestrators in ``sheepshead.training``.
"""
