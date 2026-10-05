"""Reconciliation methods.

Every method module exposes ``run(path_a, path_b, **params) -> float`` returning
the row-level match rate (matching rows / total rows). A module may also expose
``prepare()``, which the harness calls *outside* the timed region (used to warm
a JIT cache, for example).

Method metadata lives in :mod:`tabrecon.methods.registry` so it can be read
without importing the heavy libraries.
"""
