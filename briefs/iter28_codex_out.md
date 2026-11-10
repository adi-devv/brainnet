Added [iter28.py](iter28.py): upper-layer leaky tanh with cached exact derivatives, startup checks, validation statistics, requested filenames/reference values, and preserved iter24 evaluation fields.

Static syntax and field-preservation checks passed.

**Smoke verification is blocked:** the requested Python executable returns “Access is denied”; both accessible alternatives lack `torch`. No losses or timing were produced, and runtime checks remain unverified.

No full training ran. `iter27.py` was not read, edited, or run.