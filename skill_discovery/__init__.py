"""Viability-Gated Feedforward Compensation (VGFC) for learned continuous-control policies.

Core pipeline: excite a converged policy -> identify its command-response model ->
invert the model at the command interface behind viability gates. The package also
contains the retrieval-based variant (VGSR): an online archive of behavior chunks
with body-frame outcome descriptors, reused as gated command residuals.
"""

__version__ = "1.0.0"
