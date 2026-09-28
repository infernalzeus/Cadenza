"""Cadenza — ML-driven generative music pipeline."""
import os

# PyTorch bundles its own Intel OpenMP; Anaconda's numpy/MKL loads another. When
# both are imported in one process they abort with "OMP Error #15". This is the
# documented, widely-used guard — safe here (light torch use; demucs is a subprocess).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.0.1"
