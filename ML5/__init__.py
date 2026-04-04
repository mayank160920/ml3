"""
Milestone 5 batch pipeline helpers.

This package keeps Mayank Dode's M5 deliverables isolated from the
core CMSVS implementation:
  - configuration-driven batch manifests
  - SBC / FUNSD pair collection
  - batch execution over the existing CMSVSPipeline
  - system output file generation
"""

from .configuration import BatchJobConfig, BatchJobConfigParser, BatchPair
from .pipeline import Milestone5Pipeline

__all__ = [
    "BatchJobConfig",
    "BatchJobConfigParser",
    "BatchPair",
    "Milestone5Pipeline",
]
