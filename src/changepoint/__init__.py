# src/changepoint/__init__.py

"""Change point detection module.

Core components:
- ChangePointDetector: Parallel martingale detector (Algorithm 1 from ICDM 2025 paper)
- CUSUMDetector: CUSUM baseline
- EWMADetector: EWMA baseline
"""

from .detector import ChangePointDetector, DetectorConfig
from .baselines import CUSUMDetector, CUSUMConfig, EWMADetector, EWMAConfig
from .betting import BettingConfig, create_betting_function
from .martingale import MartingaleConfig, run_parallel_detection

__all__ = [
    # Main detector
    "ChangePointDetector",
    "DetectorConfig",
    # Baselines
    "CUSUMDetector",
    "CUSUMConfig",
    "EWMADetector",
    "EWMAConfig",
    # Low-level API
    "BettingConfig",
    "MartingaleConfig",
    "run_parallel_detection",
    "create_betting_function",
]
