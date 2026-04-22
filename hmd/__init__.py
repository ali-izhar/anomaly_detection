"""Horizon Martingale Detection — Ali & Ho (ICDM 2025) reference implementation.

Public API: `HorizonDetector`, `DetectorConfig`, `DetectionResult`.
See `docs/theory/` for paper↔code mapping and `docs/usage.md` for examples.
"""

from hmd._backend import xp  # noqa: F401  (re-export for experiments)

__all__ = ["HorizonDetector", "DetectorConfig", "DetectionResult"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy import keeps `import hmd` cheap; detector pulls in scipy/networkx.
    if name in __all__:
        from hmd.detector import DetectionResult, DetectorConfig, HorizonDetector

        return {
            "HorizonDetector": HorizonDetector,
            "DetectorConfig": DetectorConfig,
            "DetectionResult": DetectionResult,
        }[name]
    raise AttributeError(name)
