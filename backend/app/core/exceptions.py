from __future__ import annotations


class AVFTRPDEError(Exception):
    """Base exception for AVF-TRPDE."""


class ConfigurationError(AVFTRPDEError):
    """Raised when application configuration is invalid."""


class DataError(AVFTRPDEError):
    """Base exception for dataset and data-processing errors."""


class DataValidationError(DataError):
    """Raised when dataset validation fails."""


class DataLoadError(DataError):
    """Raised when a dataset cannot be loaded."""


class DataCanonicalizationError(DataError):
    """Raised when data cannot be converted to canonical form."""


class DataFingerprintError(DataError):
    """Raised when dataset fingerprinting fails."""


class ResearchError(AVFTRPDEError):
    """Base exception for research-pipeline errors."""


class FeatureEngineeringError(ResearchError):
    """Raised when feature construction fails."""


class ModelError(ResearchError):
    """Base exception for model-related errors."""


class WalkForwardError(ResearchError):
    """Raised when walk-forward execution fails."""


class EvaluationError(ResearchError):
    """Raised when forecast evaluation fails."""


class RiskError(AVFTRPDEError):
    """Base exception for risk-analysis errors."""


class BacktestingError(RiskError):
    """Raised when risk backtesting fails."""


class PortfolioError(AVFTRPDEError):
    """Base exception for portfolio-processing errors."""


class StressTestError(AVFTRPDEError):
    """Raised when stress testing fails."""


class ExperimentError(AVFTRPDEError):
    """Raised when experiment execution fails."""