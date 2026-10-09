"""High-level planning and execution for multisensor AOI analysis."""

from .adapters import ProductAcquisitionError
from .limits import DEFAULT_REQUEST_LIMITS, RequestEstimate, RequestLimits, RequestTooLargeError, check_request, estimate_request
from .pairing import TemporalPair, select_temporal_pair
from .request import AnalysisRequest
from .result import AnalysisResult
from .runner import AnalysisWorkflow

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisWorkflow",
    "DEFAULT_REQUEST_LIMITS",
    "ProductAcquisitionError",
    "RequestEstimate",
    "RequestLimits",
    "RequestTooLargeError",
    "TemporalPair",
    "check_request",
    "estimate_request",
    "select_temporal_pair",
]
