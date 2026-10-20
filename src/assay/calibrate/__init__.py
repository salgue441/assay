"""
Calibration: does the composite score order documents the way human raters
do? Pure, deterministic functions with no I/O; `study` combines the
statistics over stored scores and ratings.
"""

from assay.calibrate.ablation import Ablation, AblationStudy, Outcome, ablate
from assay.calibrate.agreement import (
    AlphaLevel,
    KappaWeights,
    krippendorff_alpha,
    weighted_kappa,
    weighted_kappa_estimate,
)
from assay.calibrate.bootstrap import Estimate, bootstrap_interval, resample_indices
from assay.calibrate.correlation import RankMethod, rank_correlation
from assay.calibrate.decisions import (
    DecisionQuality,
    Routing,
    choose_routing,
    decision_quality,
)
from assay.calibrate.split import Split, grouped_split
from assay.calibrate.study import (
    CalibrationReport,
    Disagreement,
    FittedWeights,
    RatedDocument,
    calibrate,
    rated_documents,
)
from assay.calibrate.weights import choose_shrinkage, fit_weights, grouped_folds

__all__ = [
    "Ablation",
    "AblationStudy",
    "AlphaLevel",
    "CalibrationReport",
    "DecisionQuality",
    "Disagreement",
    "Estimate",
    "FittedWeights",
    "KappaWeights",
    "Outcome",
    "RankMethod",
    "RatedDocument",
    "Routing",
    "Split",
    "ablate",
    "bootstrap_interval",
    "calibrate",
    "choose_routing",
    "choose_shrinkage",
    "decision_quality",
    "fit_weights",
    "grouped_folds",
    "grouped_split",
    "krippendorff_alpha",
    "rank_correlation",
    "rated_documents",
    "resample_indices",
    "weighted_kappa",
    "weighted_kappa_estimate",
]
