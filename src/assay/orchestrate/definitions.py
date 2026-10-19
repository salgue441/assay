"""
Entry point for `dagster dev`: the pipeline for the config in
`ASSAY_CONFIG_FILE` (default `config/default.yaml`), relative to the
directory Dagster runs from.
"""

import os
from pathlib import Path

from assay.orchestrate.pipeline import DEFAULT_CONFIG, build_pipeline

config_path = Path(os.environ.get("ASSAY_CONFIG_FILE") or DEFAULT_CONFIG)
defs = build_pipeline(config_path).definitions()
