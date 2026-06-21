from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from kgcl.config.paths import ProjectPaths

@dataclass(frozen=True)
class TrainingContext:
    device: str
    paths: ProjectPaths
    run_started_at: datetime
    output_dir: Path
