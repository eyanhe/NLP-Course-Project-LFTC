import os
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT))
from research_v3.supplemental import score_trajectories
from research_v3.storage import read_json
assert os.environ.get('SLURM_JOB_ID')
root=PROJECT/'research-v3'
score_trajectories(root,read_json(root/'reports/overnight/supplement-1-4/manifest.json'))
