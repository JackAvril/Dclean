#!/bin/bash
set -e

cd "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies"
python wrong_position.py
python revise_candidates.py
python build_features.py
python build_infer_candidate_features_from_whitelist.py
python llm_ranking.py
python control_iteration.py
python evaluate.py