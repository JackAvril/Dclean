#!/bin/bash
set -e

cd "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan"
python build_rule_pool.py
python shrink_search_space.py
python summarize_full_contexts.py
python cluster_and_sample_candidates.py
python annotate_with_deepseek.py
python label_propagation_for_candidates.py
python build_final_training_set.py
python build_loading_set.py
python control_iteration.py