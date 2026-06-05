#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified predicted-error-position exporter for Hospital, Flights, Beers, Rayyan,
Adult, and Soccer.

目标：
1. 六个数据集共用同一套主逻辑：
   read inference_all_predictions.csv
   -> select predicted-error cells
   -> keep dataset-specific columns
   -> apply dataset-specific type cleanup
   -> optional dedup
   -> sort by row_id,column
   -> write predicted_error_positions.csv

2. 不改变各数据集原来的 CSV 输出逻辑：
   - Hospital / Flights: 保留最简字段，不去重。
   - Beers / Rayyan: 保留原专属字段，按 row_id,column 去重。
   - Adult / Soccer: 保留原专属字段，不去重，支持 --keep_correct_for_debug。

用法示例：
    python unified_wrong_position.py --dataset hospital
    python unified_wrong_position.py --dataset flights
    python unified_wrong_position.py --dataset beers
    python unified_wrong_position.py --dataset rayyan
    python unified_wrong_position.py --dataset adult
    python unified_wrong_position.py --dataset soccer

也可以指定输入输出：
    python unified_wrong_position.py --dataset soccer \
      --input_file "/path/to/inference_all_predictions.csv" \
      --output_file predicted_error_positions.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {'hospital': {'default_inputs': ['/mnt/mydata/dq/projects/Splittree/test/Error Detection new/active_cleaning_loop_runs_v4/iter_3/infer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'final_pred_error_prob', 'final_pred_label_name'], 'numeric_cols': [], 'int_cols': [], 'dedup': False, 'simple_label_only': True, 'title': 'Hospital'}, 'flights': {'default_inputs': ['/mnt/mydata/dq/projects/Splittree/test/Error Detection flights/active_cleaning_loop_runs_flights_v4/iter_3/infer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'final_pred_error_prob', 'final_pred_label_name'], 'numeric_cols': [], 'int_cols': [], 'dedup': False, 'simple_label_only': True, 'title': 'Flights'}, 'beers': {'default_inputs': ['/mnt/mydata/dq/projects/Splittree/test/Error Detection beers/active_cleaning_loop_runs_beers_v4/iter_3/infer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'label', 'label_binary', 'label_source', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'sample_role', 'is_sampled', 'error_type', 'reason_short', 'reason_detailed', 'suggested_correct_value', 'needs_human_review', 'evidence_used', 'semantic_type', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'numeric_value', 'neighbor_majority_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'main_rule_type', 'main_usage_role', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'numeric_window_rule_count', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'candidate_rule_dominant', 'pattern_bucket', 'rarity_bucket', 'neighbor_bucket', 'numeric_window_bucket', 'numeric_value_bucket', 'bucket_id', 'cluster_id', 'dist_to_center', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'rule_neighbor_conflict', 'numeric_borderline_hardcase', 'semantic_recovery_hardcase', 'persistent_fp_high_risk', 'ounces_canonical_reject_hardcase', 'is_hard_case', 'is_high_risk', 'raw_pred_error_prob', 'raw_pred_label', 'raw_pred_label_name', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_pred_label_name', 'detector_margin_to_threshold', 'detector_threshold_used', 'is_stage_disagree', 'verifier_keep_error_prob', 'verifier_decision', 'verifier_threshold_used', 'group_verifier_threshold', 'final_pred_label', 'final_pred_label_name', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline'], 'numeric_cols': ['label_binary', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'numeric_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'numeric_window_rule_count', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'bucket_id', 'cluster_id', 'dist_to_center', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'raw_pred_error_prob', 'raw_pred_label', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_margin_to_threshold', 'detector_threshold_used', 'is_stage_disagree', 'verifier_keep_error_prob', 'verifier_threshold_used', 'group_verifier_threshold', 'final_pred_label', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline', 'rule_neighbor_conflict', 'numeric_borderline_hardcase', 'semantic_recovery_hardcase', 'persistent_fp_high_risk', 'ounces_canonical_reject_hardcase', 'is_hard_case', 'is_high_risk'], 'int_cols': [], 'dedup': True, 'simple_label_only': False, 'title': 'Beers'}, 'rayyan': {'default_inputs': ['/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/active_cleaning_loop_runs_rayyan_v4/iter_3/infer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'label', 'label_binary', 'label_source', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'sample_role', 'is_sampled', 'error_type', 'reason_short', 'reason_detailed', 'suggested_correct_value', 'needs_human_review', 'evidence_used', 'semantic_type', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'numeric_value', 'date_value', 'language_canonical', 'issn_canonical', 'neighbor_majority_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'main_rule_type', 'main_usage_role', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'canonical_rule_count', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'candidate_rule_dominant', 'pattern_bucket', 'rarity_bucket', 'neighbor_bucket', 'canonical_bucket', 'date_value_bucket', 'language_bucket', 'bucket_id', 'cluster_id', 'dist_to_center', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'rule_neighbor_conflict', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'is_high_risk', 'raw_pred_error_prob', 'raw_pred_label', 'raw_pred_label_name', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_pred_label_name', 'detector_margin_to_threshold', 'detector_threshold_used', 'is_stage_disagree', 'verifier_keep_error_prob', 'verifier_decision', 'verifier_threshold_used', 'group_verifier_threshold', 'verifier_bypass', 'final_pred_label', 'final_pred_label_name', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline'], 'numeric_cols': ['label_binary', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'numeric_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'canonical_rule_count', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'bucket_id', 'cluster_id', 'dist_to_center', 'is_sampled', 'needs_human_review', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'rule_neighbor_conflict', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'is_high_risk', 'raw_pred_error_prob', 'raw_pred_label', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_margin_to_threshold', 'detector_threshold_used', 'is_stage_disagree', 'verifier_keep_error_prob', 'verifier_threshold_used', 'group_verifier_threshold', 'verifier_bypass', 'final_pred_label', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline'], 'int_cols': [], 'dedup': True, 'simple_label_only': False, 'title': 'Rayyan'}, 'adult': {'default_inputs': ['/mnt/mydata/dq/projects/Splittree/test/Error Detection Adult/active_cleaning_loop_runs_adult_v4/iter_3_infer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'label', 'label_binary', 'label_source', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'final_pred_label', 'final_pred_label_name', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_pred_label_name', 'detector_margin_to_threshold', 'detector_threshold_used', 'is_stage_disagree', 'raw_pred_error_prob', 'raw_pred_label', 'raw_pred_label_name', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'verifier_keep_error_prob', 'verifier_decision', 'verifier_threshold_used', 'group_verifier_threshold', 'semantic_type', 'detected_type', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'neighbor_majority_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'main_rule_type', 'main_usage_role', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'time_window_rule_count', 'time_value_minutes', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'candidate_rule_dominant', 'rule_neighbor_conflict', 'adult_domain_rule_count', 'adult_format_rule_count', 'adult_consistency_rule_count', 'adult_rule_total_count', 'adult_domain_rule_ratio', 'adult_format_rule_ratio', 'adult_consistency_rule_ratio', 'domain_invalid_flag', 'adult_consistency_flag', 'high_adult_consistency_score', 'has_adult_domain_signal', 'has_adult_format_signal', 'has_adult_consistency_signal', 'adult_signal_hardcase', 'age_lower', 'age_upper', 'hours_lower', 'hours_upper', 'canonical_income', 'domain_valid', 'education_order', 'adult_value_bucket', 'adult_profile_inconsistency_count', 'adult_consistency_score', 'adult_evidence_flags', 'row_age_lower', 'row_age_upper', 'row_hours_lower', 'row_hours_upper', 'row_education_order', 'row_income_canonical', 'row_missing_count', 'row_non_missing_count', 'age_span', 'hours_span', 'is_minor_age_bucket', 'is_high_hours_bucket', 'target_is_primary_column', 'target_is_context_only_column', 'is_context_only_weak_signal', 'is_relationship_spouse_value', 'relationship_marital_conflict', 'relationship_age_conflict', 'relationship_sex_conflict', 'sex_value_invalid', 'education_age_extreme_conflict', 'has_relationship_v2_rule', 'has_education_v2_rule', 'adult_v2_attribution_bucket', 'relationship_v2_conflict_count', 'has_relationship_v2_signal', 'has_education_v2_signal', 'adult_v2_signal_strength', 'adult_v2_primary_target_signal', 'adult_v2_relationship_hardcase', 'adult_v2_sex_invalid_hardcase', 'adult_v2_context_fp_hardcase', 'pattern_bucket', 'rarity_bucket', 'neighbor_bucket', 'domain_bucket', 'adult_consistency_bucket', 'age_sampling_bucket', 'hours_sampling_bucket', 'time_window_bucket', 'time_value_bucket', 'bucket_id', 'cluster_id', 'dist_to_center', 'sample_role', 'is_sampled', 'signal_priority', 'error_type', 'reason_short', 'reason_detailed', 'suggested_correct_value', 'needs_human_review', 'evidence_used', 'current_value_pattern', 'dominant_pattern', 'dominant_pattern_ratio', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'high_prob_but_neighbor_support_current', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'is_high_risk'], 'numeric_cols': ['sample_weight', 'propagation_confidence', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_margin_to_threshold', 'detector_threshold_used', 'raw_pred_error_prob', 'raw_margin_to_threshold', 'verifier_keep_error_prob', 'verifier_threshold_used', 'group_verifier_threshold', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'time_window_rule_count', 'time_value_minutes', 'candidate_rule_ratio', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'adult_domain_rule_count', 'adult_format_rule_count', 'adult_consistency_rule_count', 'adult_rule_total_count', 'adult_domain_rule_ratio', 'adult_format_rule_ratio', 'adult_consistency_rule_ratio', 'age_lower', 'age_upper', 'hours_lower', 'hours_upper', 'education_order', 'adult_profile_inconsistency_count', 'adult_consistency_score', 'row_age_lower', 'row_age_upper', 'row_hours_lower', 'row_hours_upper', 'row_education_order', 'row_missing_count', 'row_non_missing_count', 'age_span', 'hours_span', 'relationship_v2_conflict_count', 'adult_v2_signal_strength', 'dist_to_center', 'dominant_pattern_ratio'], 'int_cols': ['label_binary', 'is_outside_candidate', 'final_pred_label', 'detector_pred_label', 'raw_pred_label', 'stage1_high_conf_error', 'stage1_mid_zone', 'is_stage_disagree', 'domain_invalid_flag', 'adult_consistency_flag', 'high_adult_consistency_score', 'has_adult_domain_signal', 'has_adult_format_signal', 'has_adult_consistency_signal', 'target_is_primary_column', 'target_is_context_only_column', 'is_relationship_spouse_value', 'relationship_marital_conflict', 'relationship_age_conflict', 'relationship_sex_conflict', 'sex_value_invalid', 'education_age_extreme_conflict', 'has_relationship_v2_rule', 'has_education_v2_rule', 'is_minor_age_bucket', 'is_high_hours_bucket', 'has_relationship_v2_signal', 'has_education_v2_signal', 'adult_v2_primary_target_signal', 'is_context_only_weak_signal', 'is_sampled', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'high_prob_but_neighbor_support_current', 'is_mid_prob', 'is_near_threshold', 'rule_neighbor_conflict', 'adult_signal_hardcase', 'verifier_borderline', 'verifier_reject_borderline', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'adult_v2_relationship_hardcase', 'adult_v2_sex_invalid_hardcase', 'adult_v2_context_fp_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'is_high_risk', 'selected_as_error'], 'dedup': False, 'simple_label_only': False, 'title': 'Adult'}, 'soccer': {'default_inputs': ['inference_all_predictions.csv', './inference_all_predictions.csv', './infer/inference_all_predictions.csv', './active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv', './active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_2_infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_2/infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_1_infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/active_cleaning_loop_runs_soccer_v4/iter_1/infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection Soccer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/active_cleaning_loop_runs_soccer_v4/iter_3_infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/active_cleaning_loop_runs_soccer_v4/iter_3/infer/inference_all_predictions.csv', '/mnt/mydata/dq/projects/Splittree/test/Error Detection soccer/inference_all_predictions.csv'], 'output': 'predicted_error_positions.csv', 'wanted_cols': ['row_id', 'column', 'value', 'label', 'label_binary', 'label_source', 'sample_weight', 'is_outside_candidate', 'propagation_confidence', 'final_pred_label', 'final_pred_label_name', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_pred_label', 'detector_pred_label_name', 'detector_margin_to_threshold', 'detector_threshold', 'detector_threshold_used', 'is_stage_disagree', 'raw_pred_error_prob', 'raw_pred_label', 'raw_pred_label_name', 'raw_margin_to_threshold', 'stage1_high_conf_error', 'stage1_mid_zone', 'verifier_used', 'verifier_keep_error_prob', 'verifier_decision', 'verifier_threshold_used', 'group_verifier_threshold', 'verifier_rejected', 'semantic_type', 'detected_type', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'neighbor_majority_value', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'main_rule_type', 'main_usage_role', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'candidate_rule_ratio', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'candidate_rule_dominant', 'rule_neighbor_conflict', 'evidence_only_fd_count', 'soccer_domain_rule_count', 'soccer_format_rule_count', 'soccer_numeric_rule_count', 'soccer_consistency_rule_count', 'soccer_rule_total_count', 'soccer_domain_rule_ratio', 'soccer_format_rule_ratio', 'soccer_numeric_rule_ratio', 'soccer_consistency_rule_ratio', 'domain_invalid_flag', 'soccer_consistency_flag', 'high_soccer_consistency_score', 'has_soccer_domain_signal', 'has_soccer_format_signal', 'has_soccer_numeric_signal', 'has_soccer_consistency_signal', 'soccer_signal_strength', 'soccer_primary_target_signal', 'year_value', 'canonical_position', 'domain_valid', 'soccer_value_bucket', 'row_birthyear_int', 'row_season_int', 'row_age_at_season', 'row_position_canonical', 'birthyear_season_gap', 'is_young_player_age', 'is_old_player_age', 'position_invalid_or_normalization', 'target_is_primary_column', 'target_is_context_only_column', 'is_context_only_weak_signal', 'birthyear_value_invalid', 'season_value_invalid', 'position_value_invalid', 'position_needs_canonicalization', 'birthyear_season_age_conflict', 'team_context_conflict', 'format_rule_signal', 'fd_like_signal', 'has_birthyear_or_season_signal', 'has_team_context_signal', 'has_fd_like_signal', 'soccer_profile_inconsistency_count', 'soccer_consistency_score', 'soccer_evidence_flags', 'soccer_attribution_bucket', 'soccer_consistency_bucket', 'soccer_precision_gate_applied', 'soccer_precision_gate_reason', 'row_team', 'row_city', 'row_stadium', 'row_manager', 'pattern_bucket', 'rarity_bucket', 'neighbor_bucket', 'domain_bucket', 'soccer_consistency_bucket', 'bucket_id', 'cluster_id', 'dist_to_center', 'sample_role', 'is_sampled', 'signal_priority', 'error_type', 'reason_short', 'reason_detailed', 'suggested_correct_value', 'needs_human_review', 'evidence_used', 'current_value_pattern', 'dominant_pattern', 'dominant_pattern_ratio', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'high_prob_but_neighbor_support_current', 'is_mid_prob', 'is_near_threshold', 'verifier_borderline', 'verifier_reject_borderline', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'hard_case_reason', 'is_high_risk', 'high_risk', 'adult_domain_rule_count', 'adult_format_rule_count', 'adult_consistency_rule_count', 'age_lower', 'age_upper', 'hours_lower', 'hours_upper', 'canonical_income', 'education_order', 'adult_value_bucket', 'adult_profile_inconsistency_count', 'adult_consistency_score', 'adult_evidence_flags', 'adult_consistency_bucket', 'age_sampling_bucket', 'hours_sampling_bucket', 'adult_v2_attribution_bucket', 'time_window_rule_count', 'time_value_minutes', 'time_window_bucket', 'time_value_bucket', 'adult_rule_total_count', 'adult_domain_rule_ratio', 'adult_format_rule_ratio', 'adult_consistency_rule_ratio', 'adult_consistency_flag', 'high_adult_consistency_score', 'has_adult_domain_signal', 'has_adult_format_signal', 'has_adult_consistency_signal', 'age_span', 'hours_span', 'is_minor_age_bucket', 'is_high_hours_bucket', 'relationship_v2_conflict_count', 'has_relationship_v2_signal', 'has_education_v2_signal', 'adult_v2_signal_strength', 'adult_v2_primary_target_signal', 'row_missing_count', 'row_non_missing_count', 'row_age_lower', 'row_age_upper', 'row_hours_lower', 'row_hours_upper', 'row_education_order', 'row_income_canonical'], 'numeric_cols': ['sample_weight', 'propagation_confidence', 'final_pred_error_prob_before_adjust', 'final_pred_error_prob', 'detector_pred_error_prob_before_adjust', 'detector_pred_error_prob', 'detector_margin_to_threshold', 'detector_threshold', 'detector_threshold_used', 'raw_pred_error_prob', 'raw_margin_to_threshold', 'verifier_keep_error_prob', 'verifier_threshold_used', 'group_verifier_threshold', 'violation_count', 'conflict_score', 'value_frequency', 'value_frequency_rank', 'neighbor_majority_ratio', 'prior_error_probability', 'posterior_error_probability', 'strong_rule_count', 'candidate_generation_rule_count', 'fd_like_count', 'context_rule_count', 'global_rule_count', 'rare_value_count', 'typo_rule_count', 'pattern_rule_count', 'schema_rule_count', 'candidate_rule_ratio', 'neighbor_agreement_score', 'rule_total_count', 'strong_rule_ratio', 'context_rule_ratio', 'fd_like_ratio', 'global_rule_ratio', 'rule_strength_sum', 'evidence_only_fd_count', 'soccer_domain_rule_count', 'soccer_format_rule_count', 'soccer_numeric_rule_count', 'soccer_consistency_rule_count', 'soccer_rule_total_count', 'soccer_domain_rule_ratio', 'soccer_format_rule_ratio', 'soccer_numeric_rule_ratio', 'soccer_consistency_rule_ratio', 'soccer_signal_strength', 'year_value', 'row_birthyear_int', 'row_season_int', 'row_age_at_season', 'birthyear_season_gap', 'soccer_profile_inconsistency_count', 'soccer_consistency_score', 'dist_to_center', 'dominant_pattern_ratio', 'adult_domain_rule_count', 'adult_format_rule_count', 'adult_consistency_rule_count', 'age_lower', 'age_upper', 'hours_lower', 'hours_upper', 'education_order', 'adult_profile_inconsistency_count', 'adult_consistency_score', 'time_window_rule_count', 'time_value_minutes', 'adult_rule_total_count', 'adult_domain_rule_ratio', 'adult_format_rule_ratio', 'adult_consistency_rule_ratio', 'age_span', 'hours_span', 'relationship_v2_conflict_count', 'adult_v2_signal_strength', 'row_missing_count', 'row_non_missing_count', 'row_age_lower', 'row_age_upper', 'row_hours_lower', 'row_hours_upper', 'row_education_order'], 'int_cols': ['label_binary', 'is_outside_candidate', 'final_pred_label', 'detector_pred_label', 'raw_pred_label', 'stage1_high_conf_error', 'stage1_mid_zone', 'is_stage_disagree', 'verifier_used', 'verifier_rejected', 'target_is_primary_column', 'target_is_context_only_column', 'birthyear_value_invalid', 'season_value_invalid', 'position_value_invalid', 'position_needs_canonicalization', 'birthyear_season_age_conflict', 'team_context_conflict', 'format_rule_signal', 'fd_like_signal', 'has_neighbor_signal', 'high_neighbor_consistency', 'is_rare_value', 'neighbor_disagree', 'is_equal_to_neighbor_majority', 'candidate_rule_dominant', 'domain_invalid_flag', 'soccer_consistency_flag', 'high_soccer_consistency_score', 'has_soccer_domain_signal', 'has_soccer_format_signal', 'has_soccer_numeric_signal', 'has_soccer_consistency_signal', 'is_young_player_age', 'is_old_player_age', 'position_invalid_or_normalization', 'has_birthyear_or_season_signal', 'has_team_context_signal', 'has_fd_like_signal', 'is_context_only_weak_signal', 'soccer_primary_target_signal', 'soccer_precision_gate_applied', 'is_sampled', 'rare_only_signal', 'is_weak_only', 'is_fp_risk_col', 'high_prob_but_neighbor_support_current', 'is_mid_prob', 'is_near_threshold', 'rule_neighbor_conflict', 'verifier_borderline', 'verifier_reject_borderline', 'fp_conflict_fd_hardcase', 'sample_borderline_hardcase', 'recall_recovery_hardcase', 'persistent_fp_high_risk', 'is_hard_case', 'is_high_risk', 'high_risk', 'adult_consistency_flag', 'high_adult_consistency_score', 'has_adult_domain_signal', 'has_adult_format_signal', 'has_adult_consistency_signal', 'is_minor_age_bucket', 'is_high_hours_bucket', 'has_relationship_v2_signal', 'has_education_v2_signal', 'adult_v2_primary_target_signal', 'selected_as_error'], 'dedup': False, 'simple_label_only': False, 'title': 'Soccer'}}


def safe_col_list(cols: List[str], df: pd.DataFrame) -> List[str]:
    """Return existing columns in the configured order, removing duplicates."""
    seen = set()
    out = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def to_numeric_if_exists(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def to_int_if_exists(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def resolve_input_file(input_file: str, default_inputs: List[str]) -> str:
    """
    Resolve input file robustly.

    If input_file is:
      - a CSV file: use it directly
      - a directory: search inference_all_predictions.csv inside it
      - empty: try dataset-specific default paths
    """
    p = str(input_file).strip() if input_file is not None else ""

    if p:
        if os.path.isdir(p):
            direct = os.path.join(p, "inference_all_predictions.csv")
            if os.path.exists(direct) and os.path.isfile(direct):
                return direct

            recursive_hits = list(Path(p).glob("**/inference_all_predictions.csv"))
            recursive_hits = [x for x in recursive_hits if x.is_file()]
            if len(recursive_hits) == 1:
                return str(recursive_hits[0])
            if len(recursive_hits) > 1:
                print("[WARN] 你传入的是目录，且里面找到多个 inference_all_predictions.csv：")
                for x in recursive_hits[:30]:
                    print(f"  - {x}")
                raise ValueError("请用 --input_file 指定其中一个具体 CSV 文件。")

            raise IsADirectoryError(
                f"你传入的 input_file 是目录，不是 CSV 文件: {p}\n"
                f"并且该目录下没有找到 inference_all_predictions.csv。"
            )

        if not os.path.exists(p):
            raise FileNotFoundError(f"指定的 input_file 不存在: {p}")
        if not os.path.isfile(p):
            raise ValueError(f"指定的 input_file 不是普通文件: {p}")
        return p

    for cand in default_inputs:
        if os.path.exists(cand) and os.path.isfile(cand):
            return cand

    msg = [
        "没有找到输入文件。请用 --input_file 指定 inference_all_predictions.csv。",
        "",
        "脚本尝试过以下路径：",
    ]
    for cand in default_inputs:
        msg.append(f"  - {cand}")
    raise FileNotFoundError("\n".join(msg))


def build_error_mask(df: pd.DataFrame, simple_label_only: bool = False) -> pd.Series:
    """
    Build predicted-error mask.

    For Hospital/Flights simple exporter, original logic only checked:
      final_pred_label or final_pred_label_name.
    For other datasets, fallback order is:
      final -> detector -> raw.
    """
    if "final_pred_label" in df.columns:
        if not simple_label_only:
            print("[INFO] 使用 final_pred_label == 1 作为最终错误位置")
        return pd.to_numeric(df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "final_pred_label_name" in df.columns:
        if not simple_label_only:
            print("[INFO] 使用 final_pred_label_name == error 作为最终错误位置")
        return df["final_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    if simple_label_only:
        raise ValueError("文件中没有找到 final_pred_label 或 final_pred_label_name 字段")

    if "detector_pred_label" in df.columns:
        print("[WARN] 没有 final_pred_label，退回使用 detector_pred_label")
        return pd.to_numeric(df["detector_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "detector_pred_label_name" in df.columns:
        print("[WARN] 没有 final_pred_label_name，退回使用 detector_pred_label_name")
        return df["detector_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    if "raw_pred_label" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label")
        return pd.to_numeric(df["raw_pred_label"], errors="coerce").fillna(0).astype(int) == 1

    if "raw_pred_label_name" in df.columns:
        print("[WARN] 没有 final/detector 标签，退回使用 raw_pred_label_name")
        return df["raw_pred_label_name"].astype(str).str.strip().str.lower().eq("error")

    raise ValueError(
        "文件中没有找到可用预测标签字段："
        "final_pred_label / final_pred_label_name / detector_pred_label / "
        "detector_pred_label_name / raw_pred_label / raw_pred_label_name"
    )


def export_predicted_error_positions(
    dataset: str,
    input_file: str = "",
    output_file: str = "",
    keep_correct_for_debug: bool = False,
) -> pd.DataFrame:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"未知 dataset={dataset}，可选：{sorted(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[dataset]
    title = cfg.get("title", dataset)

    resolved_input = resolve_input_file(input_file, cfg.get("default_inputs", []))
    resolved_output = output_file or cfg.get("output", "predicted_error_positions.csv")

    df = pd.read_csv(resolved_input, encoding="utf-8-sig")

    print("读取完成")
    print(f"dataset: {dataset}")
    print(f"输入文件: {resolved_input}")
    print(f"总记录数: {len(df)}")
    print(f"字段数: {len(df.columns)}")

    required = {"row_id", "column"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要定位字段: {missing}")

    error_mask = build_error_mask(df, simple_label_only=bool(cfg.get("simple_label_only", False)))

    if keep_correct_for_debug:
        error_df = df.copy()
        error_df["selected_as_error"] = error_mask.astype(int)
    else:
        error_df = df.loc[error_mask].copy()

    wanted_cols = list(cfg.get("wanted_cols", []))
    if keep_correct_for_debug and "selected_as_error" not in wanted_cols:
        wanted_cols = wanted_cols + ["selected_as_error"]

    keep_cols = safe_col_list(wanted_cols, error_df)
    result_df = error_df[keep_cols].copy()

    if "row_id" in result_df.columns:
        result_df["row_id"] = pd.to_numeric(result_df["row_id"], errors="raise").astype(int)

    if dataset in {"rayyan"} and "column" in result_df.columns:
        # Rayyan original script strips column; doing it only where the original did.
        result_df["column"] = result_df["column"].astype(str).str.strip()

    to_numeric_if_exists(result_df, cfg.get("numeric_cols", []))
    to_int_if_exists(result_df, cfg.get("int_cols", []))

    before_dedup = len(result_df)
    if cfg.get("dedup", False) and {"row_id", "column"}.issubset(set(result_df.columns)):
        result_df = result_df.drop_duplicates(subset=["row_id", "column"], keep="first")

    sort_cols = [c for c in ["row_id", "column"] if c in result_df.columns]
    if sort_cols:
        result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    if result_df.empty:
        print("[WARN] 没有筛选出任何预测错误单元格，请检查预测标签字段。")

    result_df.to_csv(resolved_output, index=False, encoding="utf-8-sig")

    print(f"\n========== {title} 预测错误单元格定位结果 ==========")
    if cfg.get("dedup", False):
        print(f"最终预测为错误的原始记录数: {len(error_df)}")
        print(f"去重前记录数: {before_dedup}")
        print(f"最终预测为错误的唯一单元格数: {len(result_df)}")
    else:
        print(f"最终预测为错误的单元格数: {len(result_df)}")
    print(f"输出文件: {os.path.abspath(resolved_output)}")

    if "column" in result_df.columns:
        topn = 30 if dataset in {"beers", "rayyan", "soccer"} else 20
        print(f"\n---- 各列预测错误数量 Top {topn} ----")
        print(result_df["column"].value_counts().head(topn).to_string())

    for col in [
        "semantic_type",
        "main_rule_type",
        "main_usage_role",
        "canonical_bucket",
        "date_value_bucket",
        "language_bucket",
        "adult_v2_attribution_bucket",
        "adult_consistency_bucket",
        "soccer_attribution_bucket",
        "soccer_consistency_bucket",
        "soccer_precision_gate_reason",
        "soccer_evidence_flags",
        "verifier_decision",
    ]:
        if col in result_df.columns:
            print(f"\n---- {col} 分布 Top 30 ----")
            print(result_df[col].value_counts(dropna=False).head(30).to_string())

    for flag_col in [
        "is_hard_case",
        "is_high_risk",
        "high_risk",
        "numeric_borderline_hardcase",
        "semantic_recovery_hardcase",
        "persistent_fp_high_risk",
        "ounces_canonical_reject_hardcase",
        "fp_conflict_fd_hardcase",
        "sample_borderline_hardcase",
        "recall_recovery_hardcase",
        "adult_signal_hardcase",
        "verifier_rejected",
    ]:
        if flag_col in result_df.columns:
            cnt = pd.to_numeric(result_df[flag_col], errors="coerce").fillna(0).astype(int).sum()
            print(f"{flag_col}=1 数量: {cnt}")

    print("\n---- 前 20 个错误位置 ----")
    if len(result_df) > 0:
        # Keep console display compact; CSV output is not affected.
        display_prefer = [
            "row_id", "column", "value",
            "final_pred_label_name", "final_pred_error_prob",
            "detector_pred_label_name", "detector_pred_error_prob",
            "semantic_type", "main_rule_type", "main_usage_role",
            "adult_evidence_flags", "soccer_evidence_flags",
            "soccer_attribution_bucket", "soccer_precision_gate_reason",
            "is_hard_case", "hard_case_reason", "is_high_risk", "high_risk",
        ]
        display_cols = safe_col_list(display_prefer, result_df)
        if not display_cols:
            display_cols = list(result_df.columns[:20])
        print(result_df[display_cols].head(20).to_string(index=False))
    else:
        print("[EMPTY]")

    return result_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset config to use.",
    )
    p.add_argument("--input_file", default="", help="inference_all_predictions.csv or a directory containing it.")
    p.add_argument("--output_file", default="", help="Output CSV. Default keeps original predicted_error_positions.csv.")
    p.add_argument(
        "--keep_correct_for_debug",
        action="store_true",
        help="输出全部 cell，并增加 selected_as_error 字段；默认只输出预测为 error 的 cell。",
    )
    return p.parse_args()


def main():
    args = parse_args()
    export_predicted_error_positions(
        dataset=args.dataset,
        input_file=args.input_file,
        output_file=args.output_file,
        keep_correct_for_debug=args.keep_correct_for_debug,
    )


if __name__ == "__main__":
    main()
