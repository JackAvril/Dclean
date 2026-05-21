"""
Rayyan candidate generation script, logic-migrated from the Flights candidate generation pipeline.

Migration principle:
- Keep generic candidate-generation framework:
  predicted error positions -> rich context map -> rule/neighbor/column/similarity/hint candidates -> expanded/grouped outputs.
- Convert Flights-specific consensus logic:
  flight/source/time consensus
      -> Rayyan journal/article metadata consensus
         journal_title / jounral_abbreviation / journal_issn -> article_language
         journal_title / jounral_abbreviation -> journal_issn
         journal_issn -> journal_title / jounral_abbreviation
         journal_title + article_jissue -> article_jvolumn
         journal_title + article_jvolumn -> article_jissue
- Preserve output compatibility:
  repair_candidates_expanded.csv
  repair_candidates_grouped.csv
  plus Rayyan consensus diagnostics and augmented whitelist.
"""

import os
import json
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# 1. 配置区（直接改这里）
# ============================================================

# 1) 错误位置文件
# 建议使用上一段 Rayyan 定位脚本导出的 predicted_error_positions.csv
ERROR_PRED_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction rayyan/predicted_error_positions.csv"

# 2) Rayyan rich context jsonl
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_rayyan.jsonl"

# 3) 可选 flat csv fallback；没有就留空
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection rayyan/candidate_llm_contexts_flat_rayyan.csv"

# 4) 可选原始脏表。用于构建 journal/article 的经验字典
DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/rayyan/rayyan_dirty.csv"

# 输出到当前运行目录
OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped.csv"

# Rayyan metadata-aware consensus 输出
OUTPUT_RAYYAN_CONSENSUS_CSV = "rayyan_metadata_consensus.csv"
OUTPUT_RAYYAN_CONSENSUS_SUSPICIOUS_CELLS_CSV = "rayyan_consensus_suspicious_cells.csv"
OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV = "predicted_error_positions_consensus_augmented.csv"

# 是否启用 Rayyan metadata consensus
ENABLE_RAYYAN_METADATA_CONSENSUS = True

# consensus 阈值
MIN_RAYYAN_CONSENSUS_SUPPORT_COUNT = 2
MIN_RAYYAN_CONSENSUS_CONFIDENCE = 0.62
MIN_RAYYAN_CONSENSUS_MARGIN = 0.20

# 自动扩展白名单时更保守：
# 只对 article_language / journal_issn / journal_title / jounral_abbreviation 这类强 metadata 字段扩展。
RAYYAN_CONSENSUS_ALLOW_COLUMNS = {
    "article_language",
    "journal_issn",
    "journal_title",
    "jounral_abbreviation",
    "journal_abbreviation",
}

RAYYAN_METADATA_CONSENSUS_STRONG_SCORE = 0.97
RAYYAN_METADATA_CONSENSUS_MEDIUM_SCORE = 0.89

# 参数
TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 8
MAX_CANDIDATES_PER_CELL = 25
MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.42
INCLUDE_DIRTY_VALUE = False

# 是否启用 Rayyan 专属候选生成
ENABLE_RAYYAN_SPECIFIC_GENERATION = True


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "none", "null"}

LANGUAGE_CANONICAL_MAP = {
    "eng": "eng",
    "english": "eng",
    "en": "eng",
    "engl": "eng",
    "anglais": "eng",

    "fre": "fre",
    "french": "fre",
    "fr": "fre",
    "fra": "fre",

    "ger": "ger",
    "german": "ger",
    "de": "ger",
    "deu": "ger",

    "spa": "spa",
    "spanish": "spa",
    "es": "spa",

    "jpn": "jpn",
    "japanese": "jpn",
    "ja": "jpn",

    "chi": "chi",
    "chinese": "chi",
    "zh": "chi",

    "ita": "ita",
    "italian": "ita",
    "it": "ita",

    "dut": "dut",
    "dutch": "dut",
    "nl": "dut",

    "pol": "pol",
    "polish": "pol",

    "por": "por",
    "portuguese": "por",

    "rus": "rus",
    "russian": "rus",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in EMPTY_TOKENS:
        return "empty"
    return s


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def path_exists_nonempty(path):
    return path is not None and str(path).strip() != "" and os.path.exists(str(path))


def resolve_output_path(output_path):
    if os.path.isabs(str(output_path)):
        return str(output_path)
    return os.path.abspath(str(output_path))


def string_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def clean_key(x):
    return re.sub(r"\s+", " ", norm_text(x)).strip()


# ============================================================
# 3. Rayyan 专属规范化
# ============================================================

def normalize_language_candidate(x):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    lower = norm_text(s)

    # 处理 "English; ABSTRACT LANGUAGE:English"
    if "abstract language" in lower:
        parts = re.split(r"abstract language\s*:\s*", lower)
        if len(parts) >= 2:
            lower = parts[-1].strip(";: ,")

    lower = lower.strip(";: ,.")
    if lower in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[lower]

    # 处理 ENG / eng.
    token = re.sub(r"[^a-z]", "", lower)
    if token in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[token]

    if len(token) == 3 and token.isalpha():
        return token

    return None


def normalize_issn_candidate(x):
    """
    规范化 ISSN:
    - 14620324 -> 1462-0324
    - 1462 0324 -> 1462-0324
    - 1462-032X -> 1462-032X
    """
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    raw = s.upper().strip()
    raw = raw.replace("ISSN", "").replace(":", " ").strip()
    chars = re.sub(r"[^0-9X]", "", raw)
    if len(chars) != 8:
        return None

    return f"{chars[:4]}-{chars[4:]}"


def is_valid_issn(x):
    fixed = normalize_issn_candidate(x)
    if not fixed:
        return False

    digits = fixed.replace("-", "")
    total = 0
    for i, ch in enumerate(digits):
        val = 10 if ch == "X" else int(ch)
        total += val * (8 - i)
    return total % 11 == 0


def normalize_date_candidate(x):
    """
    Rayyan 中 article_jcreated_at 常见格式：1/1/05, 1/1/2005, 2005-01-01。
    统一成 M/D/YY，尽量贴合原数据风格。
    """
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    s = s.strip()

    # yyyy-mm-dd
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{str(y)[-2:]}"

    # m/d/yy or m/d/yyyy
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{mo}/{d}/{y[-2:]}"

    # Excel 被误读的 ISSN 类值如 Feb-65 不在这里修
    return None


def normalize_numeric_int_candidate(x, min_v=None, max_v=None):
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None

    m = re.search(r"-?\d+", s)
    if not m:
        return None

    v = int(m.group(0))
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return str(v)


def normalize_volume_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=10000)


def normalize_issue_candidate(x):
    return normalize_numeric_int_candidate(x, min_v=0, max_v=1000)


def normalize_pagination_candidate(x):
    """
    轻量规范化 pagination:
    - 1442–6 / 1442—6 -> 1442-6
    - 1343S–1353S -> 1343S-1353S
    """
    s = canonical_empty_text(x)
    if s in {"", "empty"}:
        return None
    s2 = s.strip()
    s2 = s2.replace("–", "-").replace("—", "-").replace("−", "-")
    s2 = re.sub(r"\s*-\s*", "-", s2)
    if re.search(r"\d", s2):
        return s2
    return None


def normalize_by_column(column, value):
    col = norm_text(column)
    if col == "article_language":
        return normalize_language_candidate(value)
    if col == "journal_issn":
        return normalize_issn_candidate(value)
    if col == "article_jcreated_at":
        return normalize_date_candidate(value)
    if col == "article_jvolumn":
        return normalize_volume_candidate(value)
    if col == "article_jissue":
        return normalize_issue_candidate(value)
    if col == "article_pagination":
        return normalize_pagination_candidate(value)
    if col in {"index", "id"}:
        return normalize_numeric_int_candidate(value, min_v=0)
    return None


def value_equal_for_column(column, a, b):
    fa = normalize_by_column(column, a)
    fb = normalize_by_column(column, b)
    if fa is not None and fb is not None:
        return norm_text(fa) == norm_text(fb)
    return norm_text(a) == norm_text(b)


# ============================================================
# 4. 输入读取
# ============================================================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}, line={line_no}, err={e}") from e
    return rows


def load_context_records(path):
    lower = str(path).lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")

    try:
        return load_jsonl(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")


# ============================================================
# 5. 构建 Rayyan 全局字典
# ============================================================

def build_global_dictionaries(context_records, dirty_table_df=None):
    column_value_counter = defaultdict(Counter)

    journal_title_to_language = defaultdict(Counter)
    journal_abbr_to_language = defaultdict(Counter)
    issn_to_language = defaultdict(Counter)
    journal_title_to_issn = defaultdict(Counter)
    journal_abbr_to_issn = defaultdict(Counter)
    issn_to_journal_title = defaultdict(Counter)
    issn_to_journal_abbr = defaultdict(Counter)

    journal_issue_to_volume = defaultdict(Counter)
    journal_volume_to_issue = defaultdict(Counter)
    journal_to_volume = defaultdict(Counter)
    journal_to_issue = defaultdict(Counter)
    article_id_to_title = defaultdict(Counter)
    title_to_language = defaultdict(Counter)

    def consume(row_vals):
        if not isinstance(row_vals, dict):
            return

        row = {str(k): canonical_empty_text(v) for k, v in row_vals.items()}

        for c, v in row.items():
            if v not in {"", "empty"}:
                column_value_counter[str(c)][v] += 1

        journal_title = row.get("journal_title", "")
        journal_abbr = row.get("jounral_abbreviation", row.get("journal_abbreviation", ""))
        issn_raw = row.get("journal_issn", "")
        issn = normalize_issn_candidate(issn_raw) or canonical_empty_text(issn_raw)
        lang_raw = row.get("article_language", "")
        lang = normalize_language_candidate(lang_raw) or canonical_empty_text(lang_raw)
        vol = normalize_volume_candidate(row.get("article_jvolumn", "")) or canonical_empty_text(row.get("article_jvolumn", ""))
        issue = normalize_issue_candidate(row.get("article_jissue", "")) or canonical_empty_text(row.get("article_jissue", ""))
        article_id = canonical_empty_text(row.get("id", ""))
        title = canonical_empty_text(row.get("article_title", ""))

        if journal_title and lang not in {"", "empty"}:
            journal_title_to_language[clean_key(journal_title)][lang] += 1
        if journal_abbr and lang not in {"", "empty"}:
            journal_abbr_to_language[clean_key(journal_abbr)][lang] += 1
        if issn and lang not in {"", "empty"}:
            issn_to_language[clean_key(issn)][lang] += 1

        if journal_title and issn not in {"", "empty"}:
            journal_title_to_issn[clean_key(journal_title)][issn] += 1
            issn_to_journal_title[clean_key(issn)][journal_title] += 1
        if journal_abbr and issn not in {"", "empty"}:
            journal_abbr_to_issn[clean_key(journal_abbr)][issn] += 1
            issn_to_journal_abbr[clean_key(issn)][journal_abbr] += 1

        if journal_title and vol not in {"", "empty"}:
            journal_to_volume[clean_key(journal_title)][vol] += 1
        if journal_title and issue not in {"", "empty"}:
            journal_to_issue[clean_key(journal_title)][issue] += 1

        if journal_title and issue not in {"", "empty"} and vol not in {"", "empty"}:
            journal_issue_to_volume[(clean_key(journal_title), issue)][vol] += 1
            journal_volume_to_issue[(clean_key(journal_title), vol)][issue] += 1

        if article_id and title not in {"", "empty"}:
            article_id_to_title[article_id][title] += 1
        if title and lang not in {"", "empty"}:
            title_to_language[clean_key(title)][lang] += 1

    for obj in context_records:
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            consume(row_ctx.get("row_values", {}))

        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            for item in sim_ctx.get("topk_similar_rows", []) or []:
                if isinstance(item, dict):
                    consume(item.get("row_values", {}))

        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            for t in col_ctx.get("top_values_with_counts", []) or []:
                if isinstance(t, dict) and t.get("value", None) is not None:
                    val = canonical_empty_text(t.get("value"))
                    cnt = safe_int(t.get("count", 1), 1)
                    if col and val not in {"", "empty"}:
                        column_value_counter[col][val] += cnt

        if "column" in obj and "value" in obj:
            col = str(obj.get("column"))
            val = canonical_empty_text(obj.get("value"))
            if col and val not in {"", "empty"}:
                column_value_counter[col][val] += 1

    if dirty_table_df is not None and not dirty_table_df.empty:
        for _, r in dirty_table_df.iterrows():
            consume({c: r[c] for c in dirty_table_df.columns})

    return {
        "column_value_counter": column_value_counter,
        "journal_title_to_language": journal_title_to_language,
        "journal_abbr_to_language": journal_abbr_to_language,
        "issn_to_language": issn_to_language,
        "journal_title_to_issn": journal_title_to_issn,
        "journal_abbr_to_issn": journal_abbr_to_issn,
        "issn_to_journal_title": issn_to_journal_title,
        "issn_to_journal_abbr": issn_to_journal_abbr,
        "journal_issue_to_volume": journal_issue_to_volume,
        "journal_volume_to_issue": journal_volume_to_issue,
        "journal_to_volume": journal_to_volume,
        "journal_to_issue": journal_to_issue,
        "article_id_to_title": article_id_to_title,
        "title_to_language": title_to_language,
    }


# ============================================================
# 6. 构建 context_map
# ============================================================

def build_context_map_rich(context_records, global_dict):
    context_map = {}
    column_value_counter = global_dict["column_value_counter"]

    for obj in context_records:
        if "row_id" not in obj or "column" not in obj:
            continue

        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue

        column = str(obj["column"])
        value_str = canonical_empty_text(obj.get("value", ""))

        rule_expected_values = []
        alternative_values_from_column = []
        similar_row_target_values = []

        conflict_context = obj.get("conflict_context", {})
        if isinstance(conflict_context, dict):
            for vr in conflict_context.get("violated_rules_detailed", []) or []:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in {"", "empty"}:
                        rule_expected_values.append(ev)

            for item in conflict_context.get("expected_values_from_rules", []) or []:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in {"", "empty"}:
                        rule_expected_values.append(ev)

            for key in ["alternative_values_from_column", "all_alternative_values"]:
                for item in conflict_context.get(key, []) or []:
                    if isinstance(item, dict):
                        val = canonical_empty_text(item.get("value", None))
                        if val not in {"", "empty"}:
                            alternative_values_from_column.append({
                                "value": val,
                                "count": safe_int(item.get("count", 1), 1),
                                "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                            })

        neighbor_majority_value = ""
        neighbor_majority_ratio = 0.0
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            neighbor_majority_value = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            neighbor_majority_ratio = safe_float(sim_ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            for item in sim_ctx.get("topk_similar_rows", [])[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in {"", "empty"}:
                        similar_row_target_values.append(tv)

        column_top_values = []
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            for item in col_ctx.get("top_values_with_counts", [])[:TOP_K_COLUMN_VALUES]:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item.get("value"))
                    if val not in {"", "empty"}:
                        column_top_values.append({
                            "value": val,
                            "count": safe_int(item.get("count", 0), 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0),
                        })

        if not column_top_values:
            counter = column_value_counter.get(column, Counter())
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                column_top_values.append({"value": val, "count": cnt, "ratio": cnt / total})

        row_values = {}
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            rv = row_ctx.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}

        bayes_ctx = obj.get("bayesian_context", {})
        prior_error_probability = obj.get("prior_error_probability", None)
        posterior_error_probability = obj.get("posterior_error_probability", None)
        if isinstance(bayes_ctx, dict):
            prior_error_probability = bayes_ctx.get("prior_error_probability", prior_error_probability)
            posterior_error_probability = bayes_ctx.get("posterior_error_probability", posterior_error_probability)

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value_str,
            "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
            "neighbor_majority_value": neighbor_majority_value,
            "neighbor_majority_ratio": neighbor_majority_ratio,
            "value_frequency": obj.get("value_frequency", None),
            "value_frequency_rank": obj.get("value_frequency_rank", None),
            "numeric_value": obj.get("numeric_value", None),
            "date_value": obj.get("date_value", None),
            "language_canonical": obj.get("language_canonical", None),
            "issn_canonical": obj.get("issn_canonical", None),
            "semantic_type": obj.get("semantic_type", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "prior_error_probability": prior_error_probability,
            "posterior_error_probability": posterior_error_probability,
            "canonical_rule_count": conflict_context.get("canonical_rule_count", obj.get("canonical_rule_count", None)) if isinstance(conflict_context, dict) else obj.get("canonical_rule_count", None),
            "rule_expected_values": rule_expected_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "similar_row_target_values": similar_row_target_values,
            "row_values": row_values,
        }

    return context_map


def merge_flat_into_context_map(context_map, flat_records):
    for obj in flat_records:
        if "row_id" not in obj or "column" not in obj:
            continue

        try:
            row_id = int(obj["row_id"])
        except Exception:
            continue

        column = str(obj["column"])
        key = (row_id, column)

        if key not in context_map:
            context_map[key] = {
                "row_id": row_id,
                "column": column,
                "value": canonical_empty_text(obj.get("value", "")),
                "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
                "neighbor_majority_value": canonical_empty_text(obj.get("neighbor_majority_value", "")),
                "neighbor_majority_ratio": safe_float(obj.get("neighbor_majority_ratio", 0.0), 0.0),
                "value_frequency": obj.get("value_frequency", None),
                "value_frequency_rank": obj.get("value_frequency_rank", None),
                "numeric_value": obj.get("numeric_value", None),
                "date_value": obj.get("date_value", None),
                "language_canonical": obj.get("language_canonical", None),
                "issn_canonical": obj.get("issn_canonical", None),
                "semantic_type": obj.get("semantic_type", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "prior_error_probability": obj.get("prior_error_probability", None),
                "posterior_error_probability": obj.get("posterior_error_probability", None),
                "canonical_rule_count": obj.get("canonical_rule_count", None),
                "rule_expected_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "similar_row_target_values": [],
                "row_values": {},
            }
        else:
            ctx = context_map[key]
            for field in [
                "date_value", "language_canonical", "issn_canonical",
                "numeric_value", "semantic_type", "violation_count",
                "conflict_score", "prior_error_probability",
                "posterior_error_probability", "canonical_rule_count"
            ]:
                if ctx.get(field) in [None, "", []] and field in obj:
                    ctx[field] = obj.get(field)


# ============================================================
# 7. 候选添加与 Rayyan 专属生成
# ============================================================

def add_candidate(candidate_dict, value, source, score, extra=None):
    if value is None:
        return

    value = str(value).strip()
    if value == "":
        return

    key = norm_text(value)
    if key == "":
        return

    if key not in candidate_dict:
        candidate_dict[key] = {
            "candidate_value": value,
            "candidate_sources": set(),
            "source_score": float(score),
            "is_from_rule": 0,
            "is_from_neighbor": 0,
            "is_from_column_top": 0,
            "is_from_similarity": 0,
            "is_from_hint": 0,
            "is_from_dictionary": 0,
            "is_from_pattern_restore": 0,
            "is_from_canonical": 0,
            "extra_info": [],
        }

    item = candidate_dict[key]
    item["candidate_sources"].add(source)
    item["source_score"] = max(item["source_score"], float(score))

    if source in {"rule_expected_value", "expected_values_from_rules"}:
        item["is_from_rule"] = 1
    elif source in {"neighbor_majority", "similar_row_target"}:
        item["is_from_neighbor"] = 1
    elif source in {"column_top_value", "alternative_values_from_column"}:
        item["is_from_column_top"] = 1
    elif source in {"similar_column_value"}:
        item["is_from_similarity"] = 1
    elif source in {"suggested_correct_value"}:
        item["is_from_hint"] = 1
    elif source in {
        "journal_title_to_language",
        "journal_abbr_to_language",
        "issn_to_language",
        "journal_title_to_issn",
        "journal_abbr_to_issn",
        "issn_to_journal_title",
        "issn_to_journal_abbr",
        "journal_issue_to_volume",
        "journal_volume_to_issue",
        "journal_to_volume",
        "journal_to_issue",
        "article_id_to_title",
        "title_to_language",
        "rayyan_metadata_consensus",
    }:
        item["is_from_dictionary"] = 1
    elif source in {
        "canonical_language_restore",
        "canonical_issn_restore",
        "canonical_date_restore",
        "canonical_volume_restore",
        "canonical_issue_restore",
        "canonical_pagination_restore",
    }:
        item["is_from_pattern_restore"] = 1
        item["is_from_canonical"] = 1

    if extra is not None:
        item["extra_info"].append(str(extra))


def counter_top(counter, topk=5):
    if not counter:
        return []
    return counter.most_common(topk)


# ============================================================
# Rayyan metadata-aware consensus
# ============================================================

def _counter_best(counter):
    if not counter:
        return None, 0, 0, 0.0, 0.0
    total = sum(counter.values())
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    best_val, best_cnt = sorted_items[0]
    second_cnt = sorted_items[1][1] if len(sorted_items) > 1 else 0
    conf = best_cnt / max(total, 1)
    margin = (best_cnt - second_cnt) / max(total, 1)
    return best_val, best_cnt, total, conf, margin


def _row_get(row, col):
    if col in row:
        return canonical_empty_text(row.get(col, ""))
    return ""


def build_rayyan_metadata_consensus(dirty_table_df):
    """
    Rayyan 版 consensus，迁移自 Flights 的 source-aware consensus。
    Flights 里基于同一 flight 多 source 的时间共识；
    Rayyan 里基于 journal/article metadata 的字段共识。

    目前构建的高置信映射：
    - journal_title -> article_language
    - jounral_abbreviation -> article_language
    - journal_issn -> article_language
    - journal_title -> journal_issn
    - jounral_abbreviation -> journal_issn
    - journal_issn -> journal_title
    - journal_issn -> jounral_abbreviation
    - (journal_title, article_jissue) -> article_jvolumn
    - (journal_title, article_jvolumn) -> article_jissue
    """
    consensus_map = {}
    consensus_rows = []
    suspicious_rows = []

    if dirty_table_df is None or dirty_table_df.empty:
        return consensus_map, pd.DataFrame(), pd.DataFrame()

    work = dirty_table_df.copy()
    work["__row_id__"] = range(len(work))

    # Normalize all row values only when used; do not mutate raw display values globally.
    relation_specs = [
        {
            "name": "journal_title_to_language",
            "lhs_cols": ["journal_title"],
            "rhs_col": "article_language",
            "normalizer": normalize_language_candidate,
            "score": 0.94,
        },
        {
            "name": "journal_abbr_to_language",
            "lhs_cols": ["jounral_abbreviation"],
            "rhs_col": "article_language",
            "normalizer": normalize_language_candidate,
            "score": 0.92,
        },
        {
            "name": "issn_to_language",
            "lhs_cols": ["journal_issn"],
            "rhs_col": "article_language",
            "lhs_normalizers": {"journal_issn": normalize_issn_candidate},
            "normalizer": normalize_language_candidate,
            "score": 0.95,
        },
        {
            "name": "journal_title_to_issn",
            "lhs_cols": ["journal_title"],
            "rhs_col": "journal_issn",
            "normalizer": normalize_issn_candidate,
            "score": 0.94,
        },
        {
            "name": "journal_abbr_to_issn",
            "lhs_cols": ["jounral_abbreviation"],
            "rhs_col": "journal_issn",
            "normalizer": normalize_issn_candidate,
            "score": 0.91,
        },
        {
            "name": "issn_to_journal_title",
            "lhs_cols": ["journal_issn"],
            "rhs_col": "journal_title",
            "lhs_normalizers": {"journal_issn": normalize_issn_candidate},
            "normalizer": None,
            "score": 0.87,
        },
        {
            "name": "issn_to_journal_abbr",
            "lhs_cols": ["journal_issn"],
            "rhs_col": "jounral_abbreviation",
            "lhs_normalizers": {"journal_issn": normalize_issn_candidate},
            "normalizer": None,
            "score": 0.84,
        },
        {
            "name": "journal_issue_to_volume",
            "lhs_cols": ["journal_title", "article_jissue"],
            "rhs_col": "article_jvolumn",
            "lhs_normalizers": {"article_jissue": normalize_issue_candidate},
            "normalizer": normalize_volume_candidate,
            "score": 0.78,
        },
        {
            "name": "journal_volume_to_issue",
            "lhs_cols": ["journal_title", "article_jvolumn"],
            "rhs_col": "article_jissue",
            "lhs_normalizers": {"article_jvolumn": normalize_volume_candidate},
            "normalizer": normalize_issue_candidate,
            "score": 0.78,
        },
    ]

    # Alias: some files may spell journal_abbreviation correctly.
    if "jounral_abbreviation" not in work.columns and "journal_abbreviation" in work.columns:
        work["jounral_abbreviation"] = work["journal_abbreviation"]

    for spec in relation_specs:
        lhs_cols = spec["lhs_cols"]
        rhs_col = spec["rhs_col"]

        if rhs_col not in work.columns:
            continue
        if any(c not in work.columns for c in lhs_cols):
            continue

        lhs_normalizers = spec.get("lhs_normalizers", {}) or {}
        rhs_normalizer = spec.get("normalizer", None)

        group_counter = defaultdict(Counter)
        group_rows = defaultdict(list)

        for _, r in work.iterrows():
            lhs_parts = []
            skip = False
            for c in lhs_cols:
                raw = _row_get(r, c)
                if raw in {"", "empty"}:
                    skip = True
                    break
                if c in lhs_normalizers:
                    fixed = lhs_normalizers[c](raw)
                    raw = fixed if fixed else raw
                lhs_parts.append(clean_key(raw))

            if skip:
                continue

            rhs_raw = _row_get(r, rhs_col)
            if rhs_raw in {"", "empty"}:
                continue

            rhs_val = rhs_raw
            if rhs_normalizer is not None:
                fixed = rhs_normalizer(rhs_raw)
                rhs_val = fixed if fixed else rhs_raw

            if rhs_val in {"", "empty"}:
                continue

            lhs_key = tuple(lhs_parts)
            group_counter[lhs_key][rhs_val] += 1
            group_rows[lhs_key].append(int(r["__row_id__"]))

        # Consensus rows
        for lhs_key, counter in group_counter.items():
            best_val, best_cnt, total, conf, margin = _counter_best(counter)
            if best_val is None:
                continue

            high_conf = (
                best_cnt >= MIN_RAYYAN_CONSENSUS_SUPPORT_COUNT
                and conf >= MIN_RAYYAN_CONSENSUS_CONFIDENCE
                and margin >= MIN_RAYYAN_CONSENSUS_MARGIN
            )

            consensus_rows.append({
                "consensus_name": spec["name"],
                "lhs_key": " || ".join(lhs_key),
                "rhs_column": rhs_col,
                "consensus_value": best_val,
                "support_count": best_cnt,
                "total_count": total,
                "confidence": conf,
                "margin": margin,
                "is_high_confidence": int(high_conf),
            })

            if not high_conf:
                continue

            # Find suspicious rows under this lhs.
            for _, r in work.iterrows():
                lhs_parts = []
                skip = False
                for c in lhs_cols:
                    raw = _row_get(r, c)
                    if raw in {"", "empty"}:
                        skip = True
                        break
                    if c in lhs_normalizers:
                        fixed = lhs_normalizers[c](raw)
                        raw = fixed if fixed else raw
                    lhs_parts.append(clean_key(raw))
                if skip or tuple(lhs_parts) != lhs_key:
                    continue

                row_id = int(r["__row_id__"])
                dirty_val = _row_get(r, rhs_col)
                dirty_norm = dirty_val
                if rhs_normalizer is not None:
                    fixed = rhs_normalizer(dirty_val)
                    dirty_norm = fixed if fixed else dirty_val

                if value_equal_for_column(rhs_col, dirty_norm, best_val):
                    continue

                suspicious_rows.append({
                    "row_id": row_id,
                    "column": rhs_col,
                    "value": dirty_val,
                    "consensus_name": spec["name"],
                    "lhs_key": " || ".join(lhs_key),
                    "consensus_value": best_val,
                    "consensus_confidence": conf,
                    "consensus_margin": margin,
                    "consensus_support_count": best_cnt,
                    "consensus_total_count": total,
                    "consensus_reason": "rayyan_metadata_consensus",
                })

                consensus_map[(row_id, rhs_col)] = {
                    "candidate_value": best_val,
                    "consensus_name": spec["name"],
                    "lhs_key": " || ".join(lhs_key),
                    "confidence": conf,
                    "margin": margin,
                    "support_count": best_cnt,
                    "total_count": total,
                    "score": spec.get("score", 0.89),
                }

    consensus_df = pd.DataFrame(consensus_rows)
    suspicious_df = pd.DataFrame(suspicious_rows)

    if not suspicious_df.empty:
        suspicious_df = suspicious_df.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)

    return consensus_map, consensus_df, suspicious_df


def generate_rayyan_consensus_candidates(row_id, column, dirty_value, consensus_map):
    info = consensus_map.get((int(row_id), str(column)))
    if not info:
        return []

    val = canonical_empty_text(info.get("candidate_value", ""))
    if val in {"", "empty"}:
        return []
    if value_equal_for_column(column, dirty_value, val):
        return []

    conf = safe_float(info.get("confidence", 0.0), 0.0)
    margin = safe_float(info.get("margin", 0.0), 0.0)
    support_count = safe_int(info.get("support_count", 0), 0)

    if support_count < MIN_RAYYAN_CONSENSUS_SUPPORT_COUNT:
        return []

    score = RAYYAN_METADATA_CONSENSUS_STRONG_SCORE if conf >= 0.75 else RAYYAN_METADATA_CONSENSUS_MEDIUM_SCORE
    score = max(score, safe_float(info.get("score", 0.0), 0.0))
    extra = (
        f"consensus_name={info.get('consensus_name', '')}|"
        f"lhs_key={info.get('lhs_key', '')}|"
        f"confidence={conf:.4f}|margin={margin:.4f}|support_count={support_count}"
    )
    return [(val, "rayyan_metadata_consensus", score, extra)]


def generate_rayyan_context_candidates(column, dirty_value, ctx, global_dict):
    cands = []
    col = norm_text(column)
    row_values = ctx.get("row_values", {}) or {}

    journal_title = canonical_empty_text(row_values.get("journal_title", ""))
    journal_abbr = canonical_empty_text(row_values.get("jounral_abbreviation", row_values.get("journal_abbreviation", "")))
    journal_issn_raw = canonical_empty_text(row_values.get("journal_issn", ""))
    journal_issn = normalize_issn_candidate(journal_issn_raw) or journal_issn_raw
    article_language = canonical_empty_text(row_values.get("article_language", ""))
    article_id = canonical_empty_text(row_values.get("id", ""))
    article_title = canonical_empty_text(row_values.get("article_title", ""))
    volume = normalize_volume_candidate(row_values.get("article_jvolumn", "")) or canonical_empty_text(row_values.get("article_jvolumn", ""))
    issue = normalize_issue_candidate(row_values.get("article_jissue", "")) or canonical_empty_text(row_values.get("article_jissue", ""))

    if col == "article_language":
        fixed = normalize_language_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "canonical_language_restore", 0.94, "normalize_language"))

        # 如果 flat/context 中已经给出 canonical，优先使用
        lang_can = normalize_language_candidate(ctx.get("language_canonical", None))
        if lang_can and not value_equal_for_column(column, dirty_value, lang_can):
            cands.append((lang_can, "canonical_language_restore", 0.96, "context_language_canonical"))

        for v, cnt in counter_top(global_dict["journal_title_to_language"].get(clean_key(journal_title), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_title_to_language", 0.90, f"journal_title={journal_title}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["journal_abbr_to_language"].get(clean_key(journal_abbr), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_abbr_to_language", 0.88, f"journal_abbr={journal_abbr}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["issn_to_language"].get(clean_key(journal_issn), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "issn_to_language", 0.92, f"issn={journal_issn}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["title_to_language"].get(clean_key(article_title), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "title_to_language", 0.80, f"title_match|cnt={cnt}"))

    elif col == "journal_issn":
        fixed = normalize_issn_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            score = 0.96 if is_valid_issn(fixed) else 0.88
            cands.append((fixed, "canonical_issn_restore", score, "normalize_issn"))

        issn_can = normalize_issn_candidate(ctx.get("issn_canonical", None))
        if issn_can and not value_equal_for_column(column, dirty_value, issn_can):
            score = 0.98 if is_valid_issn(issn_can) else 0.90
            cands.append((issn_can, "canonical_issn_restore", score, "context_issn_canonical"))

        for v, cnt in counter_top(global_dict["journal_title_to_issn"].get(clean_key(journal_title), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_title_to_issn", 0.91, f"journal_title={journal_title}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["journal_abbr_to_issn"].get(clean_key(journal_abbr), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_abbr_to_issn", 0.88, f"journal_abbr={journal_abbr}|cnt={cnt}"))

    elif col == "journal_title":
        for v, cnt in counter_top(global_dict["issn_to_journal_title"].get(clean_key(journal_issn), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "issn_to_journal_title", 0.86, f"issn={journal_issn}|cnt={cnt}"))

    elif col in {"jounral_abbreviation", "journal_abbreviation"}:
        for v, cnt in counter_top(global_dict["issn_to_journal_abbr"].get(clean_key(journal_issn), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "issn_to_journal_abbr", 0.84, f"issn={journal_issn}|cnt={cnt}"))

    elif col == "article_jcreated_at":
        fixed = normalize_date_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "canonical_date_restore", 0.92, "normalize_date"))

        date_can = normalize_date_candidate(ctx.get("date_value", None))
        if date_can and not value_equal_for_column(column, dirty_value, date_can):
            cands.append((date_can, "canonical_date_restore", 0.93, "context_date_value"))

    elif col == "article_jvolumn":
        fixed = normalize_volume_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "canonical_volume_restore", 0.88, "normalize_volume"))

        for v, cnt in counter_top(global_dict["journal_issue_to_volume"].get((clean_key(journal_title), issue), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_issue_to_volume", 0.78, f"journal={journal_title}|issue={issue}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["journal_to_volume"].get(clean_key(journal_title), Counter()), topk=3):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_to_volume", 0.42, f"journal={journal_title}|cnt={cnt}"))

    elif col == "article_jissue":
        fixed = normalize_issue_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "canonical_issue_restore", 0.88, "normalize_issue"))

        for v, cnt in counter_top(global_dict["journal_volume_to_issue"].get((clean_key(journal_title), volume), Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_volume_to_issue", 0.78, f"journal={journal_title}|volume={volume}|cnt={cnt}"))

        for v, cnt in counter_top(global_dict["journal_to_issue"].get(clean_key(journal_title), Counter()), topk=3):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "journal_to_issue", 0.42, f"journal={journal_title}|cnt={cnt}"))

    elif col == "article_pagination":
        fixed = normalize_pagination_candidate(dirty_value)
        if fixed and not value_equal_for_column(column, dirty_value, fixed):
            cands.append((fixed, "canonical_pagination_restore", 0.86, "normalize_pagination"))

    elif col == "article_title":
        for v, cnt in counter_top(global_dict["article_id_to_title"].get(article_id, Counter())):
            if not value_equal_for_column(column, dirty_value, v):
                cands.append((v, "article_id_to_title", 0.88, f"id={article_id}|cnt={cnt}"))

    return cands


# ============================================================
# 8. 主流程
# ============================================================

def main():
    pred_df = pd.read_csv(ERROR_PRED_FILE, encoding="utf-8-sig")

    required_base = {"row_id", "column"}
    if not required_base.issubset(set(pred_df.columns)):
        raise ValueError(f"错误位置文件必须至少包含 {required_base}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="raise").astype(int)

    if "final_pred_label" in pred_df.columns:
        error_df = pred_df[pd.to_numeric(pred_df["final_pred_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif "final_pred_label_name" in pred_df.columns:
        error_df = pred_df[pred_df["final_pred_label_name"].astype(str).str.strip().str.lower() == "error"].copy()
    else:
        # 如果输入已经是 predicted_error_positions.csv，则默认全是错误位置
        error_df = pred_df.copy()

    keep_cols = ["row_id", "column"]
    if "value" in error_df.columns:
        keep_cols.append("value")

    error_df = error_df[keep_cols].drop_duplicates().copy()
    print(f"[INFO] 预测错误单元格数量: {len(error_df)}")

    context_records = load_context_records(CONTEXT_FILE)
    print(f"[INFO] 主上下文记录数: {len(context_records)}")

    dirty_table_df = None
    if str(DIRTY_TABLE_CSV).strip():
        if path_exists_nonempty(DIRTY_TABLE_CSV):
            dirty_table_df = pd.read_csv(DIRTY_TABLE_CSV, encoding="utf-8-sig")
            print(f"[INFO] 已读取 Rayyan 原始脏表: {DIRTY_TABLE_CSV}, shape={dirty_table_df.shape}")
        else:
            print(f"[WARN] DIRTY_TABLE_CSV 不存在，已跳过: {DIRTY_TABLE_CSV}")

    rayyan_consensus_map = {}
    rayyan_consensus_df = pd.DataFrame()
    rayyan_suspicious_df = pd.DataFrame()

    if ENABLE_RAYYAN_METADATA_CONSENSUS:
        rayyan_consensus_map, rayyan_consensus_df, rayyan_suspicious_df = build_rayyan_metadata_consensus(dirty_table_df)
        if not rayyan_consensus_df.empty:
            rayyan_consensus_df.to_csv(resolve_output_path(OUTPUT_RAYYAN_CONSENSUS_CSV), index=False, encoding="utf-8-sig")
            print(f"[INFO] Rayyan metadata consensus 已保存: {resolve_output_path(OUTPUT_RAYYAN_CONSENSUS_CSV)}")
        if not rayyan_suspicious_df.empty:
            rayyan_suspicious_df.to_csv(resolve_output_path(OUTPUT_RAYYAN_CONSENSUS_SUSPICIOUS_CELLS_CSV), index=False, encoding="utf-8-sig")
            print(f"[INFO] Rayyan consensus suspicious cells 已保存: {resolve_output_path(OUTPUT_RAYYAN_CONSENSUS_SUSPICIOUS_CELLS_CSV)}")

            # 迁移 Flights augmented whitelist 思想：
            # 将高置信 metadata consensus 发现的 suspicious cells 合并进 allowed cells。
            allow_base = error_df.copy()
            suspicious_allow = rayyan_suspicious_df[
                rayyan_suspicious_df["column"].astype(str).str.strip().str.lower().isin(RAYYAN_CONSENSUS_ALLOW_COLUMNS)
            ].copy()
            if not suspicious_allow.empty:
                augmented = pd.concat([
                    allow_base[["row_id", "column"] + (["value"] if "value" in allow_base.columns else [])],
                    suspicious_allow[["row_id", "column", "value"]],
                ], ignore_index=True, sort=False)
                augmented = augmented.drop_duplicates(subset=["row_id", "column"], keep="first").sort_values(["row_id", "column"]).reset_index(drop=True)
                augmented.to_csv(resolve_output_path(OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV), index=False, encoding="utf-8-sig")
                print(f"[INFO] Rayyan consensus augmented whitelist 已保存: {resolve_output_path(OUTPUT_AUGMENTED_ALLOWED_CELLS_CSV)}")
                print(f"[INFO] augmented allowed cells: {len(augmented)}")

    global_dict = build_global_dictionaries(context_records, dirty_table_df=dirty_table_df)
    context_map = build_context_map_rich(context_records, global_dict)

    if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
        if path_exists_nonempty(FALLBACK_CONTEXT_FLAT_FILE):
            flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
            print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
            merge_flat_into_context_map(context_map, flat_records)
        else:
            print(f"[WARN] FALLBACK_CONTEXT_FLAT_FILE 不存在，已跳过: {FALLBACK_CONTEXT_FLAT_FILE}")

    print(f"[INFO] 可索引上下文条数: {len(context_map)}")

    expanded_rows = []
    grouped_rows = []
    missing_context_count = 0

    for _, row in error_df.iterrows():
        row_id = int(row["row_id"])
        column = str(row["column"])
        ctx = context_map.get((row_id, column), None)

        if ctx is None:
            missing_context_count += 1
            dirty_value = "" if "value" not in row or pd.isna(row["value"]) else str(row["value"]).strip()
            grouped_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_count": 0,
                "candidates_joined": "",
                "sources_joined": "",
                "top_candidate": "",
                "top_candidate_score": None,
            })
            continue

        dirty_value = ctx.get("value", "")
        if ("value" in error_df.columns) and (pd.notna(row.get("value", None))) and str(row.get("value")).strip() != "":
            dirty_value = canonical_empty_text(row["value"])

        candidate_dict = {}

        if INCLUDE_DIRTY_VALUE and dirty_value != "":
            add_candidate(candidate_dict, dirty_value, "dirty_value", 0.01)

        # A. rule expected values
        expected_counter = Counter()
        for ev in ctx.get("rule_expected_values", []):
            ev = canonical_empty_text(ev)
            if ev in {"", "empty"} or value_equal_for_column(column, dirty_value, ev):
                continue
            fixed = normalize_by_column(column, ev)
            if fixed is not None:
                ev = fixed
            expected_counter[ev] += 1

        total_votes = sum(expected_counter.values())
        for ev, cnt in expected_counter.items():
            score = 0.76 + 0.22 * (cnt / max(total_votes, 1))
            add_candidate(candidate_dict, ev, "expected_values_from_rules", score, extra=f"rule_vote_count={cnt}")

        # B. neighbor majority
        neighbor_majority_value = ctx.get("neighbor_majority_value", None)
        if neighbor_majority_value and not value_equal_for_column(column, dirty_value, neighbor_majority_value):
            fixed = normalize_by_column(column, neighbor_majority_value)
            if fixed is not None:
                neighbor_majority_value = fixed
            nr = safe_float(ctx.get("neighbor_majority_ratio", 0.0), 0.0)
            add_candidate(candidate_dict, neighbor_majority_value, "neighbor_majority", 0.78 + 0.16 * min(nr, 1.0), extra=f"neighbor_ratio={nr:.4f}")

        # C. similar row target values
        sim_counter = Counter()
        for v in ctx.get("similar_row_target_values", []):
            v = canonical_empty_text(v)
            if v and not value_equal_for_column(column, dirty_value, v):
                fixed = normalize_by_column(column, v)
                if fixed is not None:
                    v = fixed
                sim_counter[v] += 1

        for v, cnt in sim_counter.items():
            score = 0.72 + 0.12 * min(cnt, TOP_K_SIMILAR_ROW_VALUES) / max(TOP_K_SIMILAR_ROW_VALUES, 1)
            add_candidate(candidate_dict, v, "similar_row_target", score, extra=f"count={cnt}")

        # D. suggested correct value
        suggested_correct_value = ctx.get("suggested_correct_value", None)
        if suggested_correct_value and not value_equal_for_column(column, dirty_value, suggested_correct_value):
            fixed = normalize_by_column(column, suggested_correct_value)
            if fixed is not None:
                suggested_correct_value = fixed
            add_candidate(candidate_dict, suggested_correct_value, "suggested_correct_value", 0.93)

        # E. column top / alternative values
        alt_items = []
        alt_items.extend(ctx.get("column_top_values", [])[:TOP_K_COLUMN_VALUES])
        alt_items.extend(ctx.get("alternative_values_from_column", [])[:TOP_K_COLUMN_VALUES])

        dedup_alt = {}
        for item in alt_items:
            if not isinstance(item, dict):
                continue

            val = canonical_empty_text(item.get("value", ""))
            if val in {"", "empty"} or value_equal_for_column(column, dirty_value, val):
                continue

            fixed = normalize_by_column(column, val)
            if fixed is not None:
                val = fixed

            k = norm_text(val)
            prev = dedup_alt.get(k, {"value": val, "count": 0, "ratio": 0.0})
            prev["count"] = max(prev["count"], safe_int(item.get("count", 0), 0))
            prev["ratio"] = max(prev["ratio"], safe_float(item.get("ratio", 0.0), 0.0))
            dedup_alt[k] = prev

        for item in dedup_alt.values():
            val = item["value"]
            ratio = safe_float(item.get("ratio", 0.0), 0.0)
            sim = string_similarity(dirty_value, val)

            top_score = 0.30 + 0.18 * min(ratio, 1.0)
            add_candidate(candidate_dict, val, "column_top_value", top_score, extra=f"ratio={ratio:.4f}")

            if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE:
                sim_score = 0.46 + 0.22 * sim + 0.10 * min(ratio, 1.0)
                add_candidate(candidate_dict, val, "similar_column_value", sim_score, extra=f"ratio={ratio:.4f}|sim={sim:.4f}")

        # F0. Rayyan metadata consensus 候选
        if ENABLE_RAYYAN_METADATA_CONSENSUS:
            for val, source, score, extra in generate_rayyan_consensus_candidates(row_id, column, dirty_value, rayyan_consensus_map):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        # F. Rayyan 专属候选
        if ENABLE_RAYYAN_SPECIFIC_GENERATION:
            for val, source, score, extra in generate_rayyan_context_candidates(column, dirty_value, ctx, global_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        # G. 删除与 dirty 等价的候选
        for k in list(candidate_dict.keys()):
            if value_equal_for_column(column, candidate_dict[k]["candidate_value"], dirty_value):
                candidate_dict.pop(k, None)

        # H. 排序
        final_candidates = []
        for _, item in candidate_dict.items():
            cand = item["candidate_value"]
            sim = string_similarity(dirty_value, cand)
            source_count = len(item["candidate_sources"])
            support_bonus = 0.04 * max(source_count - 1, 0)

            semantic_bonus = 0.0
            fixed_cand = normalize_by_column(column, cand)
            if fixed_cand is not None and norm_text(fixed_cand) == norm_text(cand):
                semantic_bonus += 0.06

            col = norm_text(column)
            if col == "article_language" and normalize_language_candidate(cand) is not None:
                semantic_bonus += 0.10
            if col == "journal_issn" and normalize_issn_candidate(cand) is not None:
                semantic_bonus += 0.10
                if is_valid_issn(cand):
                    semantic_bonus += 0.04
            if col == "article_jcreated_at" and normalize_date_candidate(cand) is not None:
                semantic_bonus += 0.08
            if col in {"article_jvolumn", "article_jissue"} and normalize_numeric_int_candidate(cand, min_v=0) is not None:
                semantic_bonus += 0.06
            if col == "article_pagination" and normalize_pagination_candidate(cand) is not None:
                semantic_bonus += 0.05

            final_score = item["source_score"] + 0.16 * sim + support_bonus + semantic_bonus

            item["edit_similarity_to_dirty"] = round(sim, 6)
            item["final_score"] = round(final_score, 6)
            item["candidate_sources"] = sorted(list(item["candidate_sources"]))
            item["candidate_source_joined"] = "|".join(item["candidate_sources"])
            final_candidates.append(item)

        final_candidates = sorted(
            final_candidates,
            key=lambda x: (
                -x["final_score"],
                -x["is_from_rule"],
                -x["is_from_canonical"],
                -x["is_from_dictionary"],
                -x["is_from_pattern_restore"],
                -x["is_from_neighbor"],
                x["candidate_value"],
            ),
        )[:MAX_CANDIDATES_PER_CELL]

        for rank_idx, item in enumerate(final_candidates, start=1):
            expanded_rows.append({
                "row_id": row_id,
                "column": column,
                "dirty_value": dirty_value,
                "candidate_rank": rank_idx,
                "candidate_value": item["candidate_value"],
                "candidate_source": item["candidate_source_joined"],
                "source_score": item["source_score"],
                "final_score": item["final_score"],
                "edit_similarity_to_dirty": item["edit_similarity_to_dirty"],
                "is_from_rule": item["is_from_rule"],
                "is_from_neighbor": item["is_from_neighbor"],
                "is_from_column_top": item["is_from_column_top"],
                "is_from_similarity": item["is_from_similarity"],
                "is_from_hint": item["is_from_hint"],
                "is_from_dictionary": item["is_from_dictionary"],
                "is_from_pattern_restore": item["is_from_pattern_restore"],
                "is_from_canonical": item["is_from_canonical"],
                "source_count": len(item["candidate_sources"]),
                "extra_info": " || ".join(item["extra_info"][:15]),
                "violation_count": ctx.get("violation_count", None),
                "conflict_score": ctx.get("conflict_score", None),
                "semantic_type": ctx.get("semantic_type", None),
                "main_rule_type": ctx.get("main_rule_type", None),
                "main_usage_role": ctx.get("main_usage_role", None),
                "numeric_value": ctx.get("numeric_value", None),
                "date_value": ctx.get("date_value", None),
                "language_canonical": ctx.get("language_canonical", None),
                "issn_canonical": ctx.get("issn_canonical", None),
                "prior_error_probability": ctx.get("prior_error_probability", None),
                "posterior_error_probability": ctx.get("posterior_error_probability", None),
                "canonical_rule_count": ctx.get("canonical_rule_count", None),
                "rayyan_consensus_candidate": int("rayyan_metadata_consensus" in item["candidate_sources"]),
                "rayyan_consensus_info": " || ".join([x for x in item["extra_info"] if "consensus_name=" in x]),

                # Rayyan 诊断字段，供后续特征构建 / 采样 / 推理使用
                "rayyan_is_language_candidate": int(norm_text(column) == "article_language" and normalize_language_candidate(item["candidate_value"]) is not None),
                "rayyan_is_issn_candidate": int(norm_text(column) == "journal_issn" and normalize_issn_candidate(item["candidate_value"]) is not None),
                "rayyan_is_valid_issn_candidate": int(norm_text(column) == "journal_issn" and is_valid_issn(item["candidate_value"])),
                "rayyan_is_date_candidate": int(norm_text(column) == "article_jcreated_at" and normalize_date_candidate(item["candidate_value"]) is not None),
                "rayyan_is_volume_candidate": int(norm_text(column) == "article_jvolumn" and normalize_volume_candidate(item["candidate_value"]) is not None),
                "rayyan_is_issue_candidate": int(norm_text(column) == "article_jissue" and normalize_issue_candidate(item["candidate_value"]) is not None),
                "rayyan_is_pagination_candidate": int(norm_text(column) == "article_pagination" and normalize_pagination_candidate(item["candidate_value"]) is not None),
            })

        grouped_rows.append({
            "row_id": row_id,
            "column": column,
            "dirty_value": dirty_value,
            "candidate_count": len(final_candidates),
            "candidates_joined": " || ".join([x["candidate_value"] for x in final_candidates]),
            "sources_joined": " || ".join([x["candidate_source_joined"] for x in final_candidates]),
            "top_candidate": final_candidates[0]["candidate_value"] if final_candidates else "",
            "top_candidate_score": final_candidates[0]["final_score"] if final_candidates else None,
        })

    expanded_df = pd.DataFrame(expanded_rows)
    grouped_df = pd.DataFrame(grouped_rows)

    if not expanded_df.empty:
        expanded_df = expanded_df.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)

    if not grouped_df.empty:
        grouped_df = grouped_df.sort_values(["row_id", "column"]).reset_index(drop=True)

    out_expanded = resolve_output_path(OUTPUT_CANDIDATES_EXPANDED_CSV)
    out_grouped = resolve_output_path(OUTPUT_CANDIDATES_GROUPED_CSV)

    expanded_df.to_csv(out_expanded, index=False, encoding="utf-8-sig")
    grouped_df.to_csv(out_grouped, index=False, encoding="utf-8-sig")

    print(f"[INFO] 缺失上下文的错误 cell 数量: {missing_context_count}")
    print(f"[INFO] 候选展开文件已保存: {out_expanded}")
    print(f"[INFO] 候选汇总文件已保存: {out_grouped}")

    if not grouped_df.empty:
        print("\n[INFO] 前 20 个错误 cell 的候选示例：")
        print(grouped_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
