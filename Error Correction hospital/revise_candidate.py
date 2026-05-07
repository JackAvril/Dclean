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
# 支持：
#   A. 只有 row_id,column
#   B. 包含 row_id,column,value
#   C. 检测推理结果全表（带 final_pred_label / final_pred_label_name）
ERROR_PRED_FILE = "predicted_error_positions.csv"

# 2) 修复候选构造的主输入：优先用 rich jsonl
#    例如 detection 阶段的 detailed context jsonl
CONTEXT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection new/candidate_llm_contexts_v2.jsonl"

# 3) 可选的 flat csv fallback（没有就留空）
#    例如 candidate_llm_contexts_flat_v2.csv
FALLBACK_CONTEXT_FLAT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Detection new/candidate_llm_contexts_flat_v2.csv"

# 4) 可选：原始脏表（如果有，建议填）
#    用于补全列字典、state prefix 等
DIRTY_TABLE_CSV = "/mnt/mydata/dq/projects/Splittree/hospital/hospital_dirty_no_address23.csv"

# 输出
OUTPUT_CANDIDATES_EXPANDED_CSV = "repair_candidates_expanded_v2.csv"
OUTPUT_CANDIDATES_GROUPED_CSV = "repair_candidates_grouped_v2.csv"

# 参数
TOP_K_COLUMN_VALUES = 12
TOP_K_SIMILAR_ROW_VALUES = 6
MAX_CANDIDATES_PER_CELL = 20
MIN_STRING_SIM_FOR_COLUMN_VALUE = 0.45
MIN_STRING_SIM_FOR_CANONICAL = 0.35
INCLUDE_DIRTY_VALUE = False

# 是否启用列专属补候选
ENABLE_COLUMN_SPECIFIC_GENERATION = True


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing"}
CONDITION_CANONICAL = [
    "heart attack",
    "heart failure",
    "pneumonia",
    "surgical infection prevention",
]

MEASURECODE_CANONICAL_FAMILIES = ["ami", "hf", "pn", "scip"]

KNOWN_MEASURE_CODES = [
    "ami-1", "ami-2", "ami-3", "ami-4", "ami-5", "ami-7a", "ami-8a",
    "hf-1", "hf-2", "hf-3", "hf-4",
    "pn-2", "pn-3b", "pn-4", "pn-5c", "pn-6", "pn-7",
    "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3",
    "scip-vte-1", "scip-vte-2",
]

def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_empty_text(x):
    s = norm_text(x)
    return "empty" if s in EMPTY_TOKENS else ("" if pd.isna(x) else str(x).strip())


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def string_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def looks_like_measure_code(x: str) -> bool:
    s = norm_text(x)
    if not s:
        return False
    return (
        s.startswith("ami") or s.startswith("hf") or s.startswith("pn") or s.startswith("scip")
        or ("ami" in s) or ("hf" in s) or ("pn" in s) or ("scip" in s)
    )


def looks_like_stateavg(x: str) -> bool:
    s = norm_text(x)
    return "_" in s and looks_like_measure_code(s)


def extract_state_prefix_from_stateavg(x: str) -> str:
    s = canonical_empty_text(x)
    if "_" not in s:
        return ""
    return s.split("_", 1)[0].strip().lower()


def normalize_measurecode_like(x: str) -> str:
    s = canonical_empty_text(x).lower().strip()
    # stateavg like al_hf-2 -> hf-2
    if "_" in s:
        s = s.split("_", 1)[1]
    s = s.replace("_", "-")
    s = re.sub(r"-+", "-", s)
    return s


def measure_family_name(code: str) -> str:
    c = normalize_measurecode_like(code)
    if c.startswith("hf"):
        return "heart failure"
    if c.startswith("ami"):
        return "heart attack"
    if c.startswith("pn"):
        return "pneumonia"
    if c.startswith("scip"):
        return "surgical infection prevention"
    return ""


def normalize_percent_candidate(x: str):
    s = canonical_empty_text(x).lower().replace(" ", "")
    if s == "" or s == "empty":
        return None

    # 已合法
    if re.fullmatch(r"\d{1,3}%", s):
        return s.upper() if False else s

    # 去掉百分号后处理
    raw = s.replace("%", "")
    # 常见 x typo
    raw = raw.replace("x", "")
    raw = raw.replace("o", "0")
    raw = raw.replace("O", "0")
    raw = raw.strip()

    if raw.isdigit():
        raw_int = int(raw)
        if 0 <= raw_int <= 100:
            return f"{raw_int}%"
    return None


def normalize_sample_candidate(x: str):
    s = canonical_empty_text(x).lower().strip()
    if s == "" or s == "empty":
        return None

    # 统一 patients typo
    digit_part = "".join(ch for ch in s if ch.isdigit())
    if digit_part == "":
        return None

    if "patient" in s or "patien" in s or "paxien" in s or "pxtient" in s or "patixnt" in s or "xpatients" in s or "patients" in s:
        return f"{int(digit_part)} patients"
    return None


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
            "extra_info": []
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
    elif source in {"measurecode_dictionary", "stateavg_from_measurecode", "condition_from_measurecode",
                    "measurename_from_measurecode", "measurename_from_condition"}:
        item["is_from_dictionary"] = 1
    elif source in {"pattern_restore_score", "pattern_restore_sample"}:
        item["is_from_pattern_restore"] = 1

    if extra is not None:
        item["extra_info"].append(str(extra))


# ============================================================
# 3. 输入读取
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
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(path)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df.to_dict(orient="records")

    try:
        return load_jsonl(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df.to_dict(orient="records")


# ============================================================
# 4. 构造字典（全局候选补全用）
# ============================================================

def build_global_dictionaries(context_records, dirty_table_df=None):
    """
    从 rich context 中尽量抽取全局 canonical dictionaries：
    - 列 top values
    - row_context.row_values
    - similarity_context.topk_similar_rows[*].row_values
    """
    column_value_counter = defaultdict(Counter)
    measurecode_to_names = defaultdict(Counter)
    condition_to_names = defaultdict(Counter)
    state_prefix_to_measurecodes = defaultdict(Counter)
    hospital_stateavg_from_code = defaultdict(Counter)

    def consume_row_values(row_vals):
        if not isinstance(row_vals, dict):
            return

        for c, v in row_vals.items():
            v = canonical_empty_text(v)
            if v != "":
                column_value_counter[str(c)][v] += 1

        mc = canonical_empty_text(row_vals.get("MeasureCode", ""))
        mn = canonical_empty_text(row_vals.get("MeasureName", ""))
        cond = canonical_empty_text(row_vals.get("Condition", ""))
        sa = canonical_empty_text(row_vals.get("Stateavg", ""))
        state = canonical_empty_text(row_vals.get("State", ""))
        hosp = canonical_empty_text(row_vals.get("HospitalName", ""))

        if mc and mn:
            measurecode_to_names[mc][mn] += 1
        if cond and mn:
            condition_to_names[cond][mn] += 1
        if state and mc:
            state_prefix_to_measurecodes[state.lower()][mc] += 1
        if hosp and mc and sa:
            hospital_stateavg_from_code[(hosp.lower(), mc)][sa] += 1

    for obj in context_records:
        # rich jsonl current row
        row_context = obj.get("row_context", {})
        if isinstance(row_context, dict):
            consume_row_values(row_context.get("row_values", {}))

        # similar rows
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            topk = sim_ctx.get("topk_similar_rows", []) or []
            for item in topk:
                if isinstance(item, dict):
                    consume_row_values(item.get("row_values", {}))

        # column top values
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            col = str(col_ctx.get("column", obj.get("column", "")))
            tops = col_ctx.get("top_values_with_counts", []) or []
            for t in tops:
                if isinstance(t, dict) and t.get("value", None) is not None:
                    val = canonical_empty_text(t.get("value"))
                    cnt = int(t.get("count", 1) or 1)
                    if col and val:
                        column_value_counter[col][val] += cnt

        # flat csv fallback
        if "column" in obj and "value" in obj:
            col = str(obj["column"])
            val = canonical_empty_text(obj["value"])
            if col and val:
                column_value_counter[col][val] += 1

    # optional whole dirty table
    if dirty_table_df is not None:
        for col in dirty_table_df.columns:
            for val in dirty_table_df[col].map(canonical_empty_text).tolist():
                if val != "":
                    column_value_counter[str(col)][val] += 1

        # structured dictionaries from dirty table rows
        for _, r in dirty_table_df.iterrows():
            row_vals = {c: canonical_empty_text(r[c]) for c in dirty_table_df.columns}
            consume_row_values(row_vals)

    global_dict = {
        "column_value_counter": column_value_counter,
        "measurecode_to_names": measurecode_to_names,
        "condition_to_names": condition_to_names,
        "state_prefix_to_measurecodes": state_prefix_to_measurecodes,
        "hospital_stateavg_from_code": hospital_stateavg_from_code,
    }
    return global_dict


# ============================================================
# 5. context 标准化
# ============================================================

def merge_flat_into_context_map(context_map, flat_records):
    """
    如果 rich jsonl 没有某些字段，可以用 flat csv 补摘要字段
    """
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
            value_str = canonical_empty_text(obj.get("value", ""))
            context_map[key] = {
                "row_id": row_id,
                "column": column,
                "value": value_str,
                "suggested_correct_value": canonical_empty_text(obj.get("suggested_correct_value", "")),
                "neighbor_majority_value": canonical_empty_text(obj.get("neighbor_majority_value", "")),
                "value_frequency": obj.get("value_frequency", None),
                "value_frequency_rank": obj.get("value_frequency_rank", None),
                "semantic_type": obj.get("semantic_type", None),
                "main_rule_type": obj.get("main_rule_type", None),
                "main_usage_role": obj.get("main_usage_role", None),
                "violation_count": obj.get("violation_count", None),
                "conflict_score": obj.get("conflict_score", None),
                "rule_expected_values": [],
                "column_top_values": [],
                "alternative_values_from_column": [],
                "similar_row_target_values": [],
                "row_values": {},
                "state_value": "",
                "hospital_name": "",
                "measure_code_value": "",
                "condition_value": "",
                "measure_name_value": "",
                "stateavg_value": "",
            }
        ctx = context_map[key]

        for field in [
            "suggested_correct_value", "neighbor_majority_value", "value_frequency",
            "value_frequency_rank", "semantic_type", "main_rule_type",
            "main_usage_role", "violation_count", "conflict_score"
        ]:
            if (ctx.get(field, None) in [None, "", []]) and (field in obj):
                val = obj[field]
                if isinstance(val, str):
                    val = val.strip()
                ctx[field] = val


def build_context_map_rich(context_records, global_dict):
    """
    richer jsonl -> unified context_map
    """
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

        # A. conflict_context.violated_rules_detailed[*].expected_value
        conflict_context = obj.get("conflict_context", {})
        if isinstance(conflict_context, dict):
            violated = conflict_context.get("violated_rules_detailed", []) or []
            for vr in violated:
                if isinstance(vr, dict):
                    ev = canonical_empty_text(vr.get("expected_value", None))
                    if ev not in ["", "empty"]:
                        rule_expected_values.append(ev)

            # expected_values_from_rules
            evs = conflict_context.get("expected_values_from_rules", []) or []
            for item in evs:
                if isinstance(item, dict):
                    ev = canonical_empty_text(item.get("value", None))
                    if ev not in ["", "empty"]:
                        rule_expected_values.append(ev)

            # alternative_values_from_column
            avs = conflict_context.get("alternative_values_from_column", []) or []
            for item in avs:
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in ["", "empty"]:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": 0.0
                        })

            # all_alternative_values 里也可能混有 column_top / expected value
            all_avs = conflict_context.get("all_alternative_values", []) or []
            for item in all_avs:
                if isinstance(item, dict):
                    val = canonical_empty_text(item.get("value", None))
                    if val not in ["", "empty"]:
                        alternative_values_from_column.append({
                            "value": val,
                            "count": int(item.get("count", 1) or 1),
                            "ratio": 0.0
                        })

        # B. suggested_correct_value
        suggested_correct_value = canonical_empty_text(obj.get("suggested_correct_value", ""))
        if suggested_correct_value == "empty":
            suggested_correct_value = ""

        # C. neighbor majority
        neighbor_majority_value = ""
        sim_ctx = obj.get("similarity_context", {})
        if isinstance(sim_ctx, dict):
            neighbor_majority_value = canonical_empty_text(sim_ctx.get("neighbor_majority_value", ""))
            topk_similar = sim_ctx.get("topk_similar_rows", []) or []
            for item in topk_similar[:TOP_K_SIMILAR_ROW_VALUES]:
                if isinstance(item, dict):
                    tv = canonical_empty_text(item.get("target_column_value", ""))
                    if tv not in ["", "empty"]:
                        similar_row_target_values.append(tv)

        # D. column top values
        column_top_values = []
        col_ctx = obj.get("column_context", {})
        if isinstance(col_ctx, dict):
            top_vals = col_ctx.get("top_values_with_counts", []) or []
            for item in top_vals[:TOP_K_COLUMN_VALUES]:
                if isinstance(item, dict) and item.get("value", None) is not None:
                    val = canonical_empty_text(item["value"])
                    if val not in ["", "empty"]:
                        column_top_values.append({
                            "value": val,
                            "count": int(item.get("count", 0) or 0),
                            "ratio": safe_float(item.get("ratio", 0.0), 0.0)
                        })

        # fallback: if no top vals, use global counter
        if not column_top_values:
            counter = column_value_counter.get(column, Counter())
            total = max(sum(counter.values()), 1)
            for val, cnt in counter.most_common(TOP_K_COLUMN_VALUES):
                column_top_values.append({"value": val, "count": cnt, "ratio": cnt / total})

        # row_context.row_values
        row_values = {}
        row_ctx = obj.get("row_context", {})
        if isinstance(row_ctx, dict):
            rv = row_ctx.get("row_values", {})
            if isinstance(rv, dict):
                row_values = {str(k): canonical_empty_text(v) for k, v in rv.items()}

        context_map[(row_id, column)] = {
            "row_id": row_id,
            "column": column,
            "value": value_str,
            "suggested_correct_value": suggested_correct_value,
            "neighbor_majority_value": neighbor_majority_value,
            "value_frequency": obj.get("value_frequency", None),
            "value_frequency_rank": obj.get("value_frequency_rank", None),
            "semantic_type": obj.get("semantic_type", None),
            "main_rule_type": obj.get("main_rule_type", None),
            "main_usage_role": obj.get("main_usage_role", None),
            "violation_count": obj.get("violation_count", None),
            "conflict_score": obj.get("conflict_score", None),
            "rule_expected_values": rule_expected_values,
            "column_top_values": column_top_values,
            "alternative_values_from_column": alternative_values_from_column,
            "similar_row_target_values": similar_row_target_values,
            "row_values": row_values,
            "state_value": canonical_empty_text(row_values.get("State", "")),
            "hospital_name": canonical_empty_text(row_values.get("HospitalName", "")),
            "measure_code_value": canonical_empty_text(row_values.get("MeasureCode", "")),
            "condition_value": canonical_empty_text(row_values.get("Condition", "")),
            "measure_name_value": canonical_empty_text(row_values.get("MeasureName", "")),
            "stateavg_value": canonical_empty_text(row_values.get("Stateavg", "")),
        }

    return context_map


# ============================================================
# 6. 读取错误位置
# ============================================================

pred_df = pd.read_csv(ERROR_PRED_FILE, encoding="utf-8-sig")

required_base = {"row_id", "column"}
if not required_base.issubset(set(pred_df.columns)):
    raise ValueError(f"错误位置文件必须至少包含 {required_base}")

pred_df["row_id"] = pred_df["row_id"].astype(int)

if "final_pred_label" in pred_df.columns:
    error_df = pred_df[pred_df["final_pred_label"] == 1].copy()
elif "final_pred_label_name" in pred_df.columns:
    error_df = pred_df[pred_df["final_pred_label_name"].astype(str).str.lower() == "error"].copy()
else:
    error_df = pred_df.copy()

keep_cols = ["row_id", "column"]
if "value" in error_df.columns:
    keep_cols.append("value")

error_df = error_df[keep_cols].drop_duplicates().copy()
print(f"[INFO] 预测错误单元格数量: {len(error_df)}")


# ============================================================
# 7. 读取上下文 + 构造 context_map
# ============================================================

context_records = load_context_records(CONTEXT_FILE)
print(f"[INFO] 主上下文记录数: {len(context_records)}")

dirty_table_df = None
if str(DIRTY_TABLE_CSV).strip():
    dirty_table_df = pd.read_csv(DIRTY_TABLE_CSV, encoding="utf-8-sig")
    dirty_table_df = dirty_table_df.apply(lambda col: col.map(canonical_empty_text))
    print(f"[INFO] 已读取原始脏表: {DIRTY_TABLE_CSV}, shape={dirty_table_df.shape}")

global_dict = build_global_dictionaries(context_records, dirty_table_df=dirty_table_df)
context_map = build_context_map_rich(context_records, global_dict)

if str(FALLBACK_CONTEXT_FLAT_FILE).strip():
    flat_records = load_context_records(FALLBACK_CONTEXT_FLAT_FILE)
    print(f"[INFO] fallback flat 上下文记录数: {len(flat_records)}")
    merge_flat_into_context_map(context_map, flat_records)

print(f"[INFO] 可索引上下文条数: {len(context_map)}")


# ============================================================
# 8. 列专属补候选
# ============================================================

def generate_measurecode_candidates(dirty_value, ctx, global_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)

    # 1) known dictionary
    for code in KNOWN_MEASURE_CODES:
        sim = string_similarity(dirty, code)
        if sim >= MIN_STRING_SIM_FOR_CANONICAL or normalize_measurecode_like(dirty) == normalize_measurecode_like(code):
            cands.append((code, "measurecode_dictionary", 0.82, f"sim={sim:.4f}"))

    # 2) column frequent values
    col_counter = global_dict["column_value_counter"].get("MeasureCode", Counter())
    for val, cnt in col_counter.most_common(20):
        if looks_like_measure_code(val):
            sim = string_similarity(dirty, val)
            if sim >= MIN_STRING_SIM_FOR_CANONICAL:
                cands.append((val, "measurecode_dictionary", 0.78, f"freq={cnt}|sim={sim:.4f}"))

    return cands


def generate_stateavg_candidates(dirty_value, ctx, global_dict, existing_candidate_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    state_prefix = ctx.get("state_value", "").lower()

    if not state_prefix:
        # 尝试从 dirty 本身抽 prefix
        state_prefix = extract_state_prefix_from_stateavg(dirty)

    measurecode_candidates = []
    # 从已有候选里拿 measurecode
    if ctx.get("measure_code_value", "") and looks_like_measure_code(ctx["measure_code_value"]):
        measurecode_candidates.append(ctx["measure_code_value"])
    for item in existing_candidate_dict.values():
        v = canonical_empty_text(item["candidate_value"])
        if looks_like_measure_code(v):
            measurecode_candidates.append(v)

    # 从全局字典拿
    for v, _ in global_dict["state_prefix_to_measurecodes"].get(state_prefix, Counter()).most_common(10):
        measurecode_candidates.append(v)

    measurecode_candidates = list(dict.fromkeys([canonical_empty_text(x) for x in measurecode_candidates if x]))

    for mc in measurecode_candidates:
        if state_prefix:
            sa = f"{state_prefix}_{canonical_empty_text(mc)}"
            sim = string_similarity(dirty, sa)
            if sim >= 0.25 or dirty == "" or dirty == "empty":
                cands.append((sa, "stateavg_from_measurecode", 0.84, f"state={state_prefix}|mc={mc}|sim={sim:.4f}"))

    return cands


def generate_condition_candidates(dirty_value, ctx):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    mc = ctx.get("measure_code_value", "")
    fam = measure_family_name(mc)

    if fam:
        sim = string_similarity(dirty, fam)
        cands.append((fam, "condition_from_measurecode", 0.86, f"mc={mc}|sim={sim:.4f}"))

    for cond in CONDITION_CANONICAL:
        sim = string_similarity(dirty, cond)
        if sim >= MIN_STRING_SIM_FOR_CANONICAL or fam == cond:
            cands.append((cond, "condition_from_measurecode", 0.76, f"sim={sim:.4f}"))

    return cands


def generate_measurename_candidates(dirty_value, ctx, global_dict, existing_candidate_dict):
    cands = []
    dirty = canonical_empty_text(dirty_value)

    mc = canonical_empty_text(ctx.get("measure_code_value", ""))
    cond = canonical_empty_text(ctx.get("condition_value", ""))

    # 1) from measurecode dictionary
    if mc:
        for name, cnt in global_dict["measurecode_to_names"].get(mc, Counter()).most_common(10):
            sim = string_similarity(dirty, name)
            cands.append((name, "measurename_from_measurecode", 0.88, f"mc={mc}|cnt={cnt}|sim={sim:.4f}"))

    # 2) from condition dictionary
    if cond:
        for name, cnt in global_dict["condition_to_names"].get(cond, Counter()).most_common(10):
            sim = string_similarity(dirty, name)
            cands.append((name, "measurename_from_condition", 0.82, f"cond={cond}|cnt={cnt}|sim={sim:.4f}"))

    # 3) from measurecode candidates already generated
    for item in existing_candidate_dict.values():
        v = canonical_empty_text(item["candidate_value"])
        if looks_like_measure_code(v):
            for name, cnt in global_dict["measurecode_to_names"].get(v, Counter()).most_common(6):
                sim = string_similarity(dirty, name)
                cands.append((name, "measurename_from_measurecode", 0.80, f"mc_candidate={v}|cnt={cnt}|sim={sim:.4f}"))

    return cands


def generate_score_candidates(dirty_value):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    fixed = normalize_percent_candidate(dirty)
    if fixed and fixed != dirty:
        cands.append((fixed, "pattern_restore_score", 0.90, "pattern_restore"))
    return cands


def generate_sample_candidates(dirty_value):
    cands = []
    dirty = canonical_empty_text(dirty_value)
    fixed = normalize_sample_candidate(dirty)
    if fixed and fixed != dirty:
        cands.append((fixed, "pattern_restore_sample", 0.90, "pattern_restore"))
    return cands


# ============================================================
# 9. 为每个错误 cell 构造候选修复值
# ============================================================

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
            "top_candidate_score": None
        })
        continue

    dirty_value = ctx.get("value", "")
    if ("value" in error_df.columns) and (pd.notna(row.get("value", None))) and str(row.get("value")).strip() != "":
        dirty_value = canonical_empty_text(row["value"])

    candidate_dict = {}

    if INCLUDE_DIRTY_VALUE and dirty_value != "":
        add_candidate(candidate_dict, dirty_value, "dirty_value", 0.01)

    # -------------------------------------------------
    # A. 规则 expected values（rich jsonl 主力）
    # -------------------------------------------------
    expected_counter = Counter()
    for ev in ctx.get("rule_expected_values", []):
        ev = canonical_empty_text(ev)
        if ev == "" or norm_text(ev) == norm_text(dirty_value):
            continue
        expected_counter[ev] += 1

    total_votes = sum(expected_counter.values())
    for ev, cnt in expected_counter.items():
        score = 0.74 + 0.22 * (cnt / max(total_votes, 1))
        add_candidate(candidate_dict, ev, "expected_values_from_rules", score, extra=f"rule_vote_count={cnt}")

    # -------------------------------------------------
    # B. 邻居 / 相似行 target values
    # -------------------------------------------------
    neighbor_majority_value = ctx.get("neighbor_majority_value", None)
    if neighbor_majority_value and norm_text(neighbor_majority_value) != norm_text(dirty_value):
        add_candidate(candidate_dict, neighbor_majority_value, "neighbor_majority", 0.88)

    sim_counter = Counter()
    for v in ctx.get("similar_row_target_values", []):
        v = canonical_empty_text(v)
        if v and norm_text(v) != norm_text(dirty_value):
            sim_counter[v] += 1
    for v, cnt in sim_counter.items():
        score = 0.76 + 0.10 * min(cnt, TOP_K_SIMILAR_ROW_VALUES) / max(TOP_K_SIMILAR_ROW_VALUES, 1)
        add_candidate(candidate_dict, v, "similar_row_target", score, extra=f"count={cnt}")

    # -------------------------------------------------
    # C. suggested_correct_value
    # -------------------------------------------------
    suggested_correct_value = ctx.get("suggested_correct_value", None)
    if suggested_correct_value and norm_text(suggested_correct_value) != norm_text(dirty_value):
        add_candidate(candidate_dict, suggested_correct_value, "suggested_correct_value", 0.93)

    # -------------------------------------------------
    # D. 列内 top values / 替代值
    # -------------------------------------------------
    alt_items = []
    alt_items.extend(ctx.get("column_top_values", [])[:TOP_K_COLUMN_VALUES])
    alt_items.extend(ctx.get("alternative_values_from_column", [])[:TOP_K_COLUMN_VALUES])

    dedup_alt = {}
    for item in alt_items:
        if not isinstance(item, dict):
            continue
        val = canonical_empty_text(item.get("value", ""))
        if val == "" or norm_text(val) == norm_text(dirty_value):
            continue
        k = norm_text(val)
        prev = dedup_alt.get(k, {"value": val, "count": 0, "ratio": 0.0})
        prev["count"] = max(prev["count"], int(item.get("count", 0) or 0))
        prev["ratio"] = max(prev["ratio"], safe_float(item.get("ratio", 0.0), 0.0))
        dedup_alt[k] = prev

    for item in dedup_alt.values():
        val = item["value"]
        ratio = safe_float(item.get("ratio", 0.0), 0.0)
        sim = string_similarity(dirty_value, val)

        top_score = 0.34 + 0.18 * min(ratio, 1.0)
        add_candidate(candidate_dict, val, "column_top_value", top_score, extra=f"ratio={ratio:.4f}")

        if sim >= MIN_STRING_SIM_FOR_COLUMN_VALUE:
            sim_score = 0.48 + 0.22 * sim + 0.12 * min(ratio, 1.0)
            add_candidate(candidate_dict, val, "similar_column_value", sim_score, extra=f"ratio={ratio:.4f}|sim={sim:.4f}")

    # -------------------------------------------------
    # E. 列专属补候选
    # -------------------------------------------------
    if ENABLE_COLUMN_SPECIFIC_GENERATION:
        if column == "MeasureCode":
            for val, source, score, extra in generate_measurecode_candidates(dirty_value, ctx, global_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Stateavg":
            for val, source, score, extra in generate_stateavg_candidates(dirty_value, ctx, global_dict, candidate_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Condition":
            for val, source, score, extra in generate_condition_candidates(dirty_value, ctx):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "MeasureName":
            for val, source, score, extra in generate_measurename_candidates(dirty_value, ctx, global_dict, candidate_dict):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Score":
            for val, source, score, extra in generate_score_candidates(dirty_value):
                add_candidate(candidate_dict, val, source, score, extra=extra)

        elif column == "Sample":
            for val, source, score, extra in generate_sample_candidates(dirty_value):
                add_candidate(candidate_dict, val, source, score, extra=extra)

    # -------------------------------------------------
    # F. 去掉与脏值完全相同的候选
    # -------------------------------------------------
    remove_keys = []
    for k, item in candidate_dict.items():
        if norm_text(item["candidate_value"]) == norm_text(dirty_value):
            remove_keys.append(k)
    for k in remove_keys:
        candidate_dict.pop(k, None)

    # -------------------------------------------------
    # G. 最终打分与排序
    # -------------------------------------------------
    final_candidates = []
    for _, item in candidate_dict.items():
        cand = item["candidate_value"]
        sim = string_similarity(dirty_value, cand)
        source_count = len(item["candidate_sources"])
        support_bonus = 0.04 * max(source_count - 1, 0)

        # structured semantic bonus
        semantic_bonus = 0.0
        if column == "MeasureCode" and looks_like_measure_code(cand):
            semantic_bonus += 0.08
        if column == "Stateavg" and looks_like_stateavg(cand):
            semantic_bonus += 0.08
        if column == "Condition" and cand.lower() in CONDITION_CANONICAL:
            semantic_bonus += 0.07
        if column == "Score" and normalize_percent_candidate(cand) == cand:
            semantic_bonus += 0.07
        if column == "Sample" and normalize_sample_candidate(cand) == cand:
            semantic_bonus += 0.07
        if column == "MeasureName":
            mc = ctx.get("measure_code_value", "")
            fam = measure_family_name(mc)
            if fam and fam in cand.lower():
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
            -x["is_from_dictionary"],
            -x["is_from_pattern_restore"],
            -x["is_from_neighbor"],
            x["candidate_value"]
        )
    )[:MAX_CANDIDATES_PER_CELL]

    # -------------------------------------------------
    # H. 展开输出
    # -------------------------------------------------
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
            "source_count": len(item["candidate_sources"]),
            "extra_info": " || ".join(item["extra_info"][:15]),
            "violation_count": ctx.get("violation_count", None),
            "conflict_score": ctx.get("conflict_score", None),
            "semantic_type": ctx.get("semantic_type", None),
            "main_rule_type": ctx.get("main_rule_type", None),
            "main_usage_role": ctx.get("main_usage_role", None),
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


# ============================================================
# 10. 保存结果
# ============================================================

expanded_df = pd.DataFrame(expanded_rows)
grouped_df = pd.DataFrame(grouped_rows)

if not expanded_df.empty:
    expanded_df = expanded_df.sort_values(["row_id", "column", "candidate_rank"]).reset_index(drop=True)

if not grouped_df.empty:
    grouped_df = grouped_df.sort_values(["row_id", "column"]).reset_index(drop=True)

expanded_df.to_csv(OUTPUT_CANDIDATES_EXPANDED_CSV, index=False, encoding="utf-8-sig")
grouped_df.to_csv(OUTPUT_CANDIDATES_GROUPED_CSV, index=False, encoding="utf-8-sig")

print(f"[INFO] 缺失上下文的错误 cell 数量: {missing_context_count}")
print(f"[INFO] 候选展开文件已保存: {OUTPUT_CANDIDATES_EXPANDED_CSV}")
print(f"[INFO] 候选汇总文件已保存: {OUTPUT_CANDIDATES_GROUPED_CSV}")

if not grouped_df.empty:
    print("\n[INFO] 前 20 个错误 cell 的候选示例：")
    print(grouped_df.head(20).to_string(index=False))
