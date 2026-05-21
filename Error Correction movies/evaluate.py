import os
import re
import pandas as pd


# ============================================================
# 1. 用户配置区
# ============================================================

# Movies clean / dirty 表
# 按你的实际文件名改这里即可。
CLEAN_FILE = "/mnt/mydata/dq/projects/Splittree/movies/movies_clean.csv"
DIRTY_FILE = "/mnt/mydata/dq/projects/Splittree/movies/movies_dirty.csv"

# 修复结果文件，适配 infer_ltr_movies.py / infer_ltr_movies_current_dir.py 输出：
# inference_top1_repairs.csv
REPAIR_RESULT_FILE = "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/inference_top1_repairs.csv"

OUTPUT_DIR = (
    "/mnt/mydata/dq/projects/Splittree/test/Error Correction movies/"
    "repair_eval_output_movies"
)

OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_summary.csv")
OUTPUT_CELL_DETAILS_CSV = os.path.join(OUTPUT_DIR, "full_table_eval_all_cells.csv")
OUTPUT_TP_CSV = os.path.join(OUTPUT_DIR, "full_table_tp_details.csv")
OUTPUT_FP_CSV = os.path.join(OUTPUT_DIR, "full_table_fp_details.csv")
OUTPUT_FN_CSV = os.path.join(OUTPUT_DIR, "full_table_fn_details.csv")
OUTPUT_TN_CSV = os.path.join(OUTPUT_DIR, "full_table_tn_details.csv")
OUTPUT_REPAIRED_TABLE_CSV = os.path.join(OUTPUT_DIR, "repaired_table.csv")
OUTPUT_TOUCHED_CSV = os.path.join(OUTPUT_DIR, "touched_cells_details.csv")
OUTPUT_COLUMN_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "column_level_eval_summary.csv")
OUTPUT_GATE_REASON_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "gate_reason_eval_summary.csv")
OUTPUT_SOURCE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "candidate_source_eval_summary.csv")
OUTPUT_MOVIES_TYPE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "movies_column_type_eval_summary.csv")
OUTPUT_FORMAT_ONLY_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "movies_format_only_difference_summary.csv")

# 是否使用 Movies 专属 normalized comparison。
# 建议 True：消除 Year/Duration/Rating/Release Date/List 等格式差异造成的无意义误判。
USE_MOVIES_NORMALIZED_COMPARISON = True

# 普通文本列是否大小写不敏感。
# 默认 False：Name / Description 等普通文本严格比较大小写。
# 如果确认 clean/dirty 中大小写差异不算错误，可改 True。
TEXT_COLUMNS_CASE_INSENSITIVE = False

# 列表型字段是否忽略顺序。
# 默认 False：Actors/Cast/Genre 等列表顺序仍然作为语义的一部分。
# 如果你的 clean/dirty 中列表顺序不重要，可改 True。
LIST_COLUMNS_ORDER_INSENSITIVE = False

# 是否把多个空格压缩为一个空格。
COMPRESS_TEXT_WHITESPACE = True

# 是否在 repaired table 中写入 norm_text 后的值。
# 一般保持 True，避免 NaN/None 混入。
STRINGIFY_REPAIRED_TABLE = True


# ============================================================
# 2. 基础工具函数
# ============================================================

EMPTY_TOKENS = {"", "empty", "null", "none", "nan", "na", "n/a", "missing", "{null}"}

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_NUM_TO_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

COUNTRY_CANONICAL_MAP = {
    "usa": "USA",
    "us": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "uk": "UK",
    "u.k.": "UK",
    "united kingdom": "UK",
    "england": "UK",
    "france": "France",
    "germany": "Germany",
    "india": "India",
    "canada": "Canada",
    "australia": "Australia",
    "italy": "Italy",
    "spain": "Spain",
    "japan": "Japan",
    "china": "China",
    "hong kong": "Hong Kong",
    "south korea": "South Korea",
    "korea": "South Korea",
    "russia": "Russia",
    "soviet union": "Soviet Union",
    "mexico": "Mexico",
    "brazil": "Brazil",
    "argentina": "Argentina",
    "sweden": "Sweden",
    "denmark": "Denmark",
    "norway": "Norway",
    "netherlands": "Netherlands",
    "ireland": "Ireland",
    "kuwait": "Kuwait",
}

LANGUAGE_CANONICAL_MAP = {
    "english": "English",
    "eng": "English",
    "en": "English",
    "french": "French",
    "fre": "French",
    "fr": "French",
    "spanish": "Spanish",
    "spa": "Spanish",
    "es": "Spanish",
    "german": "German",
    "ger": "German",
    "de": "German",
    "italian": "Italian",
    "ita": "Italian",
    "japanese": "Japanese",
    "jpn": "Japanese",
    "chinese": "Chinese",
    "chi": "Chinese",
    "mandarin": "Mandarin",
    "hindi": "Hindi",
    "korean": "Korean",
    "russian": "Russian",
    "arabic": "Arabic",
    "portuguese": "Portuguese",
}

MOVIE_TIME_COLUMNS = {"Year", "Release Date", "Duration"}
MOVIE_RATING_COLUMNS = {"RatingValue", "RatingCount", "ReviewCount"}
MOVIE_LIST_COLUMNS = {"Actors", "Cast", "Creator", "Director", "Genre", "Filming Locations"}
MOVIE_TEXT_COLUMNS = {"Name", "Description"}
MOVIE_METADATA_COLUMNS = {"Country", "Language"}
MOVIE_CANONICAL_COLUMNS = MOVIE_TIME_COLUMNS | MOVIE_RATING_COLUMNS | MOVIE_LIST_COLUMNS | MOVIE_METADATA_COLUMNS


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_text(x):
    """
    基础字符串清洗。只用于展示/写回，不代表 Movies normalized comparison。
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in EMPTY_TOKENS:
        return ""
    return s


def lower_clean(x):
    return norm_text(x).strip().lower()


def compact_text(s: str) -> str:
    s = norm_text(s)
    if COMPRESS_TEXT_WHITESPACE:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "").replace("%", ""))
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(str(x).strip().replace(",", "")))
    except Exception:
        return default


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"表文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_same_shape(clean_df: pd.DataFrame, dirty_df: pd.DataFrame):
    if clean_df.shape != dirty_df.shape:
        raise ValueError(
            f"clean 与 dirty 形状不一致: clean={clean_df.shape}, dirty={dirty_df.shape}"
        )

    if list(clean_df.columns) != list(dirty_df.columns):
        raise ValueError(
            "clean 与 dirty 列名不一致，请先对齐列顺序和列名。\n"
            f"clean columns={list(clean_df.columns)}\n"
            f"dirty columns={list(dirty_df.columns)}"
        )


def stringify_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(norm_text).astype(object)
    return out


# ============================================================
# 3. Movies 专属规范化比较
# ============================================================

def normalize_year_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""
    m = re.search(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", s)
    if not m:
        return compact_text(s)
    y = int(m.group(1))
    if 1800 <= y <= 2100:
        return str(y)
    return compact_text(s)


def normalize_country_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""
    k = lower_clean(s)
    if k in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[k]
    s = re.sub(r"\s+", " ", s).strip()
    if TEXT_COLUMNS_CASE_INSENSITIVE:
        return s.lower()
    return s.title() if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", s) else s


def split_list_tokens(x):
    s = norm_text(x)
    if s == "":
        return []
    parts = re.split(r"[,;/|]+", s)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", " ", str(p).strip())
        if not p:
            continue
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def normalize_list_candidate(x):
    toks = split_list_tokens(x)
    if not toks:
        return ""
    if TEXT_COLUMNS_CASE_INSENSITIVE:
        toks = [t.lower() for t in toks]
    if LIST_COLUMNS_ORDER_INSENSITIVE:
        toks = sorted(toks, key=lambda z: z.lower())
    return ",".join(toks)


def normalize_language_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""
    toks = split_list_tokens(s)
    fixed = []
    for t in toks:
        k = t.strip().lower()
        fixed.append(LANGUAGE_CANONICAL_MAP.get(k, t.strip().title()))
    if LIST_COLUMNS_ORDER_INSENSITIVE:
        fixed = sorted(fixed, key=lambda z: z.lower())
    return ",".join(fixed)


def parse_release_date_parts(x):
    s = norm_text(x)
    if s == "":
        return None

    country = None
    cm = re.search(r"\(([^()]+)\)", s)
    if cm:
        country = normalize_country_candidate(cm.group(1))

    s_no_country = re.sub(r"\([^()]+\)", "", s).strip()

    # 29 August 2014
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        day = int(m.group(1))
        month = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    # August 2014
    m = re.search(r"\b([A-Za-z]+)\s+(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        month = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        year = int(m.group(2))
        if month:
            return {"year": year, "month": month, "day": None, "country": country}

    # yyyy-mm-dd / yyyy/mm/dd
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s_no_country)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    # mm/dd/yyyy
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", s_no_country)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return {"year": year, "month": month, "day": day, "country": country}

    y = normalize_year_candidate(s_no_country)
    if re.fullmatch(r"\d{4}", y):
        return {"year": int(y), "month": None, "day": None, "country": country}

    return None


def normalize_release_date_candidate(x):
    parts = parse_release_date_parts(x)
    if not parts:
        return compact_text(x).lower() if TEXT_COLUMNS_CASE_INSENSITIVE else compact_text(x)

    year = parts.get("year")
    month = parts.get("month")
    day = parts.get("day")
    country = parts.get("country")

    if not year:
        return compact_text(x)

    if month:
        month_name = MONTH_NUM_TO_NAME.get(int(month))
    else:
        month_name = None

    if month_name and day:
        base = f"{int(day)} {month_name} {int(year)}"
    elif month_name:
        base = f"{month_name} {int(year)}"
    else:
        base = str(int(year))

    if country:
        base = f"{base} ({country})"
    return base


def normalize_duration_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    hours = 0
    minutes = 0
    mh = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", s, flags=re.I)
    mm = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", s, flags=re.I)
    if mh or mm:
        if mh:
            hours = int(mh.group(1))
        if mm:
            minutes = int(mm.group(1))
        total = hours * 60 + minutes
        if 1 <= total <= 500:
            return f"{total} min"

    mnum = re.search(r"\b(\d{1,3})(?:\.0)?\b", s)
    if mnum:
        num = int(mnum.group(1))
        if 1 <= num <= 500:
            return f"{num} min"

    return compact_text(s)


def normalize_rating_value_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""
    raw = s.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return compact_text(s)
    v = float(m.group(1))
    if "%" in raw and 0 <= v <= 100:
        v = v / 10.0
    if 0 <= v <= 10:
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}.0"
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return compact_text(s)


def normalize_count_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return compact_text(s)
    return str(int(m.group(1).replace(",", "")))


def normalize_review_count_candidate(x):
    s = norm_text(x)
    if s == "":
        return ""

    nums = [int(v.replace(",", "")) for v in re.findall(r"\d[\d,]*", s)]
    if not nums:
        return compact_text(s)

    lower = s.lower()
    if "user" in lower or "critic" in lower:
        parts = []
        um = re.search(r"(\d[\d,]*)\s*user", lower)
        cm = re.search(r"(\d[\d,]*)\s*critic", lower)
        if um:
            u = int(um.group(1).replace(",", ""))
            parts.append(f"{u} user" + ("" if u == 1 else "s"))
        if cm:
            c = int(cm.group(1).replace(",", ""))
            parts.append(f"{c} critic" + ("" if c == 1 else "s"))
        if parts:
            return ",".join(parts)

    return str(sum(nums))


def normalize_for_compare(column: str, value):
    s = norm_text(value)

    if not USE_MOVIES_NORMALIZED_COMPARISON:
        text = compact_text(s)
        return text.lower() if TEXT_COLUMNS_CASE_INSENSITIVE else text

    col = str(column).strip()
    cl = col.lower()

    if cl == "year":
        return normalize_year_candidate(s)
    if cl == "release date":
        return normalize_release_date_candidate(s)
    if cl == "duration":
        return normalize_duration_candidate(s)
    if cl == "ratingvalue":
        return normalize_rating_value_candidate(s)
    if cl in {"ratingcount", "id"}:
        return normalize_count_candidate(s)
    if cl == "reviewcount":
        return normalize_review_count_candidate(s)
    if cl == "country":
        return normalize_country_candidate(s)
    if cl == "language":
        return normalize_language_candidate(s)
    if col in MOVIE_LIST_COLUMNS:
        return normalize_list_candidate(s)

    text = compact_text(s)
    if TEXT_COLUMNS_CASE_INSENSITIVE and (col in MOVIE_TEXT_COLUMNS or col in MOVIE_METADATA_COLUMNS):
        return text.lower()
    return text


def equal_for_eval(column: str, a, b) -> bool:
    return normalize_for_compare(column, a) == normalize_for_compare(column, b)


def infer_movies_column_type(column: str) -> str:
    col = str(column)
    if col in MOVIE_TIME_COLUMNS:
        return "time"
    if col in MOVIE_RATING_COLUMNS:
        return "rating"
    if col in MOVIE_LIST_COLUMNS:
        return "list_or_people"
    if col in MOVIE_METADATA_COLUMNS:
        return "metadata"
    if col in MOVIE_TEXT_COLUMNS:
        return "text"
    return "other"


# ============================================================
# 4. 修复结果读取与写回
# ============================================================

def find_repair_value_column(df: pd.DataFrame) -> str:
    """
    兼容不同版本修复结果输出字段。
    优先级说明：
    - repaired_value / predicted_repair_value：推理脚本经过 gate 后真正决定写回表的值
    - predicted_repair / repair_value / predicted_value：通用修复值字段
    - student_best_candidate / student_top_candidate / candidate_value：模型 top 候选，不一定经过 gate
    """
    candidates = [
        "repaired_value",
        "predicted_repair_value",
        "predicted_repair",
        "repair_value",
        "predicted_value",
        "student_best_candidate",
        "student_top_candidate",
        "candidate_value",
        "teacher_best_candidate",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "修复结果文件中未找到修复值字段。支持字段包括："
        "repaired_value / predicted_repair_value / predicted_repair / repair_value / predicted_value / "
        "student_best_candidate / student_top_candidate / candidate_value / teacher_best_candidate"
    )


def load_repair_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"修复结果文件不存在: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "row_id" not in df.columns or "column" not in df.columns:
        raise ValueError("修复结果文件必须包含 row_id 和 column")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["column"] = df["column"].astype(str).str.strip()

    repair_col = find_repair_value_column(df)
    keep_cols = ["row_id", "column", repair_col]

    optional_cols = [
        "dirty_value",
        "student_top_candidate",
        "student_best_candidate",
        "candidate_value",
        "student_score",
        "student_margin",
        "margin",
        "top2_score",
        "evidence_strength",
        "movie_evidence_strength",
        "movie_consistency_strength",
        "movie_consistency_confidence",
        "movie_consistency_support_count",
        "gate_passed",
        "pass_gate",
        "gate_reason",
        "student_reason_pred",
        "candidate_count",
        "candidate_source",
        "candidate_source_text",
        "decision_source",
        "decision_confidence",
        "is_hard_case",
        "movie_candidate_is_duration",
        "movie_candidate_is_year",
        "movie_candidate_is_release_date",
        "movie_candidate_is_rating_value",
        "movie_candidate_is_count",
        "movie_candidate_is_review_count",
        "candidate_equals_allowed_consensus_value",
        "candidate_and_allowed_both_movie_consistency",
        "contains_movie_consistency_source",
        "contains_movie_dictionary_source",
        "contains_movie_canonical_source",
        "allowed_consensus_value",
        "allowed_consensus_confidence",
        "allowed_consensus_margin",
        "allowed_consensus_support_count",
        "allowed_consensus_reason",
        "allowed_consistency_name",
        "allowed_lhs_key",
        "allowed_is_movie_metadata_consistency",
        "allowed_is_strong_movie_consistency",
    ]

    for c in optional_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out = out.rename(columns={repair_col: "predicted_repair"})
    out["predicted_repair"] = out["predicted_repair"].map(norm_text)

    out = out.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return out


def apply_repairs_to_dirty(dirty_df: pd.DataFrame, repair_df: pd.DataFrame) -> pd.DataFrame:
    if STRINGIFY_REPAIRED_TABLE:
        repaired_df = stringify_table(dirty_df)
    else:
        repaired_df = dirty_df.copy()

    n_rows = len(repaired_df)
    valid_columns = set(repaired_df.columns)

    skipped_bad_row = 0
    skipped_bad_col = 0

    for _, r in repair_df.iterrows():
        row_id = int(r["row_id"])
        column = str(r["column"]).strip()
        pred_val = norm_text(r["predicted_repair"])

        if row_id < 0 or row_id >= n_rows:
            skipped_bad_row += 1
            continue

        if column not in valid_columns:
            skipped_bad_col += 1
            continue

        repaired_df.at[row_id, column] = pred_val

    if skipped_bad_row > 0:
        print(f"[WARN] 跳过越界 row_id 修复记录数: {skipped_bad_row}")
    if skipped_bad_col > 0:
        print(f"[WARN] 跳过不存在 column 修复记录数: {skipped_bad_col}")

    return repaired_df


# ============================================================
# 5. 全表评估逻辑
# ============================================================

def build_full_table_eval(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> pd.DataFrame:
    repair_pos_set = set(
        (int(r["row_id"]), str(r["column"]).strip())
        for _, r in repair_df.iterrows()
    )

    repair_info_map = {}
    for _, r in repair_df.iterrows():
        key = (int(r["row_id"]), str(r["column"]).strip())
        repair_info_map[key] = r.to_dict()

    rows = []

    for row_id in range(len(clean_df)):
        for column in clean_df.columns:
            column = str(column)

            clean_val = norm_text(clean_df.at[row_id, column])
            dirty_val = norm_text(dirty_df.at[row_id, column])
            repaired_val = norm_text(repaired_df.at[row_id, column])

            clean_cmp = normalize_for_compare(column, clean_val)
            dirty_cmp = normalize_for_compare(column, dirty_val)
            repaired_cmp = normalize_for_compare(column, repaired_val)

            is_true_error = int(dirty_cmp != clean_cmp)
            is_touched = int((row_id, column) in repair_pos_set or repaired_cmp != dirty_cmp)
            is_changed_strict = int(repaired_val != dirty_val)
            is_changed_normalized = int(repaired_cmp != dirty_cmp)
            is_final_correct = int(repaired_cmp == clean_cmp)

            if is_true_error == 1 and is_final_correct == 1:
                eval_type = "TP"
            elif is_true_error == 0 and repaired_cmp != clean_cmp:
                eval_type = "FP"
            elif is_true_error == 1 and is_final_correct == 0:
                eval_type = "FN"
            else:
                eval_type = "TN"

            info = repair_info_map.get((row_id, column), {})

            row = {
                "row_id": row_id,
                "column": column,
                "movies_column_type": infer_movies_column_type(column),

                "dirty_value": dirty_val,
                "clean_value": clean_val,
                "repaired_value": repaired_val,

                "dirty_value_cmp": dirty_cmp,
                "clean_value_cmp": clean_cmp,
                "repaired_value_cmp": repaired_cmp,

                "is_true_error": is_true_error,
                "is_touched": is_touched,
                "is_changed_strict": is_changed_strict,
                "is_changed_normalized": is_changed_normalized,
                "is_final_correct": is_final_correct,
                "eval_type": eval_type,

                "strict_equals_clean": int(repaired_val == clean_val),
                "normalized_equals_clean": int(repaired_cmp == clean_cmp),
                "strict_format_only_difference": int((repaired_val != clean_val) and (repaired_cmp == clean_cmp)),
                "dirty_strict_format_only_difference": int((dirty_val != clean_val) and (dirty_cmp == clean_cmp)),
            }

            for extra_col in [
                "predicted_repair",
                "student_top_candidate",
                "student_best_candidate",
                "candidate_value",
                "student_score",
                "student_margin",
                "margin",
                "top2_score",
                "evidence_strength",
                "movie_evidence_strength",
                "movie_consistency_strength",
                "movie_consistency_confidence",
                "movie_consistency_support_count",
                "gate_passed",
                "pass_gate",
                "gate_reason",
                "student_reason_pred",
                "candidate_count",
                "candidate_source",
                "candidate_source_text",
                "decision_source",
                "decision_confidence",
                "is_hard_case",
                "movie_candidate_is_duration",
                "movie_candidate_is_year",
                "movie_candidate_is_release_date",
                "movie_candidate_is_rating_value",
                "movie_candidate_is_count",
                "movie_candidate_is_review_count",
                "candidate_equals_allowed_consensus_value",
                "candidate_and_allowed_both_movie_consistency",
                "contains_movie_consistency_source",
                "contains_movie_dictionary_source",
                "contains_movie_canonical_source",
                "allowed_consensus_value",
                "allowed_consensus_confidence",
                "allowed_consensus_margin",
                "allowed_consensus_support_count",
                "allowed_consensus_reason",
                "allowed_consistency_name",
                "allowed_lhs_key",
                "allowed_is_movie_metadata_consistency",
                "allowed_is_strong_movie_consistency",
            ]:
                if extra_col in info:
                    row[extra_col] = info.get(extra_col)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(eval_df: pd.DataFrame) -> dict:
    tp = int((eval_df["eval_type"] == "TP").sum())
    fp = int((eval_df["eval_type"] == "FP").sum())
    fn = int((eval_df["eval_type"] == "FN").sum())
    tn = int((eval_df["eval_type"] == "TN").sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    total_cells = len(eval_df)
    total_true_errors = int(eval_df["is_true_error"].sum())
    touched_cells = int(eval_df["is_touched"].sum())
    remaining_wrong_cells = int((eval_df["repaired_value_cmp"] != eval_df["clean_value_cmp"]).sum())
    changed_cells = int((eval_df["repaired_value_cmp"] != eval_df["dirty_value_cmp"]).sum())
    strict_changed_cells = int((eval_df["repaired_value"] != eval_df["dirty_value"]).sum())
    strict_remaining_wrong_cells = int((eval_df["repaired_value"] != eval_df["clean_value"]).sum())

    unchanged_true_errors = int(
        ((eval_df["is_true_error"] == 1) & (eval_df["repaired_value_cmp"] == eval_df["dirty_value_cmp"])).sum()
    )

    format_only_differences_after_repair = int(eval_df["strict_format_only_difference"].sum())
    dirty_format_only_differences = int(eval_df["dirty_strict_format_only_difference"].sum())

    return {
        "total_cells": total_cells,
        "true_error_cells": total_true_errors,
        "touched_cells": touched_cells,
        "changed_cells": changed_cells,
        "remaining_wrong_cells_after_repair": remaining_wrong_cells,
        "unchanged_true_errors": unchanged_true_errors,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "strict_changed_cells": strict_changed_cells,
        "strict_remaining_wrong_cells_after_repair": strict_remaining_wrong_cells,
        "format_only_differences_after_repair": format_only_differences_after_repair,
        "dirty_format_only_differences": dirty_format_only_differences,
        "use_movies_normalized_comparison": int(USE_MOVIES_NORMALIZED_COMPARISON),
        "text_columns_case_insensitive": int(TEXT_COLUMNS_CASE_INSENSITIVE),
        "list_columns_order_insensitive": int(LIST_COLUMNS_ORDER_INSENSITIVE),
    }


def build_column_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column, g in eval_df.groupby("column", sort=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        rows.append({
            "column": column,
            "movies_column_type": infer_movies_column_type(column),
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "changed_cells": int((g["repaired_value_cmp"] != g["dirty_value_cmp"]).sum()),
            "strict_changed_cells": int((g["repaired_value"] != g["dirty_value"]).sum()),
            "remaining_wrong_cells_after_repair": int((g["repaired_value_cmp"] != g["clean_value_cmp"]).sum()),
            "strict_remaining_wrong_cells_after_repair": int((g["repaired_value"] != g["clean_value"]).sum()),
            "format_only_differences_after_repair": int(g["strict_format_only_difference"].sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "F1"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_gate_reason_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    if "gate_reason" not in eval_df.columns:
        return pd.DataFrame()

    touched = eval_df[eval_df["is_touched"] == 1].copy()
    if touched.empty:
        return pd.DataFrame()

    rows = []
    for (column, gate_reason), g in touched.groupby(["column", "gate_reason"], dropna=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        rows.append({
            "column": column,
            "movies_column_type": infer_movies_column_type(column),
            "gate_reason": gate_reason,
            "count": len(g),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision_within_reason": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall_contribution": tp,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["FP", "FN", "count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_source_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    source_col = None
    for c in ["candidate_source", "candidate_source_text", "decision_source"]:
        if c in eval_df.columns:
            source_col = c
            break
    if source_col is None:
        return pd.DataFrame()

    touched = eval_df[eval_df["is_touched"] == 1].copy()
    if touched.empty:
        return pd.DataFrame()

    touched[source_col] = touched[source_col].fillna("").astype(str)
    rows = []
    for source, g in touched.groupby(source_col, dropna=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        rows.append({
            "candidate_source": source,
            "count": len(g),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision_within_source": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["FP", "FN", "count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_movies_type_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for type_name, g in eval_df.groupby("movies_column_type", sort=False):
        tp = int((g["eval_type"] == "TP").sum())
        fp = int((g["eval_type"] == "FP").sum())
        fn = int((g["eval_type"] == "FN").sum())
        tn = int((g["eval_type"] == "TN").sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({
            "movies_column_type": type_name,
            "total_cells": len(g),
            "true_error_cells": int(g["is_true_error"].sum()),
            "touched_cells": int(g["is_touched"].sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "format_only_differences_after_repair": int(g["strict_format_only_difference"].sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["true_error_cells", "F1"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_format_only_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    g = eval_df[(eval_df["repaired_value"] != eval_df["clean_value"]) & (eval_df["repaired_value_cmp"] == eval_df["clean_value_cmp"])].copy()
    if g.empty:
        return pd.DataFrame()

    rows = []
    for column, sub in g.groupby("column", sort=False):
        rows.append({
            "column": column,
            "movies_column_type": infer_movies_column_type(column),
            "format_only_count": len(sub),
            "examples": " || ".join(
                [
                    f"r{int(r.row_id)}: repaired={r.repaired_value} | clean={r.clean_value}"
                    for _, r in sub.head(5).iterrows()
                ]
            ),
        })
    return pd.DataFrame(rows).sort_values("format_only_count", ascending=False).reset_index(drop=True)


# ============================================================
# 6. main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading clean / dirty tables...")
    clean_df_raw = read_table(CLEAN_FILE)
    dirty_df_raw = read_table(DIRTY_FILE)
    validate_same_shape(clean_df_raw, dirty_df_raw)

    clean_df = stringify_table(clean_df_raw)
    dirty_df = stringify_table(dirty_df_raw)

    print("[INFO] Loading repair results...")
    repair_df = load_repair_results(REPAIR_RESULT_FILE)

    print("[INFO] Applying repairs...")
    repaired_df = apply_repairs_to_dirty(dirty_df, repair_df)
    repaired_df.to_csv(OUTPUT_REPAIRED_TABLE_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] Building full-table evaluation...")
    eval_df = build_full_table_eval(clean_df, dirty_df, repaired_df, repair_df)
    metrics = compute_metrics(eval_df)
    column_summary_df = build_column_summary(eval_df)
    gate_reason_summary_df = build_gate_reason_summary(eval_df)
    source_summary_df = build_source_summary(eval_df)
    movies_type_summary_df = build_movies_type_summary(eval_df)
    format_only_summary_df = build_format_only_summary(eval_df)

    eval_df.to_csv(OUTPUT_CELL_DETAILS_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TP"].to_csv(OUTPUT_TP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FP"].to_csv(OUTPUT_FP_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "FN"].to_csv(OUTPUT_FN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["eval_type"] == "TN"].to_csv(OUTPUT_TN_CSV, index=False, encoding="utf-8-sig")
    eval_df[eval_df["is_touched"] == 1].to_csv(OUTPUT_TOUCHED_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    column_summary_df.to_csv(OUTPUT_COLUMN_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    movies_type_summary_df.to_csv(OUTPUT_MOVIES_TYPE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    if not gate_reason_summary_df.empty:
        gate_reason_summary_df.to_csv(OUTPUT_GATE_REASON_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    if not source_summary_df.empty:
        source_summary_df.to_csv(OUTPUT_SOURCE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    if not format_only_summary_df.empty:
        format_only_summary_df.to_csv(OUTPUT_FORMAT_ONLY_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n========== Movies 全表端到端修复评估结果 ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n---- 各列评估摘要 Top ----")
    if not column_summary_df.empty:
        show_cols = [
            "column",
            "movies_column_type",
            "true_error_cells",
            "touched_cells",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
        ]
        print(column_summary_df[show_cols].head(30).to_string(index=False))

    print("\n---- Movies 字段类型评估摘要 ----")
    if not movies_type_summary_df.empty:
        show_cols = [
            "movies_column_type",
            "true_error_cells",
            "touched_cells",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
        ]
        print(movies_type_summary_df[show_cols].to_string(index=False))

    if not gate_reason_summary_df.empty:
        print("\n---- Gate Reason 摘要 Top ----")
        print(gate_reason_summary_df.head(30).to_string(index=False))

    if not source_summary_df.empty:
        print("\n---- Candidate Source 摘要 Top ----")
        print(source_summary_df.head(20).to_string(index=False))

    if not format_only_summary_df.empty:
        print("\n---- Format-only Difference 摘要 Top ----")
        print(format_only_summary_df.head(20).to_string(index=False))

    print(f"\n[OK] 评估摘要已保存: {OUTPUT_SUMMARY_CSV}")
    print(f"[OK] 列级评估摘要已保存: {OUTPUT_COLUMN_SUMMARY_CSV}")
    print(f"[OK] Movies 字段类型摘要已保存: {OUTPUT_MOVIES_TYPE_SUMMARY_CSV}")
    print(f"[OK] 修复后的整表已保存: {OUTPUT_REPAIRED_TABLE_CSV}")
    print(f"[OK] 全量单元格评估明细已保存: {OUTPUT_CELL_DETAILS_CSV}")
    print(f"[OK] touched 明细已保存: {OUTPUT_TOUCHED_CSV}")
    print(f"[OK] TP 明细已保存: {OUTPUT_TP_CSV}")
    print(f"[OK] FP 明细已保存: {OUTPUT_FP_CSV}")
    print(f"[OK] FN 明细已保存: {OUTPUT_FN_CSV}")
    print(f"[OK] TN 明细已保存: {OUTPUT_TN_CSV}")
    if not gate_reason_summary_df.empty:
        print(f"[OK] gate reason 摘要已保存: {OUTPUT_GATE_REASON_SUMMARY_CSV}")
    if not source_summary_df.empty:
        print(f"[OK] candidate source 摘要已保存: {OUTPUT_SOURCE_SUMMARY_CSV}")
    if not format_only_summary_df.empty:
        print(f"[OK] format-only difference 摘要已保存: {OUTPUT_FORMAT_ONLY_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
