import os
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Iterable, Optional
import pandas as pd

# =========================
# 공통 유틸
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_")

def save_class_csv(items: List[Tuple[str, int]], out_path: Path):
    ensure_dir(out_path.parent)
    rows = [(str(g), int(c)) for g, c in items]
    rows.sort(key=lambda t: (-t[1], t[0]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_ngram", "count"])
        w.writerows(rows)

def save_debug_csv(items: Iterable, out_path: Path, header: Iterable[str]):
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for row in items:
            if isinstance(row, (list, tuple)):
                w.writerow(list(row))
            else:
                w.writerow([row])

# =========================
# 1) 라벨 로드: id -> class
# =========================
def load_labels(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if 'id' not in cols or 'class' not in cols:
        raise ValueError("trainLabels.csv에는 'Id'와 'Class' 컬럼이 필요합니다. (대소문자 무관)")
    id_col = cols['id']; cls_col = cols['class']
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[cls_col].astype(str)))

# =========================
# 2) cfg/<id>.txt 읽기
#  - 파일 내 중복 제거(set)로 '등장 여부'만 카운트
# =========================
def read_ngrams_file(txt_path: Path) -> Set[str]:
    grams = set()
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            g = line.strip()
            if g:
                grams.add(g)
    return grams

# =========================
# 3) 클래스별 집계
#    (x-3그램이 ‘등장한 파일 수’를 카운트)
# =========================
def aggregate_by_class(cfg_dir: str, id_to_class: Dict[str, str]):
    cfg_path = Path(cfg_dir)
    if not cfg_path.is_dir():
        raise FileNotFoundError(f"cfg 디렉토리 없음: {cfg_dir}")

    class_counters = defaultdict(Counter)   # class -> Counter(gram -> 등장 '파일 수')
    class_file_counts = defaultdict(int)    # class -> 파일 수
    class_nonempty_counts = defaultdict(int)# class -> 비어있지 않은 파일 수
    class_file_ids = defaultdict(list)      # class -> [id, id, ...]
    ids_without_label = []                  # cfg에 있는데 라벨에 없는 id

    for txt in cfg_path.glob("*.txt"):
        file_id = txt.stem
        cls = id_to_class.get(file_id)
        if cls is None:
            ids_without_label.append(file_id)
            continue

        grams = read_ngrams_file(txt)
        class_file_counts[cls] += 1
        class_file_ids[cls].append(file_id)

        if grams:
            class_nonempty_counts[cls] += 1
            class_counters[cls].update(grams)

    return (class_counters, class_file_counts, class_nonempty_counts,
            class_file_ids, ids_without_label)

# =========================
# 4) 50% 이상 출현 필터
# =========================
def filter_by_threshold(class_counters, class_file_counts, threshold_ratio: float = 0.5):
    filtered = {}
    thresholds = {}
    for cls, counter in class_counters.items():
        total = class_file_counts.get(cls, 0)
        thr = math.ceil(total * threshold_ratio) if total > 0 else 0
        thresholds[cls] = thr

        kept = []
        dropped = []
        for g, c in counter.items():
            if thr > 0 and c >= thr:
                kept.append((g, c))
            else:
                dropped.append((g, c))
        filtered[cls] = {'kept': kept, 'dropped_below_thr': dropped}
    return filtered, thresholds

# =========================
# 5-A) 클래스 간 중복 제거 (완전 고유만 유지)
# =========================
def drop_shared_ngrams(filtered_counts_by_class):
    gram_to_classes = defaultdict(set)
    for cls, parts in filtered_counts_by_class.items():
        for g, _ in parts['kept']:
            gram_to_classes[g].add(cls)

    unique_per_class = {}
    dropped_shared = {}
    for cls, parts in filtered_counts_by_class.items():
        kept = []
        dropped = []
        for g, c in parts['kept']:
            if len(gram_to_classes[g]) == 1:
                kept.append((g, c))
            else:
                dropped.append((g, c))
        unique_per_class[cls] = kept
        dropped_shared[cls] = dropped
    return unique_per_class, dropped_shared

# =========================
# 5-B) 지원율 비율로 재배정 (공유 n-gram을 단일 클래스로 배정)
#   - 후보: 임계 통과(kept)에 오른 클래스들
#   - best_sup / second_sup >= min_ratio AND (best_sup - second_sup) >= min_delta 일 때만 배정
#   - 단일 후보면 바로 배정
# =========================
def allocate_shared_by_ratio(class_counters, class_file_counts, filtered_counts_by_class,
                             min_ratio: float = 1.25, min_delta: float = 0.05):
    # 각 클래스의 kept 목록 set
    kept_by_class = {cls: {g for g, _ in parts['kept']}
                     for cls, parts in filtered_counts_by_class.items()}
    # 후보 gram 집합
    all_kept = set().union(*kept_by_class.values()) if kept_by_class else set()

    assigned = {cls: [] for cls in filtered_counts_by_class.keys()}       # class -> list[(g, cnt)]
    ambiguous_global = []  # [(g, best_cls, best_sup, second_cls, second_sup)]
    winner_of_gram = {}    # g -> winner class or None

    for g in sorted(all_kept):
        # 후보 클래스: 해당 gram이 kept에 포함된 클래스
        candidates = [cls for cls, s in kept_by_class.items() if g in s]
        if not candidates:
            continue

        # 지원율 계산: sup = (등장 파일 수 / total_files)
        supports = []
        for cls in candidates:
            total = class_file_counts.get(cls, 0)
            cnt = class_counters[cls].get(g, 0)
            sup = (cnt / total) if total > 0 else 0.0
            supports.append((cls, sup, cnt))
        supports.sort(key=lambda t: (-t[1], t[0]))

        best_cls, best_sup, best_cnt = supports[0][0], supports[0][1], supports[0][2]
        second_sup = supports[1][1] if len(supports) >= 2 else 0.0
        second_cls = supports[1][0] if len(supports) >= 2 else None

        # 단일 후보 → 바로 배정
        if len(candidates) == 1:
            assigned[best_cls].append((g, best_cnt))
            winner_of_gram[g] = best_cls
            continue

        # 비율/차이 기준으로 배정
        if (second_sup == 0 and best_sup > 0) or \
           (second_sup > 0 and best_sup / second_sup >= min_ratio and (best_sup - second_sup) >= min_delta):
            assigned[best_cls].append((g, best_cnt))
            winner_of_gram[g] = best_cls
        else:
            ambiguous_global.append((g, best_cls, best_sup, second_cls, second_sup))
            winner_of_gram[g] = None

    # 디버그용: 각 클래스 관점에서 “남의 승리로 잃은 항목”/“애매해서 보류된 항목”
    lost_to_other = {cls: [] for cls in filtered_counts_by_class.keys()}
    ambiguous_per_class = {cls: [] for cls in filtered_counts_by_class.keys()}

    for g in all_kept:
        winner = winner_of_gram.get(g)
        for cls in kept_by_class.keys():
            if g not in kept_by_class[cls]:
                continue
            cnt = class_counters[cls].get(g, 0)
            if winner is None:
                # 애매해서 배정 보류
                ambiguous_per_class[cls].append((g, cnt))
            elif winner != cls:
                # 다른 클래스에 배정됨
                lost_to_other[cls].append((g, cnt))

    return assigned, lost_to_other, ambiguous_per_class, ambiguous_global

# =========================
# 6) 클래스별 지원율 테이블 (진단)
# =========================
def analyze_shared_for_class(class_counters, class_file_counts, filtered_by_thr,
                             target_cls: str, out_csv: Path):
    """target_cls의 임계 통과 n-gram에 대해 모든 클래스 지원율(support)을 표로 저장"""
    ensure_dir(out_csv.parent)
    target_kept = {g for g, _ in filtered_by_thr.get(target_cls, {}).get('kept', [])}
    classes = sorted(class_counters.keys())

    rows = []
    header = ["x_ngram"] + [f"{cls}_support" for cls in classes]
    for g in sorted(target_kept):
        row = [g]
        for cls in classes:
            total = class_file_counts.get(cls, 0)
            cnt = class_counters[cls].get(g, 0)
            sup = (cnt / total) if total > 0 else 0.0
            row.append(f"{sup:.6f}")
        rows.append(row)
    save_debug_csv(rows, out_csv, header=header)

# =========================
# 7) 요약 저장
# =========================
def save_debug_summary(cls: str,
                       out_dir: Path,
                       class_file_counts: Dict[str, int],
                       class_nonempty_counts: Dict[str, int],
                       thresholds: Dict[str, int],
                       kept_after_thr: List[Tuple[str,int]],
                       dropped_below_thr: List[Tuple[str,int]],
                       final_items: List[Tuple[str,int]],
                       dropped_shared_or_lost: List[Tuple[str,int]],
                       ambiguous_for_cls: Optional[List[Tuple[str,int]]] = None,
                       class_file_ids: Optional[Dict[str, List[str]]] = None,
                       reason_override: Optional[str] = None):
    ensure_dir(out_dir)
    p = out_dir / f"{safe_name(cls)}__summary.txt"

    total_files = class_file_counts.get(cls, 0)
    nonempty = class_nonempty_counts.get(cls, 0)
    thr = thresholds.get(cls, 0)

    # 자동 reason
    reason = []
    if reason_override:
        reason.append(reason_override)
    else:
        if total_files == 0:
            reason.append("NO_FILES_FOR_CLASS")
        elif nonempty == 0:
            reason.append("ALL_FILES_EMPTY")
        elif len(kept_after_thr) == 0:
            reason.append("ALL_DROPPED_AT_THRESHOLD")
        elif len(final_items) == 0:
            reason.append("NO_ITEMS_AFTER_SHARED_HANDLING")
        else:
            reason.append("HAS_FEATURES")

    with open(p, "w", encoding="utf-8") as f:
        f.write(f"class: {cls}\n")
        f.write(f"files_total: {total_files}\n")
        f.write(f"files_nonempty: {nonempty}\n")
        f.write(f"threshold(ceil(files*ratio)): {thr}\n")
        f.write(f"kept_after_threshold: {len(kept_after_thr)}\n")
        f.write(f"dropped_below_threshold: {len(dropped_below_thr)}\n")
        f.write(f"final_items: {len(final_items)}\n")
        f.write(f"dropped_shared_or_lost: {len(dropped_shared_or_lost)}\n")
        if ambiguous_for_cls is not None:
            f.write(f"ambiguous_shared_for_cls: {len(ambiguous_for_cls)}\n")
        if class_file_ids:
            f.write(f"file_ids: {', '.join(class_file_ids.get(cls, []))}\n")
        f.write(f"reason: {', '.join(reason)}\n")

# =========================
# 8) 메인 파이프라인 (+ 진단/전략 선택)
# =========================
def build_class_cfg_with_diagnostics(
    labels_csv: str,
    cfg_dir: str = "result/cft",              # x-3그램 파일들
    out_dir: str = "result/cfg_class",        # 최종 클래스별 CSV
    debug_dir: str = "result/cfg_debug",      # 디버그 산출물
    threshold_ratio: float = 0.5,
    shared_strategy: str = "drop",            # "drop" | "allocate_ratio" | "keep_all"
    min_ratio: float = 1.25,                  # allocate_ratio용
    min_delta: float = 0.05,                  # allocate_ratio용
    supports_for_classes: Optional[List[str]] = None  # 예: ["7"] or ["3","6","7"] or None
):
    """
    shared_strategy:
      - "drop"           : 공유 n-gram은 완전히 제거(고유만 유지)
      - "allocate_ratio" : 지원율 비율/차이 기준으로 단일 클래스로 배정
      - "keep_all"       : 공유도 유지(임계 통과만 하면 모두 유지)
    """
    out_base = Path(out_dir)
    dbg_base = Path(debug_dir)
    ensure_dir(out_base); ensure_dir(dbg_base)

    # 1) 라벨 로드
    id_to_class = load_labels(labels_csv)

    # 2) 집계
    (class_counters,
     class_file_counts,
     class_nonempty_counts,
     class_file_ids,
     ids_without_label) = aggregate_by_class(cfg_dir, id_to_class)

    if ids_without_label:
        save_debug_csv(sorted(ids_without_label),
                       dbg_base / "__ids_without_label.txt",
                       header=["ids_without_label"])

    # 3) 임계 필터
    filtered_by_thr, thresholds = filter_by_threshold(class_counters, class_file_counts, threshold_ratio)

    # 4) 공유 처리 전략
    if shared_strategy == "drop":
        final_per_class, dropped_shared = drop_shared_ngrams(filtered_by_thr)
        # 디버그 저장
        for cls, parts in filtered_by_thr.items():
            safe_cls = safe_name(cls)
            save_debug_csv(list(class_counters.get(cls, Counter()).items()),
                           dbg_base / f"{safe_cls}__raw_counts.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['kept'],
                           dbg_base / f"{safe_cls}__kept_threshold.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['dropped_below_thr'],
                           dbg_base / f"{safe_cls}__dropped_below_threshold.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(dropped_shared.get(cls, []),
                           dbg_base / f"{safe_cls}__dropped_shared.csv",
                           header=["x_ngram","file_count"])

            # 요약
            save_debug_summary(
                cls=cls,
                out_dir=dbg_base,
                class_file_counts=class_file_counts,
                class_nonempty_counts=class_nonempty_counts,
                thresholds=thresholds,
                kept_after_thr=parts['kept'],
                dropped_below_thr=parts['dropped_below_thr'],
                final_items=final_per_class.get(cls, []),
                dropped_shared_or_lost=dropped_shared.get(cls, []),
                ambiguous_for_cls=None,
                class_file_ids=class_file_ids,
                reason_override=None
            )

    elif shared_strategy == "allocate_ratio":
        assigned, lost_to_other, ambiguous_per_class, ambiguous_global = allocate_shared_by_ratio(class_counters, class_file_counts, filtered_by_thr,
                                     min_ratio=min_ratio, min_delta=min_delta)

        final_per_class = assigned
        # 디버그 저장
        save_debug_csv(ambiguous_global,
                       dbg_base / "__ambiguous_global.csv",
                       header=["x_ngram","best_cls","best_sup","second_cls","second_sup"])

        for cls, parts in filtered_by_thr.items():
            safe_cls = safe_name(cls)
            save_debug_csv(list(class_counters.get(cls, Counter()).items()),
                           dbg_base / f"{safe_cls}__raw_counts.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['kept'],
                           dbg_base / f"{safe_cls}__kept_threshold.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['dropped_below_thr'],
                           dbg_base / f"{safe_cls}__dropped_below_threshold.csv",
                           header=["x_ngram","file_count"])

            save_debug_csv(final_per_class.get(cls, []),
                           dbg_base / f"{safe_cls}__allocated.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(lost_to_other.get(cls, []),
                           dbg_base / f"{safe_cls}__lost_to_other.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(ambiguous_per_class.get(cls, []),
                           dbg_base / f"{safe_cls}__ambiguous_shared.csv",
                           header=["x_ngram","file_count"])

            # reason 판단
            reason_override = None
            if len(parts['kept']) > 0 and len(final_per_class.get(cls, [])) == 0:
                # kept는 있었으나 최종 배정 0인 경우
                if len(ambiguous_per_class.get(cls, [])) > 0 and len(lost_to_other.get(cls, [])) == 0:
                    reason_override = "ALL_AMBIGUOUS_SHARED"
                elif len(lost_to_other.get(cls, [])) > 0:
                    reason_override = "ALL_ALLOCATED_TO_OTHERS"

            save_debug_summary(
                cls=cls,
                out_dir=dbg_base,
                class_file_counts=class_file_counts,
                class_nonempty_counts=class_nonempty_counts,
                thresholds=thresholds,
                kept_after_thr=parts['kept'],
                dropped_below_thr=parts['dropped_below_thr'],
                final_items=final_per_class.get(cls, []),
                dropped_shared_or_lost=lost_to_other.get(cls, []),
                ambiguous_for_cls=ambiguous_per_class.get(cls, []),
                class_file_ids=class_file_ids,
                reason_override=reason_override
            )

    elif shared_strategy == "keep_all":
        # 임계 통과한 것 전부 유지 (공유도 허용)
        final_per_class = {cls: parts['kept'] for cls, parts in filtered_by_thr.items()}
        for cls, parts in filtered_by_thr.items():
            safe_cls = safe_name(cls)
            save_debug_csv(list(class_counters.get(cls, Counter()).items()),
                           dbg_base / f"{safe_cls}__raw_counts.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['kept'],
                           dbg_base / f"{safe_cls}__kept_threshold.csv",
                           header=["x_ngram","file_count"])
            save_debug_csv(parts['dropped_below_thr'],
                           dbg_base / f"{safe_cls}__dropped_below_threshold.csv",
                           header=["x_ngram","file_count"])
            save_debug_summary(
                cls=cls,
                out_dir=dbg_base,
                class_file_counts=class_file_counts,
                class_nonempty_counts=class_nonempty_counts,
                thresholds=thresholds,
                kept_after_thr=parts['kept'],
                dropped_below_thr=parts['dropped_below_thr'],
                final_items=final_per_class.get(cls, []),
                dropped_shared_or_lost=[],
                ambiguous_for_cls=None,
                class_file_ids=class_file_ids,
                reason_override=None
            )
    else:
        raise ValueError("shared_strategy는 'drop' | 'allocate_ratio' | 'keep_all' 중 하나여야 합니다.")

    # 5) 최종 CSV 저장
    for cls, items in final_per_class.items():
        save_class_csv(items, out_base / f"{safe_name(cls)}.csv")

    # 6) (선택) 특정 클래스 지원율 테이블 저장
    if supports_for_classes:
        for c in supports_for_classes:
            analyze_shared_for_class(
                class_counters, class_file_counts, filtered_by_thr,
                target_cls=c,
                out_csv=dbg_base / f"{safe_name(c)}__shared_supports.csv"
            )

    print(f"✅ 최종 CSV → {out_base}")
    print(f"🧪 디버그 산출물 → {dbg_base}")

# =========================
# 사용 예시
# =========================
if __name__ == "__main__":
    # 예) Kelihos_ver1(클래스 '7')의 빈 이유를 줄이고 싶다면,
    #    공유 제거 대신 'allocate_ratio'로 재배정 + 지원율 테이블도 생성
    build_class_cfg_with_diagnostics(
        labels_csv=r"D:\malware-classification\trainLabels.csv",
        cfg_dir="result/train/cfg",
        out_dir="result/train/cfg_class",
        debug_dir="result/train/cfg_debug",
        threshold_ratio=0.25,
        shared_strategy="allocate_ratio",   # <- 핵심
        min_ratio=1.25,                     # 1.25~1.5 권장
        min_delta=0.05,
        supports_for_classes=["3", "6", "7"]          # 7번 클래스 지원율 표 덤프
    )
    pass
