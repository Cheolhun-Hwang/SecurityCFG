import os
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# 1) 라벨 로드: id -> class
def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if 'id' not in cols or 'class' not in cols:
        raise ValueError("CSV에 'Id'와 'Class' (대소문자 무관) 컬럼이 필요합니다.")
    id_col = cols['id']; cls_col = cols['class']
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[cls_col].astype(str)))

# 2) cfg/<id>.txt 에서 x-3그램 읽기 (파일 내 중복 제거)
def read_ngrams_file(txt_path: Path):
    grams = set()
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            g = line.strip()
            if g:
                grams.add(g)
    return grams

# 3) 클래스별 집계: x-3그램이 ‘등장한 파일 수’를 카운트
def aggregate_by_class(cfg_dir: str, id_to_class: dict):
    cfg_path = Path(cfg_dir)
    if not cfg_path.is_dir():
        raise FileNotFoundError(f"cfg 디렉토리 없음: {cfg_dir}")

    class_counters = defaultdict(Counter)   # class -> Counter(gram -> 등장 파일 수)
    class_file_counts = defaultdict(int)    # class -> 파일 수

    for txt in cfg_path.glob("*.txt"):
        file_id = txt.stem
        cls = id_to_class.get(file_id)
        if cls is None:
            # 라벨에 없는 파일은 스킵
            continue
        grams = read_ngrams_file(txt)  # 파일 내 중복 제거
        class_file_counts[cls] += 1
        if grams:
            class_counters[cls].update(grams)

    return class_counters, class_file_counts

# 4) 50% 이상 출현으로 1차 필터링
def filter_by_threshold(class_counters, class_file_counts, threshold_ratio: float = 0.5):
    """
    반환: dict[class] -> list[(gram, count)]  (임계 이상만)
    """
    filtered = {}
    for cls, counter in class_counters.items():
        total_files = class_file_counts.get(cls, 0)
        if total_files == 0:
            filtered[cls] = []
            continue
        threshold = math.ceil(total_files * threshold_ratio)
        kept = [(g, c) for g, c in counter.items() if c >= threshold]
        filtered[cls] = kept
    return filtered

# 5) 클래스 간 중복 제거 (여러 클래스에 동시에 등장하는 gram 제거)
def drop_shared_ngrams(filtered_counts_by_class):
    """
    입력: dict[class] -> list[(gram, count)]
    출력: dict[class] -> list[(gram, count)] (클래스 고유 gram만 남김)
    """
    gram_to_classes = defaultdict(set)
    for cls, items in filtered_counts_by_class.items():
        for g, _ in items:
            gram_to_classes[g].add(cls)

    unique_per_class = {}
    for cls, items in filtered_counts_by_class.items():
        unique_items = [(g, c) for (g, c) in items if len(gram_to_classes[g]) == 1]
        unique_per_class[cls] = unique_items
    return unique_per_class

# 6) 저장
def save_class_csvs(class_items, class_file_counts, out_dir: str):
    """
    class_items: dict[class] -> list[(gram, count)]
    CSV 컬럼: x_ngram,count
    count는 '등장 파일 수' 그대로 유지
    """
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    for cls, items in class_items.items():
        total_files = class_file_counts.get(cls, 0)
        safe_cls = str(cls).replace("/", "_").replace("\\", "_")
        out_path = out_base / f"{safe_cls}.csv"

        # 정렬: count desc, gram asc
        items = sorted(items, key=lambda x: (-x[1], x[0]))

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_ngram", "count"])
            writer.writerows(items)

        print(f"✅ saved class '{cls}' → {out_path} (files={total_files}, rows={len(items)})")

# 7) 전체 파이프라인
def build_class_cfg(
    labels_csv: str = r"D:\malware-classification\trainLabels.csv",
    cfg_dir: str = "result/train/cfg",         # x-3그램 파일들이 위치한 폴더
    out_dir: str = "result/train/cfg_class",   # 클래스별 집계 CSV 내보낼 폴더
    threshold_ratio: float = 0.5,
    drop_shared: bool = True
):
    # 라벨 로드
    id_to_class = load_labels(labels_csv)
    # 클래스별 집계
    class_counters, class_file_counts = aggregate_by_class(cfg_dir, id_to_class)
    # 50% 이상 출현 필터
    filtered = filter_by_threshold(class_counters, class_file_counts, threshold_ratio)
    # 클래스 간 중복 제거
    if drop_shared:
        filtered = drop_shared_ngrams(filtered)
    # 저장
    save_class_csvs(filtered, class_file_counts, out_dir)

if __name__ == "__main__":
    # 예시 실행
    build_class_cfg(
        labels_csv=r"D:\malware-classification\trainLabels.csv",
        cfg_dir="result/train/cfg",
        out_dir="result/train/cfg_class",
        threshold_ratio=0.1,
        drop_shared=True
    )
    pass
