import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import math
import hashlib

# =============================
# 경로 및 하이퍼파라미터 설정
# =============================
# jaccard cosine intersection pearson nhk veo
type = "veo"
CLASS_DIR = "result/train/cfg_class"
FILE_DIR = "result/train/cfg"
TRAIN_LABELS = "data/trainLabels.csv"
OUTPUT_CSV = f"result/cfg_graph_{type}_results.csv"

# 클래스 개수 (1 ~ N 까지의 csv 존재한다고 가정)
USE_CLASSES = [1, 2, 3, 4, 6, 7, 8, 9]
HASH_DIM = 1024

# =============================
# 유틸 함수
# =============================

def jaccard_similarity(set_a, set_b):
    """
    두 노드 집합에 대한 Jaccard Similarity 계산
    """
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union

def cosine_similarity(counter_a, counter_b):
    """
    두 3-gram 카운트 딕셔너리에 대한 Cosine Similarity 계산
    counter_a, counter_b: dict[str, int or float]
    """
    if not counter_a or not counter_b:
        return 0.0

    # dot product
    common_keys = set(counter_a.keys()) & set(counter_b.keys())
    dot = sum(counter_a[k] * counter_b[k] for k in common_keys)

    # norms
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def intersection_similarity(counter_a, counter_b):
    """
    두 3-gram 카운트 딕셔너리에 대한 Intersection 기반 Similarity 계산
    여기서는:
      - counter_a: 파일(샘플)의 3-gram -> count
      - counter_b: 클래스의 3-gram -> count
    정의: |A ∩ B| / |A|   (A = 파일 3-gram 집합)
    """
    if not counter_a:
        return 0.0

    set_a = set(counter_a.keys())
    set_b = set(counter_b.keys())

    inter = len(set_a & set_b)
    return inter / len(set_a)

def pearson_similarity(counter_a, counter_b):
    """
    두 3-gram 카운트 딕셔너리에 대한 Pearson 상관계수 기반 유사도.
    counter_a, counter_b: dict[str, int or float]
    반환값: -1 ~ 1 범위의 Pearson r
    """
    if not counter_a or not counter_b:
        return 0.0

    # 공통 key 기준이 아니라 전체 union 기준으로 벡터 맞춤
    keys = set(counter_a.keys()) | set(counter_b.keys())
    if not keys:
        return 0.0

    a_vals = [counter_a.get(k, 0.0) for k in keys]
    b_vals = [counter_b.get(k, 0.0) for k in keys]

    n = len(keys)
    mean_a = sum(a_vals) / n
    mean_b = sum(b_vals) / n

    num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a_vals, b_vals))
    den_a = sum((ai - mean_a) ** 2 for ai in a_vals)
    den_b = sum((bi - mean_b) ** 2 for bi in b_vals)

    if den_a == 0 or den_b == 0:
        return 0.0

    return num / math.sqrt(den_a * den_b)

def nhk_vector(counter, dim=HASH_DIM):
    """
    Neighborhood Hash Kernel용 해시 벡터 생성
    counter: dict[str, float]  (3-gram -> count)
    반환: numpy.ndarray (shape: [dim])
    """
    vec = np.zeros(dim, dtype=float)
    for k, c in counter.items():
        # 해시 → 버킷 인덱스
        h = int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += c
    return vec


def nhk_similarity(counter_a, counter_b, dim=HASH_DIM):
    """
    Neighborhood Hash Kernel 기반 유사도 (코사인 유사도 사용)
    counter_a, counter_b: dict[str, float]
    """
    if not counter_a or not counter_b:
        return 0.0

    va = nhk_vector(counter_a, dim)
    vb = nhk_vector(counter_b, dim)

    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(va, vb) / (na * nb))

def get_vertices_edges_from_counter(counter):
    """
    3-gram 카운트 딕셔너리에서
    - vertex 집합 V
    - edge 집합 E
    를 추출한다.

    counter: dict[str, float]  (예: "x22_x6_x7" -> count)
    """
    vertices = set()
    edges = set()

    for gram in counter.keys():
        # "x22_x6_x7" -> ["x22", "x6", "x7"]
        parts = gram.split("_")
        if not parts:
            continue

        # vertex: 각 토큰
        for p in parts:
            if p:
                vertices.add(p)

        # edge: 인접 토큰 쌍
        for i in range(len(parts) - 1):
            u = parts[i]
            v = parts[i + 1]
            if u and v:
                edges.add((u, v))

    return vertices, edges


def vertex_edge_overlap_similarity(counter_a, counter_b):
    """
    Vertex Edge Overlap 기반 유사도
    - counter_a, counter_b: dict[str, float] (3-gram -> count)
    - 반환값: 0 ~ 1

    정의:
      VEO(A,B) = (|V_A ∩ V_B| + |E_A ∩ E_B|) / (|V_A ∪ V_B| + |E_A ∪ E_B|)
    """
    if not counter_a or not counter_b:
        return 0.0

    Va, Ea = get_vertices_edges_from_counter(counter_a)
    Vb, Eb = get_vertices_edges_from_counter(counter_b)

    inter_v = len(Va & Vb)
    inter_e = len(Ea & Eb)
    union_v = len(Va | Vb)
    union_e = len(Ea | Eb)

    denom = union_v + union_e
    if denom == 0:
        return 0.0

    return (inter_v + inter_e) / denom

def load_class_node_sets_jaccard(class_dir, n_classes):
    """
    result/cfg_class 내 클래스별 CSV를 읽어서
    각 클래스에 대한 노드 집합(3-gram set)을 만든다.

    CSV 형식:
    x_ngram,count
    x22_x6_x7,1346
    ...
    """
    class_nodes = {}

    for c in USE_CLASSES:
        csv_path = os.path.join(class_dir, f"{c}.csv")
        if not os.path.exists(csv_path):
            print(f"[경고] {csv_path} 가 존재하지 않습니다. 건너뜀.")
            class_nodes[c] = set()
            continue

        df = pd.read_csv(csv_path)
        if "x_ngram" not in df.columns:
            raise ValueError(f"{csv_path} 에 x_ngram 컬럼이 없습니다.")

        # 노드 집합 = 등장한 모든 3-gram
        node_set = set(df["x_ngram"].astype(str).tolist())
        class_nodes[c] = node_set
        print(f"[INFO] 클래스 {c}: 노드 수 = {len(node_set)}")

    return class_nodes

def load_class_node_sets_cossine(class_dir, n_classes):
    """
    result/cfg_class 내 클래스별 CSV를 읽어서
    각 클래스에 대한 3-gram -> count 벡터(dict)를 만든다.

    CSV 형식:
    x_ngram,count
    x22_x6_x7,1346
    ...
    """
    class_nodes = {}

    for c in USE_CLASSES:
        csv_path = os.path.join(class_dir, f"{c}.csv")
        if not os.path.exists(csv_path):
            print(f"[경고] {csv_path} 가 존재하지 않습니다. 건너뜀.")
            class_nodes[c] = {}
            continue

        df = pd.read_csv(csv_path)
        if "x_ngram" not in df.columns:
            raise ValueError(f"{csv_path} 에 x_ngram 컬럼이 없습니다.")

        # count 컬럼이 없는 경우 대비
        if "count" in df.columns:
            counts = df["count"]
        else:
            # 없으면 1로 채움
            counts = pd.Series([1] * len(df))

        grams = df["x_ngram"].astype(str).tolist()
        cnts = counts.astype(float).tolist()

        # dict: x_ngram -> count
        node_vec = {g: c for g, c in zip(grams, cnts)}
        class_nodes[c] = node_vec
        print(f"[INFO] 클래스 {c}: 3-gram 수 = {len(node_vec)}")

    return class_nodes

def load_file_node_set_jaccard(txt_path):
    """
    result/cfg 내 txt 파일 하나를 읽어서
    해당 파일의 노드 집합(3-gram set)을 만든다.

    txt 한 줄 형식 예:
    x22_x6_x7,1346
    또는 x22_x6_x7  (count 없이 오는 경우도 고려)
    """
    node_set = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 콤마 기준으로 x_ngram, count 분리 (count는 안 써도 됨)
            parts = line.split(",")
            x_ngram = parts[0].strip()
            if x_ngram:
                node_set.add(x_ngram)
    return node_set

def load_file_node_set_cosine(txt_path):
    """
    result/cfg 내 txt 파일 하나를 읽어서
    해당 파일의 3-gram -> count 벡터(dict)를 만든다.

    txt 한 줄 형식 예:
    x22_x6_x7,1346
    또는 x22_x6_x7  (count 없이 오는 경우 count=1로 처리)
    """
    counter = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            x_ngram = parts[0].strip()
            if not x_ngram:
                continue

            if len(parts) >= 2:
                try:
                    cnt = float(parts[1].strip())
                except ValueError:
                    cnt = 1.0
            else:
                cnt = 1.0

            # 같은 3-gram이 여러 번 나오면 누적
            counter[x_ngram] = counter.get(x_ngram, 0.0) + cnt

    return counter

def load_train_labels(label_csv_path):
    """
    data/trainLabels.csv 에서 id, class 정보를 읽어
    id -> class 매핑 딕셔너리를 만든다.

    trainLabels.csv 형식 예:
    id,class
    1,1
    2,3
    ...
    """
    df = pd.read_csv(label_csv_path)
    # 컬럼 이름이 다르다면 여기서 수정
    if "Id" not in df.columns or "Class" not in df.columns:
        raise ValueError("trainLabels.csv 에 'id' 또는 'class' 컬럼이 없습니다.")

    # id를 문자열로 맞춰두는 게 안전 (파일명과 비교)
    df["Id"] = df["Id"].astype(str)
    id_to_class = dict(zip(df["Id"], df["Class"]))
    print(f"[INFO] trainLabels 라벨 수 = {len(id_to_class)}")
    return id_to_class


def compute_metrics(result_df, n_classes):
    """
    result_df (filename, class1~classN, pred_class, true_class)를 받아
    Accuracy, Log Loss, F1-score 등을 계산 후 출력
    """
    df = result_df.copy()

    # 1) true_class / pred_class 없는 행 제거
    df = df[df["true_class"].notna()]
    df = df[df["pred_class"].notna()]

    # 2) 클래스 5 (제외 클래스) 제거
    df = df[df["true_class"].isin(USE_CLASSES)]
    # (pred_class는 USE_CLASSES로만 나오므로 따로 필터할 필요는 거의 없음)

    if df.empty:
        print("[WARN] 메트릭을 계산할 샘플이 없습니다. (true_class ∈ USE_CLASSES)")
        return

    # y_true, y_pred
    y_true = df["true_class"].astype(int).tolist()
    y_pred = df["pred_class"].astype(int).tolist()

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # === Log loss 계산 ===
    #  - class{c} 점수를 softmax로 확률로 변환
    class_cols = [f"class{c}" for c in USE_CLASSES]
    scores = df[class_cols].values.astype(float)  # shape: (num_samples, n_classes)

    probs_list = []
    for row in scores:
        if np.all(row == 0):
            # 모든 점수가 0이면 균등 분포로 처리
            probs = np.ones(n_classes) / n_classes
        else:
            # softmax
            row_shift = row - np.max(row)  # 수치 안정성
            exps = np.exp(row_shift)
            probs = exps / exps.sum()
        probs_list.append(probs)

    probs = np.vstack(probs_list)  # shape: (num_samples, n_classes)

    # log_loss는 y_true 라벨과 확률행렬을 입력받음
    labels = USE_CLASSES[:]  # [1,2,3,4,6,7,8,9]
    ll = log_loss(y_true, probs, labels=labels)

    print("\n===== Metrics (labelled subset only, class 5 제외) =====")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Log Loss       : {ll:.4f}")
    print(f"Macro F1-score : {macro_f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


# =============================
# 메인 파이프라인
# =============================

def main():
    # 1) 클래스별 그래프(노드 집합) 로드
    if type == "jaccard":
        class_nodes_dict = load_class_node_sets_jaccard(CLASS_DIR, USE_CLASSES)
    else :
        class_nodes_dict = load_class_node_sets_cossine(CLASS_DIR, USE_CLASSES)

    # 2) 정답 라벨 로드
    id_to_class = load_train_labels(TRAIN_LABELS)

    # 3) 파일별 그래프(노드 집합) 생성 & 클래스와 Jaccard 계산
    records = []

    txt_files = sorted(glob.glob(os.path.join(FILE_DIR, "*.txt")))
    print(f"[INFO] 분석 대상 txt 파일 수 = {len(txt_files)}")

    for idx, txt_path in enumerate(txt_files, 1):
        filename = os.path.basename(txt_path)
        file_id = os.path.splitext(filename)[0]  # "1234.txt" -> "1234"

        # 3) 파일의 3-gram 표현 로드 (type별로 다르게)
        if type == "jaccard":
            file_nodes = load_file_node_set_jaccard(txt_path)   # set
        else:  # cosine, intersection
            file_nodes = load_file_node_set_cosine(txt_path)    # dict

        # 각 클래스와의 Jaccard 유사도 계산
        sims = {}
        for c in USE_CLASSES:
            if type == "jaccard":
                class_nodes = class_nodes_dict.get(c, {})
                sim = jaccard_similarity(class_nodes, file_nodes)
            elif type == "cosine":
                class_vec = class_nodes_dict.get(c, {})
                sim = cosine_similarity(class_vec, file_nodes)
            elif type == "intersection":
                class_vec = class_nodes_dict.get(c, {})
                sim = intersection_similarity(file_nodes, class_vec)
            elif type == "pearson":
                class_vec = class_nodes_dict.get(c, {})
                sim = pearson_similarity(class_vec, file_nodes)
            elif type == "nhk":
                class_vec = class_nodes_dict.get(c, {})
                sim = nhk_similarity(class_vec, file_nodes)
            elif type == "veo":
                class_vec = class_nodes_dict.get(c, {})
                sim = vertex_edge_overlap_similarity(class_vec, file_nodes)
            sims[c] = sim

        # 가장 유사한 클래스
        pred_class = max(sims, key=sims.get) if sims else None

        # 정답 라벨 (없으면 None)
        true_class = id_to_class.get(file_id, None)

        # 결과 레코드 구성
        row = {
            "filename": filename,
        }
        for c in USE_CLASSES:
            row[f"class{c}"] = sims.get(c, 0.0)

        row["pred_class"] = pred_class
        row["true_class"] = true_class

        records.append(row)

        if idx % 100 == 0 or idx == len(txt_files):
            print(f"[INFO] 진행 상황: {idx}/{len(txt_files)} 파일 처리 완료")

    # 4) DataFrame으로 정리
    result_df = pd.DataFrame(records)

    # 컬럼 순서 정리
    cols = ["filename"] + [f"class{c}" for c in USE_CLASSES] + [
        "pred_class",
        "true_class",
    ]
    result_df = result_df[cols]

    # 5) 메트릭 계산 (정답 라벨 있는 샘플만 대상으로)
    compute_metrics(result_df,  len(USE_CLASSES))

    # 6) CSV 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] 결과 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
