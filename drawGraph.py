import os
import csv
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------
# 1) CSV 로드: x_ngram,count
# ---------------------------
def load_class_csv(csv_path: str) -> List[Tuple[List[str], int]]:
    """
    returns: [(tokens, count), ...]
      - tokens: ["x1","x2","x3"]  (구분자 '_', '-' 자동 대응)
      - count : 등장 파일 수 (없으면 1로 처리)
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if 'x_ngram' not in cols:
        raise ValueError("CSV에 'x_ngram' 컬럼이 필요합니다.")

    xcol = cols['x_ngram']
    ccol = cols.get('count')  # count가 없을 수도 있음

    rows = []
    for _, r in df.iterrows():
        gram = str(r[xcol]).strip()
        if not gram:
            continue
        # '_' 또는 '-' 구분자 모두 허용
        if '_' in gram:
            tokens = gram.split('_')
        else:
            tokens = gram.split('-')
        tokens = [t.strip() for t in tokens if t.strip()]
        if len(tokens) < 2:
            # 최소 2개 연결이 있어야 간선이 생김
            continue
        cnt = int(r[ccol]) if ccol else 1
        rows.append((tokens, cnt))
    return rows

# ---------------------------
# 2) 그래프 생성 (방향 그래프)
# ---------------------------
def build_graph_from_ngrams(ngrams_with_count: List[Tuple[List[str], int]]) -> nx.DiGraph:
    """
    - 노드: x 토큰
    - 간선: 각 n-gram에서 연속 쌍 (x[i] -> x[i+1])
    - 동일 간선은 count 합산(가중치 weight)
    """
    G = nx.DiGraph()
    for tokens, cnt in ngrams_with_count:
        # 노드 추가
        for t in tokens:
            G.add_node(t)
        # 연속 간선 추가 (가중치 누적)
        for i in range(len(tokens) - 1):
            u, v = tokens[i], tokens[i+1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += cnt
            else:
                G.add_edge(u, v, weight=cnt)
    return G

# ---------------------------
# 3) 그리기 & 저장
# ---------------------------
def draw_and_save_graph(G: nx.DiGraph, out_png: str, title: str = ""):
    """
    - 노드: 파란색, 라벨 표시
    - 레이아웃: spring_layout (시드 고정)
    - 이미지 저장
    """
    # 노드/엣지 수에 따라 그림 크기 가변
    n_nodes = max(len(G.nodes), 1)
    n_edges = max(len(G.edges), 1)
    width = min(24, max(8, n_nodes / 8))
    height = min(24, max(6, n_nodes / 10))

    plt.figure(figsize=(width, height), dpi=300)
    pos = nx.spring_layout(G, seed=42, k=None)

    # 엣지 두께를 weight 기반으로 (선택)
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    # 정규화(너무 굵지 않게)
    max_w = max(edge_weights) if edge_weights else 1
    widths = [0.5 + 3.0 * (w / max_w) for w in edge_weights]

    # 노드 (파란색)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#1f77b4", alpha=0.9)
    # 엣지
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle='-|>', arrowsize=10, alpha=0.6)
    # 라벨: x 인자
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")

    plt.axis("off")
    if title:
        plt.title(title, fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------------------
# 4) 통계 저장
# ---------------------------
def save_stats(G: nx.DiGraph, out_txt: str):
    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"nodes: {n_nodes}\n")
        f.write(f"edges: {n_edges}\n")
        f.write("\n# nodes\n")
        for n in sorted(G.nodes()):
            f.write(f"{n}\n")
    print(f"📊 nodes={n_nodes}, edges={n_edges}")
    print(f"📝 stats saved → {out_txt}")

# ---------------------------
# 5) 파이프라인
# ---------------------------
def build_and_save_graph_from_class_csv(
    class_csv_path: str,
    out_dir: str = "result/graphs"
):
    """
    class_csv_path: result/cfg_class/<클래스명>.csv  (x_ngram,count)
    out_dir       : 결과 이미지/통계 저장 폴더
    """
    ngrams_with_count = load_class_csv(class_csv_path)
    G = build_graph_from_ngrams(ngrams_with_count)

    cls_name = Path(class_csv_path).stem
    out_png = str(Path(out_dir) / f"{cls_name}.png")
    out_txt = str(Path(out_dir) / f"{cls_name}_stats.txt")

    title = f"Class: {cls_name}  |  nodes={len(G.nodes)}  edges={len(G.edges)}"
    draw_and_save_graph(G, out_png, title=title)
    save_stats(G, out_txt)
    print(f"🖼️ graph image saved → {out_png}")

# ---------------------------
# 6) 예시 실행
# ---------------------------
if __name__ == "__main__":
    # 단일 클래스 CSV로부터 그래프 생성
    # build_and_save_graph_from_class_csv("result/cfg_class/APT1.csv", out_dir="result/graphs")

    # 디렉터리 내 모든 클래스 CSV 처리
    for p in Path("result/train/cfg_class").glob("*.csv"):
        build_and_save_graph_from_class_csv(str(p), out_dir="result/train/graphs")
    pass
