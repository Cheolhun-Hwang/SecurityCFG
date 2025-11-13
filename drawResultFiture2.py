import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict

# 라벨 CSV 로드 함수
def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if 'id' not in cols or 'class' not in cols:
        raise ValueError("CSV에 'Id'와 'Class' (대소문자 무관) 컬럼이 필요합니다.")
    id_col = cols['id']
    cls_col = cols['class']
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[cls_col].astype(str)))

# 노드 색상 결정 함수
def get_node_colors(G, red_nodes):
    return ['red' if node in red_nodes else 'skyblue' for node in G.nodes]

# 디렉토리에서 .txt 파일마다 그래프 그리기
def draw_graphs_from_txt(out_dir, labels_csv, xgram_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 파일명 -> 클래스 ID
    id_to_class = load_labels(labels_csv)

    for filename in os.listdir(out_dir):
        if not filename.endswith(".txt"):
            continue

        file_id = filename.replace('.txt', '')
        class_id = id_to_class.get(file_id)

        if class_id is None:
            print(f"라벨 없음: {file_id}")
            continue

        if class_id == "6" or class_id =="7":
            # print("class_id", class_id)

            txt_path = os.path.join(out_dir, filename)
            xgram_path = os.path.join(xgram_dir, f"{class_id}.csv")

            if not os.path.exists(xgram_path):
                print(f"Xgram CSV 없음: {xgram_path}")
                continue

            # 빨간 노드 지정: 해당 클래스 CSV에 나오는 3-gram
            red_nodes = set()
            df = pd.read_csv(xgram_path)
            for xgram in df.iloc[:, 0].values:
                parts = xgram.strip().split('_')
                red_nodes.update(parts)

            # 그래프 생성
            G = nx.DiGraph()
            with open(txt_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('_')
                    if len(parts) != 3:
                        continue
                    G.add_edge(parts[0], parts[1])
                    G.add_edge(parts[1], parts[2])

            # 색상 및 위치
            pos = nx.spring_layout(G, k=0.8, iterations=300)
            node_colors = get_node_colors(G, red_nodes)

            plt.figure(figsize=(20, 20))
            nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='black', font_size=8, node_size=500)
            plt.title(f"{file_id} - Class {class_id}")
            save_path = os.path.join(save_dir, f"{file_id}_{class_id}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"저장 완료: {save_path}")

# 실행 파라미터
out_dir = "result/train/cfg"
labels_csv = r"D:\malware-classification\trainLabels.csv"
xgram_dir = "result/train/cfg_class"
save_dir = "result/train/figures2"

draw_graphs_from_txt(out_dir, labels_csv, xgram_dir, save_dir)
