import os
import csv
import networkx as nx
import matplotlib.pyplot as plt

def load_cfg_csv(path):
    edges = []
    nodes_in_csv = set()
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parts = row['x_ngram'].split('_')
            if len(parts) == 3:
                a, b, c = parts
                edges.append((a, b))
                edges.append((b, c))
                nodes_in_csv.update([a, b, c])
    return edges, nodes_in_csv

def load_node_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    node_section = False
    node_list = []
    for line in lines:
        if line.strip() == "# nodes":
            node_section = True
            continue
        if node_section:
            line = line.strip()
            if line:
                node_list.append(line)
    return set(node_list)

def draw_graph(edges, highlight_nodes, all_nodes, output_path):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 위치 설정 (spring layout: 노드 간 겹침 최소화)
    pos = nx.spring_layout(G, seed=42)

    # 색상 지정
    node_colors = []
    for node in G.nodes():
        if node in highlight_nodes:
            node_colors.append("red")
        else:
            node_colors.append("blue")

    # 그래프 그리기
    plt.figure(figsize=(18, 14))
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight='bold')

    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ 그래프 저장 완료: {output_path}")

if __name__ == "__main__":
    cfg_csv_path = "result/train/cfg/0A32eTdBKayjCWhZqDOQ.txt.csv"
    stat_txt_path = "result/train/graphs_10/1_stats.txt"
    output_img_path = "result/train/class_result_10/graph_1.png"

    edges, nodes_from_csv = load_cfg_csv(cfg_csv_path)
    nodes_from_stat = load_node_list(stat_txt_path)

    # 실제 노드 집합: stat에 존재하는 모든 노드 중에서 csv에 등장한 노드 강조
    draw_graph(edges, highlight_nodes=nodes_from_csv & nodes_from_stat,
               all_nodes=nodes_from_stat, output_path=output_img_path)