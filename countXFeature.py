import os
import pandas as pd
from collections import Counter

# 입력 및 출력 디렉토리
xgram_dir = "result/train/cfg_class"
output_dir = "result/train/fcount3"
os.makedirs(output_dir, exist_ok=True)

# 모든 클래스 파일 처리
for fname in os.listdir(xgram_dir):
    if not fname.endswith('.csv'):
        continue

    class_id = os.path.splitext(fname)[0]  # 예: "1"
    fpath = os.path.join(xgram_dir, fname)
    df = pd.read_csv(fpath)

    feature_counter = Counter()

    for _, row in df.iterrows():
        ngram = row['x_ngram']  # 예: x22_x6_x7
        count = int(row['count'])
        features = ngram.split('_')  # ['x22', 'x6', 'x7']
        for feature in features:
            feature_counter[feature] += count

    # Counter를 DataFrame으로 변환 및 저장
    out_df = pd.DataFrame(feature_counter.items(), columns=['x_feature', 'count'])
    out_df.sort_values(by='count', ascending=False, inplace=True)
    out_df.to_csv(os.path.join(output_dir, f"{class_id}.csv"), index=False)

print("✅ 클래스별 x_feature 등장 횟수 집계 및 저장 완료.")
