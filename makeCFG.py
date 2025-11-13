import os

def read_xparams(file_path: str):
    """한 줄에 하나씩 기록된 x인자(txt)를 읽어 리스트로 반환"""
    tokens = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 공백으로 여러 토큰이 들어온 경우도 방지 차원에서 분리 처리
            parts = line.split()
            tokens.extend(parts)
    return tokens

def make_ngrams(seq, n=3, sep="_"):
    """seq에서 길이 n의 슬라이딩 n-그램 생성 (구분자 기본 '_')"""
    if len(seq) < n:
        return []
    return [sep.join(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def save_lines(out_dir: str, src_fname: str, lines):
    """/result/cfg/<입력파일명>.txt 로 저장 (입력 파일명 유지)"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(src_fname))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return out_path

def process_xparam_file(input_path: str, out_dir: str = "/result/train/cfg", n: int = 3, sep: str = "_"):
    """
    x 인자 txt를 읽어 3그램(기본)으로 묶어 저장.
    - 입력: x인자 한 줄씩
    - 출력: 각 줄이 x그램 (예: x8_x14_x13)
    """
    x_tokens = read_xparams(input_path)
    x_ngrams = make_ngrams(x_tokens, n=n, sep=sep)
    out_path = save_lines(out_dir, os.path.basename(input_path), x_ngrams)
    print(f"✅ saved {len(x_ngrams)} {n}-grams → {out_path}")

def process_path(path: str, out_dir: str = "/result/cft", n: int = 3, sep: str = "_"):
    """
    파일 또는 디렉터리 처리:
    - 파일: 해당 파일만 변환
    - 디렉터리: 내부 .txt 전부 변환
    """
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.lower().endswith(".txt"):
                process_xparam_file(os.path.join(path, fname), out_dir=out_dir, n=n, sep=sep)
    else:
        process_xparam_file(path, out_dir=out_dir, n=n, sep=sep)

if __name__ == "__main__":
    # 사용 예시: 단일 파일
    # process_path("/mnt/data/0A32eTdBKayjCWhZqDOQ.txt", out_dir="/result/cft", n=3, sep="_")

    # 사용 예시: 디렉터리 내 모든 .txt
    process_path("result/train/cvt", out_dir="result/train/cfg", n=3, sep="_")
