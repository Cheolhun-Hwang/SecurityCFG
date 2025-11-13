import os
import pandas as pd


def extract_opcodes_msasm(file_path):
    """Microsoft asm 파일에서 실제 opcode만 추출"""
    opcodes = []
    skip_keywords = {'format', 'imagebase', 'section', 'virtual', 'offset', 'flags',
                     'alignment', 'os', 'application', 'flat', 'segment'}

    junk_opcodes = {
        'align', 'nop', 'int3', 'cc', 'db', 'dd', 'dq', 'dw', 'dt',
        'extrn', 'public', 'assume', 'endp', 'proc', 'ends'
    }

    valid_opcodes = {
        'mov', 'push', 'pop', 'add', 'sub', 'cmp', 'jmp', 'call', 'retn',
        'inc', 'dec', 'lea', 'xor', 'and', 'or', 'test', 'nop',
        'shr', 'shl', 'imul', 'idiv', 'int', 'not', 'neg', 'jz', 'jnz', 'jecxz',
        'jge', 'jle', 'jg', 'jl', 'je', 'jne', 'jb', 'ja', 'jbe', 'jae',
        'stos', 'lods', 'scas', 'movs', 'cmps', 'rep', 'repe', 'repne',
        'leave', 'ret', 'cdq', 'cwd', 'setne', 'sete', 'movzx', 'movsx',
        'sbb', 'adc', 'xchg', 'bswap', 'sar', 'sal', 'ror', 'rol', 'rcr', 'rcl'
    }

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('.text:'):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            opcode = parts[2].lower()
            if not opcode.isalpha():
                continue
            if opcode in skip_keywords or opcode in junk_opcodes:
                continue
            if opcode.startswith(('sub_', 'loc_')) or opcode.endswith(':'):
                continue
            if opcode not in valid_opcodes:
                continue
            opcodes.append(opcode)
    return opcodes

def generate_ngrams(opcodes, n=3):
    return ['_'.join(opcodes[i:i+n]) for i in range(len(opcodes) - n + 1)]

def convert_ngrams_to_x(ngrams, df_map):
    """
    n-gram 리스트를 x 시퀀스로 변환.
    - df_map은 누적 변환 테이블(DataFrame: n_gram, xparam)
    - 새 n-gram이면 x{len(df_map)+1}로 추가
    """
    x_seq = []
    # 빠른 조회를 위해 dict 캐시
    lookup = dict(zip(df_map['n_gram'], df_map['xparam']))

    for ng in ngrams:
        if ng in lookup:
            x_val = lookup[ng]
        else:
            # 새 항목 추가
            next_idx = len(df_map) + 1
            x_val = f"x{next_idx}"
            df_map.loc[len(df_map)] = {'n_gram': ng, 'xparam': x_val}
            lookup[ng] = x_val  # 캐시 동기화
        x_seq.append(x_val)
    return x_seq, df_map

# ===== n-gram → x 변환 테이블 관리 =====
def load_or_init_map(map_path):
    """CSV를 로드하거나 없으면 새 DataFrame 생성"""
    if os.path.isfile(map_path):
        df = pd.read_csv(map_path)
        # 안전장치: 필수 컬럼 보정
        if not {'n_gram', 'xparam'}.issubset(df.columns):
            df = pd.DataFrame(columns=['n_gram', 'xparam'])
    else:
        df = pd.DataFrame(columns=['n_gram', 'xparam'])
    return df

def save_map(df, map_path):
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    df.to_csv(map_path, index=False, encoding='utf-8')

# ===== 메인 파이프라인 =====
def save_converted_sequence(output_dir, src_fname, x_seq):
    """
    파일별 변환 결과 저장:
    - result/cvt/<원본파일명>.txt
    - 한 줄에 하나의 xparam
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.splitext(src_fname)[0]}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(x_seq))

def save_ngrams_only(output_dir, src_fname, ngrams):
    """순수 n-gram 목록만 저장 (파일명 동일, 확장자만 .txt)"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.splitext(src_fname)[0]}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(ngrams))


def process_ms_asm_files(directory, out_dir, map_csv):
    # 누적 변환 테이블 로드
    df_map = load_or_init_map(map_csv)

    for fname in os.listdir(directory):
        if fname.endswith('.asm'):
            path = os.path.join(directory, fname)
            print(f"\n📂 {fname}")
            opcodes = extract_opcodes_msasm(path)
            ngrams = generate_ngrams(opcodes, n=3)

            print(f"🔢 총 3-gram 개수: {len(ngrams)}")
            print("🧩 상위 10개:")
            print('\n'.join(ngrams[:10]))

            # 🔹 n-gram 원본을 result/data/<파일명>.txt 로 저장 (요청 사항)
            save_ngrams_only('result/train/data', fname, ngrams)

            # 3) n-gram → x로 전체 변환 (미등록이면 x 새로 할당 및 테이블에 추가)
            x_seq, df_map = convert_ngrams_to_x(ngrams, df_map)

            # 4) 변환된 전체 시퀀스를 파일로 저장
            save_converted_sequence(out_dir, fname, x_seq)

            # 콘솔 안내(원하면 제거 가능)
            print(f"📂 {fname} | n-grams: {len(ngrams)} → x: {len(x_seq)} 저장 완료: {out_dir}")
    # 5) 모든 파일 처리 후 변환 테이블 누적 저장
    save_map(df_map, map_csv)
    print(f"✅ 변환 테이블 누적 저장: {map_csv} (총 항목 {len(df_map)})")

if __name__ == "__main__":
    asm_dir = 'D:\\malware-classification\\train\\train'
    out_dir = 'result/train/cvt'
    map_csv = 'result/train/cvt/ngram_map.csv'
    if not os.path.isdir(asm_dir):
        print(f"❌ '{asm_dir}' 디렉토리가 존재하지 않습니다.")
    else:
        process_ms_asm_files(asm_dir, out_dir, map_csv)