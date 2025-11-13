import os
import csv
from collections import Counter
from tqdm import tqdm
import argparse

def extract_opcodes(file_path):
    opcodes = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.text:'):
                parts = line.split()
                if len(parts) >= 3:
                    opcode = parts[2]
                    if opcode.isalpha():  # remove hex or db
                        opcodes.append(opcode.lower())
    return opcodes

def generate_ngrams(opcodes, n=3):
    return ['_'.join(opcodes[i:i+n]) for i in range(len(opcodes) - n + 1)]

def load_labels(label_csv_path):
    labels = {}
    with open(label_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            labels[row[0]] = row[1]
    return labels

def main(asm_dir, label_csv, output_csv, n=3, top_k=1000):
    print("🔍 Loading labels...")
    labels = load_labels(label_csv)

    print("📁 Scanning .asm files...")
    files = [f for f in os.listdir(asm_dir) if f.endswith('.asm')]

    global_counter = Counter()

    print("🔧 Extracting global n-gram frequency...")
    for fname in tqdm(files):
        opcodes = extract_opcodes(os.path.join(asm_dir, fname))
        ngrams = generate_ngrams(opcodes, n)
        global_counter.update(ngrams)

    most_common = [ng for ng, _ in global_counter.most_common(top_k)]

    print("💾 Writing feature vectors to:", output_csv)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'label'] + most_common
        writer.writerow(header)

        for fname in tqdm(files):
            file_id = fname.replace('.asm', '')
            label = labels.get(file_id, 'unknown')
            opcodes = extract_opcodes(os.path.join(asm_dir, fname))
            ngrams = generate_ngrams(opcodes, n)
            ngram_count = Counter(ngrams)

            row = [file_id, label] + [ngram_count.get(k, 0) for k in most_common]
            writer.writerow(row)

    print("✅ Done. CSV saved:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opcode n-gram extractor")
    parser.add_argument('--asm_dir', required=True, help='Path to asm files (train/)')
    parser.add_argument('--label_csv', required=True, help='Path to trainLabels.csv')
    parser.add_argument('--output_csv', default='opcode_features.csv', help='Output CSV path')
    parser.add_argument('--ngram', type=int, default=3, help='Size of n-gram')
    parser.add_argument('--top_k', type=int, default=1000, help='Top K n-gram features to include')
    args = parser.parse_args()

    main(args.asm_dir, args.label_csv, args.output_csv, args.ngram, args.top_k)