## Microsoft Malware Classification Challenge
### 해당 데이터를 기반으로 한 Feature Graph 만드는 프로젝트 입니다.

### 1. checkFile.py

- 파일을 읽어 opcode 명령을 별도의 convert 파일로 만듬
- 변환 시 ngram_map을 통해 3-gram 기반 Feature(X) 로 변환
- out_dir = 'result/train/cvt' : 변환된 txt 파일 저장 경로
- map_csv = 'result/train/cvt/ngram_map.csv'

### 