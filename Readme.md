## Microsoft Malware Classification Challenge
### 해당 데이터를 기반으로 한 Feature Graph 만드는 프로젝트 입니다.

### 1. checkFile.py

- 파일을 읽어 opcode 명령을 별도의 convert 파일로 만듬
- 변환 시 ngram_map을 통해 3-gram 기반 Feature(X) 로 변환
- out_dir = 'result/train/cvt' : 변환된 txt 파일 저장 경로
- map_csv = 'result/train/cvt/ngram_map.csv'

### 2. makeCFG.py

- 변환된 X feature 텍스트 파일을 3-gram으로 묶어 행위로 변경
- x-x-x 변경

### 3. defineCFG.py

- 변환된 x feature -gram 텍스트 파일을 클래스별로 정의
- result/train/cfg_class

### Support

- 일부 코드 기능 중복이 있을 수 있으며, 분석 및 테스트를 위해 추가된 파일들입니다.
- countXFeature.py : 클래스별 x_feature 등장 횟수 집계 및 저장
- drawGraph.py : 클래스 x-feature flow graph 그리기
- drawGraph2.py : 클래스 x-feature flow graph 임계값(유사도)을 기반으로 그리기
- drowResultFigure.py : 단일 대상 그래프 그리기
- drowResultFigure2.py : 클래스 x-feature flow graph 그리기