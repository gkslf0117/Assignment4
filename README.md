# Assignment4


𝛼,𝛽-Crown 검증기를 사용하여 TinyMNIST 모델의 Robustness을 검증하는 과제

## 프로젝트 구조
- `alpha-beta-CROWN/`: 𝛼,𝛽-Crown 오픈소스 리포지토리
- `mnist_verification.yaml`: 검증 설정을 위한 YAML 구성 파일
- `test.py`: 모델 생성(ONNX export) 및 검증 실행 통합 스크립트
- `mnist_small.onnx`: 생성된 외부 모델 파일 (실행 시 자동 생성)
- `report.pdf`: 검증 결과 및 Marabou와의 비교 분석 보고서

## 설치 및 실행 방법

### 1. 환경 설정
프로젝트에 필요한 의존성 패키지를 설치합니다.
```bash
pip install -r requirements.txt



2. 검증 실행
test.py 스크립트를 실행하면 자동으로 TinyMNIST 모델을 ONNX 파일로 내보낸 뒤, 𝛼,𝛽-Crown 검증기가 즉시 실행됩니다.

python test.py



참고사항
본 프로젝트는 NVIDIA GPU가 없는 환경을 고려하여 mnist_verification.yaml 설정 파일에서 device: cpu로 강제 지정되어 있습니다.

검증 결과는 터미널에 출력되며, 최종 요약본에서 성공 여부를 확인할 수 있습니다.



#Acknowledge 과제 구현 과정에서 OpenAI의 도움을 받았음 밝힙니다.
