# TrafficFlow: 실시간 차량 속도 감지 시스템

테스트입니다.

TrafficFlow는 CCTV 스트림에서 차량 속도를 실시간으로 측정하고 분석하는 프로젝트입니다. 이 시스템은 OpenCV와 YOLO 모델을 사용하여 차량을 감지하고, 특정 구간을 통과하는 시간을 측정하여 속도를 계산합니다.

## 주요 기능

1. **실시간 차량 감지**: YOLOv3-tiny 모델을 사용하여 차량을 감지합니다.
2. **속도 측정**: 지정된 영역을 차량이 통과하는 시간을 측정하여 속도를 계산합니다.
3. **다중 차선 분석**: 최대 6개 차선의 차량을 동시에 분석합니다.
4. **인터랙티브 영역 설정**: 마우스로 측정 영역을 선택하고 드래그하여 위치를 조정할 수 있습니다.
5. **설정 저장/로드**: 측정 영역 설정을 JSON 파일로 저장하고 로드할 수 있습니다.

## 프로젝트 구조

```
TrafficFlow/
│
├── main.py                   # 메인 애플리케이션 진입점
├── config.py                 # 설정 파일
│
├── models/
│   ├── __init__.py
│   ├── yolo_detector.py      # YOLO 기반 차량 감지 모듈
│   └── data/                 # YOLO 모델 파일
│       ├── yolov3-tiny.weights
│       ├── yolov3-tiny.cfg
│       └── coco.names
│
├── processors/
│   ├── __init__.py
│   ├── video_processor.py    # 비디오 스트림 처리
│   └── lane_processor.py     # 차선 및 영역 처리
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # 시각화 관련 유틸리티
│
├── zones.json                # 저장된 측정 영역 설정
├── README.md
└── requirements.txt
```

## 설치 방법

1. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

2. YOLO 모델 파일 다운로드:
   ```
   mkdir -p models/data
   curl -L https://pjreddie.com/media/files/yolov3-tiny.weights -o models/data/yolov3-tiny.weights
   curl -L https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -o models/data/yolov3-tiny.cfg
   curl -L https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -o models/data/coco.names
   ```

## 사용 방법

1. 기본 실행:
   ```
   python main.py
   ```

2. 특정 스트림 URL 지정:
   ```
   python main.py --stream_url "https://your-stream-url"
   ```

3. 녹화 기능 활성화:
   ```
   python main.py --record
   ```

## 단축키

- **q**: 프로그램 종료
- **e**: 편집 모드 활성화/비활성화
- **s**: 편집 모드에서 현재 측정 영역 설정 저장
- **l**: 저장된 측정 영역 설정 로드
- **n**: 야간 모드 활성화/비활성화

## 편집 모드 사용법

1. 'e' 키를 눌러 편집 모드 활성화
2. 측정 영역 클릭하여 선택
3. 선택한 영역 드래그하여 이동:
   - 상단 가장자리: 시작선만 이동
   - 하단 가장자리: 종료선만 이동
   - 중앙: 전체 영역 이동
4. 's' 키를 눌러 변경사항 저장
5. 'e' 키를 다시 눌러 편집 모드 종료

## 기술적 세부사항

1. **YOLO 모델**: Mac M1 호환성을 위해 YOLOv3-tiny 사용
2. **속도 계산**: 실제 거리와 통과 시간을 기반으로 계산 (m/s → km/h 변환)
3. **UI 스타일**: 참조 이미지에 기반한 최소한의 UI 설계
   - 화면 상단: 차선별 속도 및 차량 수 정보
   - 도로 중앙: 컬러 측정 영역과 빨간색 경계선
   - 화면 하단: 프로젝트 정보

이 프로젝트는 교통 모니터링, 속도 위반 감지, 교통 흐름 분석 등 다양한 응용 분야에 활용될 수 있습니다.