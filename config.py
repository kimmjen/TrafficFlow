"""
TrafficFlow 프로젝트의 설정 파일 (고속도로 CCTV 최적화)
전역 설정 및 매개변수 정의
"""

import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "data")

# YOLO 모델 설정
YOLO_CONFIG = {
    # YOLOv3-tiny 사용 (더 나은 호환성)
    "weights": os.path.join(MODEL_DIR, "yolov3-tiny.weights"),
    "config": os.path.join(MODEL_DIR, "yolov3-tiny.cfg"),
    "names": os.path.join(MODEL_DIR, "coco.names"),
    "confidence_threshold": 0.4,  # 신뢰도 임계값 약간 낮춤
    "nms_threshold": 0.5,
    "target_classes": [2, 5, 7]  # 2: 자동차, 5: 버스, 7: 트럭
}

# 비디오 스트림 설정
VIDEO_CONFIG = {
    "default_stream_url": "https://strm2.spatic.go.kr/live/200.stream/chunklist_w1191338775.m3u8",
    "process_every_n_frames": 5,  # 처리 빈도 낮춤 (3에서 5로)
    "output_dir": os.path.join(BASE_DIR, "output"),
    "resize_factor": 0.7  # 프레임 크기 70%로 축소
}

# 속도 측정 설정
SPEED_CONFIG = {
    "real_distance_meters": 8.0,  # 측정 라인 사이의 실제 거리 (미터) - 고속도로 기준 조정
    "max_matching_distance": 70,  # 같은 차량으로 인식할 최대 픽셀 거리
    "line_crossing_threshold": 15  # 라인 통과로 간주할 픽셀 거리
}

# 표시 설정
DISPLAY_CONFIG = {
    "lane_colors": [
        (255, 0, 0),     # 빨간색
        (0, 255, 255),   # 노란색
        (0, 165, 255),   # 주황색
        (0, 255, 0),     # 녹색
        (255, 0, 255),   # 보라색
        (255, 255, 0)    # 청록색
    ],
    "vehicle_box_color": (0, 255, 0),
    "text_color": (255, 255, 255),
    "zone_alpha": 0.5,    # 측정 영역 투명도
    "night_mode": True    # 야간 모드 활성화
}

# 디스플레이 설정
UI_CONFIG = {
    "show_fps": True,
    "show_vehicle_class": True,
    "show_confidence": False,
    "display_scale": 1.0,  # UI 요소 크기 조정
    "font_scale": 0.5,
    "line_thickness": 2
}