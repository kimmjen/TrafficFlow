"""
YOLO 기반 차량 감지 모듈 (야간 조건 최적화)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class YOLODetector:
    def __init__(self, weights_path: str, config_path: str, names_path: str,
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4,
                 target_classes: List[int] = None,
                 input_size: Tuple[int, int] = (320, 320),
                 enable_night_mode: bool = True):
        """
        YOLO 차량 감지기 초기화

        Args:
            weights_path: YOLO 가중치 파일 경로
            config_path: YOLO 설정 파일 경로
            names_path: 클래스 이름 파일 경로
            confidence_threshold: 감지 신뢰도 임계값
            nms_threshold: 비최대 억제 임계값
            target_classes: 감지할 클래스 ID 목록 (None인 경우 모든 클래스)
            input_size: YOLO 입력 크기
            enable_night_mode: 야간 모드 활성화 여부
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes
        self.input_size = input_size
        self.enable_night_mode = enable_night_mode

        # YOLO 모델 로드
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Mac M1에서는 CUDA를 사용할 수 없으므로 CPU 백엔드 사용
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 클래스 이름 로드
        self.classes = []
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 출력 레이어 이름 가져오기
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def _preprocess_night_image(self, image: np.ndarray) -> np.ndarray:
        """
        야간 이미지 전처리

        Args:
            image: 입력 이미지

        Returns:
            전처리된 이미지
        """
        # 대비 제한 적응형 히스토그램 평활화 (CLAHE) 적용
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 밝기 향상
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 밝기(V) 채널 향상
        lim = 255 - 30
        v[v > lim] = 255
        v[v <= lim] = v[v <= lim] + 30

        final_hsv = cv2.merge((h, s, v))
        final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return final_image

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        프레임에서 차량 감지

        Args:
            frame: 처리할 비디오 프레임

        Returns:
            감지된 차량 목록 (박스, 클래스, 신뢰도 포함)
        """
        # 야간 모드가 활성화된 경우 이미지 전처리
        if self.enable_night_mode:
            processed_frame = self._preprocess_night_image(frame)
        else:
            processed_frame = frame

        height, width, _ = processed_frame.shape

        # 이미지를 YOLO 입력 형식으로 변환
        blob = cv2.dnn.blobFromImage(np.zeros((416, 416, 3), dtype=np.uint8), 1 / 255.0, (416, 416), swapRB=True)
        self.net.setInput(blob)
        # blob = cv2.dnn.blobFromImage(processed_frame, 1/255.0, self.input_size, swapRB=True, crop=False)
        # self.net.setInput(blob)

        # 감지 실행
        outs = self.net.forward(self.output_layers)

        # 결과 처리
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 타겟 클래스에 포함되고 신뢰도가 임계값보다 높은 경우만 처리
                if (self.target_classes is None or class_id in self.target_classes) and confidence > self.confidence_threshold:
                    # 객체 좌표 계산
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 사각형 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제로 중복 박스 제거
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        vehicles = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                vehicles.append({
                    'box': boxes[i],
                    'class_id': class_ids[i],
                    'class': self.classes[class_ids[i]],
                    'confidence': confidences[i]
                })

        return vehicles