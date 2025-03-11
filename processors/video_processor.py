"""
비디오 스트림 처리 모듈 (성능 최적화)
"""

import cv2
import time
import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime

class VideoProcessor:
    def __init__(self, stream_url: str, process_every_n_frames: int = 5,
                 output_dir: Optional[str] = None,
                 resize_factor: float = 1.0):
        """
        비디오 처리기 초기화

        Args:
            stream_url: 비디오 스트림 URL
            process_every_n_frames: 처리할 프레임 간격
            output_dir: 출력 디렉토리 (녹화 시)
            resize_factor: 프레임 크기 조정 비율 (1.0 = 원본 크기)
        """
        self.stream_url = stream_url
        self.process_every_n_frames = process_every_n_frames
        self.output_dir = output_dir
        self.resize_factor = resize_factor
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.previous_time = time.time()
        self.current_fps = 0

        # 해상도 및 출력 파일 정보
        self.frame_width = 0
        self.frame_height = 0
        self.output_file = None

        # 네트워크 스트림 버퍼 캐시 설정
        self.frame_buffer = []
        self.max_buffer_size = 5

    def open_stream(self) -> bool:
        """
        비디오 스트림 열기

        Returns:
            성공 여부
        """
        # 이미 열려있으면 닫기
        if self.cap is not None:
            self.cap.release()

        # 새로운 스트림 열기
        self.cap = cv2.VideoCapture(self.stream_url)

        # 버퍼 크기 설정 (네트워크 지연 감소를 위해)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        if not self.cap.isOpened():
            print(f"비디오 스트림을 열 수 없습니다: {self.stream_url}")
            return False

        # 프레임 크기 및 FPS 정보 가져오기
        orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 리사이즈 적용
        self.frame_width = int(orig_width * self.resize_factor)
        self.frame_height = int(orig_height * self.resize_factor)

        # 녹화 설정
        if self.output_dir:
            self._setup_recording()

        return True

    def _setup_recording(self) -> None:
        """
        녹화 설정
        """
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 현재 시간을 파일명으로 사용
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"traffic_flow_{timestamp}.mp4")

        # 비디오 작성기 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_file,
            fourcc,
            20.0,  # 출력 프레임 속도
            (self.frame_width, self.frame_height)
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        프레임 읽기

        Returns:
            (성공 여부, 프레임)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("비디오 스트림에서 프레임을 읽을 수 없습니다.")
            return False, None

        # FPS 계산
        current_time = time.time()
        elapsed_time = current_time - self.previous_time
        self.previous_time = current_time

        if elapsed_time > 0:
            self.current_fps = 1 / elapsed_time

        self.frame_count += 1

        # 리사이즈 적용
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        # 야간 영상 밝기 향상 (야간 조명 강화)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 차량 헤드라이트 등 밝은 부분 강조를 위한 감마 보정
        lut = np.array([((i / 255.0) ** 0.8) * 255 for i in np.arange(0, 256)]).astype("uint8")
        v = cv2.LUT(v, lut)

        hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return True, frame

    def should_process_frame(self) -> bool:
        """
        현재 프레임을 처리해야 하는지 여부

        Returns:
            처리 여부
        """
        return self.frame_count % self.process_every_n_frames == 0

    def write_frame(self, frame: np.ndarray) -> None:
        """
        프레임 녹화

        Args:
            frame: 녹화할 프레임
        """
        if self.writer is not None:
            self.writer.write(frame)

    def get_fps(self) -> float:
        """
        현재 FPS 반환

        Returns:
            초당 프레임 수
        """
        return self.current_fps

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        프레임 크기 반환

        Returns:
            (너비, 높이)
        """
        return self.frame_width, self.frame_height

    def release(self) -> None:
        """
        리소스 해제
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()