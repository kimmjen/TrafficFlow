"""
속도 계산 관련 유틸리티
"""

import math
from typing import Tuple, List, Dict, Any


class SpeedCalculator:
    def __init__(self, real_distance_meters: float = 5.0):
        """
        속도 계산기 초기화

        Args:
            real_distance_meters: 측정 라인 사이의 실제 거리 (미터)
        """
        self.real_distance_meters = real_distance_meters

    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        두 점 사이의 거리 계산

        Args:
            point1: 첫 번째 점 (x, y)
            point2: 두 번째 점 (x, y)

        Returns:
            두 점 사이의 유클리드 거리
        """
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def calculate_speed(self, time_seconds: float) -> float:
        """
        속도 계산

        Args:
            time_seconds: 측정 구간을 통과하는 데 걸린 시간 (초)

        Returns:
            속도 (km/h)
        """
        if time_seconds <= 0:
            return 0

        # 초당 미터를 시간당 킬로미터로 변환
        return (self.real_distance_meters / time_seconds) * 3.6

    def process_vehicle_data(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        차량 데이터 처리

        Args:
            vehicle_data: 차량 추적 데이터

        Returns:
            처리된 차량 데이터 (속도 포함)
        """
        result = vehicle_data.copy()

        if vehicle_data.get('start_time') and vehicle_data.get('end_time'):
            time_taken = vehicle_data['end_time'] - vehicle_data['start_time']
            speed = self.calculate_speed(time_taken)
            result['speed'] = speed
        else:
            result['speed'] = 0

        return result

    def calculate_average_speed(self, vehicles: List[Dict[str, Any]]) -> float:
        """
        여러 차량의 평균 속도 계산

        Args:
            vehicles: 처리된 차량 데이터 목록

        Returns:
            평균 속도 (km/h)
        """
        completed_vehicles = [v for v in vehicles if 'speed' in v and v['speed'] > 0]

        if not completed_vehicles:
            return 0

        total_speed = sum(v['speed'] for v in completed_vehicles)
        return total_speed / len(completed_vehicles)