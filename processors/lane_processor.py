"""
Lane and measurement zone processing module (Highway CCTV custom version with interactive editing)
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import time
import json

class LaneProcessor:
    def __init__(self, frame_width: int, frame_height: int,
                 real_distance_meters: float = 5.0,
                 max_matching_distance: int = 50,
                 line_crossing_threshold: int = 10,
                 colors: List[Tuple[int, int, int]] = None,
                 use_predefined_zones: bool = True):
        """
        Initialize lane processor

        Args:
            frame_width: Frame width
            frame_height: Frame height
            real_distance_meters: Actual distance between measurement lines (meters)
            max_matching_distance: Maximum pixel distance to consider as same vehicle
            line_crossing_threshold: Pixel distance to consider as line crossing
            colors: List of colors for lanes
            use_predefined_zones: Whether to use predefined zones
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.real_distance_meters = real_distance_meters
        self.max_matching_distance = max_matching_distance
        self.line_crossing_threshold = line_crossing_threshold
        self.use_predefined_zones = use_predefined_zones

        # Set lane colors
        if colors is None:
            self.colors = [(255, 0, 0), (0, 255, 255), (0, 165, 255),
                           (0, 255, 0), (255, 0, 255)]
        else:
            self.colors = colors

        # Set up measurement zones
        self.zones = self._setup_zones()

    def _setup_zones(self) -> List[Dict[str, Any]]:
        """
        Set up measurement zones for each lane

        Returns:
            List of measurement zones
        """
        zones = []

        if self.use_predefined_zones:
            # Predefined areas based on highway CCTV image
            # Based on red boxes shown in the image

            # Define 6 measurement areas from left to right
            # Each area is defined as (x1, y1, x2, y2)
            # Top and bottom measurement lines are set 10 pixels apart

            # These values should be adjusted to match the actual image
            predefined_areas = [
                # Left areas (from left to right)
                (250, 390, 300, 400),  # Lane 1
                (320, 390, 370, 400),  # Lane 2
                (390, 390, 440, 400),  # Lane 3

                # Right areas (from left to right)
                (510, 390, 560, 400),  # Lane 4
                (580, 390, 630, 400),  # Lane 5
                (650, 390, 700, 400),  # Lane 6
            ]

            for i, area in enumerate(predefined_areas):
                x1, y1, x2, y2 = area
                start_line_y = y1
                end_line_y = y2

                zone = {
                    'lane_id': i + 1,
                    'start_line': (x1, start_line_y, x2, start_line_y),
                    'end_line': (x1, end_line_y, x2, end_line_y),
                    'polygon': np.array([
                        [x1, start_line_y],
                        [x2, start_line_y],
                        [x2, end_line_y],
                        [x1, end_line_y]
                    ], np.int32),
                    'color': self.colors[i % len(self.colors)],
                    'speed': 0,
                    'vehicles': {},  # Dictionary for tracking vehicles
                    'count': 0       # Vehicle count
                }
                zones.append(zone)
        else:
            # Original method: equal lane division
            lane_count = 6
            lane_width = self.frame_width // lane_count

            # Set measurement line positions (based on screen ratio)
            measurement_line_y1 = int(self.frame_height * 0.6)  # Top measurement line
            measurement_line_y2 = int(self.frame_height * 0.7)  # Bottom measurement line

            for i in range(lane_count):
                start_x = i * lane_width
                end_x = (i + 1) * lane_width

                zone = {
                    'lane_id': i + 1,
                    'start_line': (start_x, measurement_line_y1, end_x, measurement_line_y1),
                    'end_line': (start_x, measurement_line_y2, end_x, measurement_line_y2),
                    'polygon': np.array([
                        [start_x, measurement_line_y1],
                        [end_x, measurement_line_y1],
                        [end_x, measurement_line_y2],
                        [start_x, measurement_line_y2]
                    ], np.int32),
                    'color': self.colors[i % len(self.colors)],
                    'speed': 0,
                    'vehicles': {},  # Dictionary for tracking vehicles
                    'count': 0       # Vehicle count
                }
                zones.append(zone)

        return zones

    def update_vehicle_tracking(self, vehicles: List[Dict[str, Any]], current_time: float) -> None:
        """
        Update vehicle tracking and speed measurement

        Args:
            vehicles: List of detected vehicles
            current_time: Current time
        """
        for vehicle in vehicles:
            box = vehicle['box']
            x, y, w, h = box

            # Bottom center point of vehicle
            center_x = x + w // 2
            center_y = y + h

            # Check for each measurement zone
            for zone in self.zones:
                # Check if passed the start line
                start_line = zone['start_line']
                start_x1, start_y1, start_x2, start_y1 = start_line

                # Check if vehicle is near this zone's start line
                if (start_x1 <= center_x <= start_x2 and
                    abs(center_y - start_y1) < self.line_crossing_threshold):

                    vehicle_id = f"{center_x}_{center_y}_{current_time}"
                    if vehicle_id not in zone['vehicles']:
                        zone['vehicles'][vehicle_id] = {
                            'start_time': current_time,
                            'end_time': None,
                            'position': (center_x, center_y)
                        }

                # Check if passed the end line
                end_line = zone['end_line']
                end_x1, end_y1, end_x2, end_y1 = end_line

                # Check if vehicle is near this zone's end line
                if (end_x1 <= center_x <= end_x2 and
                    abs(center_y - end_y1) < self.line_crossing_threshold):

                    # Find closest vehicle that passed start line
                    min_distance = float('inf')
                    closest_vehicle_id = None

                    for vehicle_id, timings in zone['vehicles'].items():
                        if timings['end_time'] is None:
                            veh_x = timings['position'][0]
                            distance = abs(center_x - veh_x)
                            if distance < min_distance:
                                min_distance = distance
                                closest_vehicle_id = vehicle_id

                    if closest_vehicle_id and min_distance < self.max_matching_distance:
                        zone['vehicles'][closest_vehicle_id]['end_time'] = current_time

                        # Calculate speed
                        start_time = zone['vehicles'][closest_vehicle_id]['start_time']
                        end_time = zone['vehicles'][closest_vehicle_id]['end_time']
                        time_taken = end_time - start_time

                        if time_taken > 0:
                            # Convert m/s to km/h
                            speed = (self.real_distance_meters / time_taken) * 3.6
                            zone['speed'] = speed
                            zone['count'] += 1

    def get_zone_polygons(self) -> List[Dict]:
        """
        Get zone polygon information for visualization

        Returns:
            List of zone polygon information
        """
        return [
            {
                'polygon': zone['polygon'].reshape((-1, 1, 2)),
                'color': zone['color'],
                'lane_id': zone['lane_id'],
                'speed': zone['speed'],
                'count': zone['count']
            }
            for zone in self.zones
        ]

    def get_lane_data(self) -> List[Dict]:
        """
        Get lane data

        Returns:
            List of lane data
        """
        return [
            {
                'lane_id': zone['lane_id'],
                'speed': zone['speed'],
                'count': zone['count'],
                'color': zone['color']
            }
            for zone in self.zones
        ]

    def update_zone(self, zone_id: int, new_position: Tuple[int, int], anchor_point: str):
        """
        Update the position of a specific zone

        Args:
            zone_id: Zone ID to update (starting from 0)
            new_position: New position (x, y)
            anchor_point: Point to move ('start', 'end', 'all')
        """
        if zone_id < 0 or zone_id >= len(self.zones):
            return

        x, y = new_position
        zone = self.zones[zone_id]

        if anchor_point == 'start':
            # Move only start line
            start_x1, start_y1, start_x2, start_y1 = zone['start_line']
            width = start_x2 - start_x1

            # Calculate new position
            new_x1 = x
            new_x2 = x + width

            # Update
            zone['start_line'] = (new_x1, y, new_x2, y)

            # Update polygon
            zone['polygon'][0] = [new_x1, y]
            zone['polygon'][1] = [new_x2, y]

        elif anchor_point == 'end':
            # Move only end line
            end_x1, end_y1, end_x2, end_y1 = zone['end_line']
            width = end_x2 - end_x1

            # Calculate new position
            new_x1 = x
            new_x2 = x + width

            # Update
            zone['end_line'] = (new_x1, y, new_x2, y)

            # Update polygon
            zone['polygon'][2] = [new_x2, y]
            zone['polygon'][3] = [new_x1, y]

        elif anchor_point == 'all':
            # Move entire zone
            start_x1, start_y1, start_x2, start_y1 = zone['start_line']
            end_x1, end_y1, end_x2, end_y1 = zone['end_line']

            width = start_x2 - start_x1
            height = end_y1 - start_y1

            # Update
            zone['start_line'] = (x, y, x + width, y)
            zone['end_line'] = (x, y + height, x + width, y + height)

            # Update polygon
            zone['polygon'][0] = [x, y]
            zone['polygon'][1] = [x + width, y]
            zone['polygon'][2] = [x + width, y + height]
            zone['polygon'][3] = [x, y + height]

        # Update polygon shape
        zone['polygon'] = np.array(zone['polygon'], np.int32)

    def save_zones(self, filepath: str):
        """
        Save current zone settings to a JSON file
        """
        # Prepare data to save
        zones_data = []
        for zone in self.zones:
            zone_data = {
                'lane_id': zone['lane_id'],
                'start_line': zone['start_line'],
                'end_line': zone['end_line'],
                'color': zone['color']
            }
            zones_data.append(zone_data)

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(zones_data, f, indent=4)

        print(f"Zones saved to {filepath}")

    def load_zones(self, filepath: str):
        """
        Load zone settings from a JSON file
        """
        try:
            with open(filepath, 'r') as f:
                zones_data = json.load(f)

            # Reset existing zones
            self.zones = []

            # Reconstruct from loaded zone data
            for zone_data in zones_data:
                start_line = tuple(zone_data['start_line'])
                end_line = tuple(zone_data['end_line'])

                x1, y1, x2, y1 = start_line
                x3, y2, x4, y2 = end_line

                polygon = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x4, y2],
                    [x3, y2]
                ], np.int32)

                zone = {
                    'lane_id': zone_data['lane_id'],
                    'start_line': start_line,
                    'end_line': end_line,
                    'polygon': polygon,
                    'color': tuple(zone_data['color']),
                    'speed': 0,
                    'vehicles': {},
                    'count': 0
                }

                self.zones.append(zone)

            print(f"Zones loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading zones: {e}")
            return False