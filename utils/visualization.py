"""
Visualization utilities (Highway CCTV optimized with edit mode UI)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from PIL import ImageFont, ImageDraw, Image

class Visualizer:
    def __init__(self, text_color: Tuple[int, int, int] = (255, 255, 255),
                 vehicle_box_color: Tuple[int, int, int] = (0, 255, 0),
                 zone_alpha: float = 0.4,
                 font_scale: float = 0.5,
                 line_thickness: int = 2,
                 show_confidence: bool = False,
                 show_vehicle_class: bool = True):
        """
        Initialize visualization tools

        Args:
            text_color: Text color
            vehicle_box_color: Vehicle box color
            zone_alpha: Measurement zone transparency
            font_scale: Font size scale
            line_thickness: Line thickness
            show_confidence: Whether to display confidence
            show_vehicle_class: Whether to display vehicle class
        """
        self.text_color = text_color
        self.vehicle_box_color = vehicle_box_color
        self.zone_alpha = zone_alpha
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.show_confidence = show_confidence
        self.show_vehicle_class = show_vehicle_class

        # Default font size
        self.font_size = int(16 * self.font_scale)
        try:
            # Try to load default font
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.font_size)
        except:
            try:
                # Try to load another common font
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
            except:
                # Use default font if font loading fails
                self.font = ImageFont.load_default()
                print("Could not find specific font, using default font.")

    def put_text_pil(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                 color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Display text on image using PIL

        Args:
            frame: Original image
            text: Text to display
            position: Text position (x, y)
            color: Text color

        Returns:
            Image with added text
        """
        if color is None:
            color = self.text_color

        # Convert NumPy array to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Draw text
        draw.text(position, text, font=self.font, fill=color[::-1])  # OpenCV BGR -> RGB color order change

        # Convert PIL image back to NumPy array
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_zones(self, frame: np.ndarray, zones: List[Dict], selected_zone: int = -1) -> np.ndarray:
        """
        Draw measurement zones

        Args:
            frame: Original frame
            zones: Zone information list
            selected_zone: Currently selected zone index (-1 if none)

        Returns:
            Frame with zones drawn
        """
        overlay = frame.copy()

        for i, zone in enumerate(zones):
            # Fill zone
            cv2.fillPoly(overlay, [zone['polygon']], zone['color'])

            # If this zone is selected, draw a highlight
            if i == selected_zone:
                # Draw thicker border for selected zone
                cv2.polylines(overlay, [zone['polygon']], True, (255, 255, 255), 3)

        # Apply transparency
        cv2.addWeighted(overlay, self.zone_alpha, frame, 1 - self.zone_alpha, 0, frame)

        return frame

    def draw_lane_info(self, frame: np.ndarray, lane_data: List[Dict]) -> np.ndarray:
        """
        Draw lane information directly on the frame without background panel

        Args:
            frame: Original frame
            lane_data: Lane data list

        Returns:
            Frame with information drawn
        """
        # Draw lane speed and count information directly on the frame
        for data in lane_data:
            lane_id = data['lane_id']
            speed = data.get('speed', 0)
            count = data.get('count', 0)

            # Use the same color as the lane zone
            color = data.get('color', self.text_color)

            # Lane speed text
            speed_text = f"Lane {lane_id} speed: {speed:.1f} km/h"

            # Vehicle count text
            count_text = f"Vehicle{lane_id}: {count}"

            # Position text at the top of the frame, aligned with each lane
            x_pos = 20 + (lane_id - 1) * 120  # Adjust horizontal spacing

            # Draw lane info with color matching the lane
            cv2.putText(frame, speed_text, (10, 30 + (lane_id - 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw count info on the right side
            text_width = len(count_text) * 8  # Approximate text width
            cv2.putText(frame, count_text, (frame.shape[1] - text_width - 10, 30 + (lane_id - 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def draw_vehicles(self, frame: np.ndarray, vehicles: List[Dict]) -> np.ndarray:
        """
        Draw detected vehicles

        Args:
            frame: Original frame
            vehicles: Detected vehicle list

        Returns:
            Frame with vehicles drawn
        """
        for vehicle in vehicles:
            x, y, w, h = vehicle['box']

            # Change color based on vehicle type
            box_color = self.vehicle_box_color
            if 'class_id' in vehicle:
                class_id = vehicle['class_id']
                if class_id == 2:  # Car
                    box_color = (0, 255, 0)  # Green
                elif class_id == 5:  # Bus
                    box_color = (0, 165, 255)  # Orange
                elif class_id == 7:  # Truck
                    box_color = (0, 0, 255)  # Red

            # Draw vehicle box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, self.line_thickness)

            # Display class and confidence
            label_parts = []

            if self.show_vehicle_class and 'class' in vehicle:
                label_parts.append(vehicle['class'])

            if self.show_confidence and 'confidence' in vehicle:
                label_parts.append(f"{vehicle['confidence']:.2f}")

            if label_parts:
                label = ": ".join(label_parts)
                # Draw text background
                label_width = len(label) * self.font_size // 2
                label_height = self.font_size + 5
                cv2.rectangle(
                    frame,
                    (x, y - label_height - 5),
                    (x + label_width, y),
                    box_color,
                    -1
                )
                frame = self.put_text_pil(frame, label, (x, y - label_height), (255, 255, 255))

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS information

        Args:
            frame: Original frame
            fps: Frames per second

        Returns:
            Frame with FPS information drawn
        """
        # FPS text background
        cv2.rectangle(frame, (10, frame.shape[0] - 40), (130, frame.shape[0] - 10), (0, 0, 0), -1)

        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        frame = self.put_text_pil(frame, fps_text, (15, frame.shape[0] - 35), (0, 255, 255))

        return frame

    def add_title(self, frame: np.ndarray, title: str) -> np.ndarray:
        """
        Add title

        Args:
            frame: Original frame
            title: Title text

        Returns:
            Frame with title added
        """
        # Title background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)

        # Title text (center aligned)
        title_pos = (frame.shape[1] // 2 - len(title) * self.font_size // 4, 10)
        frame = self.put_text_pil(frame, title, title_pos, (255, 255, 255))

        return frame

    def draw_edit_mode(self, frame: np.ndarray, edit_mode: bool, selected_zone: int = -1,
                        mouse_pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Draw edit mode UI

        Args:
            frame: Original frame
            edit_mode: Whether edit mode is active
            selected_zone: Currently selected zone index
            mouse_pos: Current mouse position

        Returns:
            Frame with edit mode UI drawn
        """
        if edit_mode:
            # Edit mode background
            cv2.rectangle(frame, (10, frame.shape[0] - 80), (250, frame.shape[0] - 45), (0, 0, 200), -1)

            # Edit mode text
            edit_text = f"EDIT MODE - Zone: {selected_zone if selected_zone >= 0 else 'None'}"
            frame = self.put_text_pil(frame, edit_text, (15, frame.shape[0] - 60), (255, 255, 255))

            # Help text
            cv2.rectangle(frame, (10, frame.shape[0] - 45), (380, frame.shape[0] - 10), (0, 0, 200), -1)
            help_text = "Press 'e': Toggle edit, 's': Save, 'l': Load"
            frame = self.put_text_pil(frame, help_text, (15, frame.shape[0] - 25), (255, 255, 255))

            # Mouse position
            mouse_text = f"Mouse: ({mouse_pos[0]}, {mouse_pos[1]})"
            cv2.rectangle(frame, (frame.shape[1] - 200, frame.shape[0] - 40),
                         (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 200), -1)
            frame = self.put_text_pil(frame, mouse_text,
                                     (frame.shape[1] - 190, frame.shape[0] - 25), (255, 255, 255))

        return frame