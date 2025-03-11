#!/usr/bin/env python3
"""
TrafficFlow: Real-time Vehicle Speed Detection System
Main application entry point (Highway CCTV optimized version with interactive zone editing)
"""

import cv2
import time
import argparse
import os

from models.yolo_detector import YOLODetector
from processors.video_processor import VideoProcessor
from processors.lane_processor import LaneProcessor
from utils.visualization import Visualizer
from config import YOLO_CONFIG, VIDEO_CONFIG, SPEED_CONFIG, DISPLAY_CONFIG, UI_CONFIG

# Global variables for edit mode
edit_mode = False
selected_zone = -1
drag_point = None
mouse_pos = (0, 0)
last_mouse_x, last_mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for interactive editing
    """
    global edit_mode, selected_zone, drag_point, mouse_pos, last_mouse_x, last_mouse_y
    lane_processor = param

    if not edit_mode:
        return

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 버튼 누를 때: 존 선택
        selected_zone = -1  # 선택 초기화
        for i, zone in enumerate(lane_processor.zones):
            polygon = zone['polygon'].reshape(-1, 2)
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                selected_zone = i

                # 드래그 포인트 결정 (시작선, 종료선, 또는 전체)
                start_y = zone['start_line'][1]
                end_y = zone['end_line'][1]

                if abs(y - start_y) < 15:
                    drag_point = 'start'
                elif abs(y - end_y) < 15:
                    drag_point = 'end'
                else:
                    drag_point = 'all'  # 중간 부분을 클릭하면 전체 이동

                print(f"Selected zone {selected_zone}, drag point: {drag_point}")

                # 마우스 클릭 위치 저장
                last_mouse_x, last_mouse_y = x, y
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동 시: 선택된 존이 있으면 이동
        if selected_zone >= 0 and drag_point:
            # 마우스 이동량 계산
            dx = x - last_mouse_x
            dy = y - last_mouse_y

            # 존 데이터 가져오기
            zone = lane_processor.zones[selected_zone]

            if drag_point == 'start':
                # 시작선만 이동
                x1, y1, x2, y2 = zone['start_line']
                zone['start_line'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                # 다각형 업데이트
                zone['polygon'][0][0] += dx
                zone['polygon'][0][1] += dy
                zone['polygon'][1][0] += dx
                zone['polygon'][1][1] += dy

            elif drag_point == 'end':
                # 종료선만 이동
                x1, y1, x2, y2 = zone['end_line']
                zone['end_line'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                # 다각형 업데이트
                zone['polygon'][2][0] += dx
                zone['polygon'][2][1] += dy
                zone['polygon'][3][0] += dx
                zone['polygon'][3][1] += dy

            elif drag_point == 'all':
                # 전체 존 이동
                x1, y1, x2, y2 = zone['start_line']
                zone['start_line'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

                x1, y1, x2, y2 = zone['end_line']
                zone['end_line'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

                # 다각형 전체 업데이트
                for i in range(4):
                    zone['polygon'][i][0] += dx
                    zone['polygon'][i][1] += dy

            # 마우스 위치 업데이트
            last_mouse_x, last_mouse_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 버튼 뗄 때: 드래그 종료
        if selected_zone >= 0:
            print(f"Updated zone {selected_zone}")
            # 마우스 버튼을 떼도 선택 상태는 유지 (다시 클릭할 때까지)


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrafficFlow: Real-time Vehicle Speed Detection System')

    parser.add_argument('--stream_url', type=str,
                        default=VIDEO_CONFIG['default_stream_url'],
                        help='Video stream URL')

    parser.add_argument('--record', action='store_true',
                        help='Enable video recording')

    parser.add_argument('--output_dir', type=str,
                        default=VIDEO_CONFIG['output_dir'],
                        help='Output directory path')

    parser.add_argument('--show_ui', action='store_true', default=True,
                        help='Show UI')

    parser.add_argument('--night_mode', action='store_true',
                        default=DISPLAY_CONFIG.get('night_mode', False),
                        help='Enable night mode')

    parser.add_argument('--use_predefined_zones', action='store_true', default=True,
                        help='Use predefined measurement zones')

    parser.add_argument('--zones_file', type=str, default="zones.json",
                        help='Path to zones configuration file')

    return parser.parse_args()


def main():
    """
    Main function
    """
    global edit_mode, selected_zone, drag_point, mouse_pos

    # Parse arguments
    args = parse_arguments()

    # Set output directory
    output_dir = args.output_dir if args.record else None

    # Initialize video processor
    video_processor = VideoProcessor(
        stream_url=args.stream_url,
        process_every_n_frames=VIDEO_CONFIG['process_every_n_frames'],
        output_dir=output_dir,
        resize_factor=VIDEO_CONFIG.get('resize_factor', 1.0)
    )

    # Open video stream
    if not video_processor.open_stream():
        print("Could not open video stream. Exiting program.")
        return

    # Get frame size information
    frame_width, frame_height = video_processor.get_frame_dimensions()

    # Initialize YOLO vehicle detector
    detector = YOLODetector(
        weights_path=YOLO_CONFIG['weights'],
        config_path=YOLO_CONFIG['config'],
        names_path=YOLO_CONFIG['names'],
        confidence_threshold=YOLO_CONFIG['confidence_threshold'],
        nms_threshold=YOLO_CONFIG['nms_threshold'],
        target_classes=YOLO_CONFIG['target_classes'],
        input_size=(320, 320),  # Smaller input size for better performance
        enable_night_mode=args.night_mode
    )

    # Initialize lane processor
    lane_processor = LaneProcessor(
        frame_width=frame_width,
        frame_height=frame_height,
        real_distance_meters=SPEED_CONFIG['real_distance_meters'],
        max_matching_distance=SPEED_CONFIG['max_matching_distance'],
        line_crossing_threshold=SPEED_CONFIG['line_crossing_threshold'],
        colors=DISPLAY_CONFIG['lane_colors'],
        use_predefined_zones=args.use_predefined_zones
    )

    # Try to load zones from file if it exists
    if os.path.exists(args.zones_file):
        lane_processor.load_zones(args.zones_file)
        print(f"Loaded zones from {args.zones_file}")

    # Initialize visualizer
    visualizer = Visualizer(
        text_color=DISPLAY_CONFIG['text_color'],
        vehicle_box_color=DISPLAY_CONFIG['vehicle_box_color'],
        zone_alpha=DISPLAY_CONFIG['zone_alpha'],
        font_scale=UI_CONFIG.get('font_scale', 0.5),
        line_thickness=UI_CONFIG.get('line_thickness', 2),
        show_confidence=UI_CONFIG.get('show_confidence', False),
        show_vehicle_class=UI_CONFIG.get('show_vehicle_class', True)
    )

    print(f"TrafficFlow started. Stream: {args.stream_url}")
    print("Controls:")
    print("  'q' : Quit")
    print("  'n' : Toggle night mode")
    print("  'e' : Toggle edit mode")
    print("  's' : Save zone configuration")
    print("  'l' : Load zone configuration")

    # Vehicle list (global)
    vehicles = []

    # Create window and set mouse callback
    cv2.namedWindow("TrafficFlow")
    cv2.setMouseCallback("TrafficFlow", mouse_callback, lane_processor)

    try:
        while True:
            # Read frame
            ret, frame = video_processor.read_frame()

            if not ret:
                print("Stream ended")
                break

            # Process selected frames only (performance optimization)
            if video_processor.should_process_frame():
                # Detect vehicles
                vehicles = detector.detect(frame)

                # Update vehicle tracking and speed measurement
                current_time = time.time()
                lane_processor.update_vehicle_tracking(vehicles, current_time)

            # Visualization
            if args.show_ui:
                # Draw zones like in the reference image
                frame = visualizer.draw_zones(frame, lane_processor.get_zone_polygons(), selected_zone)

                # Draw lane info directly on frame with matching colors
                frame = visualizer.draw_lane_info(frame, lane_processor.get_lane_data())

                # Draw vehicles (on processed frames only)
                if video_processor.should_process_frame():
                    frame = visualizer.draw_vehicles(frame, vehicles)

                # Show FPS in bottom left corner
                cv2.putText(frame, f"FPS: {video_processor.get_fps():.1f}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Add Korean location text if needed
                cv2.putText(frame, "동호대교", (frame.shape[1] // 4, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "성동교", (frame.shape[1] * 3 // 4, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Only show edit mode indicator when in edit mode
                if edit_mode:
                    frame = visualizer.draw_edit_mode(frame, edit_mode, selected_zone, mouse_pos)

                # Display result
                cv2.imshow("TrafficFlow", frame)

            # Recording
            if args.record:
                video_processor.write_frame(frame)

            # Key input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit
                break
            elif key == ord('n'):
                # Toggle night mode
                detector.enable_night_mode = not detector.enable_night_mode
                print(f"Night mode: {'Enabled' if detector.enable_night_mode else 'Disabled'}")
            elif key == ord('e'):
                # Toggle edit mode
                edit_mode = not edit_mode
                if edit_mode:
                    print("Edit mode enabled. Click and drag zones to reposition.")
                else:
                    # Reset edit variables when exiting edit mode
                    selected_zone = -1
                    drag_point = None
                    print("Edit mode disabled.")
            elif key == ord('s') and edit_mode:
                # Save zones
                lane_processor.save_zones(args.zones_file)
            elif key == ord('l'):
                # Load zones
                lane_processor.load_zones(args.zones_file)
                # Reset edit variables
                selected_zone = -1
                drag_point = None

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        # Release resources
        video_processor.release()
        print("Program terminated.")


if __name__ == "__main__":
    main()