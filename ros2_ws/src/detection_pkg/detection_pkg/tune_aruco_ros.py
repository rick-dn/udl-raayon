#!/usr/bin/env python3
"""
Diagnostic: subscribe to a ROS2 image topic, sweep ArUco DetectorParameters
against a fixed number of incoming frames, then print an aggregated
accept/reject summary per parameter set and exit.

Usage:
    ros2 run <your_pkg> tune_aruco_ros.py --ros-args \
        -p camera_topic:=/camera -p num_frames:=100
"""
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# Each entry: (label, dict of param overrides applied on top of defaults)
PARAM_SETS = [
    ('DEFAULT', {}),
    ('maxMarkerPerimeterRate=8.0', {'maxMarkerPerimeterRate': 8.0}),          # default 4.0
    ('adaptiveThreshWinSizeMax=53', {'adaptiveThreshWinSizeMax': 53}),        # default 23
    ('cornerRefinementMethod=SUBPIX', {'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX}),
    ('ALL COMBINED', {
        'maxMarkerPerimeterRate': 8.0,
        'adaptiveThreshWinSizeMax': 53,
        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
    }),
    ('ALL COMBINED + looser polygon approx', {
        'maxMarkerPerimeterRate': 8.0,
        'adaptiveThreshWinSizeMax': 53,
        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'polygonalApproxAccuracyRate': 0.05,   # default 0.03
    }),
]

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def build_params(overrides):
    p = cv2.aruco.DetectorParameters()
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def try_detect(gray, params):
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
    corners, ids, rejected = detector.detectMarkers(gray)
    found = ids.flatten().tolist() if ids is not None else []
    return found, len(rejected)


class ArucoTuneNode(Node):

    def __init__(self):
        super().__init__('aruco_tune')

        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('num_frames', 100)

        camera_topic = self.get_parameter('camera_topic').value
        self.num_frames = self.get_parameter('num_frames').value

        self.bridge = CvBridge()
        self.frame_count = 0

        # Per-param-set aggregate stats
        self.stats = {
            label: {'accepted': 0, 'ids_seen': set(), 'total_rejected_candidates': 0}
            for label, _ in PARAM_SETS
        }

        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        self.get_logger().info(f'Subscribed to {camera_topic}')
        self.get_logger().info(f'Collecting {self.num_frames} frames, then printing summary and exiting.')

    def image_callback(self, msg: Image):
        if self.frame_count >= self.num_frames:
            return  # already done, waiting for shutdown

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1

        for label, overrides in PARAM_SETS:
            params = build_params(overrides)
            found, n_rejected = try_detect(gray, params)
            s = self.stats[label]
            if found:
                s['accepted'] += 1
                s['ids_seen'].update(found)
            s['total_rejected_candidates'] += n_rejected

        if self.frame_count % 10 == 0 or self.frame_count == self.num_frames:
            self.get_logger().info(f'Processed {self.frame_count}/{self.num_frames} frames...')

        if self.frame_count >= self.num_frames:
            self.print_summary()
            rclpy.shutdown()

    def print_summary(self):
        n = self.frame_count
        print(f'\n=== Summary over {n} frames ===\n')
        print(f'{"Parameter set":42s} {"Accept rate":>12s} {"IDs seen":>12s} {"Avg rejected cand.":>20s}')
        print('-' * 92)
        for label, _ in PARAM_SETS:
            s = self.stats[label]
            rate = 100.0 * s['accepted'] / n
            ids_str = str(sorted(s['ids_seen'])) if s['ids_seen'] else '[]'
            avg_rejected = s['total_rejected_candidates'] / n
            print(f'{label:42s} {rate:11.1f}% {ids_str:>12s} {avg_rejected:20.2f}')
        print()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTuneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
