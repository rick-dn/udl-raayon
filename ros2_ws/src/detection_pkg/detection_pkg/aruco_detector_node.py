#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # Parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('publish_debug_image', True)
        # self.declare_parameter('adaptive_thresh_win_size_max', 23)

        camera_topic = self.get_parameter('camera_topic').value
        self.publish_debug = self.get_parameter('publish_debug_image').value

        # win_max = self.get_parameter('adaptive_thresh_win_size_max').value

        # ArUco detector (OpenCV >= 4.7 API)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        # detector_params.adaptiveThreshWinSizeMax = win_max
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections/aruco',
            10
        )

        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image,
                '/detections/debug_image',
                10
            )

        # Subscriber
        self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(
            f'Params: camera_topic={camera_topic} dict=DICT_4X4_50 '
            f'publish_debug={self.publish_debug}'
        )
        self.get_logger().info(f'ArUco detector ready. Subscribing to: {camera_topic}')

    def image_callback(self, msg: Image):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # ArUco works on greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if self.publish_debug else None

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                pts = marker_corners.reshape(4, 2)  # TL, TR, BR, BL

                # Axis-aligned bbox from the four corners
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                bw = float(x2 - x1)
                bh = float(y2 - y1)

                # Yaw in image plane: angle of the top edge (TL -> TR)
                dx = float(pts[1][0] - pts[0][0])
                dy = float(pts[1][1] - pts[0][1])
                yaw = math.atan2(dy, dx)  # radians, image coords (y down)
                yaw = -yaw

                # Build detection message
                det = Detection2D()
                det.header = msg.header
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = bw
                det.bbox.size_y = bh

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = f'aruco_{int(marker_id)}'
                hypothesis.hypothesis.score = 1.0  # ArUco is binary: found or not

                # Yaw packed as a quaternion about Z
                hypothesis.pose.pose.orientation.z = math.sin(yaw / 2.0)
                hypothesis.pose.pose.orientation.w = math.cos(yaw / 2.0)

                det.results.append(hypothesis)
                detection_array.detections.append(det)

                # Draw on debug frame
                if self.publish_debug:
                    cv2.polylines(
                        debug_frame,
                        [pts.astype(np.int32)],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2
                    )
                    label = f'id={int(marker_id)} yaw={math.degrees(yaw):.1f}'
                    cv2.putText(debug_frame, label, (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.circle(debug_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        for det in detection_array.detections:
            self.get_logger().info(f'Publishing {len(detection_array.detections)} aruco detections')
            cid = det.results[0].hypothesis.class_id
            q = det.results[0].pose.pose.orientation
            yaw_deg = math.degrees(2.0 * math.atan2(q.z, q.w))
            self.get_logger().info(
                f'  {cid}: x={det.bbox.center.position.x:.1f}, '
                f'y={det.bbox.center.position.y:.1f}, '
                f'size=({det.bbox.size_x:.0f}x{det.bbox.size_y:.0f}), '
                f'yaw={yaw_deg:.1f}deg'
            )

        self.detection_pub.publish(detection_array)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
