#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np


class GreenDetectorNode(Node):

    def __init__(self):
        super().__init__('green_detector')

        # Parameters
        self.declare_parameter('camera_topic', '/drone/camera')
        self.declare_parameter('min_area', 500.0)
        self.declare_parameter('publish_debug_image', True)

        camera_topic = self.get_parameter('camera_topic').value
        self.min_area = self.get_parameter('min_area').value
        self.publish_debug = self.get_parameter('publish_debug_image').value

        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections/persons',
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

        self.get_logger().info(f'Green detector ready. Subscribing to: {camera_topic}')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # HSV thresholding for bright green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 150, 150])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        debug_frame = frame.copy() if self.publish_debug else None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(largest)
                cx = float(x + bw / 2.0)
                cy = float(y + bh / 2.0)
                self.get_logger().info(f'Detected green object at ({cx:.1f}, {cy:.1f}) bbox=({bw}x{bh})')

                det = Detection2D()
                det.header = msg.header
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = float(bw)
                det.bbox.size_y = float(bh)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'green_object'
                hypothesis.hypothesis.score = 1.0
                det.results.append(hypothesis)

                detection_array.detections.append(det)

                if self.publish_debug:
                    cv2.rectangle(debug_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    label = 'green_object 1.00'
                    cv2.putText(debug_frame, label, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.detection_pub.publish(detection_array)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GreenDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
