#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


PERSON_CLASS_ID = 1  # Index in COCO labelmap (0 = background)


class PersonDetectorNode(Node):

    def __init__(self):
        super().__init__('person_detector')

        # Parameters
        self.declare_parameter('camera_topic', '/drone/camera')
        self.declare_parameter('model_path', '/data/projects/udl-raayon/models/mobilenet_ssd.tflite')
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('publish_debug_image', True)

        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.publish_debug = self.get_parameter('publish_debug_image').value

        # TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input shape expected by model
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

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

        self.get_logger().info(
            f'Params: camera_topic={camera_topic} model={model_path} '
            f'confidence={self.confidence_threshold} publish_debug={self.publish_debug}'
        )

        self.get_logger().info(f'Person detector ready. Subscribing to: {camera_topic}')
        self.get_logger().info(f'Model input size: {self.input_width}x{self.input_height}')

    def image_callback(self, msg: Image):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h, w = frame.shape[:2]

        # Preprocess
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0)  # (1, H, W, 3)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Parse outputs (MobileNet SSD standard output order)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]       # [ymin, xmin, ymax, xmax] normalised
        class_ids = self.interpreter.get_tensor(self.output_details[1]['index'])[0]   # class ids
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]      # confidences

        # Build Detection2DArray
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if self.publish_debug else None

        for i in range(len(scores)):
            if scores[i] < self.confidence_threshold:
                continue
            if int(class_ids[i]) + 1 != PERSON_CLASS_ID:  # +1 because model outputs 0-indexed
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            # Denormalise to pixel coords
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)

            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bw = float(x2 - x1)
            bh = float(y2 - y1)

            # Build detection message
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = bw
            det.bbox.size_y = bh

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'person'
            hypothesis.hypothesis.score = float(scores[i])
            det.results.append(hypothesis)

            detection_array.detections.append(det)

            # Draw on debug frame
            if self.publish_debug:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'person {scores[i]:.2f}'
                cv2.putText(debug_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.get_logger().info(f'Publishing {len(detection_array.detections)} person detections')
        for det in detection_array.detections:
            self.get_logger().info(f'  x={det.bbox.center.position.x:.1f}, y={det.bbox.center.position.y:.1f}')

        self.detection_pub.publish(detection_array)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
