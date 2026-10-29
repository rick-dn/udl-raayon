#!/usr/bin/env python3
"""
Step through images, run the VisDrone person TFLite model on each and show the boxes.

Preprocessing and post-processing mirror person_detector_visdrone_node.py.

Controls:
    SPACE  next image
    ESC    quit

Usage:
    python view_detections.py
    python view_detections.py --images /path/to/images --conf 0.2
"""

import argparse
import glob
import os

import cv2
import numpy as np
from ai_edge_litert import interpreter as tflite


PERSON_CLASS_IDS = [0, 1]  # VisDrone: 0 = pedestrian, 1 = people
NMS_IOU_THRESHOLD = 0.45
KEY_SPACE = 32
KEY_ESC = 27


class Detector:

    def __init__(self, model_path: str, num_threads: int = 4):
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        # NCHW (1, 3, H, W)
        self.input_height = self.input_details['shape'][2]
        self.input_width = self.input_details['shape'][3]

    def detect(self, frame_bgr, conf: float):
        """Return a list of ((x1, y1, x2, y2), score) in pixel coords."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))[None]

        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()

        # (4 + num_classes, num_anchors), boxes normalised cx, cy, w, h
        output = self.interpreter.get_tensor(self.output_details['index'])[0]
        scores = output[4:][PERSON_CLASS_IDS].max(axis=0)
        keep = scores >= conf
        boxes_xywh = output[:4].T[keep]
        scores = scores[keep]

        nms_boxes = []
        for cx_n, cy_n, bw_n, bh_n in boxes_xywh:
            bw, bh = bw_n * w, bh_n * h
            nms_boxes.append([int(cx_n * w - bw / 2), int(cy_n * h - bh / 2), int(bw), int(bh)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf, NMS_IOU_THRESHOLD) if nms_boxes else []

        detections = []
        for i in np.array(indices).flatten():
            x1, y1, bw, bh = nms_boxes[i]
            detections.append(((x1, y1, x1 + bw, y1 + bh), float(scores[i])))
        return detections


def draw(frame, detections, title):
    out = frame.copy()
    for (x1, y1, x2, y2), score in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f'person {score:.2f}', (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='View VisDrone person detections one image at a time')
    parser.add_argument('--model', default=os.path.join(
        project_root, 'runs/detect/runs/detect/visdrone_person/weights/best_int8.tflite'))
    parser.add_argument('--images', default='/data/datasets/flight_pi_cam_vids/images/person',
                        help='Folder of .jpg/.png images (not recursive)')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold (default: 0.35)')
    parser.add_argument('--start', type=int, default=0, help='Index of the first image to show')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, '*.jpg')) +
                   glob.glob(os.path.join(args.images, '*.png')))
    if not files:
        raise SystemExit(f'No images found in {args.images}')

    detector = Detector(args.model)
    print(f'Model: {args.model} (input {detector.input_width}x{detector.input_height})')
    print(f'{len(files)} images, conf >= {args.conf}. SPACE = next, ESC = quit')

    window = 'detections'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for idx in range(args.start, len(files)):
        path = files[idx]
        frame = cv2.imread(path)
        if frame is None:
            print(f'Could not read {path}, skipping')
            continue

        detections = detector.detect(frame, args.conf)
        title = f'[{idx + 1}/{len(files)}] {os.path.basename(path)}  dets: {len(detections)}'
        print(title, ' '.join(f'{s:.2f}' for _, s in detections))
        cv2.imshow(window, draw(frame, detections, title))

        # Wait for SPACE or ESC; closing the window also quits
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == KEY_ESC or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return
            if key == KEY_SPACE:
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
