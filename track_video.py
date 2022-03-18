import argparse
import math
import os
import random
from collections import deque

import cv2
import numpy as np
import torch

from sort import Sort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", type=str, help="path to model checkpoint")
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-o", "--output", type=str, help="path to output video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    ap.add_argument(
        "--conf_thr", type=float, default=0.3, help="YOLOv5 confidence threshold"
    )
    ap.add_argument("--iou_thr", type=float, default=0.45, help="YOLOv5 IoU threshold")
    ap.add_argument(
        "--max_age",
        type=int,
        default=1,
        help="maximum number of frames to keep alive a track without associated detections",
    )
    ap.add_argument(
        "--min_hits",
        type=int,
        default=3,
        help="minimum number of associated detections before track is initialised",
    )
    ap.add_argument("--seed", type=int, default=8, help="random seed")

    args = vars(ap.parse_args())

    return args


def get_label_names(model):
    return model.module.names if hasattr(model, "module") else model.names


def create_colormap(names, seed):
    random.seed(seed)
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def prepare_model(weights_path, conf_thr, iou_thr):
    model = torch.hub.load("ultralytics/yolov5", "custom", weights_path)

    model.float()
    model.eval()

    model.conf = conf_thr
    model.iou = iou_thr

    return model


def main():
    # SETUP
    args = parse_args()

    model = prepare_model(
        weights_path=args["model_path"],
        conf_thr=args["conf_thr"],
        iou_thr=args["iou_thr"],
    )

    names = get_label_names(model)
    colors = create_colormap(names, args["seed"])

    cap = cv2.VideoCapture(args["video"])

    W = int(cap.get(3))
    H = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args["output"], fourcc, 30, (W, H))

    tracker = Sort(max_age=args["max_age"], min_hits=args["min_hits"])

    pts = [deque(maxlen=args["buffer"]) for _ in range(1000)]
    # Run model on video
    while True:
        frame = cap.read()
        frame = frame[1]

        if frame is None:
            break
        # Run model on frame
        preds = model(frame)
        dets = preds.pred[0].cpu().numpy()
        # Use tracker
        track_bbs_ids = tracker.update(dets)
        # Loop through tracked boxes
        for i in range(len(track_bbs_ids)):
            coords = track_bbs_ids[i]
            xmin, ymin, xmax, ymax = (
                int(coords[0]),
                int(coords[1]),
                int(coords[2]),
                int(coords[3]),
            )
            name_idx = int(coords[4])
            label = names[int(coords[5])]
            color = colors[int(coords[5])]
            center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            pts[name_idx].append(center)
            # Draw bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                frame,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            # Draw track
            for j in range(1, len(pts[name_idx])):
                if pts[name_idx][j - 1] is None or pts[name_idx][j] is None:
                    continue
                thickness = int(np.sqrt(args["buffer"] * float(j + 1)) * 0.1)
                cv2.line(
                    frame,
                    (pts[name_idx][j - 1]),
                    (pts[name_idx][j]),
                    color,
                    thickness,
                )
        # Save video to file
        out.write(frame)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release video
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
