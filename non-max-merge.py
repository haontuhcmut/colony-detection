import math
import os
import time

import cv2
import numpy as np
import supervision as sv
from supervision import Detections

from ultralytics import YOLO
import pandas as pd


image = cv2.imread(r"/Users/haonguyen/PycharmProject/colony-detection/demo/9bcc9a4bd6396b6732284.jpg")
model = YOLO(r"/Users/haonguyen/PycharmProject/colony-detection/runs/detect/train8/weights/best.pt")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(
    # A function that performs inference on a given image slice and returns detections.
    callback=callback,
    # Strategy for filtering or merging overlapping detections in slices.
    overlap_filter=sv.OverlapFilter.NON_MAX_MERGE,
    # Dimensions of each slice measured in pixels. The tuple should be in the format (width, height).
    slice_wh=(256, 256),

    overlap_ratio_wh=None,

    overlap_wh=(5, 5),

    iou_threshold=0.5,
)

detections = slicer(image)

# Check if masks are present in the predictions
if not detections.data:
    print("No predictions found. Please check your model and input image.")
else:
    #counting
    unique_classes, counts = np.unique(detections['class_name'], return_counts=True)

    #dictionary
    class_counts = dict(zip(unique_classes, counts))

    #total
    total_count = sum(counts)

    #add total in dictionary
    class_counts['total'] = total_count

    #dataframe and export csv
    df = pd.DataFrame(list(class_counts.items()), columns=['Class Name', 'Count'])
    df.to_csv(r'/Users/haonguyen/PycharmProject/colony-detection/demo/class_counts.csv', index=False)
    print(f"Found {total_count} predictions.\nDetail {class_counts}.\nData exported to class_counts.csv")

############################

labels = [
    f"{class_name} {confidence:.1f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_frame = sv.BoxAnnotator().annotate(
    scene=image.copy(),
    detections=detections
)

label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=0, text_position=sv.Position.TOP_LEFT)
#annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

sv.plot_image(annotated_frame)

