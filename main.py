import pandas as pd
from io import StringIO

from ultralytics import YOLO
import os

# Load a model
model = YOLO("runs/detect/train8/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["image-test/89b173e2586fe231bb7e.jpg"])  # return a list of Results objects
csv_result = results[0].to_csv()

#counting
csv_buffer = StringIO(csv_result)
data = pd.read_csv(csv_buffer)
class_name = data.iloc[:, 1]
total_cls = class_name.value_counts()
print(total_cls)

# #save_drop
# for result in results:
#     result.save_crop(save_dir="output", file_name="detection")

#Process results list
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()# display to screen
    # output_dir = 'output'
    # os.makedirs(output_dir, exist_ok=False)
    # result.save(os.path.join(output_dir, "result.jpg"))  # save to disk
