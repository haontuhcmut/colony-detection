from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from PIL import Image
import os
import numpy as np
import torch

# Set the path to the image and model
IMAGE_PATH = r"image-test/89b173e2586fe231bb7e.jpg" #change path
MODEL_PATH = r"runs/detect/train8/weights/best.pt" #change path

# Upload an image
image = Image.open(IMAGE_PATH)

# Initialize the model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Performing predictions with slicing the image into 512x512 parts
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.5,
    overlap_width_ratio=0.5
)

# Check if masks are present in the predictions
if not result.object_prediction_list:
    print("No predictions found. Please check your model and input image.")
else:
    print(f"Found {len(result.object_prediction_list)} predictions.")

# Counting of ObjectPrediction
class_count = {}
for prediction in result.object_prediction_list:
    category = prediction.category
    class_name = category.name
    if class_name in class_count:
        class_count[class_name] += 1
    else:
        class_count[class_name] = 1
for class_name, count in class_count.items():
    print(f"Class: {class_name}, Count: {count}")

# Saving the results
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
result.export_visuals(export_dir=output_dir, file_name='detection_image.png')

# Initialize the dictionary for merged masks by class
combined_masks_by_class = {}

# Export masks and combine them by class
for prediction in result.object_prediction_list:
    mask = prediction.mask
    category_name = prediction.category.name
    if mask is not None:
        mask_array = mask.bool_mask
        # Scale the mask to the size of the original image
        mask_image = Image.fromarray((mask_array.astype(np.uint8) * 255))
        mask_resized = mask_image.resize((image.width, image.height), resample=Image.NEAREST)
        mask_resized_np = np.array(mask_resized)

        # Add a mask to the appropriate class in the dictionary
        if category_name not in combined_masks_by_class:
            combined_masks_by_class[category_name] = np.zeros((image.height, image.width), dtype=np.uint8)
        combined_masks_by_class[category_name] = np.maximum(combined_masks_by_class[category_name], mask_resized_np)

# Store merged masks by class
for category_name, combined_mask in combined_masks_by_class.items():
    combined_mask_image = Image.fromarray(combined_mask)
    combined_mask_image.save(os.path.join(output_dir, f'combined_mask_{category_name}.png'))

print(f"Results saved in {output_dir}")
