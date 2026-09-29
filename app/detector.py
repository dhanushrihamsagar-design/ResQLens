import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from qai_hub_models.models.yolov8_det import Model


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_detector():
    """Load Qualcomm AI Hub YOLOv8-N."""
    model = Model.from_pretrained()
    model.eval()
    model.include_postprocessing = True
    return model


def prepare_image(image):
    """Convert image to YOLOv8-N input format."""
    image = image.convert("RGB")

    original_width, original_height = image.size

    resized = image.resize((640, 640))

    array = np.asarray(resized).astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    return tensor, (original_width, original_height)


def detect_objects(
    image,
    confidence_threshold=0.35,
    iou_threshold=0.50,
    model=None
):
    """
    Run YOLOv8-N object detection.

    Returns a clean list of detections after
    confidence filtering and Non-Maximum Suppression.
    """

    if model is None:
        model = load_detector()

    input_tensor, original_size = prepare_image(image)

    with torch.no_grad():
        boxes, scores, class_indices = model(input_tensor)

    boxes = boxes[0]
    scores = scores[0]
    class_indices = class_indices[0]

    # Confidence filtering first
    confidence_mask = scores >= confidence_threshold

    boxes = boxes[confidence_mask]
    scores = scores[confidence_mask]
    class_indices = class_indices[confidence_mask]

    if len(boxes) == 0:
        return []

    # Class-aware NMS
    keep_indices = []

    for class_id in torch.unique(class_indices):
        class_mask = class_indices == class_id

        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        original_indices = torch.where(class_mask)[0]

        kept = nms(
            class_boxes,
            class_scores,
            iou_threshold
        )

        keep_indices.append(original_indices[kept])

    if keep_indices:
        keep_indices = torch.cat(keep_indices)
    else:
        return []

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_indices = class_indices[keep_indices]

    # Convert coordinates back to original image size
    original_width, original_height = original_size

    scale_x = original_width / 640.0
    scale_y = original_height / 640.0

    detections = []

    for box, score, class_index in zip(
        boxes,
        scores,
        class_indices
    ):
        class_id = int(class_index.item())

        if 0 <= class_id < len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id]
        else:
            class_name = f"class_{class_id}"

        x1, y1, x2, y2 = [
            float(value.item())
            for value in box
        ]

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        detections.append({
            "class": class_name,
            "class_id": class_id,
            "confidence": float(score.item()),
            "box": [
                x1,
                y1,
                x2,
                y2
            ]
        })

    # Highest-confidence detections first
    detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True
    )

    return detections