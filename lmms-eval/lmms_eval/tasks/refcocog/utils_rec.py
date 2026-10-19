import logging
import os
import re

from datasets import Dataset

eval_logger = logging.getLogger("lmms-eval")

COCO_REC_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]

# Bbox coordinate format: MUST be set via BBOX_COORD_FORMAT env var by the
# evaluation script (run_eval.py) or shell before invoking lmms-eval.
# Strict: process_result raises ValueError if not one of the three below.
# "pixel"      = pixel coords, divide by image dims (Qwen2.5-VL)
# "scale_1000" = 0-1000 normalized, divide by 1000 (Qwen2-VL, Qwen3-VL, Qwen3.5)
# "normalized" = already [0,1], no conversion (LLaVA)
BBOX_COORD_FORMAT = os.environ.get("BBOX_COORD_FORMAT", "pixel")

# Prompt style: controls the user instruction sent to the model.
# "full"  = lmms-eval default Shikra/KOSMOS-2 prompt (long; says "bounded 0-1")
# "short" = trimmed — drop format spec, keep the ask + caption
# "nuwa"  = Nuwa official Qwen grounding prompt: "Locate X ... JSON format."
REFCOCO_PROMPT_STYLE = os.environ.get("REFCOCO_PROMPT_STYLE", "nuwa")


def refcoco_bbox_rec_preprocess_dataset(dataset: Dataset):
    # PIL image stored in dataset['image']
    # add `image_width` and `image_height` to the dataset
    dataset = dataset.map(lambda x: {"image_width": x["image"].width, "image_height": x["image"].height})

    # Original bbox format (top x, top y, width, height)
    # Convert to (top-left x, top-left y, bottom-right x, bottom-right y)
    # Normalize the bounding box coordinates to be between 0 and 1
    # using the image width and height
    dataset = dataset.map(lambda x: {"bbox": [x["bbox"][0] / x["image_width"], x["bbox"][1] / x["image_height"], (x["bbox"][0] + x["bbox"][2]) / x["image_width"], (x["bbox"][1] + x["bbox"][3]) / x["image_height"]]})

    # currently, the dataset has `answer` as a list of strings
    # each answer should be its own row
    # we will explode the dataset to have one row per answer
    # duplicate the other columns
    def explode_answers(example):
        answers = example.pop("answer")
        return [{"answer": answer, **example} for answer in answers]

    # Apply the function to each element, collecting the results
    exploded_rows = []
    for example in dataset:
        exploded_rows.extend(explode_answers(example))

    # Create a new dataset from the exploded rows
    new_dataset = Dataset.from_list(exploded_rows)
    print(f"Exploded dataset from {len(dataset)} to {len(new_dataset)} rows")

    return new_dataset


def refcoco_bbox_rec_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def refcoco_bbox_rec_doc_to_text(doc):
    assert isinstance(doc["answer"], str), "Answer must be a string"
    caption = doc["answer"]
    style = os.environ.get("REFCOCO_PROMPT_STYLE", REFCOCO_PROMPT_STYLE)
    if style == "short":
        return "Please provide the bounding box coordinate of the region this sentence describes: " + caption
    if style == "nuwa":
        return f"Locate {caption} in this image and output the bbox coordinates in JSON format."
    return (
        "Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y). All values are floating point numbers bounded between 0 and 1. Please provide the bounding box coordinate of the region this sentence describes: "
        + doc["answer"]
    )


def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # Also try parentheses format (some models like Qwen3-VL output (x, y, x, y))
    pattern_paren = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern_paren, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]


def refcoco_bbox_rec_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    raw_pred = result[0] if len(result) > 0 else ""
    pred = parse_float_sequence_within(raw_pred)

    # Normalize non-[0,1] coordinates to [0,1].
    # BBOX_COORD_FORMAT must be set explicitly (by run_eval.py or shell).
    # No silent heuristic fallback: wrong format on a model silently produces
    # garbage IoU, which is how Gate-0.4 hit 0% ACC@0.5 before. Fail loudly.
    #   "pixel"      -> divide by image dims (Qwen2.5-VL)
    #   "scale_1000" -> divide by 1000 (Qwen2-VL, Qwen3-VL, Qwen3.5)
    #   "normalized" -> already [0,1], no conversion (LLaVA)
    if BBOX_COORD_FORMAT not in ("pixel", "scale_1000", "normalized"):
        raise ValueError(
            "BBOX_COORD_FORMAT must be one of 'pixel'|'scale_1000'|'normalized'; "
            "got %r. Set it via run_eval.py model_family map or `export "
            "BBOX_COORD_FORMAT=...` in your shell before launching lmms-eval."
            % BBOX_COORD_FORMAT
        )
    if any(v > 1.0 for v in pred):
        w = doc.get("image_width", 0)
        h = doc.get("image_height", 0)
        if BBOX_COORD_FORMAT == "pixel" and w > 0 and h > 0:
            pred = [pred[0] / w, pred[1] / h, pred[2] / w, pred[3] / h]
        elif BBOX_COORD_FORMAT == "scale_1000":
            pred = [v / 1000.0 for v in pred]
        # "normalized" and any `v>1.0` survivor: model misbehavior — let it score 0

    ann_id = doc["question_id"]
    data_dict = {"answer": doc["answer"], "pred": pred, "ann_id": ann_id, "bbox": doc["bbox"]}
    return {f"refcoco_{metric}": data_dict for metric in COCO_REC_METRICS}


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def refcoco_bbox_rec_aggregation_result(results, metric):
    """
    Aggregate the results of the RefCOCO evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    scorers = {
        "IoU": compute_iou,
        "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
        "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
        "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
        "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
        "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
        "Center_ACC": compute_center_accuracy,
    }
    results_dict = {metric: []}
    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt_bbox = result["bbox"]
        pred_bbox = result["pred"]
        # Compute the specified metric between the ground truth and predicted bounding boxes
        score = scorers[metric](gt_bbox, pred_bbox)
        results_dict[metric].append(score)
    results_dict[metric] = sum(results_dict[metric]) / len(results_dict[metric])
    print(f"Aggregated {metric} score: {results_dict[metric]}")
    return results_dict[metric]


def refcoco_bbox_rec_iou(results):
    return refcoco_bbox_rec_aggregation_result(results, "IoU")


def refcoco_bbox_rec_acc01(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.1")


def refcoco_bbox_rec_acc03(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.3")


def refcoco_bbox_rec_acc05(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.5")


def refcoco_bbox_rec_acc07(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.7")


def refcoco_bbox_rec_acc09(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.9")


def refcoco_bbox_rec_center_acc(results):
    return refcoco_bbox_rec_aggregation_result(results, "Center_ACC")
