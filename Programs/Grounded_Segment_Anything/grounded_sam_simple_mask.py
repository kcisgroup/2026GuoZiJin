import cv2
import os
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from Grounded_Segment_Anything.segment_anything.segment_anything import sam_model_registry, SamPredictor,sam_hq_model_registry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def g_sam(image_path,text,config_path,checkpoint_path):

    #GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    #GROUNDING_DINO_CHECKPOINT_PATH = "/home/gzj/test/BrushNetSimple-main/grounded_sam/models/groundingdino/groundingdino_swinb_cogcoor.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "/home/gzj/test/BrushNetSimple-main/models/sams/sam_hq_vit_h.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=checkpoint_path)


    # Building SAM Model and SAM Predictor
    sam = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    #predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))

    # Predict classes and hyper-param for GroundingDINO
    SOURCE_IMAGE_PATH =image_path
    CLASSES = [text]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8


    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    combined_mask = np.any(detections.mask, axis=0).astype(np.uint8) * 255

    #保存黑白蒙版
    cv2.imwrite("outputs/mask_image.png", combined_mask)
    print(combined_mask.size)
    #combined_mask = np.any(detections.mask, axis=0).astype(np.uint8) * 255
    return combined_mask

if __name__ == "__main__":

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "/home/gzj/test/BrushNetSimple-main/grounded_sam/models/groundingdino/groundingdino_swinb_cogcoor.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "/home/gzj/test/BrushNetSimple-main/grounded_sam/models/sams/sam_hq_vit_h.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


    # Building SAM Model and SAM Predictor
    sam = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    #predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))

    # Predict classes and hyper-param for GroundingDINO
    SOURCE_IMAGE_PATH = "/home/gzj/test/BrushNetSimple-main/grounded_sam/pic/1.png"
    CLASSES = ["face"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8


    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # Prompting SAM with detected boxes


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )


    combined_mask = np.any(detections.mask, axis=0).astype(np.uint8) * 255

    #保存黑白蒙版
    cv2.imwrite("outputs/mask_image.png", combined_mask)


