from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

# HaMeR imports
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from vitpose_model import ViTPoseModel
from hamer.utils.skeleton_renderer import SkeletonRenderer

import json
from typing import Dict, Optional

import utils as utils

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument("--img_path", help="""Filepath of .jpg or .png image to load""")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    args = parser.parse_args()

    # Download and load checkpoints
    checkpoint = "/juno/u/clairech/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "/juno/u/clairech/hamer/_DATA/detectron_ckpts/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    keypoints_renderer = SkeletonRenderer(model_cfg)

    img_cv2 = cv2.imread(str(args.img_path))

    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores=det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp # [21,3], pixel coords
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        print("No hands detected")
        quit()

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        #print(out["pred_keypoints_3d"]) # [B, 21, 3]

        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2*batch['right']-1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        pred_keypoints_3d = out['pred_keypoints_3d'] # [B, 21, 3]
        world_landmarks = pred_keypoints_3d.detach().cpu().numpy()[0]

        #utils.save_as_pcd(
        #    pred_keypoints_3d[0].detach().cpu().numpy(), "test.ply"
        #)

        # Get keypoints in hand frame
        H_hand_to_world = utils.get_H_hand_to_world(world_landmarks)
        H_world_to_hand = utils.get_H_inv(H_hand_to_world)
        landmarks_hand_frame = utils.transform_pts(world_landmarks, H_world_to_hand)

        utils.save_as_pcd(
            landmarks_hand_frame, "test.ply"
        )

        # Draw hand keypoints on image

        print("Rendering keypoints on image")

        # Render the result
        pred_keypoints_3d = out['pred_keypoints_3d'] # [B, 21, 3]
        #pred_keypoints_3d = torch.unsqueeze(out['pred_keypoints_3d'].detach(), dim=1)
        pred_keypoints_2d = out["pred_keypoints_2d"]
        _placeholder_gt_keypoints_3d = torch.zeros((args.batch_size, 21, 4))
        print(batch["img"].shape)
        prediction_img = keypoints_renderer.draw_keypoints(
            pred_keypoints_3d,
            images=255*batch["img"].detach().cpu().numpy().transpose(0,2,3,1),
            camera_translation=out["pred_cam_t"],
        )
        print(prediction_img.shape)
        img_to_show = 255. * prediction_img[:, :, ::-1] # [0, 1] --> [0, 255], RGB --> BGR
        cv2.imwrite("test.png", img_to_show)


if __name__ == '__main__':
    main()