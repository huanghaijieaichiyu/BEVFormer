# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
# Modified by Zhiqi Li
# Refactored into a CLI-friendly BEVFormer detection visualization tool.
# ---------------------------------------------

import argparse
import json
import math
import os
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, view_points

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
except Exception:  # pragma: no cover - optional dependency at runtime
    NuScenesMap = None


nusc = None

CAMS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

COLOR_MAP = {
    'car': '#1f77b4',
    'truck': '#ff7f0e',
    'construction_vehicle': '#9467bd',
    'bus': '#2ca02c',
    'trailer': '#8c564b',
    'barrier': '#e377c2',
    'motorcycle': '#bcbd22',
    'bicycle': '#17becf',
    'pedestrian': '#d62728',
    'traffic_cone': '#ff9896',
    'unknown': '#7f7f7f',
}

PRED_LINE_COLOR = '#1f77b4'
GT_LINE_COLOR = '#2ca02c'
PLACEHOLDER_BG = '#f2f4f7'
FIG_BG = '#ffffff'


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: Optional[str] = 'render.png',
        extra_info: bool = False) -> None:
    """Render a selected annotation from the current global NuScenes object."""
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'], 'Error: No LIDAR_TOP in data, unable to render.'

    all_bboxes = []
    select_cams = []
    for cam_key in [key for key in sample_record['data'] if 'CAM' in key]:
        _, boxes_data, _ = nusc.get_sample_data(
            sample_record['data'][cam_key],
            box_vis_level=box_vis_level,
            selected_anntokens=[anntoken])
        if boxes_data:
            all_bboxes.append(boxes_data)
            select_cams.append(cam_key)

    num_cam = len(all_bboxes)
    if num_cam == 0:
        fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))

    lidar = sample_record['data']['LIDAR_TOP']
    data_path_lidar, boxes_lidar, _ = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path_lidar).render_height(axes[0], view=view)
    for box in boxes_lidar:
        c = np.array(_rgb_float(get_color(box.name)))
        box.render(axes[0], view=view, colors=(c, c, c))
    if boxes_lidar:
        corners = view_points(boxes_lidar[0].corners(), view, False)[:2, :]
        axes[0].set_xlim(float(np.min(corners[0, :])) - margin, float(np.max(corners[0, :])) + margin)
        axes[0].set_ylim(float(np.min(corners[1, :])) - margin, float(np.max(corners[1, :])) + margin)
    axes[0].axis('off')
    axes[0].set_aspect('equal')

    for i, cam_key in enumerate(select_cams):
        cam_data_token = sample_record['data'][cam_key]
        data_path_cam, boxes_cam, camera_intrinsic_cam = nusc.get_sample_data(
            cam_data_token, selected_anntokens=[anntoken])
        im = Image.open(data_path_cam)
        axes[i + 1].imshow(im)
        axes[i + 1].set_title(nusc.get('sample_data', cam_data_token)['channel'])
        axes[i + 1].axis('off')
        axes[i + 1].set_aspect('equal')
        for box in boxes_cam:
            c = np.array(_rgb_float(get_color(box.name)))
            if camera_intrinsic_cam is not None:
                box.render(axes[i + 1], view=camera_intrinsic_cam, normalize=True, colors=(c, c, c))
        axes[i + 1].set_xlim(0, im.size[0])
        axes[i + 1].set_ylim(im.size[1], 0)

    if extra_info:
        w, l, h = ann_record['size']
        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))
        information = ' \n'.join([
            f'category: {ann_record["category_name"]}',
            '',
            f'# lidar points: {ann_record["num_lidar_pts"]:>4}',
            f'# radar points: {ann_record["num_radar_pts"]:>4}',
            '',
            f'distance: {dist:>7.3f}m',
            '',
            f'width:  {w:>7.3f}m',
            f'length: {l:>7.3f}m',
            f'height: {h:>7.3f}m',
        ])
        plt.annotate(information, (0, 0), (0, -20),
                     xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)
    plt.close(fig)


def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Boxes are transformed into the current sensor's coordinate frame.
    """
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    boxes = list(map(nusc.get_box, selected_anntokens)) if selected_anntokens is not None \
        else nusc.get_boxes(sample_data_token)

    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(
                scalar=math.cos(float(yaw) / 2),
                vector=[0, 0, math.sin(float(yaw) / 2)]).inverse)
        else:
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and cam_intrinsic is not None and imsize is not None \
                and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None):
    """
    Returns the data path as well as predicted annotations related to that sample_data.
    Boxes are transformed into the current sensor's coordinate frame.
    """
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    boxes = pred_anns if pred_anns is not None else []
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(
                scalar=math.cos(float(yaw) / 2),
                vector=[0, 0, math.sin(float(yaw) / 2)]).inverse)
        else:
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and cam_intrinsic is not None and imsize is not None \
                and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidiar_render(sample_token, data, save_path_full: str):
    """Render lidar-top GT vs prediction visualization for one sample."""
    if data is None:
        raise ValueError('No prediction data provided for lidar render.')

    bbox_gt_list = []
    bbox_pred_list = []
    skipped_gt_annotations = 0
    skipped_pred_annotations = 0
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            detection_name = category_to_detection_name(content['category_name'])
            if detection_name is None:
                skipped_gt_annotations += 1
                continue
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=detection_name,
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except Exception as exc:
            print(f'Error processing ground truth box for lidar render: {exc}')

    bbox_anns = data.get('results', {}).get(sample_token, [])
    for content in bbox_anns:
        try:
            detection_name = content.get('detection_name')
            if not detection_name:
                skipped_pred_annotations += 1
                continue
            bbox_pred_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content.get('velocity', (0.0, 0.0))),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=detection_name,
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=content.get('attribute_name', '')))
        except Exception as exc:
            print(f'Error processing prediction box for lidar render: {exc}')

    if skipped_gt_annotations:
        print(
            f'Skipped {skipped_gt_annotations} GT annotations not mappable to '
            f'nuScenes detection classes for sample {sample_token}')
    if skipped_pred_annotations:
        print(
            f'Skipped {skipped_pred_annotations} prediction annotations with '
            f'missing detection_name for sample {sample_token}')

    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)

    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=save_path_full)


def get_color(category_name: str):
    """
    Provides default colors based on category names using the active NuScenes colormap when possible.
    """
    if nusc is not None:
        if category_name == 'bicycle':
            return nusc.colormap['vehicle.bicycle']
        if category_name == 'construction_vehicle':
            return nusc.colormap['vehicle.construction']
        if category_name == 'traffic_cone':
            return nusc.colormap['movable_object.trafficcone']
        for key in nusc.colormap.keys():
            if category_name in key:
                return nusc.colormap[key]

    if category_name in COLOR_MAP:
        return COLOR_MAP[category_name]
    return COLOR_MAP['unknown']


def _rgb_float(color):
    if isinstance(color, str):
        color = color.lstrip('#')
        return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(np.array(color, dtype=np.float32) / 255.0)


def _prediction_boxes_for_sample(sample_token: str, pred_data: dict, score_thresh: float) -> List[Box]:
    boxes = []
    for record in pred_data.get('results', {}).get(sample_token, []):
        if record.get('detection_score', 0.0) < score_thresh:
            continue
        boxes.append(Box(
            record['translation'],
            record['size'],
            Quaternion(record['rotation']),
            name=record['detection_name'],
            token='predicted'))
    return boxes


def _placeholder_panel(title: str, message: str, size=(900, 900)) -> Image.Image:
    fig, ax = plt.subplots(1, 1, figsize=(size[0] / 100, size[1] / 100), dpi=100)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLACEHOLDER_BG)
    ax.axis('off')
    ax.text(0.5, 0.9, title, ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='#555555', wrap=True)
    buf = _fig_to_image(fig)
    return buf


def _fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return Image.fromarray(buf)


def _load_image_safe(path: str, fallback_title: str, fallback_message: str) -> Image.Image:
    if os.path.exists(path):
        return Image.open(path).convert('RGB')
    return _placeholder_panel(fallback_title, fallback_message)


def render_camera_grid(sample_token: str,
                       pred_data: dict,
                       score_thresh: float = 0.2,
                       output_path: Optional[str] = None) -> Image.Image:
    """Render the 6-camera prediction/ground-truth grid as one image."""
    sample = nusc.get('sample', sample_token)
    pred_boxes = _prediction_boxes_for_sample(sample_token, pred_data, score_thresh)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), dpi=180)
    fig.patch.set_facecolor(FIG_BG)

    for ax in axes.ravel():
        ax.set_facecolor(FIG_BG)
        ax.axis('off')

    for idx, cam_channel in enumerate(CAMS):
        current_col = idx % 3
        pred_row_idx = 0 if idx < 3 else 2
        gt_row_idx = pred_row_idx + 1
        ax_pred = axes[pred_row_idx, current_col]
        ax_gt = axes[gt_row_idx, current_col]
        ax_pred.set_title(cam_channel, fontsize=11, fontweight='bold', pad=8)

        if cam_channel not in sample['data']:
            ax_pred.text(0.5, 0.5, f'{cam_channel}\nNot available', ha='center', va='center')
            ax_gt.text(0.5, 0.5, f'{cam_channel}\nNot available', ha='center', va='center')
            continue

        sample_data_token = sample['data'][cam_channel]
        try:
            data_path, boxes_pred_transformed, camera_intrinsic = get_predicted_data(
                sample_data_token,
                box_vis_level=BoxVisibility.ANY,
                pred_anns=_prediction_boxes_for_sample(sample_token, pred_data, score_thresh))
            _, boxes_gt_transformed, _ = nusc.get_sample_data(
                sample_data_token, box_vis_level=BoxVisibility.ANY)
            img_data = Image.open(data_path).convert('RGB')
            np_img = np.array(img_data)

            ax_pred.imshow(np_img)
            for box in boxes_pred_transformed:
                c = np.array(_rgb_float(get_color(box.name)))
                if camera_intrinsic is not None:
                    box.render(ax_pred, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            ax_pred.set_xlim(0, img_data.size[0])
            ax_pred.set_ylim(img_data.size[1], 0)

            ax_gt.imshow(np_img)
            for box in boxes_gt_transformed:
                c = np.array(_rgb_float(get_color(box.name)))
                if camera_intrinsic is not None:
                    box.render(ax_gt, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            ax_gt.set_xlim(0, img_data.size[0])
            ax_gt.set_ylim(img_data.size[1], 0)
        except Exception as exc:
            msg = f'Error rendering {cam_channel}\n{exc}'
            ax_pred.text(0.5, 0.5, msg, ha='center', va='center', fontsize=10)
            ax_gt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=10)

    fig.text(0.01, 0.76, 'Prediction', rotation=90, va='center', ha='center',
             fontsize=13, fontweight='bold', color=PRED_LINE_COLOR)
    fig.text(0.01, 0.54, 'Ground Truth', rotation=90, va='center', ha='center',
             fontsize=13, fontweight='bold', color=GT_LINE_COLOR)
    fig.text(0.01, 0.27, 'Prediction', rotation=90, va='center', ha='center',
             fontsize=13, fontweight='bold', color=PRED_LINE_COLOR)
    fig.text(0.01, 0.05, 'Ground Truth', rotation=90, va='center', ha='center',
             fontsize=13, fontweight='bold', color=GT_LINE_COLOR)

    plt.tight_layout(rect=[0.03, 0.02, 1.0, 0.98])
    image = _fig_to_image(fig)
    if output_path:
        image.save(output_path)
    return image


def render_map_panel(sample_token: str,
                     out_path: Optional[str] = None,
                     patch_radius: float = 55.0) -> Image.Image:
    """Render a NuScenes static map patch around the current ego pose, or a placeholder."""
    if NuScenesMap is None:
        image = _placeholder_panel('Map', 'NuScenes map API is unavailable in this environment.')
        if out_path:
            image.save(out_path)
        return image

    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log.get('location')
    if not map_name:
        image = _placeholder_panel('Map', 'Map location is missing for this sample.')
        if out_path:
            image.save(out_path)
        return image

    try:
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        x, y = ego_pose['translation'][:2]

        fig = None
        try:
            fig, ax = nusc_map.render_map_patch(
                (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius),
                figsize=(8, 8),
                render_legend=False)
        except TypeError:
            fig, ax = nusc_map.render_map_patch(
                (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius),
                layer_names=['drivable_area', 'road_segment', 'lane', 'ped_crossing'],
                figsize=(8, 8),
                render_legend=False)
        except Exception:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
            fig.patch.set_facecolor(FIG_BG)
            ax.set_facecolor(PLACEHOLDER_BG)
            ax.text(0.5, 0.5, 'Unable to render map patch.', ha='center', va='center')

        ax.scatter([x], [y], c='#d62728', s=40, marker='o')
        ax.annotate('Ego', (x, y), textcoords='offset points', xytext=(6, 6), fontsize=10)
        ax.set_title('Map', fontsize=14, fontweight='bold')
        image = _fig_to_image(fig)
    except Exception as exc:
        image = _placeholder_panel('Map', f'Map patch unavailable.\n{exc}')

    if out_path:
        image.save(out_path)
    return image


def compose_detection_figure(camera_img: Image.Image,
                             lidar_img: Image.Image,
                             map_img: Image.Image,
                             sample_token: str) -> Image.Image:
    """Compose the final paper-style output figure."""
    camera_target_h = 1500
    camera_img = camera_img.resize(
        (int(camera_img.size[0] * camera_target_h / camera_img.size[1]), camera_target_h),
        Image.Resampling.LANCZOS)

    side_width = 780
    top_h = 760
    bottom_h = 700
    lidar_img = lidar_img.resize((side_width, top_h), Image.Resampling.LANCZOS)
    map_img = map_img.resize((side_width, bottom_h), Image.Resampling.LANCZOS)

    padding = 24
    title_h = 44
    canvas_w = camera_img.size[0] + side_width + padding * 3
    canvas_h = max(camera_img.size[1], top_h + bottom_h + padding + title_h) + padding * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), FIG_BG)

    camera_x = padding
    camera_y = padding + title_h
    side_x = camera_x + camera_img.size[0] + padding
    side_y = camera_y

    canvas.paste(camera_img, (camera_x, camera_y))
    canvas.paste(lidar_img, (side_x, side_y))
    canvas.paste(map_img, (side_x, side_y + top_h + padding))

    draw = ImageDraw.Draw(canvas)
    draw.text((camera_x, padding), f'Sample: {sample_token}', fill='#222222')
    draw.text((side_x, padding), 'LIDAR_TOP / Map', fill='#222222')

    return canvas


def render_sample_data(
    sample_token: str,
    output_file_index: int,
    pred_data: dict,
    out_dir: str = 'runs/visual_detection',
    score_thresh: float = 0.2,
    with_map: bool = True,
    save_panels: bool = False,
):
    """
    Render sample data into camera, lidar, map and combined panels.
    """
    os.makedirs(out_dir, exist_ok=True)
    token_prefix = sample_token[:8]
    sample_dir = os.path.join(out_dir, f'sample_{output_file_index:03d}_{token_prefix}')
    os.makedirs(sample_dir, exist_ok=True)

    camera_path = os.path.join(sample_dir, 'camera_grid.png')
    lidar_path = os.path.join(sample_dir, 'lidar_top.png')
    map_path = os.path.join(sample_dir, 'map.png')
    combined_path = os.path.join(sample_dir, 'combined.png')
    meta_path = os.path.join(sample_dir, 'meta.json')

    camera_img = render_camera_grid(sample_token, pred_data, score_thresh, output_path=camera_path if save_panels else None)
    lidiar_render(sample_token, pred_data, lidar_path)
    lidar_img = _load_image_safe(lidar_path, 'LIDAR_TOP', 'Lidar visualization could not be generated.')

    if with_map:
        map_img = render_map_panel(sample_token, out_path=map_path if save_panels else None)
    else:
        map_img = _placeholder_panel('Map', 'Map rendering disabled.')

    combined_img = compose_detection_figure(camera_img, lidar_img, map_img, sample_token)
    combined_img.save(combined_path)

    if not save_panels:
        lidar_img.save(lidar_path)
        map_img.save(map_path)
        camera_img.save(camera_path)

    meta = {
        'sample_token': sample_token,
        'index': output_file_index,
        'output_dir': sample_dir,
        'camera_grid': os.path.basename(camera_path),
        'lidar_top': os.path.basename(lidar_path),
        'map': os.path.basename(map_path),
        'combined': os.path.basename(combined_path),
        'score_thresh': score_thresh,
        'with_map': with_map,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return meta


def _parse_sample_tokens(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    if os.path.isfile(raw):
        with open(raw, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    tokens = []
    for part in raw.replace('\n', ',').split(','):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens or None


def _select_sample_tokens(all_tokens: Sequence[str],
                          explicit_tokens: Optional[Iterable[str]],
                          num_samples: int) -> List[str]:
    if explicit_tokens:
        selected = [token for token in explicit_tokens if token in set(all_tokens)]
        return selected
    if num_samples <= 0 or num_samples >= len(all_tokens):
        return list(all_tokens)
    step = max(1, len(all_tokens) // num_samples)
    return list(all_tokens)[::step][:num_samples]


def _init_nusc(dataroot: str, version: str, verbose: bool):
    global nusc
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return nusc


def parse_args():
    parser = argparse.ArgumentParser(
        description='BEVFormer official-style detection output visualization')
    parser.add_argument('--results', required=True, help='Path to results_nusc.json')
    parser.add_argument('--dataroot', required=True, help='NuScenes dataset root')
    parser.add_argument('--nusc-version', default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--out-dir', default='runs/visual_detection', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to render')
    parser.add_argument('--sample-tokens', default=None,
                        help='Comma-separated sample tokens or a text file with one token per line')
    parser.add_argument('--score-thresh', type=float, default=0.2, help='Prediction score threshold')
    parser.add_argument('--with-map', dest='with_map', action='store_true', default=True,
                        help='Render a static NuScenes map patch when possible')
    parser.add_argument('--no-map', dest='with_map', action='store_false',
                        help='Disable map rendering')
    parser.add_argument('--save-panels', action='store_true',
                        help='Keep individual panel images alongside the combined output')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose NuScenes loading')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(f'Results file not found: {args.results}')
    os.makedirs(args.out_dir, exist_ok=True)

    _init_nusc(args.dataroot, args.nusc_version, args.verbose)
    pred_data = mmcv.load(args.results)
    if 'results' not in pred_data or not pred_data['results']:
        raise ValueError('The provided results file does not contain valid detection results.')

    all_tokens = sorted(pred_data['results'].keys())
    selected_tokens = _select_sample_tokens(
        all_tokens,
        _parse_sample_tokens(args.sample_tokens),
        args.num_samples)

    if not selected_tokens:
        raise ValueError('No valid sample tokens were selected for visualization.')

    manifest = {
        'results': os.path.abspath(args.results),
        'dataroot': os.path.abspath(args.dataroot),
        'nusc_version': args.nusc_version,
        'score_thresh': args.score_thresh,
        'with_map': args.with_map,
        'samples': [],
    }

    for idx, token in enumerate(tqdm(selected_tokens, desc='Rendering detection outputs'), start=1):
        manifest['samples'].append(render_sample_data(
            sample_token=token,
            output_file_index=idx,
            pred_data=pred_data,
            out_dir=args.out_dir,
            score_thresh=args.score_thresh,
            with_map=args.with_map,
            save_panels=args.save_panels,
        ))

    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print('=' * 68)
    print(f'Visualization complete. Output directory: {args.out_dir}')
    print(f'Rendered samples: {len(selected_tokens)}')
    print(f'Manifest: {manifest_path}')
    print('=' * 68)


if __name__ == '__main__':
    main()
