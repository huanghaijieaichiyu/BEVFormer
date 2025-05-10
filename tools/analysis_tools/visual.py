# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import math  # 添加math模块，用于三角函数
import os  # 用于文件路径操作和创建目录
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from tqdm import tqdm
from pyquaternion import Quaternion
from matplotlib.axes import Axes
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Iterable, Optional
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image
from nuscenes.nuscenes import NuScenes
import mmcv
import matplotlib
matplotlib.use('Agg')  # 使用非交互式的Agg后端，解决_tkinter错误


cams = ['CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT']


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: Optional[str] = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(
    ), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams_local = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam_key in cams_local:
        _, boxes_data, _ = nusc.get_sample_data(sample_record['data'][cam_key], box_vis_level=box_vis_level,
                                                selected_anntokens=[anntoken])
        if len(boxes_data) > 0:
            all_bboxes.append(boxes_data)
            select_cams.append(cam_key)
            # We found an image that matches. Let's abort.

    num_cam = len(all_bboxes)
    if num_cam == 0 and len(select_cams) == 0:
        fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))

    select_cams_data = [sample_record['data'][cam_key]
                        for cam_key in select_cams]
    # print('bbox in cams:', select_cams_data)

    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path_lidar, boxes_lidar, _ = nusc.get_sample_data(
        lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(
        data_path_lidar).render_height(axes[0], view=view)
    for box in boxes_lidar:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
    if boxes_lidar:
        corners = view_points(boxes_lidar[0].corners(), view, False)[:2, :]
        min_x = float(np.min(corners[0, :])) - margin
        max_x = float(np.max(corners[0, :])) + margin
        min_y = float(np.min(corners[1, :])) - margin
        max_y = float(np.max(corners[1, :])) + margin
        axes[0].set_xlim(min_x, max_x)
        axes[0].set_ylim(min_y, max_y)
    axes[0].axis('off')
    axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(num_cam):
        cam_data_token = select_cams_data[i]
        data_path_cam, boxes_cam, camera_intrinsic_cam = nusc.get_sample_data(
            cam_data_token, selected_anntokens=[anntoken])
        im = Image.open(data_path_cam)
        axes[i+1].imshow(im)
        axes[i+1].set_title(nusc.get('sample_data', cam_data_token)['channel'])
        axes[i+1].axis('off')
        axes[i+1].set_aspect('equal')
        for box in boxes_cam:
            c = np.array(get_color(box.name)) / 255.0
            if camera_intrinsic_cam is not None:
                box.render(axes[i+1], view=camera_intrinsic_cam,
                           normalize=True, colors=(c, c, c))

        axes[i+1].set_xlim(0, im.size[0])
        axes[i+1].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get(
            'sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get(
            'ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(
            np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(
                                      lidar_points),
                                  '# radar points: {0:>4}'.format(
                                      radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

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
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=math.cos(float(yaw) / 2),
                       vector=[0, 0, math.sin(float(yaw) / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        # Only check for boxes in camera images
        if sensor_record['modality'] == 'camera' and cam_intrinsic is not None and imsize is not None and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns if pred_anns is not None else []
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=math.cos(float(yaw) / 2),
                       vector=[0, 0, math.sin(float(yaw) / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        # Only check for boxes in camera images
        if sensor_record['modality'] == 'camera' and cam_intrinsic is not None and imsize is not None and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidiar_render(sample_token, data, save_path_full: str):
    if data is None:
        print("Warning: No prediction data provided for Lidar render.")
        return

    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            detection_name = category_to_detection_name(
                content['category_name'])
            if detection_name is None:
                detection_name = 'unknown'

            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-
                1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=detection_name,
                detection_score=-
                1.0 if 'detection_score' not in content else float(
                    content['detection_score']),
                attribute_name=''))
        except Exception as e:
            print(f"Error processing ground truth box for Lidar render: {e}")
            pass

    if sample_token not in data['results']:
        print(
            f"Warning: Sample token {sample_token} not found in prediction results for Lidar render.")
        return

    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        try:
            bbox_pred_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content['velocity']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-
                1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['detection_name'],
                detection_score=-
                1.0 if 'detection_score' not in content else float(
                    content['detection_score']),
                attribute_name=content['attribute_name']))
        except Exception as e:
            print(f"Error processing prediction box for Lidar render: {e}")
            pass

    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth (Lidar BEV)')
    print('blue is the predicted result (Lidar BEV)')

    visualize_sample(nusc, sample_token, gt_annotations,
                     pred_annotations, savepath=save_path_full)


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
         'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
         'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
         'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
         'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
         'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
         'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
         'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    # print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
    sample_toekn: str,
    output_file_index: int,
    with_anns: bool = True,
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
    axes_limit: float = 40,
    ax=None,
    nsweeps: int = 1,
    underlay_map: bool = True,
    use_flat_vehicle_coordinates: bool = True,
    show_lidarseg: bool = False,
    show_lidarseg_legend: bool = False,
    filter_lidarseg_labels=None,
    lidarseg_preds_bin_path: Optional[str] = None,
    verbose: bool = True,
    show_panoptic: bool = False,
    pred_data=None,
) -> None:
    """
    Render sample data, save BEV and camera images, then concatenate them.
    Images are saved to 'runs/visual/' with sequential names like 1.png, 2.png, etc.
    """
    base_save_dir = "runs/visual"
    os.makedirs(base_save_dir, exist_ok=True)
    bev_dir = os.path.join(base_save_dir, 'bev')
    camera_dir = os.path.join(base_save_dir, 'camera')
    combined_dir = os.path.join(base_save_dir, 'combined')
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    bev_image_path = os.path.join(
        bev_dir, f"{output_file_index}_bev.png")
    camera_image_path = os.path.join(
        camera_dir, f"{output_file_index}_camera.png")
    combined_image_path = os.path.join(
        combined_dir, f"{output_file_index}.png")

    if pred_data is None:
        print("Warning: No prediction data provided. Cannot render.")
        return

    # Render and save BEV image
    try:
        lidiar_render(sample_toekn, pred_data, save_path_full=bev_image_path)
    except Exception as e:
        print(f"Error in lidiar_render for {sample_toekn}: {e}")
        return

    sample = nusc.get('sample', sample_toekn)

    # Setup for camera images
    num_cameras = len(cams)

    # Create a new figure for camera images
    fig_cam, axes_cam = plt.subplots(4, 3, figsize=(24, 18))

    j = 0
    for ind, cam_channel in enumerate(cams):
        if cam_channel not in sample['data']:
            print(
                f"Warning: Camera {cam_channel} not found in sample data {sample_toekn}. Skipping.")
            col_idx = ind % 3
            row_idx_pred = j + (ind // 3) * 2
            row_idx_gt = row_idx_pred + 2

            current_row_pred = 0 if ind < 3 else 1
            current_row_gt = row_idx_pred + 2
            current_col = ind % 3

            axes_cam[current_row_pred, current_col].axis('off')
            axes_cam[current_row_gt, current_col].axis('off')
            axes_cam[current_row_pred, current_col].set_title(
                f'{cam_channel} (Not available)')
            axes_cam[current_row_gt, current_col].set_title(
                f'{cam_channel} (Not available)')
            continue

        sample_data_token = sample['data'][cam_channel]
        sd_record = nusc.get('sample_data', sample_data_token)

        current_col = ind % 3
        pred_row_idx = 0 if ind < 3 else 1
        gt_row_idx = pred_row_idx + 2

        ax_pred = axes_cam[pred_row_idx, current_col]
        ax_gt = axes_cam[gt_row_idx, current_col]

        try:
            predicted_boxes_for_sample = []
            if sample_toekn in pred_data.get('results', {}):
                predicted_boxes_for_sample = [
                    Box(record['translation'], record['size'], Quaternion(record['rotation']),
                        name=record['detection_name'], token='predicted')
                    for record in pred_data['results'][sample_toekn]
                    if record.get('detection_score', 0) > 0.2
                ]

            data_path, boxes_pred_transformed, camera_intrinsic = get_predicted_data(
                sample_data_token, box_vis_level=box_vis_level, pred_anns=predicted_boxes_for_sample
            )
            _, boxes_gt_transformed, _ = nusc.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level
            )

            img_data = Image.open(data_path)

            ax_pred.imshow(img_data)
            if with_anns:
                for box in boxes_pred_transformed:
                    c = np.array(get_color(box.name)) / 255.0
                    if camera_intrinsic is not None:
                        box.render(ax_pred, view=camera_intrinsic,
                                   normalize=True, colors=(c, c, c))
            ax_pred.set_title(f'PRED: {sd_record["channel"]}')
            ax_pred.axis('off')
            ax_pred.set_aspect('equal')
            ax_pred.set_xlim(0, img_data.size[0])
            ax_pred.set_ylim(img_data.size[1], 0)

            ax_gt.imshow(img_data)
            if with_anns:
                for box in boxes_gt_transformed:
                    c = np.array(get_color(box.name)) / 255.0
                    if camera_intrinsic is not None:
                        box.render(ax_gt, view=camera_intrinsic,
                                   normalize=True, colors=(c, c, c))
            ax_gt.set_title(f'GT: {sd_record["channel"]}')
            ax_gt.axis('off')
            ax_gt.set_aspect('equal')
            ax_gt.set_xlim(0, img_data.size[0])
            ax_gt.set_ylim(img_data.size[1], 0)

        except Exception as e:
            print(
                f"Error rendering camera {cam_channel} for sample {sample_toekn}: {e}")
            ax_pred.set_title(f'PRED: {cam_channel} (Error)')
            ax_pred.axis('off')
            ax_gt.set_title(f'GT: {cam_channel} (Error)')
            ax_gt.axis('off')
            continue

    plt.tight_layout()
    plt.savefig(camera_image_path, bbox_inches='tight',
                pad_inches=0.1, dpi=200)
    if verbose:
        plt.show()
    plt.close(fig_cam)

    # Concatenate BEV and Camera images
    try:
        if not os.path.exists(bev_image_path):
            print(
                f"BEV image not found at {bev_image_path}, skipping concatenation.")
            return

        if not os.path.exists(camera_image_path):
            print(
                f"Camera composite image not found at {camera_image_path}, skipping concatenation.")
            return

        img_bev = Image.open(bev_image_path)
        img_camera = Image.open(camera_image_path)

        TARGET_IMG_HEIGHT_PER_SECTION = 800  # Pixels

        # Determine resampling filter (Pillow 9.0.0+ syntax)
        # Fallback to older syntax might be needed if runtime errors occur due to Pillow version
        # For now, assuming a modern Pillow version for linter compatibility.
        try:
            resampling_filter = Image.Resampling.LANCZOS
        except AttributeError:
            # Fallback for older Pillow versions if Image.Resampling is not available
            # This might still cause linter issues but is a common fallback pattern.
            # If Image.LANCZOS also fails, Pillow is likely very old or installation is broken.
            print("Warning: Image.Resampling.LANCZOS not available. Falling back to older Pillow constants if possible.")
            resampling_filter = getattr(
                Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', None))
            if resampling_filter is None:
                print(
                    "Error: Suitable resampling filter not found in Pillow. Update Pillow or check installation.")
                return  # Cannot proceed without a filter

        # Resize BEV image
        bev_width, bev_height = img_bev.size
        if bev_height == 0:  # Prevent division by zero
            print(
                f"Error: BEV image {bev_image_path} has zero height. Skipping concatenation.")
            return
        bev_aspect_ratio = bev_width / bev_height
        resized_bev_height = TARGET_IMG_HEIGHT_PER_SECTION
        resized_bev_width = int(resized_bev_height * bev_aspect_ratio)

        resized_img_bev = img_bev.resize(
            (resized_bev_width, resized_bev_height), resampling_filter)

        # Resize Camera composite image
        cam_img_width, cam_img_height = img_camera.size
        if cam_img_height == 0:  # Prevent division by zero
            print(
                f"Error: Camera image {camera_image_path} has zero height. Skipping concatenation.")
            return
        cam_aspect_ratio = cam_img_width / cam_img_height
        resized_cam_img_height = TARGET_IMG_HEIGHT_PER_SECTION
        resized_cam_img_width = int(resized_cam_img_height * cam_aspect_ratio)

        resized_img_camera = img_camera.resize(
            (resized_cam_img_width, resized_cam_img_height), resampling_filter)

        # Create final canvas
        total_width = max(resized_bev_width, resized_cam_img_width)
        # Should be 2 * TARGET_IMG_HEIGHT_PER_SECTION
        total_height = resized_bev_height + resized_cam_img_height

        combined_img = Image.new(
            'RGB', (total_width, total_height), (255, 255, 255))  # White background

        # Paste resized BEV image
        combined_img.paste(
            resized_img_bev, ((total_width - resized_bev_width) // 2, 0))

        # Paste resized Camera composite image
        combined_img.paste(resized_img_camera, ((
            total_width - resized_cam_img_width) // 2, resized_bev_height))

        combined_img.save(combined_image_path)
        print(f"Saved combined image to {combined_image_path}")

    except FileNotFoundError:
        print(
            f"Error: One or both image files not found for concatenation: {bev_image_path}, {camera_image_path}")
    except Exception as e:
        print(
            f"Error concatenating images for output index {output_file_index} (sample token {sample_toekn}): {e}")


if __name__ == '__main__':
    nusc_dataroot = os.environ.get(
        'NUSCENES_DATAROOT', '/mnt/f/datasets/nuscenes')
    nusc_version = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')

    print(f"Loading NuScenes {nusc_version} from {nusc_dataroot}...")
    try:
        nusc = NuScenes(version=nusc_version,
                        dataroot=nusc_dataroot, verbose=True)
    except Exception as e:
        print(f"Error initializing NuScenes: {e}")
        print("Please ensure NuScenes dataset is correctly set up and env vars NUSCENES_DATAROOT/NUSCENES_VERSION are correct if used.")
        exit()

    results_file_path = 'test/bevformer_tiny/Wed_May__7_15_45_28_2025/pts_bbox/results_nusc.json'
    if not os.path.exists(results_file_path):
        print(f"Results file not found: {results_file_path}")
        print("Please provide a valid path to a results_nusc.json file.")
        print("Creating dummy bevformer_results for testing purposes.")
        dummy_sample_token = "dummysampletoken12345678901234567890"
        if nusc_version != 'v1.0-mini' or not any(s['token'] == dummy_sample_token for s in nusc.sample):
            print("Using a very minimal dummy NuScenes setup for dummy results.")

            class DummyNusc:
                def __init__(self):
                    self.sample = [
                        {'token': dummy_sample_token, 'data': {}, 'anns': []}]
                    self.colormap = {}

                def get(self, table_name, token):
                    if table_name == 'sample' and token == dummy_sample_token:
                        return self.sample[0]
                    return {}

                def get_sample_data_path(self, token): return "dummy_path.jpg"
                def get_box(self, token): return Box(
                    center=[0, 0, 0], size=[1, 1, 1], orientation=Quaternion())

                def get_boxes(self, sample_data_token): return []
                def box_velocity(self, token): return np.array([0, 0])

            try:
                Image.new('RGB', (10, 10), color='red').save("dummy_path.jpg")
            except Exception as img_e:
                print(f"Could not create dummy image file: {img_e}")

            if not (nusc_version == 'v1.0-mini' and any(s['token'] == dummy_sample_token for s in nusc.sample)):
                nusc = DummyNusc()

        bevformer_results = {
            'results': {
                dummy_sample_token: [
                    {
                        'sample_token': dummy_sample_token,
                        'translation': [0, 0, 0], 'size': [1, 1, 1], 'rotation': [1, 0, 0, 0],
                        'velocity': [0, 0], 'detection_name': 'car', 'detection_score': 0.9,
                        'attribute_name': 'parked'
                    }
                ]
            },
            'meta': {}
        }

    else:
        bevformer_results = mmcv.load(results_file_path)

    if not bevformer_results or 'results' not in bevformer_results or not bevformer_results['results']:
        print("Loaded results are empty or invalid. Exiting.")
        exit()

    sample_token_list = list(bevformer_results['results'].keys())

    if not sample_token_list:
        print("No sample tokens found in results. Exiting.")
        exit()

    print(
        f"Found {len(sample_token_list)} samples in results. Visualizing up to 10.")
    for i, token in enumerate(sample_token_list[:10], start=1):
        print(f"Rendering sample with output index {i} (token: {token})")
        render_sample_data(token, output_file_index=i,
                           pred_data=bevformer_results, verbose=False)
    print("Visualization complete. Check the 'runs/visual' directory.")
