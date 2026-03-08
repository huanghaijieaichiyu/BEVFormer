"""
Paper-Quality Detection Comparison Visualization (v3 - Enhanced)
================================================================
Compare BEVFormer detection on normal vs low-light images.

Features:
  ✓ 3D bounding boxes with confidence score labels
  ✓ BEV bird's-eye-view with GT overlay
  ✓ Per-sample + global metrics panel
  ✓ ALL images saved separately at HIGH RESOLUTION
  ✓ Combined figure also saved
  ✓ JSON manifest for web viewer

Output structure per sample:
  runs/visual_comparison/
  ├── sample_001/
  │   ├── ll_CAM_FRONT.png         # low-light detection, camera view
  │   ├── ll_CAM_FRONT_LEFT.png
  │   ├── ...
  │   ├── nm_CAM_FRONT.png         # normal detection on low-light image
  │   ├── nm_CAM_FRONT_LEFT.png
  │   ├── ...
  │   ├── bev_lowlight.png         # BEV: low-light detection
  │   ├── bev_normal.png           # BEV: normal detection
  │   ├── metrics.png              # per-sample metrics panel
  │   ├── combined.png             # full composition
  │   └── stats.json               # machine-readable stats
  ├── global_metrics.png
  ├── legend.png
  └── manifest.json                # for web viewer
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion
from tqdm import tqdm

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import mmcv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# ── Constants ───────────────────────────────────────────────────────────────
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']

CATEGORY_COLORS = {
    'car':                  (  0, 160, 255),
    'truck':                (255, 127,  14),
    'construction_vehicle': (148, 103, 189),
    'bus':                  ( 44, 160,  44),
    'trailer':              (140,  86,  75),
    'barrier':              (227, 119, 194),
    'motorcycle':           (255, 215,   0),
    'bicycle':              ( 23, 190, 207),
    'pedestrian':           (214,  39,  40),
    'traffic_cone':         (255, 152, 150),
}
DEFAULT_COLOR = (180, 180, 180)

# High-res DPI for publication
OUTPUT_DPI = 200
BEV_DPI = 150
BEV_FIGSIZE = 10   # inches, so 10*150=1500px


def get_color(name: str):
    for key, col in CATEGORY_COLORS.items():
        if key in name:
            return col
    return DEFAULT_COLOR


def _get_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ═══════════════════════════════════════════════════════════════════════════
#  Camera Rendering with Confidence Scores
# ═══════════════════════════════════════════════════════════════════════════

def _project_boxes(nusc, sample_data_token, pred_records, score_thresh=0.25):
    sd = nusc.get('sample_data', sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    cam_intrinsic = np.array(cs['camera_intrinsic'])
    imsize = (sd['width'], sd['height'])
    results = []
    for rec in pred_records:
        score = rec.get('detection_score', 0)
        if score < score_thresh:
            continue
        box = Box(rec['translation'], rec['size'],
                  Quaternion(rec['rotation']),
                  name=rec['detection_name'], token='pred')
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)
        if not box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
            continue
        corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)[:2]
        results.append((corners_2d, get_color(rec['detection_name']),
                        rec['detection_name'], score))
    return results


def _draw_3d_box(draw, corners_2d, color, lw=3):
    c = corners_2d.T.tolist()
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        draw.line([tuple(c[i]), tuple(c[j])], fill=color, width=lw)


def _draw_front_face(draw, corners_2d, color, lw=4):
    c = corners_2d.T.tolist()
    for i, j in [(0,1),(1,5),(5,4),(4,0)]:
        draw.line([tuple(c[i]), tuple(c[j])], fill=color, width=lw)


def render_camera_with_boxes(nusc, sample_token, cam_channel,
                             pred_records, score_thresh,
                             image_dataroot=None):
    """Render a single camera at FULL RESOLUTION with boxes and score labels."""
    sample = nusc.get('sample', sample_token)
    if cam_channel not in sample['data']:
        return None
    sd_token = sample['data'][cam_channel]
    sd = nusc.get('sample_data', sd_token)
    img_path = os.path.join(image_dataroot or nusc.dataroot, sd['filename'])
    if not os.path.exists(img_path):
        print(f"  Warning: {img_path} not found")
        return None

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    projected = _project_boxes(nusc, sd_token, pred_records, score_thresh)

    # Scale font to image resolution
    font_size = max(18, img.size[1] // 35)
    font = _get_font(font_size)
    small_font = _get_font(max(14, font_size - 4))

    for corners_2d, color, name, score in projected:
        _draw_3d_box(draw, corners_2d, color, lw=3)
        _draw_front_face(draw, corners_2d, color, lw=4)

        # Confidence label at box top
        top_x = float(np.mean(corners_2d[0, 4:8]))
        top_y = float(np.min(corners_2d[1, 4:8])) - 6
        label = f"{name[:3]} {score:.2f}"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        rx1 = top_x - tw // 2 - pad
        ry1 = top_y - th - pad * 2
        rx2 = top_x + tw // 2 + pad
        ry2 = top_y
        draw.rectangle([rx1, ry1, rx2, ry2], fill=(*color, 200))
        draw.text((rx1 + pad, ry1 + pad), label,
                  fill=(255, 255, 255), font=small_font)

    return img


# ═══════════════════════════════════════════════════════════════════════════
#  BEV Rendering (high-res)
# ═══════════════════════════════════════════════════════════════════════════

def render_bev(nusc, sample_token, pred_records, score_thresh=0.25, axes_limit=50):
    """Render BEV at high resolution."""
    fig, ax = plt.subplots(1, 1, figsize=(BEV_FIGSIZE, BEV_FIGSIZE), dpi=BEV_DPI)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Grid
    for r in range(10, int(axes_limit) + 1, 10):
        circle = plt.Circle((0, 0), r, fill=False, color='#2d2d44', lw=0.5, ls='--')
        ax.add_patch(circle)
        ax.text(r + 0.5, 0.8, f'{r}m', fontsize=8, color='#555577')

    # Ego
    ego = mpatches.FancyBboxPatch((-1, -2), 2, 4, boxstyle="round,pad=0.3",
        facecolor='#e94560', edgecolor='white', lw=1.5, zorder=10)
    ax.add_patch(ego)
    ax.text(0, 0, 'EGO', ha='center', va='center', fontsize=8,
            color='white', fontweight='bold', zorder=11)

    # GT boxes (green)
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', sd_token)
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    for ann_token in sample['anns']:
        box = nusc.get_box(ann_token)
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)
        corners = box.bottom_corners()[:2]
        xs = list(corners[0]) + [corners[0, 0]]
        ys = list(corners[1]) + [corners[1, 0]]
        ax.plot(xs, ys, color='#00ff88', lw=1.2, alpha=0.6, zorder=3)

    # Pred boxes (colored)
    for rec in pred_records:
        score = rec.get('detection_score', 0)
        if score < score_thresh:
            continue
        box = Box(rec['translation'], rec['size'], Quaternion(rec['rotation']),
                  name=rec['detection_name'])
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)
        corners = box.bottom_corners()[:2]
        xs = list(corners[0]) + [corners[0, 0]]
        ys = list(corners[1]) + [corners[1, 0]]
        c = tuple(v / 255. for v in get_color(rec['detection_name']))
        ax.fill(xs, ys, color=c, alpha=0.3, zorder=4)
        ax.plot(xs, ys, color=c, lw=1.8, zorder=5)
        cx, cy = float(np.mean(corners[0])), float(np.mean(corners[1]))
        ax.text(cx, cy, f'{score:.2f}', fontsize=7, color='white',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=c, alpha=0.7,
                          edgecolor='none'), zorder=6)

    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_aspect('equal')
    ax.tick_params(colors='#777799', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.annotate('', xy=(0, axes_limit * 0.95), xytext=(0, axes_limit * 0.8),
                arrowprops=dict(arrowstyle='->', color='#aaaacc', lw=1.5))
    ax.text(0, axes_limit * 0.97, 'Front', ha='center', fontsize=9, color='#aaaacc')

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_sample_stats(pred_records, score_thresh=0.25):
    filtered = [r for r in pred_records if r.get('detection_score', 0) >= score_thresh]
    stats = {
        'total_detections': len(filtered),
        'avg_score': round(float(np.mean([r['detection_score'] for r in filtered])), 4) if filtered else 0.0,
        'max_score': round(float(max(r['detection_score'] for r in filtered)), 4) if filtered else 0.0,
        'min_score': round(float(min(r['detection_score'] for r in filtered)), 4) if filtered else 0.0,
        'per_class': {},
    }
    for r in filtered:
        n = r['detection_name']
        if n not in stats['per_class']:
            stats['per_class'][n] = {'count': 0, 'scores': []}
        stats['per_class'][n]['count'] += 1
        stats['per_class'][n]['scores'].append(r['detection_score'])
    for cd in stats['per_class'].values():
        cd['avg_score'] = round(float(np.mean(cd['scores'])), 4)
        del cd['scores']
    return stats


def render_metrics_panel(ll_stats, norm_stats, width=600, height=800):
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=OUTPUT_DPI)
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    ax.axis('off')

    y, dy = 0.96, 0.032

    def _t(text, x, yp, fs=9, color='white', weight='normal', ha='left'):
        ax.text(x, yp, text, fontsize=fs, color=color, fontweight=weight,
                ha=ha, va='top', transform=ax.transAxes, fontfamily='monospace')

    _t("Detection Metrics Comparison", 0.5, y, fs=13, color='#e0e0ff', weight='bold', ha='center')
    y -= dy * 1.8
    _t("─" * 50, 0.03, y, color='#333366'); y -= dy
    _t(f"{'':20s} {'Low-Light':>12s} {'Normal':>12s}", 0.03, y, fs=9, color='#8888bb'); y -= dy
    _t("─" * 50, 0.03, y, color='#333366'); y -= dy

    for name, ll_v, nm_v in [
        ('Total Detections', ll_stats['total_detections'], norm_stats['total_detections']),
        ('Avg Confidence',   ll_stats['avg_score'],        norm_stats['avg_score']),
        ('Max Confidence',   ll_stats['max_score'],        norm_stats['max_score']),
        ('Min Confidence',   ll_stats['min_score'],        norm_stats['min_score']),
    ]:
        fmt = f"{name:<20s} {ll_v:>12.4f} {nm_v:>12.4f}" if isinstance(ll_v, float) \
              else f"{name:<20s} {ll_v:>12d} {nm_v:>12d}"
        c = '#44dd88' if nm_v > ll_v else '#dd4466' if nm_v < ll_v else '#aaaacc'
        _t(fmt, 0.03, y, fs=9, color=c); y -= dy

    y -= dy
    _t("─" * 50, 0.03, y, color='#333366'); y -= dy
    _t("Per-Class Breakdown", 0.5, y, fs=11, color='#e0e0ff', weight='bold', ha='center'); y -= dy
    _t(f"{'Category':<18s} {'LL#':>5s} {'LL_sc':>6s}  {'NM#':>5s} {'NM_sc':>6s}", 0.03, y, fs=8, color='#8888bb')
    y -= dy
    _t("─" * 50, 0.03, y, color='#333366'); y -= dy

    all_cls = sorted(set(list(ll_stats['per_class']) + list(norm_stats['per_class'])))
    for cn in all_cls:
        ll_c = ll_stats['per_class'].get(cn, {'count': 0, 'avg_score': 0})
        nm_c = norm_stats['per_class'].get(cn, {'count': 0, 'avg_score': 0})
        hex_c = '#{:02x}{:02x}{:02x}'.format(*get_color(cn))
        _t(f"{cn:<18s} {ll_c['count']:>5d} {ll_c['avg_score']:>6.3f}  {nm_c['count']:>5d} {nm_c['avg_score']:>6.3f}",
           0.03, y, fs=8, color=hex_c)
        y -= dy

    plt.tight_layout(pad=0.3)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


# ═══════════════════════════════════════════════════════════════════════════
#  Labels & Legend
# ═══════════════════════════════════════════════════════════════════════════

def _add_label(img, text, bg_color=(0,0,0,180), text_color=(255,255,255)):
    img = img.copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    w, h = img.size
    font = _get_font(max(18, h // 25))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    draw.rectangle([0, 0, w, th + 2 * pad], fill=bg_color)
    draw.text(((w - tw) // 2, pad), text, fill=text_color, font=font)
    return img


def make_legend_image(width=1400, height=100):
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=OUTPUT_DPI)
    ax.set_facecolor('#0f0f23'); fig.patch.set_facecolor('#0f0f23'); ax.axis('off')
    patches = [mpatches.Patch(color=tuple(v/255. for v in c), label=n)
               for n, c in CATEGORY_COLORS.items()]
    patches.append(mlines.Line2D([], [], color='#00ff88', lw=2, label='Ground Truth'))
    ax.legend(handles=patches, loc='center', ncol=min(len(patches), 11),
              fontsize=9, frameon=False, handlelength=1.5, columnspacing=1.2,
              handletextpad=0.5, labelcolor='white')
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


# ═══════════════════════════════════════════════════════════════════════════
#  Combined Figure Composition
# ═══════════════════════════════════════════════════════════════════════════

def build_combined(ll_cam_imgs, norm_cam_imgs, bev_ll, bev_norm, metrics_img):
    """Compose all separate images into one combined overview."""
    gap = 6
    cam_h = 420

    def _resize_row(imgs, th):
        return [img.resize((int(img.size[0] * th / img.size[1]), th), Image.LANCZOS)
                for img in imgs]

    ll_resized = _resize_row(ll_cam_imgs, cam_h)
    nm_resized = _resize_row(norm_cam_imgs, cam_h)

    row1_w = sum(im.size[0] for im in ll_resized) + gap * (len(ll_resized) - 1)
    row2_w = sum(im.size[0] for im in nm_resized) + gap * (len(nm_resized) - 1)
    cam_w = max(row1_w, row2_w)

    bev_h = 500
    bev_ll_r = bev_ll.resize((int(bev_ll.size[0] * bev_h / bev_ll.size[1]), bev_h), Image.LANCZOS)
    bev_nm_r = bev_norm.resize((int(bev_norm.size[0] * bev_h / bev_norm.size[1]), bev_h), Image.LANCZOS)
    metrics_r = metrics_img.resize((metrics_img.size[0], min(bev_h, metrics_img.size[1])), Image.LANCZOS)

    total_w = max(cam_w, bev_ll_r.size[0] + gap + bev_nm_r.size[0] + gap + metrics_r.size[0])
    total_h = cam_h + gap + cam_h + gap + bev_h
    canvas = Image.new('RGB', (total_w, total_h), (15, 15, 35))

    x = 0
    for im in ll_resized:
        canvas.paste(im, (x, 0)); x += im.size[0] + gap
    x = 0
    for im in nm_resized:
        canvas.paste(im, (x, cam_h + gap)); x += im.size[0] + gap

    y3 = cam_h * 2 + gap * 2
    canvas.paste(bev_ll_r, (0, y3))
    canvas.paste(bev_nm_r, (bev_ll_r.size[0] + gap, y3))
    canvas.paste(metrics_r, (bev_ll_r.size[0] + gap + bev_nm_r.size[0] + gap, y3))
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  Per-sample Processing
# ═══════════════════════════════════════════════════════════════════════════

def process_sample(nusc, sample_token, ll_preds, norm_preds,
                   lowlight_dataroot, score_thresh, cameras, out_dir, idx):
    """Process one sample: render all views, save individually + combined."""
    sample_dir = os.path.join(out_dir, f"sample_{idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    saved_files = {'token': sample_token, 'index': idx, 'cameras': {}}

    # ── Camera views ──
    ll_cam_list, nm_cam_list = [], []
    for cam in cameras:
        # Low-light detection
        img_ll = render_camera_with_boxes(
            nusc, sample_token, cam, ll_preds, score_thresh,
            image_dataroot=lowlight_dataroot)
        if img_ll:
            ll_labeled = _add_label(img_ll, f"⚡ Low-Light: {cam}", bg_color=(180, 40, 40, 200))
            fname_ll = f"ll_{cam}.png"
            ll_labeled.save(os.path.join(sample_dir, fname_ll), dpi=(OUTPUT_DPI, OUTPUT_DPI))
            ll_cam_list.append(ll_labeled)
            saved_files['cameras'][f'll_{cam}'] = fname_ll

        # Normal detection (on low-light image)
        img_nm = render_camera_with_boxes(
            nusc, sample_token, cam, norm_preds, score_thresh,
            image_dataroot=lowlight_dataroot)
        if img_nm:
            nm_labeled = _add_label(img_nm, f"☀ Normal: {cam}", bg_color=(40, 120, 40, 200))
            fname_nm = f"nm_{cam}.png"
            nm_labeled.save(os.path.join(sample_dir, fname_nm), dpi=(OUTPUT_DPI, OUTPUT_DPI))
            nm_cam_list.append(nm_labeled)
            saved_files['cameras'][f'nm_{cam}'] = fname_nm

    # ── BEV ──
    bev_ll = render_bev(nusc, sample_token, ll_preds, score_thresh)
    bev_ll_labeled = _add_label(bev_ll, "BEV: Low-Light Detection", bg_color=(180, 40, 40, 200))
    bev_ll_labeled.save(os.path.join(sample_dir, "bev_lowlight.png"), dpi=(OUTPUT_DPI, OUTPUT_DPI))
    saved_files['bev_lowlight'] = "bev_lowlight.png"

    bev_nm = render_bev(nusc, sample_token, norm_preds, score_thresh)
    bev_nm_labeled = _add_label(bev_nm, "BEV: Normal Detection", bg_color=(40, 120, 40, 200))
    bev_nm_labeled.save(os.path.join(sample_dir, "bev_normal.png"), dpi=(OUTPUT_DPI, OUTPUT_DPI))
    saved_files['bev_normal'] = "bev_normal.png"

    # ── Metrics ──
    ll_stats = compute_sample_stats(ll_preds, score_thresh)
    nm_stats = compute_sample_stats(norm_preds, score_thresh)
    metrics_img = render_metrics_panel(ll_stats, nm_stats)
    metrics_img.save(os.path.join(sample_dir, "metrics.png"), dpi=(OUTPUT_DPI, OUTPUT_DPI))
    saved_files['metrics'] = "metrics.png"

    # Save stats JSON
    stats_data = {'lowlight': ll_stats, 'normal': nm_stats}
    with open(os.path.join(sample_dir, "stats.json"), 'w') as f:
        json.dump(stats_data, f, indent=2)
    saved_files['stats'] = "stats.json"

    # ── Combined ──
    if ll_cam_list and nm_cam_list:
        combined = build_combined(ll_cam_list, nm_cam_list, bev_ll_labeled, bev_nm_labeled, metrics_img)
        combined.save(os.path.join(sample_dir, "combined.png"), dpi=(OUTPUT_DPI, OUTPUT_DPI))
        saved_files['combined'] = "combined.png"

    return saved_files


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='BEVFormer detection comparison (v3)')
    p.add_argument('--lowlight-results', required=True)
    p.add_argument('--normal-results', required=True)
    p.add_argument('--lowlight-dataroot', required=True)
    p.add_argument('--normal-dataroot', default=None)
    p.add_argument('--nusc-version', default='v1.0-mini')
    p.add_argument('--out-dir', default='runs/visual_comparison')
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--score-thresh', type=float, default=0.25)
    p.add_argument('--front-only', action='store_true')
    p.add_argument('--all-cams', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cameras = ['CAM_FRONT'] if not args.all_cams else CAMS
    meta_root = args.normal_dataroot or args.lowlight_dataroot

    print(f"Loading NuScenes {args.nusc_version} from {meta_root} ...")
    nusc = NuScenes(version=args.nusc_version, dataroot=meta_root, verbose=True)
    print(f"Loading results ...")
    ll_data = mmcv.load(args.lowlight_results)
    norm_data = mmcv.load(args.normal_results)

    common_tokens = sorted(set(ll_data['results']) & set(norm_data['results']))
    print(f"Common samples: {len(common_tokens)}")

    if args.num_samples > 0:
        step = max(1, len(common_tokens) // args.num_samples)
        selected = common_tokens[::step][:args.num_samples]
    else:
        selected = common_tokens
    print(f"Processing {len(selected)} samples ...")

    # Legend
    make_legend_image().save(os.path.join(args.out_dir, 'legend.png'))

    # Global metrics
    all_ll = [r for preds in ll_data['results'].values() for r in preds]
    all_nm = [r for preds in norm_data['results'].values() for r in preds]
    gll = compute_sample_stats(all_ll, args.score_thresh)
    gnm = compute_sample_stats(all_nm, args.score_thresh)
    render_metrics_panel(gll, gnm, width=700, height=900).save(
        os.path.join(args.out_dir, 'global_metrics.png'), dpi=(OUTPUT_DPI, OUTPUT_DPI))

    # Manifest for web viewer
    manifest = {
        'version': '3.0',
        'cameras': cameras,
        'score_threshold': args.score_thresh,
        'global_stats': {'lowlight': gll, 'normal': gnm},
        'samples': [],
    }

    for idx, token in enumerate(tqdm(selected, desc='Rendering'), 1):
        sample_info = process_sample(
            nusc, token,
            ll_data['results'][token], norm_data['results'][token],
            args.lowlight_dataroot, args.score_thresh,
            cameras, args.out_dir, idx)
        manifest['samples'].append(sample_info)

    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  ✓ Complete! Output: {args.out_dir}")
    print(f"  ✓ {len(selected)} samples, each with separate images")
    print(f"  ✓ manifest.json generated for web viewer")
    print(f"{'═' * 60}")


if __name__ == '__main__':
    main()
