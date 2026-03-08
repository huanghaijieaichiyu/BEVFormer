"""
Image Domain Analysis: Three-Way Comparison (v3 — Scientifically Correct)
=========================================================================
Compare Normal / Real-Night / Synthetic Low-Light images.

Scientific notes:
  - Grayscale: ITU-R BT.601 (Y = 0.299R + 0.587G + 0.114B)
  - Histograms computed on FULL-RESOLUTION images (ALL pixels)
  - t-SNE features use resized thumbnails (acceptable for embedding)
  - Density=True normalizes histogram area to 1 (probability density)

Sampling:
  - Normal daytime:     first N from normal dataset
  - Real nighttime:     last  N from normal dataset
  - Synthetic low-light: PAIRED with above (same N_front + N_back)

Outputs (300 DPI, PDF + PNG):
  ── Individual (per-domain) ──
  histogram_gray_Normal.pdf
  histogram_gray_RealNight.pdf
  histogram_gray_Synthetic.pdf
  histogram_rgb_Normal.pdf
  histogram_rgb_RealNight.pdf
  histogram_rgb_Synthetic.pdf
  ── Comparison (all domains) ──
  histogram_grayscale.pdf    — overlaid grayscale
  histogram_rgb.pdf          — 3-panel RGB
  brightness_boxplot.pdf
  tsne_comparison.pdf
  tsne_density.pdf
  combined_analysis.pdf      — 2×2 for paper
  statistics_table.pdf
  statistics.csv / .json
"""

import argparse
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from tqdm import tqdm

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from nuscenes.nuscenes import NuScenes

# ── Journal matplotlib style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
})

# ── Three-class color palette ──
C_NORMAL    = '#2166ac'
C_NIGHT     = '#b2182b'
C_SYNTHETIC = '#e08214'
C_NORMAL_L  = '#92c5de'
C_NIGHT_L   = '#f4a582'
C_SYNTH_L   = '#fdd49e'

DOMAIN_COLORS = [C_NORMAL, C_NIGHT, C_SYNTHETIC]
DOMAIN_LIGHT  = [C_NORMAL_L, C_NIGHT_L, C_SYNTH_L]


# ═══════════════════════════════════════════════════════════════════════════
#  Data Collection & Statistics
# ═══════════════════════════════════════════════════════════════════════════

def collect_paired_paths(nusc, normal_root, lowlight_root, camera):
    """Collect paired (normal, lowlight) paths in NuScenes sample order."""
    pairs = []
    for sample in nusc.sample:
        if camera in sample['data']:
            sd = nusc.get('sample_data', sample['data'][camera])
            rel = sd['filename']
            nm_p = os.path.join(normal_root, rel)
            ll_p = os.path.join(lowlight_root, rel)
            if os.path.exists(nm_p) and os.path.exists(ll_p):
                pairs.append((nm_p, ll_p, rel))
    return pairs


TSNE_RESIZE = (320, 180)  # only for t-SNE features


def analyze_images(paths):
    """Analyze a set of images at FULL RESOLUTION.

    For each image:
      1. Load at original resolution (no resize)
      2. Convert to grayscale via ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
      3. Compute 256-bin histogram over ALL pixels (density normalized)
      4. Compute per-channel 256-bin histograms over ALL pixels
      5. Compute mean brightness, per-channel means

    For t-SNE features only:
      Resize to TSNE_RESIZE for fixed-dimension feature extraction.

    Returns dict with:
      gray_hists  (N, 256)      — per-image grayscale histogram (density)
      rgb_hists   (N, 3, 256)   — per-image per-channel histogram (density)
      features    (N, feat_dim) — for t-SNE (from resized)
      brightness  (N,)          — per-image mean grayscale brightness
      rgb_means   (N, 3)        — per-image per-channel mean
      total_pixels int          — total pixels analyzed
    """
    gray_hists, rgb_hists = [], []
    features, brightness, rgb_means = [], [], []
    total_px = 0

    for path in tqdm(paths, desc='  Analyzing (full-res)', leave=False):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        total_px += w * h

        # ── Full-resolution analysis ──
        arr = np.array(img, dtype=np.float32)  # (H, W, 3), NO resize

        # Grayscale: BT.601
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gray_flat = gray.ravel()
        h_g, _ = np.histogram(gray_flat, bins=256, range=(0, 255), density=True)
        gray_hists.append(h_g)
        brightness.append(float(gray_flat.mean()))

        # Per-channel
        ch_h, ch_m = [], []
        for c in range(3):
            ch_flat = arr[:, :, c].ravel()
            hc, _ = np.histogram(ch_flat, bins=256, range=(0, 255), density=True)
            ch_h.append(hc)
            ch_m.append(float(ch_flat.mean()))
        rgb_hists.append(ch_h)
        rgb_means.append(ch_m)

        # ── t-SNE features (resized is OK) ──
        arr_s = np.array(img.resize(TSNE_RESIZE, Image.LANCZOS), dtype=np.float32)
        feat_parts = []
        for c in range(3):
            hf, _ = np.histogram(arr_s[:, :, c], bins=16, range=(0, 255), density=True)
            feat_parts.append(hf)
        gh, gw = 4, 4
        bh, bw = TSNE_RESIZE[1] // gh, TSNE_RESIZE[0] // gw
        for gy in range(gh):
            for gx in range(gw):
                patch = arr_s[gy*bh:(gy+1)*bh, gx*bw:(gx+1)*bw]
                feat_parts.append([patch[:, :, c].mean() / 255. for c in range(3)])
                feat_parts.append([patch[:, :, c].std() / 255. for c in range(3)])
        features.append(np.concatenate([np.array(p).ravel() for p in feat_parts]))

    return {
        'gray_hists': np.array(gray_hists),
        'rgb_hists':  np.array(rgb_hists),
        'features':   np.array(features),
        'brightness': np.array(brightness),
        'rgb_means':  np.array(rgb_means),
        'total_pixels': total_px,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Individual Per-Domain Histogram (saved separately)
# ═══════════════════════════════════════════════════════════════════════════

def plot_single_gray_histogram(ghists, bright, label, color, out_dir):
    """Single-domain grayscale histogram."""
    bins = np.arange(256)
    mean_h = ghists.mean(axis=0)
    std_h = ghists.std(axis=0)
    mu = bright.mean()

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.fill_between(bins, mean_h - std_h, mean_h + std_h,
                    alpha=0.2, color=color, linewidth=0)
    ax.plot(bins, mean_h, color=color, linewidth=1.6,
            label=f'{label} (μ={mu:.1f}, n={len(ghists)})')
    ax.axvline(mu, color=color, ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Grayscale Distribution — {label}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 255); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    tag = label.replace(' ', '')
    fig.savefig(os.path.join(out_dir, f'histogram_gray_{tag}.pdf'))
    fig.savefig(os.path.join(out_dir, f'histogram_gray_{tag}.png'))
    plt.close(fig)


def plot_single_rgb_histogram(rgb_hists, label, color, out_dir):
    """Single-domain RGB histogram (3-panel)."""
    bins = np.arange(256)
    ch_names = ['Red', 'Green', 'Blue']
    ch_pure = ['#cc3333', '#33aa33', '#3333cc']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ch, ax in enumerate(axes):
        mean_h = rgb_hists[:, ch].mean(axis=0)
        ax.fill_between(bins, mean_h, alpha=0.3, color=ch_pure[ch], linewidth=0)
        ax.plot(bins, mean_h, color=ch_pure[ch], linewidth=1.0)
        ax.set_xlabel('Intensity')
        ax.set_title(ch_names[ch], fontsize=10)
        ax.set_xlim(0, 255); ax.set_ylim(bottom=0)
        if ch == 0:
            ax.set_ylabel('Probability Density')
    fig.suptitle(f'{label}', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(w_pad=0.3)
    tag = label.replace(' ', '')
    fig.savefig(os.path.join(out_dir, f'histogram_rgb_{tag}.pdf'))
    fig.savefig(os.path.join(out_dir, f'histogram_rgb_{tag}.png'))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_grayscale_comparison(domains, out_dir):
    """Overlaid 3-class grayscale histogram."""
    bins = np.arange(256)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for i, (d, label) in enumerate(domains):
        mean_h = d['gray_hists'].mean(axis=0)
        std_h = d['gray_hists'].std(axis=0)
        mu = d['brightness'].mean()
        ax.fill_between(bins, mean_h - std_h, mean_h + std_h,
                        alpha=0.12, color=DOMAIN_COLORS[i], linewidth=0)
        ax.plot(bins, mean_h, color=DOMAIN_COLORS[i], linewidth=1.6,
                label=f'{label} (μ={mu:.1f})')
        ax.axvline(mu, color=DOMAIN_COLORS[i], ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Probability Density')
    ax.set_title('Grayscale Intensity Distribution')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc')
    ax.set_xlim(0, 255); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'histogram_grayscale.pdf'))
    fig.savefig(os.path.join(out_dir, 'histogram_grayscale.png'))
    plt.close(fig)
    print(f"  ✓ histogram_grayscale.pdf")


def plot_rgb_comparison(domains, out_dir):
    """3-panel RGB comparison."""
    bins = np.arange(256)
    ch_names = ['Red', 'Green', 'Blue']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ch, ax in enumerate(axes):
        for i, (d, label) in enumerate(domains):
            mean_h = d['rgb_hists'][:, ch].mean(axis=0)
            ax.fill_between(bins, mean_h, alpha=0.2, color=DOMAIN_COLORS[i], linewidth=0)
            ax.plot(bins, mean_h, color=DOMAIN_COLORS[i], linewidth=1.0,
                    label=label if ch == 0 else None)
        ax.set_xlabel('Intensity')
        ax.set_title(ch_names[ch], fontsize=10)
        ax.set_xlim(0, 255); ax.set_ylim(bottom=0)
        if ch == 0:
            ax.set_ylabel('Probability Density')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    plt.tight_layout(w_pad=0.3)
    fig.savefig(os.path.join(out_dir, 'histogram_rgb.pdf'))
    fig.savefig(os.path.join(out_dir, 'histogram_rgb.png'))
    plt.close(fig)
    print(f"  ✓ histogram_rgb.pdf")


def plot_brightness_boxplot(domains, out_dir):
    """Violin + box plot."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    data = [d['brightness'] for d, _ in domains]
    labels = [l for _, l in domains]
    pos = list(range(1, len(data) + 1))
    parts = ax.violinplot(data, positions=pos, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(DOMAIN_LIGHT[i]); pc.set_alpha(0.5)
    for key in ['cmins', 'cmaxes', 'cbars', 'cmeans', 'cmedians']:
        if key in parts:
            parts[key].set_color('#444')
    bp = ax.boxplot(data, positions=pos, widths=0.15, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DOMAIN_COLORS[i]); patch.set_alpha(0.7)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean Brightness (0–255)')
    ax.set_title('Brightness Distribution')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'brightness_boxplot.pdf'))
    fig.savefig(os.path.join(out_dir, 'brightness_boxplot.png'))
    plt.close(fig)
    print(f"  ✓ brightness_boxplot.pdf")


def plot_tsne(domains, out_dir):
    """t-SNE scatter + density contours."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  ⚠ pip install scikit-learn for t-SNE")
        return None

    feats_list = [d['features'] for d, _ in domains]
    labels = [l for _, l in domains]
    sizes = [len(f) for f in feats_list]
    all_f = StandardScaler().fit_transform(np.vstack(feats_list))

    perp = max(5, min(30, len(all_f) // 4))
    print(f"  Computing t-SNE (perplexity={perp}, n={len(all_f)}) ...")
    emb = TSNE(n_components=2, random_state=42, perplexity=perp,
               learning_rate='auto', init='pca', n_iter=1000).fit_transform(all_f)

    splits = []
    idx = 0
    for s in sizes:
        splits.append(emb[idx:idx+s]); idx += s

    # ── Scatter ──
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    for i, (pts, label) in enumerate(zip(splits, labels)):
        ax.scatter(pts[:, 0], pts[:, 1], c=DOMAIN_COLORS[i], s=20, alpha=0.65,
                   edgecolors='white', linewidths=0.3, label=label, zorder=3)
    ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Feature Embedding')
    ax.legend(loc='best', framealpha=0.9, edgecolor='#ccc', markerscale=1.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tsne_comparison.pdf'))
    fig.savefig(os.path.join(out_dir, 'tsne_comparison.png'))
    plt.close(fig)
    print(f"  ✓ tsne_comparison.pdf")

    # ── KDE contours ──
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    for i, (pts, label) in enumerate(zip(splits, labels)):
        ax.scatter(pts[:, 0], pts[:, 1], c=DOMAIN_COLORS[i], s=14, alpha=0.4,
                   edgecolors='none', label=label, zorder=3)
    try:
        from scipy.stats import gaussian_kde
        for i, pts in enumerate(splits):
            if len(pts) > 10:
                kde = gaussian_kde(pts.T)
                xr = pts[:, 0].max() - pts[:, 0].min()
                yr = pts[:, 1].max() - pts[:, 1].min()
                pad = max(xr, yr) * 0.15
                xmin, xmax = pts[:, 0].min() - pad, pts[:, 0].max() + pad
                ymin, ymax = pts[:, 1].min() - pad, pts[:, 1].max() + pad
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=4, colors=DOMAIN_COLORS[i],
                           linewidths=0.8, alpha=0.7, zorder=2)
    except ImportError:
        pass
    ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE with Density Contours')
    ax.legend(loc='best', framealpha=0.9, edgecolor='#ccc', markerscale=1.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tsne_density.pdf'))
    fig.savefig(os.path.join(out_dir, 'tsne_density.png'))
    plt.close(fig)
    print(f"  ✓ tsne_density.pdf")
    return splits


# ═══════════════════════════════════════════════════════════════════════════
#  Combined 2×2 Panel
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined_panel(domains, out_dir):
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return

    bins = np.arange(256)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))

    # (a) Grayscale
    ax = axes[0, 0]
    for i, (d, label) in enumerate(domains):
        mu = d['brightness'].mean()
        ax.fill_between(bins, d['gray_hists'].mean(0), alpha=0.2,
                        color=DOMAIN_COLORS[i], lw=0)
        ax.plot(bins, d['gray_hists'].mean(0), color=DOMAIN_COLORS[i], lw=1.2,
                label=f'{label} (μ={mu:.1f})')
    ax.set_xlabel('Pixel Intensity'); ax.set_ylabel('Prob. Density')
    ax.set_title('(a) Grayscale Distribution')
    ax.legend(fontsize=6.5, loc='upper right'); ax.set_xlim(0, 255)

    # (b) Brightness violin
    ax = axes[0, 1]
    data = [d['brightness'] for d, _ in domains]
    labels_b = [l for _, l in domains]
    pos = list(range(1, len(data) + 1))
    parts = ax.violinplot(data, positions=pos, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(DOMAIN_LIGHT[i]); pc.set_alpha(0.5)
    bp = ax.boxplot(data, positions=pos, widths=0.15, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DOMAIN_COLORS[i]); patch.set_alpha(0.7)
    ax.set_xticks(pos); ax.set_xticklabels(labels_b, fontsize=7)
    ax.set_ylabel('Mean Brightness'); ax.set_title('(b) Brightness Distribution')

    # (c) Per-channel bar
    ax = axes[1, 0]
    channels = ['R', 'G', 'B']
    x = np.arange(3)
    nd = len(domains)
    bw = 0.7 / nd
    for i, (d, label) in enumerate(domains):
        mc = d['rgb_means'].mean(axis=0)
        offset = (i - (nd - 1) / 2) * bw
        bars = ax.bar(x + offset, mc, bw, color=DOMAIN_COLORS[i], alpha=0.8, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(channels)
    ax.set_ylabel('Mean Intensity'); ax.set_title('(c) Per-Channel Mean')
    ax.legend(fontsize=6.5)

    # (d) t-SNE
    ax = axes[1, 1]
    fl = [d['features'] for d, _ in domains]
    ll = [l for _, l in domains]
    ss = [len(f) for f in fl]
    af = StandardScaler().fit_transform(np.vstack(fl))
    perp = max(5, min(30, len(af) // 4))
    emb = TSNE(n_components=2, random_state=42, perplexity=perp,
               learning_rate='auto', init='pca', n_iter=800).fit_transform(af)
    idx = 0
    for i, s in enumerate(ss):
        ax.scatter(emb[idx:idx+s, 0], emb[idx:idx+s, 1], c=DOMAIN_COLORS[i],
                   s=8, alpha=0.5, edgecolors='none', label=ll[i])
        idx += s
    ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
    ax.set_title('(d) t-SNE Embedding')
    ax.legend(fontsize=6.5, markerscale=2)
    ax.set_xticklabels([]); ax.set_yticklabels([])

    plt.tight_layout(h_pad=0.8, w_pad=0.8)
    fig.savefig(os.path.join(out_dir, 'combined_analysis.pdf'))
    fig.savefig(os.path.join(out_dir, 'combined_analysis.png'))
    plt.close(fig)
    print(f"  ✓ combined_analysis.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(data, label):
    b = data['brightness']
    rm = data['rgb_means']
    return {
        'domain': label,
        'num_images': int(len(b)),
        'total_pixels': int(data['total_pixels']),
        'mean_brightness': round(float(b.mean()), 2),
        'std_brightness': round(float(b.std()), 2),
        'median_brightness': round(float(np.median(b)), 2),
        'min_brightness': round(float(b.min()), 2),
        'max_brightness': round(float(b.max()), 2),
        'mean_R': round(float(rm[:, 0].mean()), 2),
        'mean_G': round(float(rm[:, 1].mean()), 2),
        'mean_B': round(float(rm[:, 2].mean()), 2),
    }


def save_statistics(stats_list, out_dir):
    # JSON
    with open(os.path.join(out_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)

    # CSV
    fields = list(stats_list[0].keys())
    with open(os.path.join(out_dir, 'statistics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(stats_list)
    print(f"  ✓ statistics.csv")

    # Table image
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axis('off')
    header = ['Metric'] + [s['domain'] for s in stats_list]
    rows = [
        ['# Images']       + [str(s['num_images']) for s in stats_list],
        ['Total Pixels']   + [f"{s['total_pixels']:,}" for s in stats_list],
        ['Mean Gray']      + [f"{s['mean_brightness']:.1f}" for s in stats_list],
        ['Std Dev']        + [f"{s['std_brightness']:.1f}" for s in stats_list],
        ['Median']         + [f"{s['median_brightness']:.1f}" for s in stats_list],
        ['Min / Max']      + [f"{s['min_brightness']:.0f} / {s['max_brightness']:.0f}" for s in stats_list],
        ['Mean R']         + [f"{s['mean_R']:.1f}" for s in stats_list],
        ['Mean G']         + [f"{s['mean_G']:.1f}" for s in stats_list],
        ['Mean B']         + [f"{s['mean_B']:.1f}" for s in stats_list],
    ]
    nc = len(header)
    cw = [0.20] + [0.80 / (nc - 1)] * (nc - 1)
    table = ax.table(cellText=rows, colLabels=header, cellLoc='center',
                     loc='center', colWidths=cw)
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    for j in range(nc):
        c = table[0, j]
        c.set_facecolor(DOMAIN_COLORS[j - 1] if j > 0 else '#333')
        c.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(nc):
            c = table[i, j]
            c.set_facecolor('#f0f4f8' if i % 2 == 0 else 'white')
            c.set_edgecolor('#ccc')
    ax.set_title('Image Domain Statistics', pad=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'statistics_table.pdf'))
    fig.savefig(os.path.join(out_dir, 'statistics_table.png'))
    plt.close(fig)
    print(f"  ✓ statistics_table.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Three-way image domain analysis (full-resolution histograms)')
    p.add_argument('--lowlight-dataroot', required=True)
    p.add_argument('--normal-dataroot', required=True)
    p.add_argument('--nusc-version', default='v1.0-mini')
    p.add_argument('--out-dir', default='runs/domain_analysis')
    p.add_argument('--num-normal', type=int, default=20,
                   help='Normal daytime count (from front of dataset)')
    p.add_argument('--num-night', type=int, default=20,
                   help='Real nighttime count (from end of dataset)')
    p.add_argument('--camera', default='CAM_FRONT')
    p.add_argument('--all-cams', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'] \
              if args.all_cams else [args.camera]

    print(f"{'═' * 65}")
    print(f"  Three-Way Image Domain Analysis (Full-Resolution)")
    print(f"{'═' * 65}")
    print(f"  Normal Root:      {args.normal_dataroot}")
    print(f"  Low-Light Root:   {args.lowlight_dataroot}")
    print(f"  Cameras:          {cameras}")
    print(f"  Normal (front):   {args.num_normal}")
    print(f"  Real Night (end): {args.num_night}")
    print(f"  Synthetic:        paired (={args.num_normal + args.num_night})")
    print(f"  Output:           {args.out_dir}")
    print(f"{'═' * 65}")
    print(f"  NOTE: Histograms use ALL pixels at ORIGINAL resolution")
    print(f"        Grayscale: Y = 0.299R + 0.587G + 0.114B (BT.601)")
    print(f"{'═' * 65}")

    print(f"\nLoading NuScenes {args.nusc_version} ...")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.normal_dataroot, verbose=True)

    # ── Paired paths ──
    print("\nCollecting paired images ...")
    all_pairs = []
    for cam in cameras:
        all_pairs.extend(collect_paired_paths(
            nusc, args.normal_dataroot, args.lowlight_dataroot, cam))
    total = len(all_pairs)
    print(f"  Total paired samples: {total}")

    front_pairs = all_pairs[:args.num_normal]
    back_pairs  = all_pairs[-args.num_night:]

    normal_paths  = [p[0] for p in front_pairs]
    night_paths   = [p[0] for p in back_pairs]
    synth_paths   = [p[1] for p in front_pairs] + [p[1] for p in back_pairs]

    print(f"  Normal:    {len(normal_paths)} (idx 0..{min(args.num_normal, total)-1})")
    print(f"  Night:     {len(night_paths)} (idx {max(0, total-args.num_night)}..{total-1})")
    print(f"  Synthetic: {len(synth_paths)} (paired with above)")

    if not normal_paths or not night_paths or not synth_paths:
        print("\nERROR: No images found."); return

    # ── Analyze ──
    print("\n[1/3] Analyzing Normal images (full resolution) ...")
    d_normal = analyze_images(normal_paths)
    print(f"       {d_normal['total_pixels']:,} total pixels")

    print("[2/3] Analyzing Real Night images ...")
    d_night = analyze_images(night_paths)
    print(f"       {d_night['total_pixels']:,} total pixels")

    print("[3/3] Analyzing Synthetic images ...")
    d_synth = analyze_images(synth_paths)
    print(f"       {d_synth['total_pixels']:,} total pixels")

    domains = [(d_normal, 'Normal'), (d_night, 'Real Night'), (d_synth, 'Synthetic')]

    # ── Generate figures ──
    print(f"\nGenerating figures to {args.out_dir}/ ...")

    # Individual per-domain
    for i, (d, label) in enumerate(domains):
        plot_single_gray_histogram(d['gray_hists'], d['brightness'],
                                   label, DOMAIN_COLORS[i], args.out_dir)
        plot_single_rgb_histogram(d['rgb_hists'], label, DOMAIN_COLORS[i], args.out_dir)
    print(f"  ✓ 6 individual histograms")

    # Comparison
    plot_grayscale_comparison(domains, args.out_dir)
    plot_rgb_comparison(domains, args.out_dir)
    plot_brightness_boxplot(domains, args.out_dir)
    plot_tsne(domains, args.out_dir)

    # Statistics
    stats = [compute_stats(d, l) for d, l in domains]
    save_statistics(stats, args.out_dir)

    # Combined panel
    plot_combined_panel(domains, args.out_dir)

    # ── Console table ──
    print(f"\n{'═' * 72}")
    print(f"  统计数据 (已保存至 statistics.csv, 可自行调整)")
    print(f"{'═' * 72}")
    header = f"{'Metric':<20s}"
    for s in stats:
        header += f" {s['domain']:>16s}"
    print(header)
    print("─" * 72)
    for key in ['num_images', 'total_pixels', 'mean_brightness', 'std_brightness',
                'median_brightness', 'min_brightness', 'max_brightness',
                'mean_R', 'mean_G', 'mean_B']:
        row = f"{key:<20s}"
        for s in stats:
            v = s[key]
            if isinstance(v, int):
                row += f" {v:>16,}"
            else:
                row += f" {v:>16.2f}"
        print(row)
    print(f"{'═' * 72}")

    print(f"\n  ✓ 全部完成! 输出: {args.out_dir}/")
    print(f"{'═' * 72}")
    print(f"  单域图:")
    for label in ['Normal', 'RealNight', 'Synthetic']:
        print(f"    histogram_gray_{label}.pdf + histogram_rgb_{label}.pdf")
    print(f"  对比图:")
    print(f"    histogram_grayscale.pdf — 三类灰度叠加")
    print(f"    histogram_rgb.pdf       — RGB通道对比")
    print(f"    brightness_boxplot.pdf  — 亮度分布")
    print(f"    tsne_comparison.pdf     — t-SNE散点")
    print(f"    tsne_density.pdf        — t-SNE+KDE")
    print(f"    combined_analysis.pdf   — 2×2论文组合")
    print(f"  数据:")
    print(f"    statistics.csv / .json  — 可编辑统计表")
    print(f"    statistics_table.pdf    — 排版好的表格")
    print(f"{'═' * 72}")


if __name__ == '__main__':
    main()
