import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Compare BEVFormer Detection Metrics")
    parser.add_argument('--ll-metrics', type=str, required=True, help='Path to low-light metrics summary json')
    parser.add_argument('--nm-metrics', type=str, required=True, help='Path to normal metrics summary json')
    parser.add_argument('--out-dir', type=str, default='runs/metrics_comparison', help='Output directory')
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def clean_label(label):
    return label.replace('_', ' ').title()

def plot_radar_chart(ll_data, nm_data, out_path):
    # NDS, mAP, and TP Error scores (scores = 1.0 - error)
    # The higher the score, the better
    categories = ['NDS', 'mAP', 'mATE (Score)', 'mASE (Score)', 'mAOE (Score)', 'mAVE (Score)', 'mAAE (Score)']
    N = len(categories)

    def extract_scores(data):
        return [
            data['nd_score'],
            data['mean_ap'],
            data['tp_scores']['trans_err'],
            data['tp_scores']['scale_err'],
            data['tp_scores']['orient_err'],
            data['tp_scores']['vel_err'],
            data['tp_scores']['attr_err']
        ]

    ll_scores = extract_scores(ll_data)
    nm_scores = extract_scores(nm_data)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ll_scores += ll_scores[:1]
    nm_scores += nm_scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=9)
    plt.ylim(0, 1)

    # Plot normal
    ax.plot(angles, nm_scores, linewidth=2, linestyle='solid', label='Normal (High-light)', color='#1f77b4')
    ax.fill(angles, nm_scores, '#1f77b4', alpha=0.1)

    # Plot low-light
    ax.plot(angles, ll_scores, linewidth=2, linestyle='solid', label='Low-light', color='#ff7f0e')
    ax.fill(angles, ll_scores, '#ff7f0e', alpha=0.1)

    plt.title('Global Detection Metrics Comparison\n(Higher is Better)', size=14, color='#333333', weight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(ll_data, nm_data, out_path):
    classes = list(nm_data['mean_dist_aps'].keys())
    
    ll_aps = [ll_data['mean_dist_aps'].get(c, 0) for c in classes]
    nm_aps = [nm_data['mean_dist_aps'].get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, nm_aps, width, label='Normal (High-light)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, ll_aps, width, label='Low-light', color='#ff7f0e')

    ax.set_ylabel('mean Average Precision (mAP)', fontsize=12, weight='bold')
    ax.set_title('Per-Class mAP Comparison', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(c) for c in classes], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)

    # Add numeric labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_csv_table(ll_data, nm_data, out_path):
    rows = []
    
    # Global
    global_metrics = [
        ('NDS', 'nd_score'),
        ('mAP', 'mean_ap'),
        ('mATE', ('tp_errors', 'trans_err')),
        ('mASE', ('tp_errors', 'scale_err')),
        ('mAOE', ('tp_errors', 'orient_err')),
        ('mAVE', ('tp_errors', 'vel_err')),
        ('mAAE', ('tp_errors', 'attr_err'))
    ]
    
    def get_val(data, key):
        if isinstance(key, tuple):
            return data[key[0]].get(key[1], float('nan'))
        return data.get(key, float('nan'))
        
    for name, key in global_metrics:
        ll_val = get_val(ll_data, key)
        nm_val = get_val(nm_data, key)
        diff = ll_val - nm_val
        rows.append({
            'Metric Category': 'Global',
            'Metric Name': name,
            'Normal': round(nm_val, 4) if pd.notna(nm_val) else 'N/A',
            'Low-Light': round(ll_val, 4) if pd.notna(ll_val) else 'N/A',
            'Absolute Diff (LL - NM)': round(diff, 4) if pd.notna(diff) else 'N/A'
        })
        
    # Per-class AP
    for cls in nm_data['mean_dist_aps']:
        nm_val = nm_data['mean_dist_aps'].get(cls, float('nan'))
        ll_val = ll_data['mean_dist_aps'].get(cls, float('nan'))
        diff = ll_val - nm_val
        rows.append({
            'Metric Category': 'Per-Class mAP',
            'Metric Name': clean_label(cls),
            'Normal': round(nm_val, 4) if pd.notna(nm_val) else 'N/A',
            'Low-Light': round(ll_val, 4) if pd.notna(ll_val) else 'N/A',
            'Absolute Diff (LL - NM)': round(diff, 4) if pd.notna(diff) else 'N/A'
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    
    # Also save as markdown for easy console viewing / GitHub integration
    md_path = out_path.replace('.csv', '.md')
    with open(md_path, 'w') as f:
        f.write("# BEVFormer Detection Metrics Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Note: For mAP and NDS, higher is better. For mATE, mASE, mAOE, mAVE, mAAE, lower is better.*\n")

def plot_pr_curves(ll_details_path, nm_details_path, out_dir):
    if not os.path.exists(ll_details_path) or not os.path.exists(nm_details_path):
        print("metrics_details.json not found, skipping PR curves.")
        return
        
    ll_det = load_json(ll_details_path)
    nm_det = load_json(nm_details_path)
    
    # We will plot PR curves for the standard 2.0m matching threshold to avoid clutter
    threshold = "2.0"
    classes = ['car', 'pedestrian', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, cls in enumerate(classes):
        key = f"{cls}:{threshold}"
        ax = axes[idx]
        
        if key in nm_det and 'recall' in nm_det[key] and 'precision' in nm_det[key]:
            ax.plot(nm_det[key]['recall'], nm_det[key]['precision'], label="Normal (High-light)", color='#1f77b4', linewidth=2)
        if key in ll_det and 'recall' in ll_det[key] and 'precision' in ll_det[key]:
            ax.plot(ll_det[key]['recall'], ll_det[key]['precision'], label="Low-light", color='#ff7f0e', linewidth=2)
            
        ax.set_title(f"PR Curve: {clean_label(cls)}\n(Dist Thresh: {threshold}m)", weight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc='lower left')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "pr_curves_comparison.png"), dpi=300)
    plt.close()

def plot_strictness_ap(ll_data, nm_data, out_path):
    # Plot AP decay across different matching thresholds (0.5m, 1.0m, 2.0m, 4.0m)
    thresholds = ["0.5", "1.0", "2.0", "4.0"]
    
    # Calculate mean AP at each threshold
    def get_mean_ap_at_thresh(data, thresh):
        aps = []
        for cls, t_dict in data.get('label_aps', {}).items():
            if thresh in t_dict:
                aps.append(t_dict[thresh])
        return np.mean(aps) if aps else 0.0

    ll_thresh_aps = [get_mean_ap_at_thresh(ll_data, t) for t in thresholds]
    nm_thresh_aps = [get_mean_ap_at_thresh(nm_data, t) for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, nm_thresh_aps, width, label='Normal (High-light)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, ll_thresh_aps, width, label='Low-light', color='#ff7f0e')

    ax.set_ylabel('Mean AP across classes', fontsize=12, weight='bold')
    ax.set_xlabel('Matching Strictness Distance Threshold (meters)', fontsize=12, weight='bold')
    ax.set_title('mAP Decay by Localization Strictness', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}m" for t in thresholds])
    ax.legend(fontsize=10)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    if not os.path.exists(args.ll_metrics):
        print(f"Error: Low-light metrics file not found: {args.ll_metrics}")
        return
    if not os.path.exists(args.nm_metrics):
        print(f"Error: Normal metrics file not found: {args.nm_metrics}")
        return
        
    print(f"Loading Low-Light Metrics: {args.ll_metrics}")
    ll_data = load_json(args.ll_metrics)
    
    print(f"Loading Normal Metrics: {args.nm_metrics}")
    nm_data = load_json(args.nm_metrics)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Generating Radar Chart (Global Metrics) ...")
    plot_radar_chart(ll_data, nm_data, os.path.join(args.out_dir, "radar_global_metrics.pdf"))
    plot_radar_chart(ll_data, nm_data, os.path.join(args.out_dir, "radar_global_metrics.png"))
    
    print("Generating Bar Chart (Per-Class mAP) ...")
    plot_bar_chart(ll_data, nm_data, os.path.join(args.out_dir, "bar_per_class_map.pdf"))
    plot_bar_chart(ll_data, nm_data, os.path.join(args.out_dir, "bar_per_class_map.png"))
    
    print("Generating Strictness AP Decay Chart ...")
    plot_strictness_ap(ll_data, nm_data, os.path.join(args.out_dir, "bar_strictness_map_decay.pdf"))
    plot_strictness_ap(ll_data, nm_data, os.path.join(args.out_dir, "bar_strictness_map_decay.png"))
    
    print("Generating Detailed CSV Table ...")
    generate_csv_table(ll_data, nm_data, os.path.join(args.out_dir, "metrics_comparison_table.csv"))
    
    # Try to plot PR curves if details are available
    ll_details = args.ll_metrics.replace('metrics_summary.json', 'metrics_details.json')
    nm_details = args.nm_metrics.replace('metrics_summary.json', 'metrics_details.json')
    print("Generating PR Curves ...")
    plot_pr_curves(ll_details, nm_details, args.out_dir)
    
    print(f"\n✅ All metric comparisons saved to {args.out_dir}/")

if __name__ == '__main__':
    main()
