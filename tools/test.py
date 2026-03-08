# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Modernized with enhanced features
# ---------------------------------------------

import os
import os.path as osp
import sys
import time
import json
import argparse
import warnings

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

# Add project root to Python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
except ImportError:
    print("ImportError: Failed to import modules from projects.mmdet3d_plugin.")
    print("Please ensure that the project structure is correct and the modules are available.")
    print("For example, you can run:")
    print("export PYTHONPATH=$PYTHONPATH:$(pwd)")


# ─── Utility: Colored Console Output ───────────────────────────────────────────
class _Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

C = _Colors()


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _info(msg):
    print(f"{C.GREEN}[{_ts()}] ✓ {msg}{C.RESET}")


def _warn(msg):
    print(f"{C.YELLOW}[{_ts()}] ⚠ {msg}{C.RESET}")


def _error(msg):
    print(f"{C.RED}[{_ts()}] ✗ {msg}{C.RESET}")


def _header(msg):
    width = 64
    print(f"\n{C.BOLD}{C.CYAN}{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}{C.RESET}\n")


# ─── Utility: Model Summary ────────────────────────────────────────────────────
def print_model_summary(model):
    """Print model parameter statistics."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _info(f'Model Parameters: {n_params:,} total, {n_trainable:,} trainable')

    # Try FLOPs estimation with fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        _info('FLOPs analysis available (fvcore installed)')
    except ImportError:
        pass


# ─── Utility: Results Formatting ───────────────────────────────────────────────
def format_eval_results(results_dict):
    """Format evaluation results as a nicely-aligned table."""
    if not isinstance(results_dict, dict):
        return str(results_dict)

    lines = []
    lines.append(f"\n{C.BOLD}{C.CYAN}{'─' * 50}")
    lines.append(f"  Evaluation Results")
    lines.append(f"{'─' * 50}{C.RESET}")

    for key, val in results_dict.items():
        if isinstance(val, float):
            lines.append(f"  {C.BOLD}{key:<30}{C.RESET} {val:.4f}")
        else:
            lines.append(f"  {C.BOLD}{key:<30}{C.RESET} {val}")

    lines.append(f"{C.CYAN}{'─' * 50}{C.RESET}\n")
    return '\n'.join(lines)


# ─── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='BEVFormer Testing & Evaluation Script (Modernized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
{C.CYAN}Examples:{C.RESET}
  # Evaluate with bbox metric
  python tools/test.py projects/configs/bevformer/bevformer_tiny.py ckpts/bevformer_tiny_epoch_24.pth --eval bbox

  # Save raw predictions to pickle
  python tools/test.py projects/configs/bevformer/bevformer_tiny.py ckpts/model.pth --out results.pkl --eval bbox

  # Save evaluation results as JSON
  python tools/test.py projects/configs/bevformer/bevformer_tiny.py ckpts/model.pth --eval bbox --save-results-dir ./eval_results

  # Fuse conv-bn for faster inference
  python tools/test.py projects/configs/bevformer/bevformer_tiny.py ckpts/model.pth --eval bbox --fuse-conv-bn
""")

    # ── Core Arguments ──
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', help='Output result file in pickle format (.pkl)')

    # ── Optimization ──
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Fuse conv and bn layers for faster inference')

    # ── Evaluation ──
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format output results without evaluation (for submission)')
    parser.add_argument(
        '--eval', type=str, nargs='+',
        help='Evaluation metrics, e.g., "bbox" for detection')
    parser.add_argument(
        '--show', action='store_true',
        help='Show results interactively')
    parser.add_argument(
        '--show-dir',
        help='Directory to save visualization results')

    # ── Result Saving ──
    parser.add_argument(
        '--save-results-dir',
        type=str, default=None,
        help='Directory to save evaluation results as JSON')

    # ── Multi-GPU ──
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='Use GPU to collect results (faster but uses more GPU memory)')
    parser.add_argument(
        '--tmpdir',
        help='Temp directory for collecting results from multiple workers')

    # ── Reproducibility ──
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic mode for CUDNN backend')

    # ── Config Overrides ──
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config settings in key=value format')
    parser.add_argument(
        '--options', nargs='+', action=DictAction,
        help='[DEPRECATED] Use --eval-options instead')
    parser.add_argument(
        '--eval-options', nargs='+', action=DictAction,
        help='Custom options for dataset.evaluate() in key=value format')

    # ── Launcher ──
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified. '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    _header('BEVFormer Evaluation')

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show) '
         'with --out, --eval, --format-only, --show, or --show-dir')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a .pkl file.')

    # ── Validate files ──
    if not osp.isfile(args.config):
        _error(f'Config file not found: {args.config}')
        sys.exit(1)

    if not osp.isfile(args.checkpoint):
        _error(f'Checkpoint file not found: {args.checkpoint}')
        sys.exit(1)

    _info(f'Config: {args.config}')
    _info(f'Checkpoint: {args.checkpoint}')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plugin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # ── Build dataset and dataloader ──
    _info('Building dataset...')
    dataset = build_dataset(cfg.data.test)
    _info(f'Test dataset: {len(dataset)} samples')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # ── Build model and load checkpoint ──
    _info('Building model...')
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        _info('FP16 inference enabled')

    _info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
        _info('Conv-BN fusion applied')

    # old versions did not save class info in checkpoints
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    # ── Print model summary ──
    print_model_summary(model)

    # ── Run inference ──
    _header('Running Inference')
    inference_start = time.time()

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    inference_time = time.time() - inference_start

    # ── Inference statistics ──
    rank, _ = get_dist_info()
    if rank == 0:
        n_samples = len(dataset)
        fps = n_samples / inference_time if inference_time > 0 else 0
        latency_ms = (inference_time / n_samples * 1000) if n_samples > 0 else 0

        _header('Inference Statistics')
        print(f"  {C.BOLD}{'Total Samples':<25}{C.RESET} {n_samples}")
        print(f"  {C.BOLD}{'Total Time':<25}{C.RESET} {inference_time:.1f}s")
        print(f"  {C.BOLD}{'Throughput (FPS)':<25}{C.RESET} {fps:.2f}")
        print(f"  {C.BOLD}{'Latency (ms/sample)':<25}{C.RESET} {latency_ms:.1f}ms")
        print()

        # ── Save raw predictions ──
        if args.out:
            _info(f'Saving raw predictions to {args.out}')
            mmcv.dump(outputs, args.out)

        # ── Evaluate ──
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))

        if args.format_only:
            dataset.format_results(outputs, **kwargs)
            _info('Results formatted for submission')

        if args.eval:
            _header('Evaluation Results')
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            eval_results = dataset.evaluate(outputs, **eval_kwargs)

            # Print formatted results
            print(format_eval_results(eval_results))

            # Save evaluation results as JSON
            if args.save_results_dir:
                mmcv.mkdir_or_exist(args.save_results_dir)
                config_name = osp.splitext(osp.basename(args.config))[0]
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                result_file = osp.join(
                    args.save_results_dir, f'eval_{config_name}_{timestamp}.json')

                save_data = {
                    'config': args.config,
                    'checkpoint': args.checkpoint,
                    'eval_metrics': args.eval,
                    'results': {k: float(v) if isinstance(v, (float, np.floating)) else v
                                for k, v in eval_results.items()},
                    'inference_stats': {
                        'total_samples': n_samples,
                        'total_time_seconds': round(inference_time, 2),
                        'fps': round(fps, 2),
                        'latency_ms': round(latency_ms, 1),
                    },
                    'timestamp': timestamp,
                }
                with open(result_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                _info(f'Evaluation results saved to: {result_file}')

        _info('Evaluation complete!')


if __name__ == '__main__':
    main()
