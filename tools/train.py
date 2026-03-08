# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Modernized with enhanced features
# ---------------------------------------------

from __future__ import division
import sys
import os
import glob
import argparse
import copy
import time
import warnings
import json

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version
from os import path as osp

# Add project root to sys.path to ensure modules can be imported
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
    """Return a formatted timestamp string."""
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


# ─── Utility: Training Summary ─────────────────────────────────────────────────
def print_training_summary(cfg, args, distributed, world_size):
    """Print a formatted training configuration summary before training."""
    _header('Training Configuration Summary')

    # Model info
    model_type = cfg.model.get('type', 'Unknown')
    backbone = 'N/A'
    if 'img_backbone' in cfg.model:
        bb = cfg.model.img_backbone
        backbone = f"{bb.get('type', '?')}-{bb.get('depth', '?')}"
    elif 'backbone' in cfg.model:
        bb = cfg.model.backbone
        backbone = f"{bb.get('type', '?')}-{bb.get('depth', '?')}"

    bev_h = cfg.model.get('pts_bbox_head', {}).get('bev_h', 'N/A')
    bev_w = cfg.model.get('pts_bbox_head', {}).get('bev_w', 'N/A')
    num_query = cfg.model.get('pts_bbox_head', {}).get('num_query', 'N/A')

    # Optimizer info
    opt_type = cfg.optimizer.get('type', 'Unknown')
    lr = cfg.optimizer.get('lr', 'N/A')
    wd = cfg.optimizer.get('weight_decay', 'N/A')

    # Data info
    samples_per_gpu = cfg.data.get('samples_per_gpu', 'N/A')
    workers_per_gpu = cfg.data.get('workers_per_gpu', 'N/A')

    # Schedule info
    total_epochs = cfg.runner.get('max_epochs', cfg.get('total_epochs', 'N/A'))
    lr_policy = cfg.lr_config.get('policy', 'N/A') if hasattr(cfg, 'lr_config') else 'N/A'

    rows = [
        ('Model Type', model_type),
        ('Backbone', backbone),
        ('BEV Size', f'{bev_h} x {bev_w}'),
        ('Num Queries', num_query),
        ('', ''),
        ('Optimizer', f'{opt_type} (lr={lr}, wd={wd})'),
        ('LR Policy', lr_policy),
        ('Total Epochs', total_epochs),
        ('', ''),
        ('Distributed', distributed),
        ('World Size', world_size),
        ('Samples/GPU', samples_per_gpu),
        ('Workers/GPU', workers_per_gpu),
        ('', ''),
        ('Work Dir', cfg.work_dir),
        ('Seed', args.seed),
        ('Deterministic', args.deterministic),
    ]

    if cfg.get('resume_from'):
        rows.append(('Resume From', cfg.resume_from))
    if cfg.get('load_from'):
        rows.append(('Load From', cfg.load_from))

    for key, val in rows:
        if key == '':
            print(f"  {C.DIM}{'─' * 50}{C.RESET}")
        else:
            print(f"  {C.BOLD}{key:<18}{C.RESET} {val}")

    print()


# ─── Utility: Auto Resume ──────────────────────────────────────────────────────
def find_latest_checkpoint(work_dir):
    """Find the latest checkpoint in work_dir by modification time."""
    if not osp.isdir(work_dir):
        return None

    pth_files = glob.glob(osp.join(work_dir, '*.pth'))
    if not pth_files:
        return None

    # Filter out 'latest.pth' symlinks and find by modification time
    pth_files = [f for f in pth_files if not osp.islink(f) or osp.basename(f) != 'latest.pth']
    # Prefer 'latest.pth' symlink if it exists
    latest_link = osp.join(work_dir, 'latest.pth')
    if osp.exists(latest_link):
        return latest_link

    if not pth_files:
        return None

    return max(pth_files, key=osp.getmtime)


# ─── Utility: GPU Info ──────────────────────────────────────────────────────────
def print_gpu_info():
    """Print GPU device information."""
    if not torch.cuda.is_available():
        _warn('CUDA is not available. Training will be very slow on CPU.')
        return

    num_gpus = torch.cuda.device_count()
    _info(f'Found {num_gpus} GPU(s):')
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
        print(f"      GPU {i}: {name} ({mem:.1f} GB)")


# ─── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='BEVFormer Training Script (Modernized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
{C.CYAN}Examples:{C.RESET}
  # Train with default settings (single GPU)
  python tools/train.py projects/configs/bevformer/bevformer_tiny.py

  # Train and auto-resume from latest checkpoint
  python tools/train.py projects/configs/bevformer/bevformer_tiny.py --auto-resume

  # Train with WandB logging
  python tools/train.py projects/configs/bevformer/bevformer_tiny.py --wandb --wandb-project bevformer

  # Train with custom work directory
  python tools/train.py projects/configs/bevformer/bevformer_tiny.py --work-dir ./experiments/run1

  # Override config options
  python tools/train.py projects/configs/bevformer/bevformer_tiny.py --cfg-options total_epochs=12 optimizer.lr=1e-4
""")

    # ── Core Arguments ──
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument(
        '--resume-from',
        help='Checkpoint file to resume training from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Automatically find and resume from the latest checkpoint in work_dir')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable validation during training')

    # ── GPU Arguments ──
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus', type=int,
        help='Number of GPUs (non-distributed training only)')
    group_gpus.add_argument(
        '--gpu-ids', type=int, nargs='+',
        help='GPU IDs to use (non-distributed training only)')

    # ── Reproducibility ──
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic mode for CUDNN backend')

    # ── Config Overrides ──
    parser.add_argument(
        '--options', nargs='+', action=DictAction,
        help='[DEPRECATED] Use --cfg-options instead')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config settings in key=value format. '
        'e.g. --cfg-options optimizer.lr=1e-4 total_epochs=12')

    # ── Launcher ──
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher (default: none)')
    parser.add_argument('--local_rank', type=int, default=0)

    # ── Learning Rate ──
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='Automatically scale learning rate based on GPU count (linear scaling rule)')

    # ── WandB Integration ──
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging')
    parser.add_argument(
        '--wandb-project',
        type=str, default='bevformer',
        help='WandB project name (default: bevformer)')
    parser.add_argument(
        '--wandb-name',
        type=str, default=None,
        help='WandB run name (default: config filename)')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified. '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    _header('BEVFormer Training')
    print_gpu_info()

    # ── Validate config file ──
    if not osp.isfile(args.config):
        _error(f'Config file not found: {args.config}')
        _info('Available configs:')
        for cfg_file in sorted(glob.glob('projects/configs/**/*.py', recursive=True)):
            print(f'      {cfg_file}')
        sys.exit(1)

    _info(f'Loading config: {args.config}')
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
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model

    # ── CUDNN / TF32 settings ──
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        _info('CUDNN benchmark mode enabled')
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _info('TF32 disabled for higher precision')

    # ── Work directory ──
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # ── Auto resume ──
    if args.auto_resume and args.resume_from is None:
        latest_ckpt = find_latest_checkpoint(cfg.work_dir)
        if latest_ckpt:
            args.resume_from = latest_ckpt
            _info(f'Auto-resume: found checkpoint {latest_ckpt}')
        else:
            _info('Auto-resume: no checkpoint found, training from scratch')

    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
        _info(f'Will resume from: {args.resume_from}')

    # ── GPU setup ──
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2'  # fix bug in AdamW

    if args.autoscale_lr:
        original_lr = cfg.optimizer['lr']
        cfg.optimizer['lr'] = original_lr * len(cfg.gpu_ids) / 8
        _info(f'Auto-scaled LR: {original_lr} → {cfg.optimizer["lr"]} '
              f'({len(cfg.gpu_ids)} GPUs / 8 base)')

    # ── Distributed init ──
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    world_size = len(cfg.gpu_ids)

    # ── Create work directory ──
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # ── Dump config ──
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # ── Logger ──
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # ── WandB integration ──
    if args.wandb:
        try:
            import wandb
            wandb_name = args.wandb_name or osp.splitext(osp.basename(args.config))[0]
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config=cfg._cfg_dict.to_dict() if hasattr(cfg._cfg_dict, 'to_dict') else dict(cfg._cfg_dict),
                resume='allow',
            )
            # Add WandB hook to log_config
            if 'log_config' in cfg:
                wandb_hook = dict(type='WandbLoggerHook',
                                  init_kwargs=dict(project=args.wandb_project,
                                                   name=wandb_name))
                cfg.log_config.hooks.append(wandb_hook)
            _info(f'WandB enabled: project={args.wandb_project}, run={wandb_name}')
        except ImportError:
            _warn('wandb not installed. Install with: pip install wandb')
        except Exception as e:
            _warn(f'WandB initialization failed: {e}')

    # ── Meta info ──
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ── Random seed ──
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # ── Print training summary ──
    print_training_summary(cfg, args, distributed, world_size)

    # ── Build model ──
    _info('Building model...')
    build_start = time.time()
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    build_time = time.time() - build_start

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _info(f'Model built in {build_time:.1f}s')
    _info(f'Parameters: {n_params:,} total, {n_trainable:,} trainable '
          f'({n_trainable/n_params*100:.1f}%)')

    logger.info(f'Model:\n{model}')

    # ── Build datasets ──
    _info('Building datasets...')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    _info(f'Training dataset: {len(datasets[0])} samples')

    # ── Checkpoint config ──
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
            if hasattr(datasets[0], 'PALETTE') else None)

    model.CLASSES = datasets[0].CLASSES

    # ── TensorBoard hint ──
    tb_logdir = osp.join(cfg.work_dir, 'tf_logs') if osp.isdir(osp.join(cfg.work_dir, 'tf_logs')) else cfg.work_dir
    _info(f'TensorBoard: tensorboard --logdir={cfg.work_dir}')

    # ── Start training ──
    _header('Starting Training')
    train_start = time.time()

    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    # ── Training complete ──
    train_duration = time.time() - train_start
    hours, remainder = divmod(int(train_duration), 3600)
    minutes, seconds = divmod(remainder, 60)

    _header('Training Complete')
    _info(f'Total training time: {hours}h {minutes}m {seconds}s')
    _info(f'Results saved to: {cfg.work_dir}')

    # Save a training summary JSON
    summary = {
        'config': args.config,
        'work_dir': cfg.work_dir,
        'total_epochs': cfg.runner.get('max_epochs', 'N/A'),
        'training_time_seconds': int(train_duration),
        'training_time_formatted': f'{hours}h {minutes}m {seconds}s',
        'n_params': n_params,
        'n_trainable': n_trainable,
        'seed': args.seed,
        'distributed': distributed,
        'world_size': world_size,
        'timestamp': timestamp,
    }
    summary_path = osp.join(cfg.work_dir, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    _info(f'Training summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
