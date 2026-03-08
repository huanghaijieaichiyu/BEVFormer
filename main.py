#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
#  BEVFormer Unified CLI Entry Point
#  Supports: python main.py train <args> / python main.py test <args>
# ═══════════════════════════════════════════════════════════════════

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ─── Colors ─────────────────────────────────────────────────────────
class _C:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    R = '\033[0m'


BANNER = f"""\
{_C.BOLD}{_C.CYAN}
  ╔══════════════════════════════════════════════════════╗
  ║                                                      ║
  ║   ██████╗ ███████╗██╗   ██╗███████╗                  ║
  ║   ██╔══██╗██╔════╝██║   ██║██╔════╝                  ║
  ║   ██████╔╝█████╗  ██║   ██║█████╗                    ║
  ║   ██╔══██╗██╔══╝  ╚██╗ ██╔╝██╔══╝                    ║
  ║   ██████╔╝███████╗ ╚████╔╝ ██║  ormer                ║
  ║   ╚═════╝ ╚══════╝  ╚═══╝  ╚═╝                       ║
  ║                                                      ║
  ║   Camera-based 3D Detection with BEV Transformers    ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
{_C.R}"""


def print_help():
    """Print the main help message."""
    print(BANNER)
    print(f"{_C.BOLD}Usage:{_C.R}")
    print(f"  python main.py <command> [args...]")
    print()
    print(f"{_C.BOLD}Commands:{_C.R}")
    print(f"  {_C.GREEN}train{_C.R}    Train a BEVFormer model")
    print(f"  {_C.GREEN}test{_C.R}     Evaluate a trained BEVFormer model")
    print(f"  {_C.GREEN}info{_C.R}     Show project information and available configs")
    print()
    print(f"{_C.BOLD}Examples:{_C.R}")
    print(f"  python main.py train projects/configs/bevformer/bevformer_tiny.py")
    print(f"  python main.py train projects/configs/bevformer/bevformer_tiny.py --auto-resume --wandb")
    print(f"  python main.py test projects/configs/bevformer/bevformer_tiny.py ckpts/model.pth --eval bbox")
    print(f"  python main.py info")
    print()
    print(f"{_C.DIM}For command-specific help: python main.py <command> --help{_C.R}")
    print()


def cmd_train(argv):
    """Run the training script."""
    sys.argv = ['tools/train.py'] + argv
    from tools.train import main
    main()


def cmd_test(argv):
    """Run the testing/evaluation script."""
    sys.argv = ['tools/test.py'] + argv
    from tools.test import main
    main()


def cmd_info():
    """Show project information."""
    print(BANNER)

    print(f"{_C.BOLD}Project Structure:{_C.R}")
    print(f"  projects/configs/bevformer/     - BEVFormer v1 configs (tiny/small/base)")
    print(f"  projects/configs/bevformerv2/   - BEVFormer v2 configs")
    print(f"  projects/configs/bevformer_fp16/- FP16 training configs")
    print(f"  projects/mmdet3d_plugin/        - Custom plugin modules")
    print(f"  tools/                          - Training/testing scripts")
    print()

    # List available configs
    print(f"{_C.BOLD}Available Configs:{_C.R}")
    import glob
    configs = sorted(glob.glob('projects/configs/**/*.py', recursive=True))
    configs = [c for c in configs if '_base_' not in c and '__pycache__' not in c]
    for cfg in configs:
        print(f"  {_C.GREEN}•{_C.R} {cfg}")
    print()

    # List available checkpoints
    print(f"{_C.BOLD}Available Checkpoints:{_C.R}")
    ckpts = sorted(glob.glob('ckpts/*.pth') + glob.glob('work_dirs/**/*.pth', recursive=True))
    if ckpts:
        for ckpt in ckpts[:15]:
            size = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"  {_C.GREEN}•{_C.R} {ckpt} ({size:.0f} MB)")
    else:
        print(f"  {_C.DIM}No checkpoints found{_C.R}")
    print()

    # Python env info
    print(f"{_C.BOLD}Environment:{_C.R}")
    try:
        import torch
        print(f"  PyTorch:  {torch.__version__}")
        print(f"  CUDA:     {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
        if torch.cuda.is_available():
            print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    except ImportError:
        print(f"  {_C.YELLOW}PyTorch not installed{_C.R}")

    for pkg_name in ['mmcv', 'mmdet', 'mmdet3d', 'mmseg']:
        try:
            pkg = __import__(pkg_name)
            print(f"  {pkg_name:10s} {pkg.__version__}")
        except ImportError:
            print(f"  {_C.YELLOW}{pkg_name:10s} not installed{_C.R}")
    print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    argv = sys.argv[2:]

    if command == 'train':
        cmd_train(argv)
    elif command == 'test':
        cmd_test(argv)
    elif command == 'info':
        cmd_info()
    else:
        print(f"{_C.YELLOW}Unknown command: {command}{_C.R}")
        print()
        print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
