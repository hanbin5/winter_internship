import os
import argparse

DATASET_DIR = '/home/hanbin5/data/'   # where datasets are stored
EXPERIMENT_DIR = '/home/hanbin5/experiments/'  # where to save the experiments
PROJECT_DIR = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]


def get_default_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # experiment path
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_name')
    parser.add_argument('--exp_id', type=str, default='exp_id')

    # training
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='accumulate gradient every N batches')
    parser.add_argument("--workers", default=12, type=int)

    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument('--gpus', type=str, default='-1', help='which gpus to use, if -1, use all')
    parser.add_argument('--save_all_models', action='store_true')
    parser.add_argument('--overwrite_models', action='store_true', help='if True, overwrite the existing checkpoints')

    parser.add_argument('--lr', default=0.0003, type=float, help='max learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--diff_lr', action="store_true", help="use different LR for different network components")
    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--validate_every', default=1e20, type=int, help='validate every N iterations, validation also happens every epoch')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualize every N iterations')

    # checkpoint (only needed when testing the model)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # arguments for testing
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--visualize', default=False, action="store_true")
    parser.add_argument('--test_path', type=str, default='/home/hanbin5/data/youtube_drone_racing/Boston-Drone.mp4')
    parser.add_argument('--save_trajectory', action='store_true')
    parser.add_argument('--output', default=None, type=str, help='Output video path')

    return parser


def convert_arg_line_to_args(arg_line):
    # 주석 라인과 빈 라인 건너뛰기
    arg_line = arg_line.strip()
    if not arg_line or arg_line.startswith('#'):
        return
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)