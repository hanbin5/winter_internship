import os
import sys
import glob
from datetime import datetime

from projects import DATASET_DIR, EXPERIMENT_DIR, get_default_parser
import utils.utils as utils

import logging
logger = logging.getLogger('root')


def get_args(test=False):
    parser = get_default_parser()

    #↓↓↓↓
    # NOTE: project-specific args (uareme)
    parser.add_argument('--model_architecture', type=str, default='dsine_v00')

    # UAREME args
    parser.add_argument('--b_kappa', type=bool, default=True)
    parser.add_argument('--kappa_threshold', type=float, default=75.0)
    parser.add_argument('--b_multiframe', type=bool, default=True)
    parser.add_argument('--b_robust', type=bool, default=True)
    parser.add_argument('--window_length', type=int, default=30)
    parser.add_argument('--interframe_sigma', type=float, default=0.75)
    #↑↑↑↑

    # read arguments from txt file
    assert '.txt' in sys.argv[1], "첫 번째 인자로 .txt config 파일을 전달해주세요"
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])

    #↓↓↓↓
    # NOTE: 프로젝트 고정값 설정
    args.exp_root = os.path.join(EXPERIMENT_DIR, 'uareme')
    #↑↑↑↑

    # output 디렉토리 생성
    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    args.output_dir = os.path.join(exp_dir, args.exp_id)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    # 기존 checkpoint 존재 시 종료 (덮어쓰기 방지)
    if not test and \
        not args.overwrite_models and \
            len(glob.glob(os.path.join(args.output_dir, 'models', '*.pt'))) > 0:
        print('checkpoints exist! --overwrite_models 플래그를 사용하세요')
        exit()

    # 학습 시 로거/텐서보드 설정
    if not test:
        global logger
        utils.change_logger_dest(logger, os.path.join(args.output_dir, 'log', '%s.log' % datetime.now()))
        from torch.utils.tensorboard.writer import SummaryWriter
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'log'))

    # args 저장 (재현성)
    args_path = os.path.join(args.output_dir, 'log', 'params.txt')
    utils.save_args(args, args_path)
    logger.info('config saved in %s' % args_path)

    logger.info('DATASET_DIR: %s' % DATASET_DIR)
    logger.info('OUTPUT_DIR: %s' % args.output_dir)

    return args