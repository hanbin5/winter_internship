import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger('root')



def save_args(args, filename):
    """args를 텍스트 파일로 저장 (재현성 보장)"""
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


def change_logger_dest(logger, new_dest):
    """로거에 파일 핸들러 추가"""
    formatter = logging.Formatter(fmt='[%(asctime)s]- %(levelname)s - %(module)s - %(message)s')
    handler = logging.FileHandler(new_dest, mode='a')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_custom_logger(name, test=False):
    """커스텀 로거 설정"""
    formatter = logging.Formatter(fmt='[%(asctime)s]- %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(name)
    if test:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_checkpoint(fpath, model):
    """체크포인트 로드 (DDP module. prefix 처리)"""
    assert os.path.exists(fpath), f"Checkpoint not found: {fpath}"
    logger.info('loading checkpoint... %s' % fpath)
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    # Check if model expects 'model.' prefix (e.g., UAREME wrapper)
    model_keys = set(model.state_dict().keys())
    load_keys = set(load_dict.keys())

    # If keys don't match, try adding 'model.' prefix
    if not model_keys.intersection(load_keys):
        sample_model_key = next(iter(model_keys))
        sample_load_key = next(iter(load_keys))
        if sample_model_key.startswith('model.') and not sample_load_key.startswith('model.'):
            load_dict = {'model.' + k: v for k, v in load_dict.items()}

    model.load_state_dict(load_dict)
    logger.info('loading checkpoint... / done')
    return model


def save_model(model, target_path, total_iter):
    """모델 저장"""
    torch.save({
        "model": model.state_dict(),
        "iter": total_iter
    }, target_path)
    logger.info('model saved / path: {}'.format(target_path))


class dotdict(dict):
    """dot notation으로 접근 가능한 dict"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RunningAverage:
    """실시간 평균 계산"""
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value, count_add=1):
        self.avg = (count_add * value + self.count * self.avg) / (count_add + self.count)
        self.count += count_add

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """dict 형태의 실시간 평균 계산"""
    def __init__(self):
        self._dict = None

    def update(self, new_dict, count_add):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value, count_add)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}
