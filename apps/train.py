import numpy as np
import random
import torch

from models.EncoderDecoder import EncoderDecoderModel
from trainers.trainer import PresenceAbsenceTrainer
from utils.util import get_config


def main():
    config = get_config('../configs/config.yaml')

    data_root = config['data_root']
    log_path = config['log_path']

    seed = config['random_seed']
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = EncoderDecoderModel(opt=config, log_path=log_path)
    trainer = PresenceAbsenceTrainer(model, config, dataset_path_root=data_root)
    print("[*] Training starts...")

    with torch.cuda.device(config['gpu_id']):
        trainer.init_data_loader()
        trainer.train()

if __name__ == "__main__":
    main()
