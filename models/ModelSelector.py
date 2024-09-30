from models.EncoderDecoder import EncoderDecoderModel
from trainers.trainer import PresenceAbsenceTrainer


class ModelSelector(object):
    def __init__(self, config, data_root, log_path):
        self.config = config
        self.data_root = data_root
        self.log_path = log_path
        self.model = None

        if self.config['strategy'] == 'Reconstruction':
            self.model = SSIMUnetModel(opt=config, log_path=log_path)
            self.trainer = PresenceAbsenceTrainer(self.model, self.config, dataset_path_root=self.data_root)

        if self.config['strategy'] == 'PACUnet':
            self.model = PACUnetModel(opt=config, log_path=log_path)
            self.trainer = PresenceAbsenceTrainer(self.model, self.config, dataset_path_root=self.data_root)

        # baseline
        if self.config['strategy'] == 'baseline':
            self.model = UnetPACModel(opt=config, log_path=log_path)
            self.trainer = PresenceAbsenceTrainer(self.model, self.config, dataset_path_root=self.data_root)


    def get_model(self):
        return self.model

    def train(self):
        print("[*] Training starts...")
        self.trainer.init_data_loader()
        self.trainer.train()

    def val(self):
        self.trainer.init_data_loader()
        self.trainer.val(epoch=self.config.which_epoch)
