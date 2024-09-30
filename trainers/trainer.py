import time
from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader

from dataset.Dataset import TrainingDataset
from dataset.FruitDataset import FruitPADataset
from trainers.BaseTrainer import BaseTrainer
from torchnet.meter import APMeter

class PresenceAbsenceTrainer(BaseTrainer):

    def __init__(self, model, opt, dataset_path_root):
        super(PresenceAbsenceTrainer, self).__init__(model, opt)
        self.dataset_path_root = dataset_path_root
        self.ap_meter = APMeter()
        self.best_val_loss = np.inf
        self.patience_counter = 0
        self.break_loop = False
        self.total_val_steps = 0
        self.dataset = {
            'train': None,
            'validation': None
            # 'test': None
        }

    def init_data_loader(self):

        # transforms = [
        # Rescale(output_size=(100,100))
        # TODO: insert other transforms
        # ]

        transforms = None

        presence_absence_trainset = FruitPADataset(
            config=self.opt,
            mode=TrainingDataset.TRAIN,
            name="PresenceAbsenceDataset_TRAIN",
            transforms=transforms,
            data_root=self.dataset_path_root
        )

        presence_absence_validset = FruitPADataset(
            self.opt,
            mode=TrainingDataset.VALIDATION,
            name="PresenceAbsenceDataset_VALIDATION",
            transforms=transforms,
            data_root=self.dataset_path_root
        )

        # presence_absence_testset = FruitPADataset(
        #     self.opt,
        #     mode=TrainingDataset.TEST,
        #     name="PresenceAbsenceDataset_TEST",
        #     transforms=transforms,
        #     data_root=self.dataset_path_root
        # )

        self.dataset['train'] = presence_absence_trainset
        self.dataset['validation'] = presence_absence_validset
        # self.dataset['test'] = presence_absence_testset


        self.trainloader = DataLoader(self.dataset['train'], batch_size=self.opt['batch_size'],
                                      shuffle=True, num_workers=self.opt['num_workers'], drop_last=True)

        self.validationloader = DataLoader(self.dataset['validation'], batch_size=self.opt['batch_size'],
                                           shuffle=False, num_workers=self.opt['num_workers'], drop_last=True)

        # self.testloader = DataLoader(self.dataset['test'], batch_size=self.opt['batch_size'],
        #                              shuffle=False, num_workers=self.opt['num_workers'], drop_last=True)

        if self.opt['is_train']:
            self.init_summary_writer()
        self.model.initialize(self.opt)
        self.model.setup()

    def train(self):
        print("[*] Training starts...")
        if self.opt['resume_train']:
            total_steps = self.opt['start_epoch'] * len(self.trainloader) * self.opt['batch_size']
        else:
            total_steps = 0

        for epoch in range(self.opt['start_epoch'] + 1, self.opt['max_epochs'] + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            self.ap_meter.reset()
            if self.opt['early_stopping'] & self.break_loop:
                break

            epoch_losses = OrderedDict()
            losses = self.model.get_loss_names()

            for label in losses:
                epoch_losses[label] = 0.0

            for i, data in enumerate(self.trainloader, 0):

                iter_start_time = time.time()
                if total_steps % self.opt['print_freq'] == 0:
                    t_data = iter_start_time - iter_data_time

                total_steps += self.opt['batch_size']
                epoch_iter += self.opt['batch_size']

                inputs, labels, img_name = data

                self.model.set_input([inputs, labels])
                # this method call the forward and backward process
                self.model.optimize_parameters(epoch=epoch)

                losses = self.model.get_current_losses()
                for label, value in losses.items():
                    epoch_losses[label] += value

                if total_steps % self.opt['print_freq'] * self.opt['batch_size'] == 0:
                    t = (time.time() - iter_start_time) / self.opt['batch_size']
                    for label, value in losses.items():
                        self.writer.add_scalar('batch_train_loss/' + label, value, total_steps)
                        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
                        print(message)
                        print("batch_loss: ", value)

                if total_steps % self.opt['print_hist_freq'] * self.opt['batch_size'] == 0:
                    histograms = self.model.get_current_histograms()
                    for name, param in histograms.items():
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), total_steps)

                iter_data_time = time.time()

            latest_loss = 0
            for label, value in epoch_losses.items():
                self.writer.add_scalar('epoch_train_loss/' + label, value / len(self.trainloader), epoch)
                latest_loss += value / len(self.trainloader)
            self.writer.add_scalar('epoch_train_loss/' + self.opt['strategy'], latest_loss, epoch)

            if epoch % self.opt['save_epoch_freq'] == 0 and epoch != 0:
                if epoch == self.opt['start_epoch']:
                    pass
                else:
                    print('saving the model at the end of epoch %d, iters %d' %
                          (epoch, total_steps))
                    self.model.save_networks(epoch)
                    self.model.save_optimizers(epoch)
                self.model.set_train_mode()

            if epoch % self.opt['val_epoch_freq'] == 0 and epoch != self.opt['start_epoch']:
                print('validating the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                self.val(epoch)
                self.model.set_train_mode()

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.opt['max_epochs'], time.time() - epoch_start_time))
            self.model.update_learning_rate()

    def val(self, epoch):
        ap_meter_val = APMeter()
        ap_meter_val.reset()

        print("[*] Validation starts...")

        val_epoch_losses = OrderedDict()
        losses = self.model.get_loss_names()

        for label in losses:
            val_epoch_losses[label] = 0.0

        for i, data in enumerate(self.validationloader, 0):

            inputs, labels, img_name = data

            self.model.set_input([inputs, labels])
            self.model.val()
            if self.opt['resume_train']:
                self.total_val_steps = self.opt['start_epoch'] * len(self.trainloader) * self.opt['batch_size']

            self.model.val_loss()
            losses = self.model.get_current_losses()
            for label, value in losses.items():
                val_epoch_losses[label] += value
            if self.total_val_steps % self.opt['print_freq'] * self.opt['batch_size'] == 0:
                for label, value in losses.items():
                    self.writer.add_scalar('batch_val_loss/' + label, value, self.total_val_steps)
            self.total_val_steps += self.opt['batch_size']

        latest_loss = 0
        for label, value in val_epoch_losses.items():
            self.writer.add_scalar('epoch_val_loss/' + label, value / len(self.validationloader), epoch)
            print("%s validation loss: %f" % (label[:-4], value / len(self.validationloader)))
            latest_loss += value / len(self.validationloader)
        self.writer.add_scalar('epoch_val_loss/' + self.opt['strategy'], latest_loss, epoch)

        if latest_loss < self.best_val_loss:
            self.best_val_loss = latest_loss
            self.patience_counter = 0
            print('Best validation loss found at epoch %d!' % epoch)
            self.model.save_networks('best')
            self.model.save_optimizers('best')
        else:
            self.patience_counter += 1
            if self.patience_counter % 1 == 0:
                print(f"Loss has not improved for {self.patience_counter} epochs")
                if (self.patience_counter >= self.opt['patience_epochs']) and self.opt['early_stopping']:
                    print(f"EARLY STOPPING")
                    print(f'Best model was found at epoch {epoch - self.patience_counter}')
                    self.break_loop = True
