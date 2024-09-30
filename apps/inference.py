import cv2
import numpy as np
import os
import random
import torch

from torch.utils.data import DataLoader
from dataset.Dataset import TrainingDataset
from dataset.FruitDataset import FruitCountDataset, FruitBboxDataset
from models.EncoderDecoder import EncoderDecoderModel
from utils.util import get_config, find_local_maxima, WSConfusionMatrix


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

    # ------ load networks and set in evaluation mode
    model = EncoderDecoderModel(opt=config, log_path=log_path)
    model.setup()
    model.initialize(opt=config)
    model.load_networks(which_epoch=config['which_epoch'])
    model.set_eval_mode()

    # ------ load count dataset
    if config['validate_on_bboxes']:
        testset = FruitBboxDataset(config=config, mode=TrainingDataset.TEST,
                                        name="PresenceAbsenceDataset_TEST", transforms=None,
                                        data_root=data_root)

        testloader = DataLoader(testset, batch_size=1,
                                     shuffle=False, num_workers=1, drop_last=False)

        confusion_matrix = WSConfusionMatrix(config)

    else:
        presence_absence_testset = FruitCountDataset(config=config, mode=TrainingDataset.TEST,
                                                     name="PresenceAbsenceDataset_TEST", transforms=None,
                                                     data_root=data_root)

        testloader = DataLoader(presence_absence_testset, batch_size=1,
                                shuffle=False, num_workers=1, drop_last=False)


    absolute_errors = []
    for i, data in enumerate(testloader, 0):
        print("-- processing sample: ", i)

        if config['validate_on_bboxes']:
            inputs, bboxes, img_name = data
            labels = torch.tensor([[len(bboxes[0])]])
        else:
            inputs, labels, img_name = data

        model.set_input([inputs, labels])
        output = model.inference_forward()

        maps = output[0, :, :, :].cpu().detach().numpy()
        maps = maps.sum(axis=0)

        # generate heatmaps
        matrix_min = np.min(maps)
        matrix_max = np.max(maps)
        norm_resp = (maps - matrix_min) / (matrix_max - matrix_min)
        image_resp_map = cv2.applyColorMap(np.uint8(norm_resp.copy() * 255.0),
                                           cv2.COLORMAP_JET)

        if config['save_heatmaps']:
            maps_dir = os.path.join(log_path, config['exp_name'], 'Test', 'Heatmaps')
            if not os.path.exists(maps_dir):
                os.makedirs(maps_dir)
            cv2.imwrite(maps_dir + '/' + img_name[0][:-4] + '_' + str(config['which_epoch']) + '.jpg', image_resp_map)

        # ----- computing detections
        detection = find_local_maxima(maps, neighborhood_size=config['r'], threshold=config['thp'])

        fruits = labels[0].cpu().detach().numpy()[0]
        absolute_errors.append(len(detection) - fruits)
        print("\tlabel: ", fruits)
        print("\tprediction: ", len(detection))

        img = (inputs[0].cpu().detach().numpy()) * 255
        img = np.transpose(img, (1, 2, 0))
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        det_dir = os.path.join(log_path, config['exp_name'], 'Test', 'Detections')
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)

        if config['save_detections']:
            for j in range(len(detection)):
                img = cv2.circle(img, (int(detection[j, 1]), int(detection[j, 0])), 22, (0, 255, 255), 4)

            if config['validate_on_bboxes']:
                confusion_matrix.process_sample(detections=detection, bboxes=bboxes.squeeze(0)[:, 1:])
                if config['save_bboxes']:
                    bboxes = bboxes.cpu().detach().numpy()[0]
                    for j in range(len(bboxes)):
                        img = cv2.rectangle(img, (int(bboxes[j, 1]), int(bboxes[j, 2])),
                                            (int(bboxes[j, 3]), int(bboxes[j, 4])), (0, 0, 250), 5)

            cv2.imwrite(det_dir + '/' + img_name[0][:-4] + '_' + str(config['which_epoch']) + '.jpg', img)

    if config['validate_on_bboxes']:
        confusion_matrix.plot(normalize=False)
        TP, FP, FN = confusion_matrix.results()
        print(f'TP = %d, FP = %d, FN = %d' % (TP, FP, FN))
        precision, recall, f1_score = confusion_matrix.f1_score()
        print(f'precision = %f, recall = %f, f1 score = %f' % (precision, recall, f1_score))
    else:
        absolute_errors = np.array(absolute_errors)
        RMSE = np.sqrt(np.mean(np.square(absolute_errors), axis=0))
        print("RMSE on test set: ", RMSE)
    print("Inference Completed")

if __name__ == "__main__":
    main()