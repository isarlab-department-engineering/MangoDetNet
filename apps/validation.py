import numpy as np
import os
import itertools
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset.Dataset import TrainingDataset
from dataset.FruitDataset import FruitCountDataset, FruitBboxDataset
from models.EncoderDecoder import EncoderDecoderModel
from utils.util import get_config, WSConfusionMatrix, find_local_maxima, result_curves


def main():
    config = get_config('../configs/config.yaml')

    data_root = config['data_root']
    log_path = config['log_path']

    save_dir = os.path.join(log_path, config['exp_name'], 'Validation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    # ------ load dataset
    if config['validate_on_bboxes']:
        testset = FruitBboxDataset(config=config, mode=TrainingDataset.VALIDATION,
                                        name="PresenceAbsenceDataset_TEST", transforms=None,
                                        data_root=data_root)

        testloader = DataLoader(testset, batch_size=1,
                                     shuffle=False, num_workers=1, drop_last=False)

        confusion_matrix = WSConfusionMatrix(config)

    else:
        presence_absence_testset = FruitCountDataset(config=config, mode=TrainingDataset.VALIDATION,
                                                     name="PresenceAbsenceDataset_TEST", transforms=None,
                                                     data_root=data_root)

        testloader = DataLoader(presence_absence_testset, batch_size=1,
                                shuffle=False, num_workers=1, drop_last=False)


    # ------ computing maps
    maps = []
    labels = []
    Bboxes = []
    for i, data in enumerate(testloader, 0):
        print("-- processing sample: ", i)
        if config['validate_on_bboxes']:
            inputs, bboxes, img_name = data
            label = torch.tensor([[len(bboxes[0])]])
        else:
            inputs, label, img_name = data
        model.set_input([inputs, label])
        output = model.inference_forward()

        map = output[0, :, :, :].cpu().detach().numpy()
        map = map.sum(axis=0)
        maps.append(map)
        labels.append(label)
        if config['validate_on_bboxes']:
            Bboxes.append(bboxes)

    # ------ computing parameters for cross validation
    neighborhood_size = np.arange(config['min_r'], config['max_r'] + 1, config['step_r']).tolist()
    threshold = np.arange(config['min_thp'], config['max_thp'] + 1, config['step_thp']).tolist()

    param = [neighborhood_size, threshold]
    parameters = list(itertools.product(*param))

    RMSEs = []
    f1_scores = []
    TPs = []
    FPs = []
    FNs = []
    if config['validate_on_bboxes']:
        header = ['N', 'T', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1 score', 'RMSE']
    else:
        header = ['N', 'T', 'RMSE']
    data = pd.DataFrame(columns=header)

    for j in range(len(parameters)):

        print(f'\nSession {j} / {len(parameters) - 1}')
        print(f'Parameters: {parameters[j]}')

        # confusion_matrix = WSConfusionMatrix(config)
        absolute_errors = []
        for i in range(len(maps)):

            detection = find_local_maxima(maps[i], neighborhood_size=parameters[j][0], threshold=parameters[j][1])
            fruits = labels[i].cpu().detach().numpy()[0]
            absolute_errors.append(len(detection) - fruits)
            if config['validate_on_bboxes']:
                confusion_matrix.process_sample(detections=detection, bboxes=Bboxes[0].squeeze(0)[:, 1:])

        if config['validate_on_bboxes']:
            TP, FP, FN = confusion_matrix.results()
            precision, recall, f1_score = confusion_matrix.f1_score()
            f1_scores.append(f1_score)
            TPs.append(TP)
            FPs.append(FP)
            FNs.append(FN)
            print(f'f1 score: {f1_score}')
        absolute_errors = np.array(absolute_errors)
        RMSE = np.sqrt(np.mean(np.square(absolute_errors), axis=0))
        RMSEs.append(RMSE)
        print(f'RMSE: {RMSE}')

        if config['validate_on_bboxes']:
            row = [parameters[j][0], parameters[j][1], TP, FP, FN, precision, recall, f1_score, RMSE[0]]
        else:
            row = [parameters[j][0], parameters[j][1], RMSE[0]]

        data.loc[len(data)] = row

    if config['validate_on_bboxes']:
        max_f1_score = max(f1_scores)
        index_f1 = f1_scores.index(max(f1_scores))
        print(f'\nBest f1 score found in Session: {index_f1}')
        print(f'f1 score: {max_f1_score}')
        print(f'TP = %d, FP = %d, FN = %d' % (TPs[index_f1], FPs[index_f1], FNs[index_f1]))
        print(f'Parameters: {parameters[index_f1]}')
        best_data = data.loc[data['N'] == parameters[index_f1][0]]
        best_data.to_csv(path_or_buf=os.path.join(save_dir, 'best_f1_score_val.csv'), index=False, header=header,
                         sep=',')
        # Precision-Recall curves computation
        print(f'Computing precision-recall curves')

        best_data = data.loc[data['N'] == parameters[index_f1][0]]
        Ps = best_data.iloc[:, 5].tolist()
        Rs = best_data.iloc[:, 6].tolist()
        F1s = best_data.iloc[:, 7].tolist()
        result_curves(config, Ps, Rs, F1s, threshold)

    min_RMSE = min(RMSEs)
    index = RMSEs.index(min(RMSEs))
    print(f'\nBest RMSE found in Session: {index}')
    print(f'RMSE: {min_RMSE}')
    print(f'Parameters: {parameters[index]}')

    data.to_csv(path_or_buf=os.path.join(save_dir, 'validation_metrics.csv'), index=False, header=header, sep=',')

if __name__ == "__main__":
    main()