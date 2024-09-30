import yaml
import numpy as np
import math
import os.path
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
from skimage.feature import peak_local_max


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def find_local_maxima(heatmap, neighborhood_size, threshold, threshold_rel=None):

    coordinates = peak_local_max(heatmap, min_distance=neighborhood_size, threshold_abs=threshold,
                                 threshold_rel=threshold_rel, exclude_border=True)
    return coordinates

class WSConfusionMatrix():

    def __init__(self, config):
        self.config = config
        self.save_dir = os.path.join(self.config['log_path'], self.config['exp_name'], 'Test')
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.absolute_errors = []
        self.matrix_indexes = config['class_name'] + ["background"]

    def compute_distances(self, Xc, Yc, points_in_bbox):
        distances = []
        for index, point in enumerate(points_in_bbox):
            dist = math.dist((Xc, Yc),point)
            distances.append(dist)
        return distances

    def find_points_in_bbox(self, detections, bbox):
        points_in_bbox = []
        if len(bbox) == 0:
            pass
        else:
            for i in range(len(detections)):
                det = detections[i]
                x = det[1]
                y = det[0]
                if (x >= bbox[0] and x <= bbox[2] and y >= bbox[1] and y <= bbox[3]):
                    points_in_bbox.append(det)
        return points_in_bbox

    def process_sample(self, detections, bboxes):

        tp = 0
        fn = 0
        for index, bbox in enumerate(bboxes):
            points_in_bbox = self.find_points_in_bbox(detections, bbox)
            if len(points_in_bbox) == 0:
                fn += 1
            elif len(points_in_bbox) == 1:
                point_to_delete = np.where(np.all(detections==points_in_bbox, axis=1))[0][0]
                tp += 1
                detections = np.delete(detections, point_to_delete, axis=0)
            else:
                Xc = bbox[0] + (bbox[2] - bbox[0])/2
                Yc = bbox[1] + (bbox[3] - bbox[1])/2
                distances = self.compute_distances(Xc, Yc, points_in_bbox)
                point_index = distances.index(min(distances))
                point_to_delete = np.where(np.all(detections==points_in_bbox[point_index], axis=1))[0][0]
                tp += 1
                detections = np.delete(detections, point_to_delete, axis=0)

        fp = len(detections)
        self.TP = self.TP + tp
        self.FP = self.FP + fp
        self.FN = self.FN + fn
        self.absolute_errors.append(len(bboxes) - tp)

    def results(self):
        return self.TP, self.FP, self.FN

    def f1_score(self):
        self.precision = self.TP / (self.TP + self. FP) if (self.TP + self.FP) else 0
        self.recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) else 0
        self.score = (2 * self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) else 0
        return self.precision, self.recall, self.score

    def confusion_matrix(self):
        self.matrix = np.array([[self.TP, self.FP], [self.FN, 0]])
        return self.matrix

    def plot(self, normalize=False):

        self.confusion_matrix()
        self.num_classes = 1
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.num_classes, len(self.matrix_indexes)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        # labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (self.matrix_indexes) if self.matrix_indexes else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8},
                       cmap='Blues',
                       fmt='.0f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        fig.savefig(save_path, dpi=250)
        plt.close(fig)

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def plot_pr_curve(px, py, ap, save_dir):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color='blue', label='AP: %.4f' % ap)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def plot_mc_curve(px, py, save_dir, xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    py = np.array(py)
    ax.plot(px, py, linewidth=3, color='blue')
    ax.plot(px[py.argmax()], py.max(), 'ro', label=f'Max: {py.max():.4f} at {px[py.argmax()]:.0f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylabel == 'Precision':
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='upper right')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    Arguments:
        recall:    The recall curve (list)
        precision: The precision curve (list)
    Returns:
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([1.0], recall, [0.0]))
    mpre = np.concatenate(([0.0], precision, [1.0]))

    # Compute the precision envelope
    mpre = np.maximum.accumulate(mpre)

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def result_curves(config, Ps, Rs, f1_scores, Ths):

    save_dir = os.path.join(config['log_path'], config['exp_name'], 'Validation')

    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    ap, mpre, mrec = compute_ap(Rs, Ps)

    plot_pr_curve(mrec, mpre, ap, os.path.join(save_dir, 'PR_curve.png'))
    plot_mc_curve(Ths, f1_scores, os.path.join(save_dir, 'F1_curve.png'), ylabel='F1')
    plot_mc_curve(Ths, Ps, os.path.join(save_dir, 'P_curve.png'), ylabel='Precision')
    plot_mc_curve(Ths, Rs, os.path.join(save_dir, 'R_curve.png'), ylabel='Recall')
