import torch
from torch import Tensor
from torch import device as Device
from ignite.metrics import Metric, MetricsLambda
from ignite.metrics.metric import reinit__is_reduced
from typing import Sequence
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal


import sys
sys.path.insert(0, '../../..')


class BinaryThresholdConfusionMatrix(Metric):

    '''
    Implements a confusion matrix over a set of thresholds (inclusive) with GPU support
    Leverage redundant computations for monotonically increasing thresholds
    Accumulate confusion in running count tensor for all thresholds (T,4)
    Supports MetricLambdas to reuse confusion matrix statistics across thresholds
    - Accuracy        ~ returns Tensor with dim (T,)
    - Precision       ~ returns Tensor with dim (T,)
    - Recall          ~ returns Tensor with dim (T,)
    - PrecisionRecall ~ returns (Precision, Recall)
    - F1              ~ returns Tensor with dim (T,)
    
    NOTE:
        Metric should not be directly attached to an engine - please attach lambdas instead as needed
        Assumes all tensors have been detached from computational graph - see .detach()
        some elements of the confusion matrix may be zero and may exhibit
        unexpected behavior. Please monitor experiments as needed
    '''
    
    def __init__(self, thresholds: Tensor, pred_key, label_key, device: Device=torch.device("cpu")):
        self.thresholds = thresholds.to(device)
        self.confusion_matrix = torch.zeros(4, len(thresholds), dtype=torch.long, device=device)
        self.device = device
        # super's init should come last since Metric overrides __getattr__ and that messes with self.<foo>'s behavior
        super().__init__(output_transform=lambda x: x, device=device)
        self.required_output_keys = (pred_key, label_key)
        self._required_output_keys = self.required_output_keys

    @reinit__is_reduced
    def reset(self):
        self.confusion_matrix.fill_(0)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        preds, labels = output                                                            # unpack 
        preds, labels = preds.detach(), labels.detach()                                   # get off graph
        if preds.device != self.device: preds = preds.to(self.device)                     # coord device
        if labels.device != self.device: labels = labels.to(self.device)                  # coord device
        preds, labels = preds.view(-1), labels.view(-1)                                   # flatten
        preds, locs = torch.sort(preds)                                                   # sort
        labels = torch.cumsum(labels[locs], 0)                                            # pool for reuse
        labels = torch.cat([torch.tensor([0], device=self.device), labels], dim=0)        # pre-pending 0
        changes = torch.searchsorted(preds, self.thresholds, right=True)                  # get threshold change index
        neg_preds = labels[changes]                                                       # get fwd change accumulation
        pos_preds = labels[-1] - neg_preds                                                # get bck change accumulation
        self.confusion_matrix[0] += (pos_preds).type(torch.long)                          # TP 
        self.confusion_matrix[1] += (len(labels) - 1 - changes - pos_preds).type(torch.long)# FP (-1 accounts for prepend. 0)
        self.confusion_matrix[2] += (changes - neg_preds).type(torch.long)                # TN
        self.confusion_matrix[3] += (neg_preds).type(torch.long)                          # FN

    def compute(self):
        return self.confusion_matrix

def btcmAccuracy(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    accuracy = (cm[0]+cm[2]) / (cm.sum(dim=0) + 1e-15)
    return accuracy.max() if reduce else accuracy

def btcmPrecision(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    precision = cm[0] / (cm[0]+cm[1] + 1e-15)
    return precision.max() if reduce else precision

def btcmRecall(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    recall = cm[0] / (cm[0]+cm[3] + 1e-15) 
    return recall.max() if reduce else recall

def btcmPrecisionRecall(cm: BinaryThresholdConfusionMatrix) -> MetricsLambda:
    precision_recall = (btcmPrecision(cm, False), btcmRecall(cm, False))
    return precision_recall

def btcmF1(cm: BinaryThresholdConfusionMatrix, reduce=True, return_thresh=False) -> MetricsLambda:
    precision, recall = btcmPrecisionRecall(cm)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    if return_thresh:
        thresh = cm.thresholds[torch.argmax(f1.compute())]
        print("thresh")
        f1 = f1.max() if reduce else f1
        return f1
    else:
        return f1.max() if reduce else f1

def btcmDice(cm: BinaryThresholdConfusionMatrix, reduce=True, return_thresh=False) -> MetricsLambda:
    precision, recall = btcmPrecisionRecall(cm)
    dice = 2 * cm[0] / (2*cm[0] + cm[1] + cm[3])
    if return_thresh:
        thresh = cm.thresholds[torch.argmax(dice.compute())]
        print("thresh")
        dice = dice.max() if reduce else dice
        return dice
    else:
        return dice.max() if reduce else dice


class ApproximateMetrics():
    '''
        This class can be used to approximate ROC type statistics via preset thresholds
        It should be known that this method may under/over estimate true AUCs depending on T
        - ApproxPR_AUC      ~ returns float of Precision-Recall AUC over T
        - ApproxROC_AUC     ~ returns float of ROC AUC over T (worst case, under-estimated)
    '''
    
    @staticmethod
    def ApproxPR_AUC(cm:BinaryThresholdConfusionMatrix) -> MetricsLambda:
        precision, recall = btcmPrecisionRecall(cm)
        auc = -1 * MetricsLambda(torch.trapz, precision, recall)
        return auc
    
    @staticmethod
    def ApproxROC_AUC(cm:BinaryThresholdConfusionMatrix) -> MetricsLambda:
        tpr = btcmRecall(cm, False)
        fpr = cm[1] / (cm[1] + cm[2] + 1e-15)
        auc = -1 * MetricsLambda(torch.trapz, tpr, fpr)
        return auc


class DetectionConfusionMatrix(Metric):
    def __init__(self, pred_key, label_key, device: Device=torch.device("cpu"), threshold=0.1, max_cost=6, argmax=True,
                 enable_dilation=False, existence_threshold=0.0, calibration=False, save_dir=None, sigma=1.0,
                 size=7, existence_lb=0.1, per_patient=False, per_size=False, sum_count=False, pbcount=False,
                 entropy=False, mahalanobis=False, cc_count=False, ovl=False):

        self.threshold = threshold
        if type(existence_threshold) == float:
            self.existence_threshold = [existence_threshold]
        else:
            self.existence_threshold = existence_threshold
        self.enable_dilation = enable_dilation

        self.calibration = calibration
        if self.calibration:
            self.true_positive_array = np.array([])
            self.probability_array = np.array([])

        self.max_cost = max_cost
        self.sigma = sigma
        self.size = size
        self.existence_lb = existence_lb
        self.mahalanobis = mahalanobis
        self.ovl = ovl

        self.statistics = {}
        for t in self.existence_threshold:
            self.statistics[t] = {'ntp': 0, 'nfp': 0, 'nfn': 0}

        self.per_patient = per_patient
        if per_patient:
            self.patient_stats = []
        self.per_size = per_size
        if self.per_size:
            self.per_size_statistics = {}
            sizes = ['1', '2', '3', '4']  # tiny, small, medium, large
            for s in sizes:
                self.per_size_statistics[s] = {'ntp': 0, 'nfp': 0, 'nfn': 0}
        self.sum_count = sum_count
        self.pbcount = pbcount
        if self.sum_count or self.pbcount:
            self.counts = []
        self.cc_count = cc_count
        if self.cc_count:
            self.cc_counts = []

        self.entropy = entropy
        if self.entropy:
            self.true_positive_array_uncertainty = np.array([])
            self.uncertainties = np.array([])
            self.patient_indexes = np.array([])
            self.patient_id = 0

        self.total_cost = 0
        self.device = device
        # super's init should come last since Metric overrides __getattr__ and that messes with self.<foo>'s behavior
        super().__init__(output_transform=lambda x: x, device=device)
        self.required_output_keys = (pred_key, label_key)
        self._required_output_keys = self.required_output_keys
        self.cca_structure = ndimage.generate_binary_structure(3, 2)
        # whether to use the argmax or the center of mass as the point prediction
        self.argmax = argmax
        self.save_dir = save_dir
        self.running_gt_count = 0
        self.running_pred_count = {}
        for t in self.existence_threshold:
            self.running_pred_count[t] = 0

    @reinit__is_reduced
    def reset(self):
        self.statistics = {}
        for t in self.existence_threshold:
            self.statistics[t] = {'ntp': 0, 'nfp': 0, 'nfn': 0}
        self.total_cost = 0
        if self.calibration:
            self.true_positive_array = []
            self.probability_array = []

    def compute(self):
        confusion_matrix = np.zeros(shape=(4, len(self.existence_threshold)))
        for t, thresh in enumerate(self.existence_threshold):
            confusion_matrix[0, t] = self.statistics[thresh]['ntp']
            confusion_matrix[1, t] = self.statistics[thresh]['nfp']
            confusion_matrix[3, t] = self.statistics[thresh]['nfn']
        return torch.from_numpy(confusion_matrix)

    def counting(self, pred, max_occurence=5):

        contribution = torch.unbind(pred, 1)

        count_prediction = torch.cuda.FloatTensor(pred.size()[0], max_occurence).fill_(0)
        count_prediction[:, 0] = 1  # (batch x max_occ)
        for increment in contribution:
            mass_movement = (count_prediction * increment.unsqueeze(1))[:, :max_occurence - 1]
            move = - torch.cat([mass_movement,
                                torch.cuda.FloatTensor(count_prediction.size()[0], 1).fill_(0)], axis=1) \
                   + torch.cat(
                [torch.cuda.FloatTensor(count_prediction.size()[0], 1).fill_(0),
                 mass_movement], axis=1)

            count_prediction = count_prediction + move

        return count_prediction

    def fit_gaussians(self, pred):
        x_loc = []
        alpha_list = []

        # initializer
        existence = 0.9
        max_ = 100

        while existence > self.existence_lb and max_ > 0:
            # 1. Find Maximum Value
            max_coord = np.unravel_index(np.argmax(pred), pred.shape)
            size = self.size
            half_size = int(size / 2)

            # 2. Get Neighbourhood around Index
            x, y, z = max_coord[0], max_coord[1], max_coord[2]
            neighbour_mask = pred[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1,
                             z - half_size:z + half_size + 1]

            # 3. Fit a Gaussian density to the neighbourhood
            # 3.1 Initialize parameters
            sigma = self.sigma * np.eye(3)
            mu_init = np.array([half_size, half_size, half_size]).astype(float)
            alpha_init = np.sum(neighbour_mask)
            # 3.2 Create Input Data
            x_data = np.zeros(shape=(size ** 3, 3))
            y_data = np.zeros(size ** 3)
            index = 0
            for i in range(len(neighbour_mask)):
                for j in range(len(neighbour_mask[0])):
                    for k in range(len(neighbour_mask[1])):
                        x_data[index] = [i, j, k]
                        y_data[index] = neighbour_mask[i, j, k]
                        index += 1

            # 3.3 Define Gaussian function
            def gaussian_density(x_values, mean_x, mean_y, mean_z, alpha):
                return alpha * multivariate_normal.pdf(x_values, mean=np.array([mean_x, mean_y, mean_z]), cov=sigma)

            # 3.4 Optimize for mu and alpha
            try:
                parameters, _ = curve_fit(gaussian_density, x_data, y_data,
                                          p0=(mu_init[0], mu_init[1], mu_init[2], alpha_init))
            except RuntimeError:
                max_ -= 1
                continue
            mean_x_fit = parameters[0]
            mean_y_fit = parameters[1]
            mean_z_fit = parameters[2]

            alpha_fit = parameters[3]

            # 5. Subtract from Image
            # 5.1 Re-center
            offset = np.array([mean_x_fit, mean_y_fit, mean_z_fit]) - mu_init
            mu = max_coord + offset
            # 5.2 Generate pdf
            x_data = np.zeros(shape=(64 * 192 * 192, 3))
            index = 0
            for i in range(64):
                for j in range(192):
                    for k in range(192):
                        x_data[index] = [i, j, k]
                        index += 1
            y_data = gaussian_density(x_data, mu[0], mu[1], mu[2], alpha_fit).reshape((64, 192, 192))

            # 5.3 Subtract
            pred = pred - y_data

            existence = alpha_fit
            if existence >= self.existence_lb and not np.isnan(alpha_fit):
                x_loc.append(mu)
                alpha_list.append(alpha_fit)
            max_ -= 1
        return np.array(x_loc), np.array(alpha_list)

    def get_detections(self, pred, output_label, output_num_features):
        x_loc = np.zeros(shape=(output_num_features, 3))
        alpha = np.zeros(shape=output_num_features)
        unique = np.unique(output_label)
        for label in unique:
            if label != 0:
                mask = (output_label == label).astype(int)
                if self.enable_dilation:
                    mask = ndimage.binary_dilation(mask, structure=self.cca_structure, iterations=5)
                masked_pred = pred * mask
                if self.argmax:
                    loc = np.argmax(masked_pred)
                    index1 = int(loc / (192 * 192))
                    index2_ = loc - index1 * 192 * 192
                    index2 = int(index2_ / 192)
                    index3 = int(index2_ - index2 * 192)
                    x_loc[label - 1] = np.array([index1, index2, index3])
                    alpha[label - 1] = np.sum(masked_pred)
                else:
                    # center of mass
                    loc = ndimage.measurements.center_of_mass(input=masked_pred)
                    x_loc[label - 1] = np.array([loc[0], loc[1], loc[2]])
                    alpha[label - 1] = np.max(masked_pred)
        return np.array(x_loc), np.array(alpha)

    def single_threshold(self, x_loc, alpha, gt_count, gt_points, existence_threshold, lesion_sizes=None):
        indexes = []
        for j, a in enumerate(alpha):
            if a >= existence_threshold:
                indexes.append(j)

        if self.cc_count:
            if type(self.threshold) == float:
                self.cc_counts.append({"predicted": len(indexes), "gt": gt_count})

        if self.pbcount:
            pred_count = self.counting(torch.cuda.FloatTensor(alpha).unsqueeze(0)).detach().cpu().numpy()
            self.counts.append({"predicted": pred_count, "gt": gt_count})

        if gt_count == 0:
            self.statistics[existence_threshold]['nfp'] += len(indexes)
            if self.per_patient:
                self.patient_stats[-1]['nfp'] += len(indexes)
        elif len(indexes) == 0:
            self.statistics[existence_threshold]['nfn'] += gt_count
            if self.per_patient:
                self.patient_stats[-1]['nfn'] += gt_count
            if self.per_size:
                for s in lesion_sizes:
                    self.per_size_statistics[s]['nfn'] += 1.0
        else:
            row_ind, col_ind = self.hungarian_algo(x_loc[indexes], gt_points, existence_threshold, lesion_sizes)

            component_difference = len(indexes) - gt_count
            if component_difference > 0:
                # there are more detected lesions than in ground truth
                self.statistics[existence_threshold]['nfp'] += component_difference
                if self.per_patient:
                    self.patient_stats[-1]['nfp'] += component_difference
            elif component_difference < 0:
                # there are more ground truth lesions that were not found
                self.statistics[existence_threshold]['nfn'] -= component_difference
                if self.per_patient:
                    self.patient_stats[-1]['nfn'] -= component_difference
                if self.per_size:
                    for index in np.arange(len(gt_points)):
                        if index not in col_ind:
                            l_size = lesion_sizes[index]
                            self.per_size_statistics[l_size]['nfn'] += 1

    def calibration_curve(self, x_loc, alpha, gt_points):
        if len(x_loc) > 0:
            if self.calibration:
                self.probability_array = np.concatenate((self.probability_array, alpha))
            else:
                indexes = []
                for j, a in enumerate(alpha):
                    if a >= self.existence_threshold[0]:
                        indexes.append(j)
                alpha = alpha[indexes]
                x_loc = x_loc[indexes]
                entropies = - 1 * (alpha * np.log(alpha) + (1 - alpha) * np.log(1 - alpha))
                self.uncertainties = np.concatenate((self.uncertainties, entropies))
                self.patient_indexes = np.concatenate((self.patient_indexes, self.patient_id * np.ones_like(entropies)))
                self.patient_id += 1
            tp = np.zeros_like(alpha)
            if len(gt_points) > 0:
                if self.mahalanobis:
                    cost_matrix = self.mahalanobis_distance(x_loc, gt_points)
                elif self.ovl:
                    cost_matrix = self.ovl_distance(x_loc, gt_points)
                else:
                    cost_matrix = cdist(x_loc, gt_points)
                # hungarian assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for k in range(len(row_ind)):
                    if cost_matrix[row_ind[k], col_ind[k]] <= self.max_cost:
                        tp[k] = 1.0
            if self.calibration:
                self.true_positive_array = np.concatenate((self.true_positive_array, tp))
            elif self.entropy:
                self.true_positive_array_uncertainty = np.concatenate((self.true_positive_array_uncertainty, tp))

        if self.calibration and self.save_dir is not None:
            pk.dump(self.probability_array, open(self.save_dir+"_probability_array.pk", "wb"))
            pk.dump(self.true_positive_array, open(self.save_dir + "_true_positive_array.pk", "wb"))
        if self.entropy:
            pk.dump(self.uncertainties, open(self.save_dir + "_uncertainties.pk", "wb"))
            pk.dump(self.true_positive_array_uncertainty, open(self.save_dir + "_true_positive_array_uncertainty.pk", "wb"))
            pk.dump(self.patient_indexes,
                    open(self.save_dir + "_patient_index_uncertainty.pk", "wb"))

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        if self.per_size:
            preds, label_tuple = output  # unpack
            labels, sizes = label_tuple
            sizes_ = sizes.detach().cpu().numpy()
            del sizes
        else:
            preds, labels = output  # unpack
        preds_, gt_points_raw = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        del preds, labels
        torch.cuda.empty_cache()
        for j, pred in enumerate(preds_):
            if self.per_patient:
                self.patient_stats.append({'ntp': 0.0, 'nfp': 0.0, 'nfn': 0.0})

            gt_count = np.count_nonzero(gt_points_raw[j, 0])
            self.running_gt_count += gt_count
            gt_points = np.zeros(shape=(gt_count, 3))
            for i in range(gt_count):
                gt_points[i] = np.array([gt_points_raw[j, 0, i], gt_points_raw[j, 1, i], gt_points_raw[j, 2, i]])
            if self.sum_count:
                self.counts.append({"predicted": np.sum(pred), "gt": gt_count})

            if self.per_size:
                lesion_sizes = sizes_[j, :gt_count].astype('str')
            else:
                lesion_sizes = None

            if type(self.threshold) == float:
                # Step 1.1: binarize using threshold
                preds_binary = (pred > self.threshold).astype(int)
                # Step 1.2: find connected components
                output_label, output_num_features = ndimage.label(preds_binary, structure=self.cca_structure)

            elif type(self.threshold) == str:
                if self.threshold == "fit_gaussians":
                    output_num_features = 1
            # Step 1.3: find x0 and alpha0 for each component (estimate of x0 is the local argmax)
            if output_num_features != 0:
                # get x_loc and alpha
                if self.threshold == "fit_gaussians":
                    x_loc, alpha = self.fit_gaussians(pred)
                else:
                    x_loc, alpha = self.get_detections(pred, output_label, output_num_features)
                # get matches or perform calibration
                if self.calibration or self.entropy:
                    self.calibration_curve(x_loc, alpha, gt_points)
                else:
                    for thresh in self.existence_threshold:
                        self.single_threshold(x_loc, alpha, gt_count, gt_points, existence_threshold=thresh, lesion_sizes=lesion_sizes)
                        fn = self.statistics[thresh]['nfn']
                        fp = self.statistics[thresh]['nfp']
                        tp = self.statistics[thresh]['ntp']
                        assert tp+fn == self.running_gt_count, f"TP({tp})+FN({fn}) must equal gt_count({self.running_gt_count})"
                        self.running_pred_count[thresh] += np.count_nonzero(alpha >= thresh)
                        assert tp+fp == self.running_pred_count[thresh],\
                            f"TP({tp})+FP({fp}) must equal number of detected points above the threshold ({self.running_pred_count[thresh]})"

            else:
                if self.pbcount:
                    pred_count = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
                    self.counts.append({"predicted": pred_count, "gt": gt_count})
                if self.cc_count:
                    self.cc_counts.append({"predicted": 0, "gt": gt_count})
                if self.calibration or self.entropy:
                    pass
                else:
                    # check if the label also has 0, then do nothing (and skip hungarian algorithm)
                    # if the label is not empty, then add that number of false negatives
                    if self.per_patient:
                        self.patient_stats[-1]['nfn'] += gt_count
                    for thresh in self.existence_threshold:
                        self.statistics[thresh]['nfn'] += gt_count
                    if self.per_size:
                        for s in lesion_sizes:
                            self.per_size_statistics[s]['nfn'] += 1.0
        if self.save_dir is not None:
            pk.dump(self.statistics, open(self.save_dir + "_statistics.pk", "wb"))
            if self.per_patient:
                pk.dump(self.patient_stats, open(self.save_dir + "_statistics_per_patient.pk", "wb"))
            if self.per_size:
                pk.dump(self.per_size_statistics, open(self.save_dir + "_statistics_per_size.pk", "wb"))
            if self.sum_count:
                pk.dump(self.counts, open(self.save_dir + "_statistics_sumcounts.pk", "wb"))
            if self.pbcount:
                pk.dump(self.counts, open(self.save_dir + "_statistics_pbcounts.pk", "wb"))
            if self.cc_count:
                pk.dump(self.cc_counts, open(self.save_dir + "_statistics_cc_counts.pk", "wb"))

    def hungarian_algo(self, x_loc, gt_points, existence_threshold, lesion_sizes=None):
        # create a matrix of size (row=output_num_features, column=gt_count)
        if self.mahalanobis:
            cost_matrix = self.mahalanobis_distance(x_loc, gt_points)
        elif self.ovl:
            cost_matrix = self.ovl_distance(x_loc, gt_points)
        else:
            cost_matrix = cdist(x_loc, gt_points)
        # hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_costs = cost_matrix[row_ind, col_ind]
        for k, a in enumerate(assignment_costs):
            if a < self.max_cost:
                self.statistics[existence_threshold]['ntp'] += 1
                if self.per_patient:
                    self.patient_stats[-1]['ntp'] += 1
                if self.per_size:
                    l_size = lesion_sizes[col_ind[k]]
                    self.per_size_statistics[l_size]['ntp'] += 1
                # self.confusion_matrix[0] += 1  # TP
                self.total_cost += a
            else:
                self.statistics[existence_threshold]['nfp'] += 1
                self.statistics[existence_threshold]['nfn'] += 1
                if self.per_patient:
                    self.patient_stats[-1]['nfp'] += 1
                    self.patient_stats[-1]['nfn'] += 1
                if self.per_size:
                    l_size = lesion_sizes[col_ind[k]]
                    self.per_size_statistics[l_size]['nfn'] += 1
                # self.confusion_matrix[1] += 1  # FP
                # self.confusion_matrix[3] += 1  # FN
        return row_ind, col_ind

    @staticmethod
    def ece(acc, total_in_bin, confidence):
        total = np.sum(total_in_bin)
        return np.sum(total_in_bin * np.abs(acc - confidence) / total)

    @staticmethod
    def mce(acc, confidence):
        return np.max(np.abs(acc - confidence))

    @staticmethod
    def plot_calibraiton_curve(experiment, name, save_dir):
        probability_array = pk.load(open(save_dir + "_probability_array.pk", "rb"))
        probability_array = np.clip(probability_array, 0.0, 1.0)
        true_positive_array = pk.load(open(save_dir + "_true_positive_array.pk", "rb"))
        # sklearn calibration
        prob_true, prob_pred = calibration_curve(true_positive_array, probability_array, n_bins=10)

        plt.figure(figsize=(5, 5))
        plt.plot(prob_pred, prob_true, label='Ours')
        plt.plot(np.arange(0.0, 1.1, 0.1), np.arange(0.0, 1.1, 0.1), label='Ideal')
        plt.legend()
        plt.grid()
        title = "Calibration_"+name
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        experiment.log_figure(figure=plt, figure_name=title)

        # from sklearn source code to get the total in each bin
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/calibration.py#L968
        bins = np.linspace(0.0, 1.0 + 1e-8, 10 + 1)
        binids = np.digitize(probability_array, bins) - 1
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        bin_total = bin_total[nonzero]

        ece = DetectionConfusionMatrix.ece(prob_true, bin_total, prob_pred)
        mce = DetectionConfusionMatrix.mce(prob_true, prob_pred)
        experiment.log_metrics({name+"_ECE": ece, name+"_MCE": mce})

    @staticmethod
    def plot_entropy_curves(experiment, name, save_dir):
        uncertainties = pk.load(open(save_dir + "_uncertainties.pk", "rb"))
        true_positive_array = pk.load(open(save_dir + "_true_positive_array_uncertainty.pk", "rb"))

        # normalize uncertainties
        min, max = np.min(uncertainties), np.max(uncertainties)
        uncertainties = (uncertainties - min) / (max - min)

        # create curves
        quantiles = []
        for q in np.arange(0.0, 1.00001, 0.00001):
            quantiles.append(np.quantile(uncertainties, q))

        accuracies_gt = []
        quantiles_gt = []
        accuracies_lt = []
        quantiles_lt = []

        for i, b in enumerate(quantiles):
            indices = np.where(uncertainties >= b)[0]
            if len(indices) != 0:
                accuracies_gt.append(np.sum(true_positive_array[indices]) / len(indices))
                quantiles_gt.append(b)

            indices = np.where(uncertainties <= b)[0]
            if len(indices) != 0:
                accuracies_lt.append(np.sum(true_positive_array[indices]) / len(indices))
                quantiles_lt.append(b)

        title = "Uncertainty_" + name
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.plot(quantiles_gt, accuracies_gt), plt.grid(), plt.title("Greater Than"),\
        plt.xlabel("Entropy Threshold"), plt.ylabel("Accuracy")
        plt.subplot(1, 2, 2), plt.plot(quantiles_lt, accuracies_lt), plt.grid(), plt.title("Less Than"),\
        plt.xlabel("Entropy Threshold"), plt.ylabel("Accuracy")
        experiment.log_figure(figure=plt, figure_name=title)


def dcmAccuracy(cm: DetectionConfusionMatrix) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    accuracy = (cm[0]+cm[2]) / (cm.sum() + 1e-15)
    return accuracy

def dcmPrecision(cm: DetectionConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    precision = cm[0] / (cm[0]+cm[1] + 1e-15)
    return precision.max() if reduce else precision

def dcmRecall(cm: DetectionConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    recall = cm[0] / (cm[0]+cm[3] + 1e-15)
    return recall.max() if reduce else recall

def dcmPrecisionRecall(cm: DetectionConfusionMatrix) -> MetricsLambda:
    precision_recall = (btcmPrecision(cm, False), btcmRecall(cm, False))
    return precision_recall

def dcmF1(cm: DetectionConfusionMatrix, reduce=True) -> MetricsLambda:
    precision, recall = btcmPrecisionRecall(cm)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return f1.max() if reduce else f1