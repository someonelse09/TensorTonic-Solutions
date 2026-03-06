import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    classes = sorted(list(set(y_true + y_pred)))
    count_classes = len(classes)

    # Initialising the multi-class confusion matrix with all entries equal to zero
    confusion_matrix = [[0 for _ in range(count_classes)] for _ in range(count_classes)]

    # Dictionary (or mapping) from classes to indices,
    # which matches every class with some natural number within [0, count_classes)
    class_to_idx = {cls : j for j, cls in enumerate(classes)}

    for t, p in zip(y_true, y_pred):
        true_idx = class_to_idx[t]
        pred_idx = class_to_idx[p]
        confusion_matrix[true_idx][pred_idx] += 1
    confusion_matrix = np.array(confusion_matrix)

    TP = np.diag(confusion_matrix)
    FN = np.sum(confusion_matrix, axis=1) - TP
    FP = np.sum(confusion_matrix, axis=0) - TP

    # Correct fraction of predictions
    accuracy = np.sum(TP) / np.sum(confusion_matrix)
    # Among predicted positives
    precision = TP / (TP + FP + 1e-12)
    # Among actual positives
    recall  = TP / (TP + FN + 1e-12)
    # Harmonic mean of recall and precision
    f1_score = 2 * recall * precision / (recall + precision + 1e-12)
    rs = np.sum(confusion_matrix, axis=1)
    if average == "micro":
        TP_sum = np.sum(TP)
        FN_sum = np.sum(FN)
        FP_sum = np.sum(FP)

        precision_avg = TP_sum / (TP_sum + FP_sum + 1e-12)
        recall_avg = TP_sum / (TP_sum + FN_sum + 1e-12)
        f1_score_avg = 2 * precision_avg * recall_avg / (precision_avg + recall_avg + 1e-12)

    elif average == "macro":
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        f1_score_avg = np.mean(f1_score)

    elif average == "weighted":
        weights = rs / np.sum(rs)

        precision_avg = np.sum(weights * precision)
        recall_avg = np.sum(weights * recall)
        f1_score_avg = np.sum(weights * f1_score)
    elif average == "binary":
        idx = class_to_idx[pos_label]

        precision_avg = precision[idx]
        recall_avg = recall[idx]
        f1_score_avg = f1_score[idx]
    else:
        raise ValueError("Unknown average type.")
    result = {}
    result["accuracy"] = float(accuracy)
    result["precision"] = float(precision_avg)
    result["recall"] = float(recall_avg)
    result["f1"] = float(f1_score_avg)

    return result
    