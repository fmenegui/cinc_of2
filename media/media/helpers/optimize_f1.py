import numpy as np
from sklearn.metrics import precision_recall_curve

def find_optimal_thresholds(predictions, true_labels):
    """
    Finds the optimal thresholds that maximize the F1 score for each class.

    Parameters:
    - predictions: Numpy array of shape (N, C) containing the predicted probabilities for C classes.
    - true_labels: Numpy array of shape (N, C) containing the true labels in a one-hot encoded format.

    Returns:
    - optimal_thresholds: Numpy array containing the optimal threshold for each class.
    - optimal_f1_scores: Numpy array containing the optimal F1 score for each class at its threshold.
    """
    num_classes = true_labels.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    optimal_f1_scores = np.zeros(num_classes)
    
    for class_index in range(num_classes):
        probs = predictions[:, class_index]
        labels = true_labels[:, class_index]
        
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * recall * precision / (recall + precision)
        
        f1_scores = np.nan_to_num(f1_scores)
        
        ix = np.argmax(f1_scores)
        optimal_thresholds[class_index] = thresholds[ix] if ix < len(thresholds) else 1.0
        optimal_f1_scores[class_index] = f1_scores[ix]
    
    return optimal_thresholds, optimal_f1_scores

def find_global_optimal_threshold(fold_preds_labels):
    """
    Finds the global optimal threshold that maximizes the mean F1 score across all folds in a cross-validation setup.

    Parameters:
    - fold_preds_labels: List of tuples, where each tuple contains (predictions, true_labels)
      for a fold. Predictions and true_labels are numpy arrays in N x C format.

    Returns:
    - global_optimal_threshold: The threshold that maximizes the mean F1 score across all classes and folds.
    - mean_f1_scores: The mean F1 score for each class at the global optimal threshold.
    """
    all_preds = np.vstack([preds for preds, _ in fold_preds_labels])
    all_labels = np.vstack([labels for _, labels in fold_preds_labels])
    
    num_classes = all_labels.shape[1]
    global_f1_scores = []
    
    for class_index in range(num_classes):
        probs = all_preds[:, class_index]
        labels = all_labels[:, class_index]
        
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1])
        f1_scores = np.nan_to_num(f1_scores)
        
        max_index = np.argmax(f1_scores)
        optimal_threshold = thresholds[max_index]
        max_f1_score = f1_scores[max_index]
        
        global_f1_scores.append((optimal_threshold, max_f1_score))
    
    mean_f1_scores = np.mean([f1 for _, f1 in global_f1_scores])
    optimal_thresholds_per_class = [threshold for threshold, _ in global_f1_scores]
    global_optimal_threshold = np.mean(optimal_thresholds_per_class)
    
    return global_optimal_threshold, mean_f1_scores


if __name__ == '__main__':
    ## Teste find_optimal_thresholds
    np.random.seed(42) 
    predictions = np.random.rand(100, 3)
    true_labels = np.zeros_like(predictions)
    for i in range(true_labels.shape[0]):
        true_labels[i, np.random.choice(3)] = 1
    optimal_thresholds, optimal_f1_scores = find_optimal_thresholds(predictions, true_labels)
    print("Optimal Thresholds per Class:", optimal_thresholds)
    print("Optimal F1 Scores per Class:", optimal_f1_scores)

    ## Teste find_global_optimal_threshold
    fold_preds_labels = []
    for _ in range(3):  # 3 folds
        # Predicted probabilities for 100 instances, 3 classes
        predictions = np.random.rand(100, 3)
        # True labels (one-hot encoded) for 100 instances, 3 classes
        true_labels = np.zeros_like(predictions)
        for i in range(true_labels.shape[0]):
            true_labels[i, np.random.choice(3)] = 1
        fold_preds_labels.append((predictions, true_labels))

    # Find the global optimal threshold across all folds
    global_optimal_threshold, mean_f1_scores = find_global_optimal_threshold(fold_preds_labels)

    print("Global Optimal Threshold:", global_optimal_threshold)
    print("Mean F1 Score at Global Optimal Threshold:", mean_f1_scores)
    pass