import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_recall_curve, cohen_kappa_score
import seaborn as sns
import numpy as np
import os
from media.helpers.optimize_f1 import find_optimal_thresholds

def binary_cm_and_metrics(df, true_col, pred_col, threshold=0.5, save_path=None, labels=None):
    labels = [1,0]
    if save_path is None: raise ValueError('save_path must be specified')
    
    dir_name = os.path.dirname(save_path)
    if dir_name: os.makedirs(dir_name, exist_ok=True)

    true_labels = df.loc[:,true_col]
    pred_labels = df.loc[:,pred_col]>threshold
    pred_probs  = df.loc[:,pred_col]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    accuracy = accuracy_score(true_labels, pred_labels)
    try: auc_score = roc_auc_score(true_labels, pred_probs, labels=labels)
    except: auc_score = np.nan
    f1 = f1_score(true_labels, pred_labels, labels=labels)
    kappa = cohen_kappa_score(true_labels, pred_labels.astype(int), labels=labels)
    
    metrics = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': PPV,
        'NPV': NPV,
        'Accuracy': accuracy,
        'AUC': auc_score,
        'F1 Score': f1,
        'Kappa':kappa,
    }
    
    optimal_thresholds, optimal_f1_scores = find_optimal_thresholds(np.array(pred_probs)[:,None],
                                                                    np.array(true_labels)[:,None])
    metrics['Optimal F1 Score'] = optimal_f1_scores
    metrics['Optimal Threshold'] = optimal_thresholds
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_records([metrics])
    metrics_csv_path = os.path.splitext(save_path)[0] + '.csv'
    metrics_df = metrics_df.T
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Value']
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_df

if __name__ == '__main__':
    df = pd.read_csv('/home/fdias/repositorios/media/logs/teste_100perc_5folds/2024-03-15_16-59-18/predictions/predictions_fold0.csv')
    metrics_df = binary_cm_and_metrics(df, 'Labels', 'Predictions', 0.5, 'tmp.png')
