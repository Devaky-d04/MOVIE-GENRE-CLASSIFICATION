import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_names):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot Precision-Recall curve for multi-label classification
def plot_precision_recall_curve(y_true, y_pred, label_names):
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=label_names)
    y_pred_bin = label_binarize(y_pred, classes=label_names)
    
    plt.figure(figsize=(10, 8))
    for i in range(y_true_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        plt.plot(recall, precision, label=f'Label {label_names[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Multi-label Classification')
    plt.legend(loc='best')
    plt.show()

# Function to plot ROC curve for multi-label classification
def plot_roc_curve(y_true, y_pred, label_names):
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=label_names)
    y_pred_bin = label_binarize(y_pred, classes=label_names)
    
    plt.figure(figsize=(10, 8))
    for i in range(y_true_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Label {label_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-label Classification')
    plt.legend(loc='best')
    plt.show()

# Function to plot XGBoost decision tree
def plot_xgboost_tree(model, num_trees=0):
    plt.figure(figsize=(20, 10))
    xgb.plot_tree(model, num_trees=num_trees)
    plt.show()

# Combined visualization function to plot all metrics
def combined_visualizations(y_true, y_pred, model, label_names):
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, label_names)
    
    # Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_pred, label_names)
    
    # ROC Curve
    plot_roc_curve(y_true, y_pred, label_names)
    
    # XGBoost Decision Tree
    plot_xgboost_tree(model)

# Example usage (replace with actual variables)
combined_visualizations(y_test, y_pred_lr, xgb, mlb.classes_)
