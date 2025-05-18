import argparse
import logging



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)



def setup_logging(log_file):
    """
    Configure logging settings.
    
    Parameters:
    - log_file: Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate and log various classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels of the data.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels by the classifier.
    - y_prob: array-like of shape (n_samples,) or (n_samples, n_classes), optional
        Predicted probabilities by the classifier. Required for log loss and ROC AUC.
    """
    logging.info("Calculating classification metrics...")

    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    logging.info(f"Precision: {precision:.4f}")

    recall = recall_score(y_true, y_pred, average='weighted')
    logging.info(f"Recall: {recall:.4f}")

    f1 = f1_score(y_true, y_pred, average='weighted')
    logging.info(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")

    report = classification_report(y_true, y_pred)
    logging.info(f"Classification Report:\n{report}")

    if y_prob is not None:
        try:
            logloss = log_loss(y_true, y_prob)
            logging.info(f"Log Loss: {logloss:.4f}")
        except ValueError as e:
            logging.error(f"Log Loss calculation error: {e}")

    if y_prob is not None:
        try:
            # Check if y_prob is for binary or multiclass classification
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_prob if y_prob.ndim == 1 else y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            logging.info(f"ROC AUC Score: {roc_auc:.4f}")
        except ValueError as e:
            logging.error(f"ROC AUC Score calculation error: {e}")



if __name__=="__main__":
    
    pass