import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def plot_confusion_matrix(X,t,lr,model_name):
    cm = confusion_matrix(t, lr.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
    disp.plot()
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix: ", np.array([[tp, fp], [fn, tn]]))
    print("Accuracy: ", ((tp + tn) / (tp + fp + fn + tn)))
    print("Precision: ", (tp / (tp + fp)))
    print("Recall: ", (tp / (tp + fn)))
    plt.title(f"Confusion Matrix for {model_name}")

    plt.show()

def plot_roc_curve(X, t, lr, model_name, color='r'):
    t_pred = lr.predict_proba(X)[::, 1]
    fpr, tpr, thresholds = roc_curve(t, t_pred)

    plt.plot(fpr, tpr, color)
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print(f"AUC for Winner Prediction ({model_name}): ", auc(fpr, tpr))