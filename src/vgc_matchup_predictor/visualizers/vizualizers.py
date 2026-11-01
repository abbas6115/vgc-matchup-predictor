import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, log_loss

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

def plot_learning_curve_log_loss(model, X_train, y_train, X_test, y_test, title: str = 'Learning Curve (Log Loss)') -> None:
    # Plots train vs validation log loss over increasing training set sizes
    train_sizes = []
    train_losses = []
    test_losses = []
    
    fractions = np.linspace(0.1, 1.0, 10)
    total_samples = len(X_train)

    for frac in fractions:
        size = int(total_samples * frac)
        X_sub, y_sub = X_train[:size], y_train[:size]

        # Fit clone/instance on subset
        model.fit_subset(X_sub, y_sub) if hasattr(model, 'fit_subset') else model.fit(X_sub, y_sub)
        
        train_proba = model.predict_proba(X_sub)
        test_proba = model.predict_proba(X_test)

        train_sizes.append(size)
        train_losses.append(log_loss(y_sub, train_proba))
        test_losses.append(log_loss(y_test, test_proba))

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_losses, 'o-', label='Train Loss')
    plt.plot(train_sizes, test_losses, 'o-', label='Test Loss')
    plt.title(f'{title} - Learning Curve')
    plt.xlabel('Training Samples')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.close()

def plot_feature_importances(
        importance_dict, # dict mapping pokemon species to feature importances 
        top_n: int, # N; number of top species to plot
        title: str = 'Top Feature Importances - Random Forest'
) -> None:
    # Plots horizontal bar chart of top N Pokemon species feature importances

    # slice top N pokemon and reverse order so highest importance is at the top
    top_items = list(importance_dict.items())[:top_n]
    pokemon = [item[0] for item in reversed(top_items)]
    importances = [item[1] for item in reversed(top_items)]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(pokemon, importances, color="skyblue", edgecolor="navy")

    # Add numeric label to the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.xlabel("Gini Feature Importance Score", fontsize=10)
    plt.ylabel("Species", fontsize=10)
    plt.title(f"{title} - Top {top_n}", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()

    plt.show()