from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os

def evaluate_model(name, model, X_test, y_test):

    os.makedirs("results", exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n==== {name} ====")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nSample Confidence Scores:")
    for i in range(3):
        print(f"Prediction: {y_pred[i]} | Confidence: {round(max(y_prob[i]), 4)}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{name} - Confusion Matrix")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"results/{name}_confusion.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.title(f"{name} - ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.savefig(f"results/{name}_roc.png", dpi=300)
    plt.close()

    return accuracy, precision, recall, f1