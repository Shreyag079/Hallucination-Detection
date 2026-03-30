from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, X_test, y_test):
    
    os.makedirs("results", exist_ok=True)
    y_pred = model.predict(X_test)
    
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Evaluation Metrics ---")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nSample Predictions with Confidence Scores -")

    for i in range(5):
        predicted_label = y_pred[i]
        confidence = max(y_prob[i])  

        print(f"Sample {i+1}:")
        print("Prediction :", "Hallucinated" if predicted_label == 1 else "Factual")
        print("Confidence :", round(confidence, 4))
        print("-" * 40)

  
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Confusion Matrix ---")
    print(cm)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.colorbar()
    plt.savefig("results/confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()