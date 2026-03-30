from train import run_training
from evaluate import evaluate_model
import matplotlib.pyplot as plt

models, X_test, y_test = run_training()

results = {}

for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(name, model, X_test, y_test)
    results[name] = acc

plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("results/model_comparison.png", dpi=300)
plt.close()