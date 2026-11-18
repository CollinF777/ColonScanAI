import matplotlib.pyplot as plt
# Seaborn is a library for statistical visualizations. You can use it to make a colored heatmap for the confusion matrix
import seaborn as sns
# This is for calculating all the stats we wanna track
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class ClinicalEvaluator:
    def print_clinical_metrics(self, y_true, y_pred, class_names, pos_label=1):
        print("Classification Report: \n")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print()

        # Evaluation for precision, recall, and f1
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)

        #Precision = how many of the images that are flagged positive are actually positive
        #Recall = how many of the actual positives did the model catch
        #F1 = combines precision and recall into one score which means its only high when both are high
        print(f"Clinical Metrics (positive = {class_names[pos_label]})\n")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print()

        print("Caution: Recall is the most important here.")
        print("Every false negative, \nor missed positive, is a cancerous sample that the model called benign. Which could be left untreated.\n")

        #Creating table using confusion matrix
        cm = confusion_matrix(y_true, y_pred)   #calculates how many images fall into each category (Predicted and true)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True, #writes the count in each cell
            fmt="d",    #shows each count as a whole number
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.tight_layout()  #Ensures the labels fit within frame and don't get cut off
        plt.show()

        return{"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}