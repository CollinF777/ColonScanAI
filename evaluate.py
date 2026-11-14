'''
Yousif this is your part, this file should be first

This is where we are getting our clinical evaluation metrics like precision, recall, F1, and a confusion matrix
The precision part is just to see how often the AI is correct, a high precision is important as it can reduce unneeded clinician reviews
The recall part is to see how many the of the images AI successfully detected cancer, this is important as fewer misses or higher
recall is important although high recall and precision can trade off each other which could lead to more alarms
The F1 scores combines precision and recall into once score which lets see how many positives predictions are correct and
how many actual positive cases were detected
Lastly, the confusion matrix is a table that shows the models correct predictions and mistakes, basically TP TN FP FN
'''
import matplotlib.pyplot as plt
# Seaborn is a library for statistical visualizations. You can use it to make a colored heatmap for the confusion matrix
import seaborn as sns
# This is for calculating all the stats we wanna track
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class ClinicalEvaluator:
    def print_clinical_metrics(self, y_true, y_pred, class_names, pos_label=0):
        print("Classification Report: \n")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print()

        # Evaluation for precision, recall, and f1
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)

        #Precision = how often prediction is correct, Recall = of flagged positives and negatives how many are actually positive
        #F1 = is what evaluates the quality of the predictions
        print(f"Clinical Metrics (positive = {class_names[pos_label]})\n")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print()

        print("Caution: Recall is the most important here.")
        print("Every false negative, \nor missed positive, is a case of benign cancer. Which could be left untreated.\n")

        #Creating heatmap using confusion matrix
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
        plt.tight_layout()  #Ensures the labels fit within frame and dont get cut off
        plt.show()

        return{"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}
'''
You only need one method in this class 
I'd call it something like print_clinical_metrics
It would have the params y_true, y_pred, and class_names
y_true would be the correct labels for images while y_pred is the labels predicted by the model

You can then print a classification report with y_true, y_pred and target_names=class_names
That would generate and print a report with precision, recall, F1 score, and support for each class

After that calculate the precision, recall, and f1_score with the library functions
Make sure you print them out after in some readable format (I'm not gonna be picky about it)
Just make sure you print a warning with it noting that recall matters most as it could be a missed cancer case

Then you should make the confusion matrix with y_true and y_pred
You should make a figure for it with plt and then a heatmap with sns
'''