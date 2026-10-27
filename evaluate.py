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