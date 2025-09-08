
# Import necessary libraries
from sklearn.preprocessing import StandardScaler  # For standardizing features by removing the mean and scaling to unit variance
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score  # For evaluating model performance
import matplotlib.pyplot as plt  # For plotting graphs
import os  # For interacting with the operating system
import numpy as np  # For numerical operations
from imblearn.metrics import classification_report_imbalanced  # For detailed classification metrics, especially useful for imbalanced datasets
import argparse  # For parsing command line arguments
from collections import Counter  # For counting hashable objects
import pandas as pd  # For data manipulation and analysis
import ipdb  # For the Python debugger
import glob

# Define a function to calculate and return various metrics based on the predictions
def metrics(mainFolder):
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Over/under sampling')
    parser.add_argument('--sampling', dest='sampling', default="under", type=str, help='Input sampling mode, over / under')
    args = parser.parse_args()
    sampling = args.sampling

    # Initialize a counter for wrong labels
    WrongLable = 0

    # Setup filenames and ranges based on the sampling argument
    if sampling == 'over':
#        FILE = "AI2ARC_validation_results_"
        Range = range(1, 100)
    else:
        FILEname = "AI2ARC*"
        Range = range(1, 1000)

    folder = mainFolder
    Files = glob.glob(os.path.join(folder, FILEname))

    # Initialize lists to store various metrics
    Precision = []
    Sensitivity = []
    Acuracy = []
    F1_score = []
    Specificity = []

    # Loop through each file in the specified range
    for FileLocation in Files:
        try:
#            FileLocation = folder + FILE + str(ii) + ".txt"

            # Skip if the file doesn't exist
            if not os.path.exists(FileLocation):
                continue

            labels = []
            # Read labels from the file
            with open(FileLocation, 'r') as file:
                for line in file:
                    tmp = line.strip().split(' ')
                    if len(tmp) < 2:
                        tmp = line.strip().split('\t')
                    labels.append(tmp)

            # Split labels into first and second elements, handling cases with missing second elements
            first_elements = []
            second_elements = []
            for sublist in labels:
                if len(sublist) == 2:
                    first_elements.append(sublist[0])
                    second_elements.append(sublist[1])
                else:
                    first_elements.append(sublist[0])
                    second_elements.append('empty')

            # Determine unique classes and correct predictions not matching any class
            classes = list(set(first_elements))
            for idx, pred in enumerate(second_elements):
                if pred not in classes:
                    if first_elements[idx] == classes[0]:
                        second_elements[idx] = classes[1]
                    elif first_elements[idx] == classes[1]:
                        second_elements[idx] = classes[0]

            # Skip file if all predictions are the same or if there are no predictions
            if len(second_elements) == 0 or (second_elements.count(second_elements[0]) == len(second_elements)):
                WrongLable += 1
                continue

            # Convert labels to binary format based on specific criteria
            y_true = []
            y_pred = []
            for ai,bi in zip(first_elements,second_elements):
                if (bi.lower().find("pharmacokinetic")>=0 or ai.lower().find("pharmacokinetic")>=0 or ai=='1' or ai=='0' or bi=='0' or bi=='1' or ai=='Yes' or ai=='No' or bi=='No' or bi=='Yes' or ai=='True' or ai=='False' or bi=='True' or bi=='False'):
                    if (ai == 'Pharmacokinetic' or ai == '1' or ai == 'Yes' or ai == 'True'):
                        y_true.append(1)
                    else:
                        y_true.append(0)

                    if (bi == 'Pharmacokinetic' or bi== '1' or bi == 'Yes' or bi == 'True'):
                        y_pred.append(1)
                    else:
                        y_pred.append(0)

            # Calculate confusion matrix and normalize it
            conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
            conf_matrix = conf_matrix * 100 / np.sum(conf_matrix)

            # Extract true negatives, false positives, false negatives, and true positives
            tn, fp, fn, tp = conf_matrix.ravel()

            # Calculate sensitivity, specificity, precision, accuracy, and F1 score
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            PRECISION = 100.0 * precision_score(y_true, y_pred)
            SENSITIVITY = 100.0 * sensitivity
            ACURACY = 100.0 * accuracy_score(y_true, y_pred)
            F1_SCORE = 100.0 * f1_score(y_true, y_pred)
            SPECIFICITY = 100.0 * specificity

            # Skip if precision is zero
            if PRECISION == 0.0:
                continue

            # Append calculated metrics to their respective lists
            Precision.append(PRECISION)
            Sensitivity.append(SENSITIVITY)
            Acuracy.append(ACURACY)
            F1_score.append(F1_SCORE)
            Specificity.append(SPECIFICITY)
        except Exception as e:
            print("Error occurred:", e)
            print(ii)
            ipdb.set_trace()  # Trigger debugger on exception

    # Print the number of trials and calculate the mean and standard deviation of each metric
    print('Number of trials is:')
    print(len(Precision))
    preMean = np.mean(Precision) / 100
    preStd = np.std(Precision) / 100
    senMean = np.mean(Sensitivity) / 100
    senStd = np.std(Sensitivity) / 100
    f1Mean = np.mean(F1_score) / 100
    f1Std = np.std(F1_score) / 100
    speMean = np.mean(Specificity) / 100
    speStd = np.std(Specificity) / 100

    # Compile metrics into a DataFrame and return
    Values = {'pre': [preMean, preStd], 'sen': [senMean, senStd], 'f1': [f1Mean, f1Std], 'spe': [speMean, speStd]}
    return pd.DataFrame(data=Values)

# Define a list of folders containing validation results
Folders = ['PKDDI/ZeroShot/AI2ARC_validation_results/', 'PKDDI/FewShot/AI2ARC_validation_results/', 'IntrinsicFactors/ZeroShot/AI2ARC_validation_results/', 'IntrinsicFactors/FewShot/AI2ARC_validation_results/']

# Initialize lists to store aggregated metrics and their errors
pre = []; spe = []; f1 = []; sen = []
pre_error = []; spe_error = []; f1_error = []; sen_error = []

# Iterate through each folder, calculate metrics, and append results to the lists
for i, folder in enumerate(Folders):
    print(folder)
    measurements = metrics(folder)
    print(measurements)
    pre.append(measurements['pre'][0])
    sen.append(measurements['sen'][0])
    f1.append(measurements['f1'][0])
    spe.append(measurements['spe'][0])
    pre_error.append(measurements['pre'][1])
    sen_error.append(measurements['sen'][1])
    f1_error.append(measurements['f1'][1])
    spe_error.append(measurements['spe'][1])

# Define x-axis labels for the plots
x0 = ['0', '1', '2', '3']

# Clear the current figure and create a new 2x2 subplot layout
plt.clf()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot Precision
axes[0, 0].bar(x0, pre, yerr=pre_error, label='Precision', capsize=4, color='blue')
axes[0, 0].set_ylabel('Precision', fontsize=20)
axes[0, 0].axhline(y=0.83, color='grey', linestyle='--')  # Joel's result for comparison
axes[0, 0].tick_params(axis='x', labelsize=20)
axes[0, 0].tick_params(axis='y', labelsize=20)
axes[0, 0].set_ylim(0, 1.05)

# Plot Specificity
axes[0, 1].bar(x0, spe, yerr=spe_error, label='Specificity', capsize=4, color='red')
axes[0, 1].set_ylabel('Specificity', fontsize=20)
axes[0, 1].tick_params(axis='x', labelsize=20)
axes[0, 1].tick_params(axis='y', labelsize=20)
axes[0, 1].set_ylim(0, 1.05)

# Plot F1 Score
axes[1, 0].bar(x0, f1, yerr=f1_error, label='F1 Score', capsize=4, color='green')
axes[1, 0].set_ylabel('F1 Score', fontsize=20)
axes[1, 0].axhline(y=0.82, color='grey', linestyle='--')  # Joel's result for comparison
axes[1, 0].tick_params(axis='x', labelsize=20)
axes[1, 0].tick_params(axis='y', labelsize=20)
axes[1, 0].set_ylim(0, 1.05)

# Plot Sensitivity
axes[1, 1].bar(x0, sen, yerr=sen_error, label='Sensitivity', capsize=4, color='orange')
axes[1, 1].set_ylabel('Sensitivity', fontsize=20)
axes[1, 1].axhline(y=0.81, color='grey', linestyle='--')  # Joel's result for comparison
axes[1, 1].tick_params(axis='x', labelsize=20)
axes[1, 1].tick_params(axis='y', labelsize=20)
axes[1, 1].set_ylim(0, 1.05)

# Adjust spacing between subplots for clarity
plt.tight_layout()

# Save the plot to a PDF file
plt.savefig('Compare_ZeroShot54.pdf', format='pdf')
# Display the plot
plt.show()
