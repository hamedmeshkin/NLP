import torch
from torch import no_grad
from torch.utils.data import DataLoader, Dataset
import os
import csv
import argparse
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
import numpy as np
import random

FileLocation = 'DeepSeek/FewShot/AI2ARC_validation_results'
WrongLable = 0
folder = FileLocation
pattern = 'FewShot_NoSampling.*'

files = [f for f in glob.glob(os.path.join(FileLocation, pattern)) if os.path.isfile(f)]


Precision =[]
Sensitivity=[]
Acuracy = []
F1_score = []
Specificity = []
for ii in files:
    try:
        FileLocation = ii
        if not (os.path.exists(FileLocation)):
            continue



        labels = []
        with open(FileLocation, 'r') as file:
            for line in file:
                if line.lower().find('1')>=0 or line.lower().find('0')>=0:
                    labels.append(line.strip().split(' '))
                else:
                    labels.append(line.strip().split('\t'))  # Split the line by a delimiter, such as a comma

        # print(labels)


        first_elements = []
        second_elements = []
        for sublist in labels:
            if (len(sublist) == 2):
                first_elements.append(sublist[0])
                second_elements.append(sublist[1])
            else:
                first_elements.append(sublist[0])
                second_elements.append('empty')




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

        #pdb.set_trace()
        # for bi in second_elements:
        #     if (bi == 'Pharmacokinetic'):
        #         y_pred.append(1)
        #     else:
        #         y_pred.append(0)
        #     if (bi != 'Pharmacokinetic' and bi != 'Non-Pharmacokinetic'):
        #         print("No Label " + str(ii))
        # if len(Counter(y_true)) != 2 or len(Counter(y_pred)) != 2:
        #     continue
        # print(classification_report_imbalanced(y_true, y_pred))

        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

        conf_matrix = conf_matrix * 100 / np.sum(conf_matrix)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)


        PRECISION = (100.0*np.double(precision_score(y_true, y_pred)))
        SENSITIVITY = 100.0*np.double(sensitivity)
        ACURACY = 100.0*np.double(accuracy_score(y_true, y_pred))
        F1_SCORE = 100.0*np.double(f1_score(y_true, y_pred))
        SPECIFICITY = 100.0*np.double(specificity)

        # if PRECISION==0.0:
        #     continue
        #     pdb.set_trace()

        Precision.append(PRECISION)
        Sensitivity.append(SENSITIVITY)
        Acuracy.append(ACURACY)
        F1_score.append(F1_SCORE)
        Specificity.append(SPECIFICITY)
    except Exception as e:
        print("Error occurred:", e)
        print(ii)
        ipdb.set_trace()

print('Number of triales is:')
print(len(Precision))
preMean = np.mean(Precision)/100
preStd  = np.std(Precision)/100

senMean = np.mean(Sensitivity)/100
senStd  = np.std(Sensitivity)/100

f1Mean = np.mean(F1_score)/100
f1Std  = np.std(F1_score)/100

speMean = np.mean(Specificity)/100
speStd  = np.std(Specificity)/100


Values = {'pre':[preMean,preStd],'sen':[senMean,senStd],'f1':[f1Mean,f1Std],'spe':[speMean,speStd]}
pd.DataFrame(data=Values)

#################
labels = []
with open(FileLocation, 'r') as file:
    for line in file:
        tmp = line.strip().split(' ')
        if len(tmp) < 2:
            tmp = line.strip().split('\t')
        labels.append(tmp)

# print(labels)
#pdb.set_trace()

first_elements = []
second_elements = []
for sublist in labels:
    if (len(sublist) == 2):
        first_elements.append(sublist[0])
        second_elements.append(sublist[1])
    else:
        first_elements.append(sublist[0])
        second_elements.append('empty')

classes = list(set(first_elements))
for idx, pred in enumerate(second_elements):
    # Check if pred is not one of the two classes and update accordingly
    if pred not in classes:
        if first_elements[idx] == classes[0]:
            second_elements[idx] = classes[1]
        elif first_elements[idx] == classes[1]:
            second_elements[idx] = classes[0]



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


classes = list(set(y_true))

for idx, pred in enumerate(y_pred):
    # Check if pred is not one of the two classes and update accordingly
    if pred not in classes:
        if y_true[idx] == classes[0]:
            y_pred[idx] = classes[1]
        elif y_true[idx] == classes[1]:
            y_pred[idx] = classes[0]

print(classification_report_imbalanced(y_true,y_pred))


