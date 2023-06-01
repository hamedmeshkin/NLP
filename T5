import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import re
import string
import time
import evaluate
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Flatten,InputLayer,BatchNormalization,Dropout,Input,LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
from transformers import (DataCollatorWithPadding,create_optimizer,AutoTokenizer,DataCollatorForSeq2Seq,
                          T5TokenizerFast,T5ForConditionalGeneration,TFAutoModelForSeq2SeqLM,TFT5ForConditionalGeneration,TFT5ForConditionalGeneration,T5Tokenizer)

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the maximum sequence length
max_length = 10

# Define the text
text = ["This is an example sentence to demonstrate left padding and right truncation."
        ,"my name is abas"]

# Tokenize and encode the text
encoded_inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=False,  # We don't want to add special tokens here
    max_length=max_length,     # Truncate if necessary
    padding="max_length",      # Apply left padding
    truncation=True            # Enable truncation
)
tokenizer.pad
# Retrieve the input IDs and attention mask
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

print("Original text:", text)
print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)

decoded_attention_mask = tokenizer.decode(input_ids, skip_special_tokens=False)


###############################################################################################################
from transformers import T5TokenizerFast

# Instantiate the tokenizer
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token
# Define the maximum sequence length
max_length = 28
prefix = "translate English to French: "
# Define the text
#text = ["This is an example sentence to demonstrate left padding and right truncation.","my name is abas"]
BATCH_SIZE=64
text=load_dataset('text', data_files='C:/Users/shmes/Downloads/fra-eng/fra.txt')
text['train'][100000]['text'].split('\t') 
# Tokenize and encode the text
# encoded_inputs = tokenizer.encode_plus(
#     text,
#     add_special_tokens=False,  # We don't want to add special tokens here
#     max_length=max_length,     # Truncate if necessary
#     padding="max_length",      # Apply left padding
#     truncation=True            # Enable truncation
# )

def preprocess_function(examples):

  inputs = [prefix + example.split('\t')[0] for example in examples['text']]
  targets = [example.split('\t')[1] for example in examples['text']]

  model_inputs = tokenizer(inputs, text_target=targets,max_length=max_length, 
                           truncation=True,add_special_tokens=True,padding="max_length")
  return model_inputs

tokenized_dataset=text.map(preprocess_function,batched=True)

TEXT = tokenized_dataset['train'][100000]['text']
input_ids = tokenized_dataset['train'][100000]['input_ids']
attention_mask = tokenized_dataset['train'][100000]['attention_mask']
labels = tokenized_dataset['train'][100000]['labels']
# Retrieve the input IDs and attention mask
# input_ids = encoded_inputs["input_ids"]
# attention_mask = encoded_inputs["attention_mask"]

print("Original text:", TEXT)
print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)
print("labels:", labels)

#tokenizer.decode(text, skip_special_tokens=True)
tokenizer.decode(input_ids, skip_special_tokens=False)
tokenizer.decode(attention_mask, skip_special_tokens=False)
tokenizer.decode(labels, skip_special_tokens=False)










