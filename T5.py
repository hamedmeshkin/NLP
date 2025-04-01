from datapreprocessing import myDataSet, AI2ARC_Intrinsic_Challenge_qaoptions_Processor  as Processor
from imblearn.metrics import classification_report_imbalanced
from transformers import AutoTokenizer, AutoModelForCausalLM
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, Dataset
from accelerate import init_empty_weights
from collections import Counter
from torch import no_grad
from tqdm import tqdm
import bitsandbytes
import numpy as np
import argparse
import random
import torch
import csv
import os
import re



parser = argparse.ArgumentParser(description='Randomly choose 4 sentences (2 per class), do the training and validation, calculate metrics, and then repeat it over and over again')
parser.add_argument('--input', dest='input_file',default = "Dataset/eval_1_DataSet.txt", type=str, help='Input file path')
parser.add_argument('--output', dest='output_file',default = "AI2ARC_validation_results_1.txt", type=str, help='Output file path')
parser.add_argument('--GPU', dest='SGE_GPU',default = 0, type=int, help='GPU Device Number')
parser.add_argument('--sampling', dest='sampling_Mode', default = "under_sampling", type=str, help='Select between under and over sampling')
parser.add_argument('--sample', dest='sample', default = True, type=str, help='Select between under and over sampling')

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file
SGE_GPU = args.SGE_GPU
sampling_Mode = args.sampling_Mode
sampling = args.sample

print("The input file is " + input_file)
print("The output file is " + output_file)
print("SGE GPU is " + str(SGE_GPU))
print("sampling Mode is " + sampling_Mode)

P_unit = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(P_unit)
print(f'device used is {device}')


tokenizer = AutoTokenizer.from_pretrained("/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/Qwen/")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

processor = Processor()

def collate_tokenize(data): #data is a list of length batch_size (defined in DataLoader later;
                            #each element of data is a tuple of sentence, label, output from _getitem_ of myDataset
  text_batch = [leading_examples + element[0] for element in data] #if preceding with leading examples
  #text_batch = [element[0] for element in data]
  label_batch = [element[1] for element in data]
  tokenized_batch = tokenizer(text_batch, padding=True, truncation=True,
                              max_length=1024, #even though T0 paper says max_len 1024, T0pp config says 512
                              return_tensors='pt')
  return tokenized_batch, label_batch

# Load the validation examples
eval_examples = processor.get_dev_examples('../../Dataset/PK-intrinsic.txt')
print(f'the 1st question is \n{eval_examples[0].text_a}')
batch_size = 5

# Create lists to store labels and rows
Y = []
Text = []
# Populate the lists with labels and Text from the examples
for lines in eval_examples:
    Y.append(lines.label)
    Text.append([lines.text_a,lines.label])


indx_class_a = [indx for  indx,label in enumerate(Y) if label == np.unique(Y)[0]]
indx_class_b = [indx for  indx,label in enumerate(Y) if label == np.unique(Y)[1]]
R1 = sorted(random.sample(indx_class_a, 2) + random.sample(indx_class_b, 2), reverse=True)

leading_examples=[]
for idx in R1:
    leading_examples.append(Text[idx])
    del Y[idx]
    del Text[idx]
leading_examples = processor.get_imblearner_examples(leading_examples)
leading_examples = processor.create_leading_examples(leading_examples)
print(f'leading examples are \n{leading_examples}')

# Choose the sampling technique based on the specified mode
if sampling_Mode=='under_sampling':
    Sampling = RandomUnderSampler()
elif sampling_Mode=='over_sampling':
    Sampling = RandomOverSampler()

# Perform resampling
indices = list(range(len(Y)))
resampled_indices, tmp = Sampling.fit_resample(torch.tensor(indices).reshape(-1, 1), Y)

# Create lists to store the resampled rows and labels
resampledText = []
resampledlabels = []
# Populate the lists with resampled rows and labels
resampledText = []
resampledlabels = []
# Populate the lists with resampled rows and labels
if False:
    for idx in resampled_indices:
        resampledText.append(Text[idx[0]])
        resampledlabels.append(Text[idx[0]][1])
else:
    for idx,tmp in enumerate(Text):
        resampledText.append(Text[idx])
        resampledlabels.append(Text[idx][1])

# Print the count of each label after resampling
print(Counter(resampledlabels))
# Get imblearner examples for the resampled rows
imb_samples = processor.get_imblearner_examples(resampledText)
print(f'the 1st question is \n{imb_samples[0].text_a}')
###############################################################################################################

test_loader = DataLoader(myDataSet(imb_samples), batch_size=batch_size, collate_fn=collate_tokenize, shuffle=False)
#I didn't use truncation=True below because in theory leading_examples will have to be << max_len
#so that the real question can be added; however this may generate an warning
# leading_examples_tokens = tokenizer(leading_examples, return_tensors='pt').input_ids
# print(f'number of tokens from leading examples is {leading_examples_tokens.size()}')

if 'model' not in locals() and 'model' not in globals():
    if P_unit=="cuda":
        with init_empty_weights():
            max_memory_mapping = {
            0: "25GiB",
            1: "25GiB",
            2: "25GiB",
            3: "25GiB",
            "cpu": "60GB"
            }
            model = AutoModelForCausalLM.from_pretrained(
                "/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/Qwen/",
                device_map="auto",
                max_memory = max_memory_mapping, #style from https://huggingface.co/docs/transformers/perf_infer_gpu_one
        #        llm_int8_enable_fp32_cpu_offload=True,
        #        load_in_8bit=True  # Enable 8-bit quantization
                )
    if  P_unit=="cpu":
        model = AutoModelForCausalLM.from_pretrained("/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/Qwen/",
                                                     device_map={"": P_unit}  # Force CPU execution
                                                     ).to(device)

    print("done loading model!")
    print(model.hf_device_map)

with torch.no_grad():
    correct = 0
    y_true = []
    y_pred = []
    #print("start batching")
    for batch_idx, (encodings, truelabels) in enumerate(tqdm(test_loader, desc="Processing batches", total=len(test_loader))):
#        if batch_idx==2:
#            break
        #print(f'at {batch_idx}, size of encoding is {encodings.input_ids.size()} and truelabel is {truelabels}')
        encodings = encodings.to(device)
        #truelabels = truelabels.to(device) #no need to move labels to GPU; doing so makes enumerate later hard
        gen_tokens = model.generate(
            **encodings,
            max_new_tokens=10,           # Limit the response length (enough for "Yes" or "No")
#            temperature=0.0,             # Ensure deterministic output
            num_beams=1,                 # Greedy search is sufficient for a yes/no answer
            do_sample=False,             # Disable sampling to remove randomness
            return_dict_in_generate=True, # Return additional information (optional)
        )
        beam_output = gen_tokens['sequences']
        #size: num of rows: batch_size*num_return_sequences,
        #num of columns: len(tokenized leading_examples)+ len(longest tokenized "question" in this batch) +len(max_new_tokens)
        #print(beam_output[0,:])
        #print(tokenizer.batch_decode(encodings['input_ids'],skip_special_tokens=True)[0])
#        size_of_full_tokens = beam_output.size(dim=1)
#        new_output = beam_output[:,(size_of_full_tokens - 17):]
        new_seq = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
        #print(tokenizer.batch_decode(beam_output,skip_special_tokens=True))

        for i, label in enumerate(truelabels): #truelabels is a tensor of true labels for this batch
            predicted = new_seq[i]

            if label == "Pharmacokinetic interaction": #to be consistent with datapreprocessing.py
                label = "Yes"
            elif label == "Non-Pharmacokinetic interaction":
                label = "No"

            Think_location = predicted.lower().find("think")
            if Think_location >= 0:
                if predicted[Think_location:].lower().find("no")>=0:
                    predicted_label='No'
                elif predicted[Think_location:].lower().find("yes")>=0:
                    predicted_label='Yes'
            else:
                matches = [match.start() for match in re.finditer("###", predicted)]
                if len(matches)==5:
                    if predicted[matches[4]-5:].lower().find("no")>=0:
                        predicted_label='No'
                    elif predicted[matches[4]-5:].lower().find("yes")>=0:
                        predicted_label='Yes'
                else:
                    if label=='Yes':
                            predicted_label='Noo'
                    if label=='No':
                            predicted_label='Yess'

            y_true.append(label)
            #print(f'label is {label}')
            y_pred.append(predicted_label)
            #print(f'predicted_label is {predicted_label}')
        if  P_unit=="cuda":
            del gen_tokens
            torch.cuda.empty_cache()  # 🔹 Clear unused memory (reduces fragmentation)
            torch.cuda.reset_peak_memory_stats()  # Optional: Reset memory stats
#print(f'y_true is {y_true}')
#print(f'y_pred is {y_pred}')



print('The final evaluation metrics is:')
print(classification_report_imbalanced(y_true,y_pred))
output_file = 'ZeroShot_NoSampling_04'
output_predict_file = os.path.join("AI2ARC_validation_results/", output_file)
with open(output_predict_file,'w') as writer:

    for sampleIdx, prediction in enumerate(y_pred):

        output_line = str(y_true[sampleIdx])+"\t"+ str(prediction)+ "\n"
        writer.write(output_line)

j=0
bb = []
with open('tmp.txt','w') as writer:
    for batch_idx, (encodings, truelabels) in enumerate(tqdm(test_loader, desc="Processing batches", total=len(test_loader))):
         seq = tokenizer.batch_decode(encodings['input_ids'],skip_special_tokens=True)
         for i,tx in enumerate(seq):
            writer.write(str(str(j) + "\t" + str(y_true[j]) + "\t"+ str(y_pred[j]) + "\t" +  "#########################################################################" + "\n" + tx+ "\n\n"))
            bb.append(str(tx))
            j = j+1
