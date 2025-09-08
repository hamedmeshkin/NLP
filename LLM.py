################################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import no_grad
from torch.utils.data import DataLoader, Dataset
import os
import csv
import bitsandbytes
from accelerate import init_empty_weights

P_unit = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(P_unit)
print(f'device used is {device}')




tokenizer = AutoTokenizer.from_pretrained("/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/LLM/snapshots/Qwen/")

if 'model' not in locals() and 'model' not in globals():
    if P_unit=="cuda":
        with init_empty_weights():
            max_memory_mapping = {
            0: "40GiB",
            1: "40GiB",
            2: "40GiB",
            3: "40GiB",
            "cpu": "60GB"
            }
            model = AutoModelForCausalLM.from_pretrained(
                "/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/LLM/snapshots/Qwen/",
                device_map="auto",
                max_memory = max_memory_mapping, #style from https://huggingface.co/docs/transformers/perf_infer_gpu_one
        #        llm_int8_enable_fp32_cpu_offload=True,
                load_in_8bit=True  # Enable 8-bit quantization
                )
    if  P_unit=="cpu":
        model = AutoModelForCausalLM.from_pretrained("/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/LLM/snapshots/Qwen/",
                                                     device_map={"": P_unit}  # Force CPU execution
                                                     ).to(device)

    print("done loading model!")
    print(model.hf_device_map)


# Question
question = "Given “Deferasirox increases the exposure of repaglinide.”\
 Is it true that this sentence relates to how the pharmacokinetics of one drug, namely absorption, distribution, metabolism, \
 or excretion, change when co-administered with another drug, indicating a pharmacokinetic drug-drug interaction?  exactly answer with Exclusively reply by Yes or No"

sentence = 'Vaginal bleeding after menopause may be a warning sign of cancer of the uterus (womb). If you have had a hysterectomy (surgical removal of the uterus'
question = "Is this correct that the sentence above relates to how the pharmacokinetics of a drug, namely absorption, distribution, metabolism, or excretion, change when coadministered with another drug, indicating a pharmacokinetic drug drug interaction (PK-DDI)?"
prompt = sentence + "\n\n" + question + "\n\n" +  "Answer exactly with 'Yes' if the sentence relates to PK-DDI or 'No' if it does not."

prompt = "What is pharmacokinetics"

print(prompt)
print("##############################################################################################")
encodings = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
gen_tokens = model.generate(
    **encodings,
    max_new_tokens=1050,           # Limit the response length (enough for "Yes" or "No")
    temperature=0.0,             # Ensure deterministic output
    num_beams=3,                 # Greedy search is sufficient for a yes/no answer
    do_sample=False,             # Disable sampling to remove randomness
    return_dict_in_generate=True, # Return additional information (optional)
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print the response
generated_text = tokenizer.decode(gen_tokens['sequences'][0], skip_special_tokens=True)
print("\n\n\n")
print(generated_text)