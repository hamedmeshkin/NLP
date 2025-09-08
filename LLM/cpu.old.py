from torch.distributed.device_mesh import DeviceMesh
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
# Get environment variables set by SLURM
MASTER_ADDR = os.getenv("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.getenv("MASTER_PORT", "29500")
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
RANK = int(os.getenv("RANK", "0"))

# Initialize Torch Distributed Processing
dist.init_process_group(
    backend="gloo",
    init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
    world_size=WORLD_SIZE,
    rank=RANK
)

# Set Device to CPU Only
device = torch.device("cpu")
print(f"Using device: {device}, Rank: {dist.get_rank()}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/model/")

# Load Model on CPU and Wrap with DDP
model = AutoModelForCausalLM.from_pretrained(
    "/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/model/",
    device_map={"": "cpu"}  # Force CPU execution
).to(device)

# Wrap Model in Distributed Data Parallel (DDP)
model = DDP(model)

if dist.get_rank() == 0:
    print("Model successfully loaded across multiple nodes!")

# Question
question = "What is pharmacokinetics drug-drug interaction?"
encodings = tokenizer(question, return_tensors="pt").to(device)


# Generate response in parallel
with torch.no_grad():
    gen_tokens = model.generate(
        **encodings,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        num_beams=5,
        do_sample=False,
        return_dict_in_generate=True
    )

# Gather results only on rank 0

generated_text = tokenizer.decode(gen_tokens['sequences'][0], skip_special_tokens=True)
print("Generated Response:\n", generated_text)

# Clean up Distributed Processing
dist.barrier()
dist.destroy_process_group()
