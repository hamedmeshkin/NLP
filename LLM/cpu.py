import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
from transformers.utils import logging
import time
start_time = time.time()  # Start the timer

# Initialize Distributed Processing
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
args = parser.parse_args()
dist.init_process_group(backend="gloo")  # Use "gloo" for CPU-based parallelism

# Define CPU device
P_unit = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(P_unit)
print(f"Using device: {device}, Rank: {dist.get_rank()}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/Qwen/"
)

# Load Model on CPU
print("Model loading ... .")
model = AutoModelForCausalLM.from_pretrained(
    "/projects01/lizhi/seyedhamed.meshkin/SuperLargeLanguageModels/models/deepseek_Llama/snapshots/Qwen/",
    device_map={"": P_unit}  # Force CPU execution
).to(device)

print("Model loaded successfully.")
# Define Device Mesh (corrected)
world_size = dist.get_world_size()  # Total number of nodes
device_mesh = DeviceMesh(P_unit, list(range(world_size)))  # Corrected version
# Define Parallelization Plan
parallelize_plan = {
    'transformer.h.0.mlp.fc_in': ColwiseParallel(),
    'transformer.h.0.mlp.fc_out': RowwiseParallel(),
    # Add more layers as needed
}
# Apply Tensor Parallelism (corrected)
model = parallelize_module(model, device_mesh, parallelize_plan)

if dist.get_rank() == 0:
    print("Model successfully parallelized across CPU nodes!")

# Prepare input
question = "What is pharmacokinetics drug-drug interaction?"
encodings = tokenizer(question, return_tensors="pt").to(device)

# Generate response in parallel
with torch.no_grad():
    gen_tokens = model.generate(
        **encodings,
        max_new_tokens=1000,
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


end_time = time.time()  # End the timer
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")