from transformers import Blip2Processor, Blip2ForConditionalGeneration
from accelerate import init_empty_weights
from PIL import Image
import torch

ModelLocation = "./model/model4/"
P_unit = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(P_unit)


processor = AutoTokenizer.from_pretrained(ModelLocation)
processor.padding_side = "left"
processor.truncation_side = "right"
#model = LlavaForConditionalGeneration.from_pretrained("./model/").to("cuda")


if 'model' not in locals() and 'model' not in globals():
    if P_unit=="cuda":
        with init_empty_weights():
            max_memory_mapping = {
            0: "25GiB",
            1: "20GiB",
            2: "15GiB",
            3: "10GiB",
            "cpu": "60GB"
            }
            model = AutoModel.from_pretrained(
                ModelLocation,
                device_map=P_unit,
                max_memory = max_memory_mapping, #style from https://huggingface.co/docs/transformers/perf_infer_gpu_one
        #        llm_int8_enable_fp32_cpu_offload=True,
        #        load_in_8bit=True  # Enable 8-bit quantization
                )
    if  P_unit=="cpu":
        model = AutoModel.from_pretrained(ModelLocation,
                                                     device_map=P_unit  # Force CPU execution
                                                     ).to(device)




# Load Image
image = Image.open("photo.jpg").convert("RGB")
   # Ensure correct input size (optional)

# Process Image Input
inputs = tokenizer(images=image, text="What is in the image?", return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs)




# Decode the generated text
caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", caption)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]