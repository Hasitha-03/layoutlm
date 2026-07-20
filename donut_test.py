
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

image = Image.open("receipt1.jpg").convert("RGB")

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "What is the amount due?",
]

for q in questions:
    prompt = f"<s_docvqa><s_question>{q}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=128,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the prompt echo from the output
    answer = answer.split("<s_answer>")[-1].strip()
    print(f"\nQ: {q}")
    print(f"A: {answer}")
