from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("receipt1.jpg").convert("RGB")

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "What is the amount due?",
]

for q in questions:
    inputs = processor(image, q, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
