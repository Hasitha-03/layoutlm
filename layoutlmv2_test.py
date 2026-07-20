from transformers import pipeline

nlp = pipeline(
    "document-question-answering",
    model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
)

image_path = "receipt1.jpg"

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "What is the amount due?",
]

for q in questions:
    result = nlp(image_path, q)
    print(f"\nQ: {q}")
    print(f"A: {result}")
