from transformers import pipeline

nlp = pipeline("document-question-answering", model="impira/layoutlm-invoices")

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "what is the amount due?"
]

for q in questions:
    result = nlp("receipt1.jpg", q)
    print(f"\nQ: {q}")
    print(f"A: {result}")
