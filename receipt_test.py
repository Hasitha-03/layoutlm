from transformers import pipeline

nlp = pipeline("document-question-answering", model="impira/layoutlm-invoices")

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "What is the ticket number?",
    "How was this paid?",
]

for q in questions:
    result = nlp("receipt.jpg", q)
    print(f"\nQ: {q}")
    print(f"A: {result}")
