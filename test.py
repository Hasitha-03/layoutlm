from transformers import pipeline

print("Loading model... this may take a minute the first time.")

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-invoices",
)

image_url = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"

questions = [
    "What is the invoice number?",
    "What is the total amount?",
    "What is the invoice 
