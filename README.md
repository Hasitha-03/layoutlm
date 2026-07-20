Testing LayoutLM on Real-World Invoices and Receipts

A hands-on exploration of impira/layoutlm-invoices (Hugging Face) for document question-answering

setup 

python3.11 -m venv venv
source venv/bin/activate
pip install torch transformers pytesseract pillow
brew install tesseract  

The experiments
1. Clean invoice template - It returned high-confidence, correct answers for invoice number, total amount, and due date (all >0.99 confidence), one exception found - OCR misread — the two-column "Bill To / Ship To" layout confused Tesseract's reading order, corrupting the extracted text before it ever reached the model.
2. Real photographed receipt — accuracy degrades:
Issue	Answer	and the Root cause
Vendor name	"Bag" instead of "Bao"	OCR character misread
Item name	"Durnplings" instead of "Dumplings"	Classic OCR rn/m confusion
Date	Wrong year (2026 vs 2025)	OCR digit misread
Ticket number	Grabbed wrong field entirely (unit number instead)	Model reasoning error — multiple similar label+number pairs confused the answer selection
"What items were purchased"	Only one item returned	Architectural limit — extractive QA returns a single text span, never a list

3. Receipt with multi-step math — genuine uncertainty of confidence
A second real receipt (with a subtotal, a discount line, a service charge, and a final "Amount due") - The model repeatedly picked an intermediate subtotal instead of the true final total, because the document contained three plausible "total-looking" numbers. the confidence scores correctly signaled this: the wrong answers came back with confidence near zero. This confirmed that confidence score is a genuinely useful, trustworthy signal for flagging bad answers

4. Comparison: general vision model (BLIP-VQA): To test whether "vision-capable" alone is enough for document understanding, the same receipt was tested against Salesforce/blip-vqa-base, a general-purpose visual QA model not fine-tuned for documents. Result: complete failure across all five questions (e.g., total amount answered as "3", amount due answered as "receipt"). This demonstrated that vision capability and document-reading capability are not the same skill — a model trained on natural photos (dogs, scenes, objects) does not automatically transfer to reading dense structured text
5. Comparison: OCR-free document model (Donut)
naver-clova-ix/donut-base-finetuned-docvqa skips OCR entirely, reading raw pixels end-to-end via a generative (not extractive) architecture.
Result: a genuinely mixed outcome — Donut correctly answered "amount due" ($18.54) where LayoutLM failed, but got the vendor name wrong where LayoutLM succeeded, and garbled the date worse than LayoutLM's OCR did.

Takeaway: removing OCR doesn't remove errors — it relocates them. LayoutLM's failures are traceable to an inspectable intermediate step (the OCR output). Donut's failures happen inside an opaque end-to-end generation process with no equivalent debugging step. This is a real tradeoff between transparency and architectural simplicity, not a strict improvement in either direction.
