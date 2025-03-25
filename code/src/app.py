import os
import hashlib
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re

# --- Configuration ---
# Folder containing PDFs
PDF_FOLDER = './emails'
# Candidate labels (request types and sub-types)
labels = [
  "Adjustment",
  "AU Transfer",
  "Closing Notice",
  "Commitment Charge",
  "Fee payment",
  "Money movement - Inbound",
  "Money movement - Outbound"
]

# Initialize the zero-shot classification pipeline using a free pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Optional: Initialize a generation model for reasoning (using a summarization or seq2seq model)
# For demonstration, we use a T5 model; adjust prompt format as needed.
reasoning_tokenizer = AutoTokenizer.from_pretrained("t5-base")
reasoning_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# --- Functions ---

def extract_text_from_pdf(pdf_path):
  """Converts PDF to images and extracts text using OCR."""
  images = convert_from_path(pdf_path)
  text = ""
  for image in images:
    text += pytesseract.image_to_string(image) + "\n"
    print("type(text) ----  ", type(text))
  return text

def detect_duplicate(text, seen_hashes):
  """Simple duplicate detection based on content hash."""
  text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
  if text_hash in seen_hashes:
    return True, text_hash
  seen_hashes.add(text_hash)
  return False, text_hash

def classify_email(email_text):
  """
  Use zero-shot classification to predict the request type.
  Returns predicted label, confidence score and explanation.
  """
  # Classify the email text against candidate labels
  result = classifier(email_text, candidate_labels=labels)
  # Determine the primary request type (highest score)
  request_type = result['labels'][0]
  confidence_score = result['scores'][0]

  # Generate reasoning explanation using a simple prompt approach.
  prompt = f"Explain why the following email is classified as '{request_type}':\n\n{email_text[:500]}\n\nExplanation:"
  input_ids = reasoning_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
  reasoning_output_ids = reasoning_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
  reasoning = reasoning_tokenizer.decode(reasoning_output_ids[0], skip_special_tokens=True)

  return request_type, confidence_score, reasoning

def extract_configurable_fields(email_text):
  """
  Extract configurable fields like deal name, amount, expiration date.
  This can be enhanced using regex or rule-based extraction.
  """
  # Example: Extract amount (this is a basic regex; adjust as needed)
  amount_pattern = re.compile(r'\b(?:USD|EUR|\$)?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b')
  amounts = amount_pattern.findall(email_text)

  # Example: Extract expiration date (pattern may need to be adjusted for your date formats)
  date_pattern = re.compile(r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](\d{2,4})\b')
  dates = date_pattern.findall(email_text)

  # Deal name extraction could be rule-based depending on how it appears in the text
  # For example, assume deal names are quoted or prefixed by "Deal Name:"
  deal_pattern = re.compile(r'Deal Name:\s*([A-Za-z0-9\s]+)')
  deals = deal_pattern.findall(email_text)

  return {
    "amounts": amounts,
    "expiration_dates": ['-'.join(date) for date in dates],
    "deal_names": deals
  }

# --- Main Processing Loop ---

def process_emails(pdf_folder):
  seen_hashes = set()  # for duplicate detection
  results = []

  for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
      pdf_path = os.path.join(pdf_folder, filename)
      email_text = extract_text_from_pdf(pdf_path)

      # Skip if duplicate
      is_dup, text_hash = detect_duplicate(email_text, seen_hashes)
      if is_dup:
        print(f"Duplicate detected: {filename}")
        continue

      # Classification step
      request_type, confidence_score, reasoning = classify_email(email_text)

      # Field extraction (for additional configurable fields)
      fields = extract_configurable_fields(email_text)

      # You can also incorporate priority rules here:
      # For example, if the email body contains clear classification hints, prioritize that over any OCR noise from attachments.
      # (Implement custom logic as required.)

      result = {
        "filename": filename,
        "request_type": request_type,
        "confidence_score": confidence_score,
        "reasoning": reasoning,
        "extracted_fields": fields
      }
      results.append(result)
      print(f"Processed {filename}: {result}")
  return results

# --- Run the processing ---
if __name__ == "__main__":
  results = process_emails(PDF_FOLDER)
  # Optionally, write the results to a JSON or CSV file for further review
  import json
  with open("email_classification_results.json", "w") as f:
    json.dump(results, f, indent=2)
