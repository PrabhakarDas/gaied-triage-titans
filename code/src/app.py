from flask import Flask, jsonify
import os
import re
import tempfile
import uuid
import hashlib
from email import policy
from email.parser import BytesParser
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from transformers import pipeline
from email_reply_parser import EmailReplyParser
from PyPDF2 import PdfReader
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Step 3: Configuration setup
REQUEST_TYPES = {
    "Adjustment": [],
    "AU Transfer": [],
    "Closing Notice": ["Reallocation fees", "Amendment fees", "Reallocation principal"],
    "Commitment Charge": ["Cashless Roll", "Decrease", "Increase"],
    "Fee payment": ["Ongoing fee", "Letter of Credit Fee"],
    "Money movement - Inbound": ["Principal", "Interest", "Principal + Interest", "Principal + Interest + fee"],
    "Money movement - Outbound": ["Timebound", "Foreign Currency"],
}

EXTRACTION_CONFIG = {
    "fields": {
        "deal_name": {
            "patterns": [r"Deal\s*Name:\s*(\w+)", r"Deal\s*#?\s*(\d+)"],
            "priority": ["email_body", "attachments"]
        },
        "amount": {
            "patterns": [r"\$(\d{1,3}(?:,\d{3})*\.\d{2})", r"Amount:\s*\$?(\d+(?:,\d{3})*\.\d{2})"],
            "priority": ["attachments", "email_body"]
        },
        "expiration_date": {
            "patterns": [r"Expiration Date:\s*(\d{2}/\d{2}/\d{4})", r"Exp\.?\s*Date:\s*(\d{4}-\d{2}-\d{2})"],
            "priority": ["email_body", "attachments"]
        }
    }
}

# Step 4: Initialize ML models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
processed_hashes = set()

# Step 5: Email processing functions
def parse_email(eml_path):
    print("********************")
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg

def get_email_body(msg):
    print("&&&&&&&&&&&&&&&&&&s",type(msg))
    import pdb;pdb.set_trace()
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()
    return msg.get_payload(decode=True).decode()

def clean_email_content(text):
    return EmailReplyParser.parse_reply(text).strip()

def is_duplicate(email_text):
    clean_text = clean_email_content(email_text)
    email_hash = hashlib.sha256(clean_text.encode()).hexdigest()
    return email_hash in processed_hashes

# Step 6: Attachment processing functions
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return '\n'.join([page.extract_text() for page in reader.pages])
    except Exception:
        images = convert_from_path(pdf_path)
        return '\n'.join([pytesseract.image_to_string(image) for image in images])

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

# Step 7: Classification functions
def generate_candidate_labels():
    candidates = []
    for main_type, sub_types in REQUEST_TYPES.items():
        candidates.append(main_type)
        for sub_type in sub_types:
            candidates.append(f"{main_type} - {sub_type}")
    return candidates

CANDIDATE_LABELS = generate_candidate_labels()

def classify_email(text):
    result = classifier(text, CANDIDATE_LABELS, multi_label=True)
    detected_requests = []
    
    for label, score in zip(result['labels'], result['scores']):
        if ' - ' in label:
            main_type, sub_type = label.split(' - ', 1)
        else:
            main_type = label
            sub_type = None
        
        detected_requests.append({
            "type": main_type,
            "sub_type": sub_type,
            "confidence": score,
            "reasoning": f"Detected '{label}' with confidence {score:.2f}"
        })
    
    detected_requests.sort(key=lambda x: x['confidence'], reverse=True)
    return detected_requests

# Step 8: Data extraction functions
def extract_fields(text, config):
    results = {}
    for field, settings in config['fields'].items():
        for pattern in settings['patterns']:
            match = re.search(pattern, text)
            if match:
                results[field] = match.group(1)
                break
    return results

def extract_data(email_body, attachments, config):
    extracted = {}
    for field, settings in config['fields'].items():
        sources = settings['priority']
        for source in sources:
            if source == 'email_body':
                field_data = extract_fields(email_body, {field: settings})
                if field_data.get(field):
                    extracted[field] = field_data[field]
                    break
            elif source == 'attachments':
                for attachment in attachments:
                    field_data = extract_fields(attachment, {field: settings})
                    if field_data.get(field):
                        extracted[field] = field_data[field]
                        break
                if extracted.get(field):
                    break
    return extracted

# Step 9: Flask endpoint
@app.route('/process_emails', methods=['POST'])
def process_emails():
    results = []
    email_folder = './emails'  # Folder containing .eml files
    
    for filename in os.listdir(email_folder):
        if not filename.endswith('.eml'):
            print(filename)
            continue
        
        eml_path = os.path.join(email_folder, filename)
        msg = parse_email(eml_path)
        print("message*****************", msg)
        email_body = get_email_body(msg)
        
        if is_duplicate(email_body):
            results.append({
                "email_id": filename,
                "status": "duplicate",
                "message": "Duplicate email detected"
            })
            continue
        
        # Process attachments
        attachment_texts = []
        temp_dir = tempfile.mkdtemp()
        for part in msg.iter_attachments():
            if part.get_filename():
                filepath = os.path.join(temp_dir, part.get_filename())
                with open(filepath, 'wb') as f:
                    f.write(part.get_payload(decode=True))
                
                if filepath.lower().endswith('.pdf'):
                    attachment_texts.append(extract_text_from_pdf(filepath))
                elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                    attachment_texts.append(extract_text_from_image(filepath))
        
        # Process email
        detected_requests = classify_email(email_body)
        extracted_data = extract_data(email_body, attachment_texts, EXTRACTION_CONFIG)
        
        # Prepare response
        email_results = []
        for req in detected_requests:
            email_results.append({
                "service_request_number": str(uuid.uuid4()),
                "confidence_score": req['confidence'],
                "reasoning": req['reasoning'],
                "request_type": req['type'],
                "sub_type": req['sub_type'],
                "extracted_data": extracted_data
            })
        
        results.append({
            "email_id": filename,
            "requests": email_results
        })
        
        # Cleanup
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    
    return jsonify({"results": results})

# Step 10: Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)