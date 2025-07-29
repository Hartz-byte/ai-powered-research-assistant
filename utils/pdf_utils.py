from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_file):
    text = ''
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    except Exception:
        text = ''
    return text
