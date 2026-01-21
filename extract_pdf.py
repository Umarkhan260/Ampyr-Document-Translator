import pdfplumber

with pdfplumber.open(r'c:\Users\UmarKhan\Downloads\Service.pdf') as pdf:
    text = ''
    for page in pdf.pages:
        text += page.extract_text() + '\n'

print(text)