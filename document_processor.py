"""
Document Processor Module

Extracts text from PDF and DOCX files for translation.
"""

import os
from typing import Optional


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        dict with success status and extracted text
    """
    try:
        import pdfplumber
        import re
        
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Try to extract with layout preservation
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                
                if page_text:
                    # Cleanup: Fix hyphenation and broken lines
                    # Remove multiple spaces
                    clean_text = re.sub(r'[ ]{2,}', ' ', page_text)
                    # Fix hyphenation at end of line (e.g. "exam-\nple" -> "example")
                    clean_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', clean_text)
                    # Join lines that don't end in punctuation (simple heuristic)
                    # clean_text = re.sub(r'(?<![.!?])\n', ' ', clean_text) 
                    # (Disabled aggressive joining for now as it might merge headers/lists)
                    
                    text_content.append(f"--- Page {page_num} ---\n{clean_text}")
        
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
             print(f"DEBUG: No text found via native extraction. Attempting Vision OCR...", file=sys.stderr)
             return ocr_pdf_with_vision(file_path)
        
        return {
            "success": True,
            "text": full_text,
            "page_count": len(text_content),
            "file_type": "pdf",
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "error": f"Failed to extract PDF text: {str(e)}"
        }

def ocr_pdf_with_vision(file_path: str) -> dict:
    """
    Extract text from PDF using OCR (Azure OpenAI Vision) via PyMuPDF rendering.
    """
    try:
        import fitz  # PyMuPDF
        from openai_client import extract_text_from_image
        
        doc = fitz.open(file_path)
        text_parts = []
        
        print(f"DEBUG: Starting OCR for {len(doc)} pages...", file=sys.stderr)
        
        for i, page in enumerate(doc):
            # Render page to image (scale 2.0 for better quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            result = extract_text_from_image(img_bytes)
            
            if result["success"]:
                text_parts.append(f"--- Page {i+1} ---\n{result['text']}")
                print(f"DEBUG: OCR Page {i+1} success", file=sys.stderr)
            else:
                text_parts.append(f"--- Page {i+1} ---\n[OCR Failed]")
                print(f"DEBUG: OCR Page {i+1} failed: {result.get('error')}", file=sys.stderr)
        
        doc.close()
        full_text = "\n\n".join(text_parts)
        
        return {
            "success": True,
            "text": full_text,
            "page_count": len(text_parts),
            "file_type": "pdf_ocr",
            "error": None
        }
        
    except Exception as e:
        print(f"DEBUG: OCR failed: {str(e)}", file=sys.stderr)
        return {"success": False, "text": None, "error": f"OCR failed: {str(e)}"}


def extract_text_from_docx(file_path: str) -> dict:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
    
    Returns:
        dict with success status and extracted text
    """
    try:
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)
        
        return {
            "success": True,
            "text": full_text,
            "paragraph_count": len(paragraphs),
            "file_type": "docx",
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "error": f"Failed to extract DOCX text: {str(e)}"
        }


def extract_text_from_txt(file_path: str) -> dict:
    """
    Read text from a plain text file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        dict with success status and text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return {
            "success": True,
            "text": text,
            "file_type": "txt",
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "error": f"Failed to read text file: {str(e)}"
        }


def extract_text(file_path: str, file_type: Optional[str] = None) -> dict:
    """
    Extract text from a document based on file type.
    
    Args:
        file_path: Path to the document
        file_type: Optional file type override ('pdf', 'docx', 'txt')
    
    Returns:
        dict with extracted text or error
    """
    if not os.path.exists(file_path):
        return {
            "success": False,
            "text": None,
            "error": f"File not found: {file_path}"
        }
    
    # Determine file type from extension if not provided
    if not file_type:
        _, ext = os.path.splitext(file_path)
        file_type = ext.lower().lstrip('.')
    
    # Route to appropriate extractor
    extractors = {
        'pdf': extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'txt': extract_text_from_txt,
        'text': extract_text_from_txt,
    }
    
    extractor = extractors.get(file_type)
    
    if not extractor:
        return {
            "success": False,
            "text": None,
            "error": f"Unsupported file type: {file_type}. Supported: pdf, docx, txt"
        }
    
    return extractor(file_path)


# ... (existing imports)
from translator import translate_text

def translate_docx_file(file_path: str, output_path: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate a DOCX file preserving formatting.
    """
import sys

def translate_docx_file(file_path: str, output_path: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate a DOCX file preserving formatting.
    """
    try:
        from docx import Document
        
        print(f"DEBUG: Opening DOCX: {file_path}", file=sys.stderr)
        doc = Document(file_path)
        
        para_count = len(doc.paragraphs)
        table_count = len(doc.tables)
        print(f"DEBUG: Found {para_count} paragraphs and {table_count} tables.", file=sys.stderr)
        
        translated_count = 0
        
        # Translate paragraphs
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                original_text = para.text
                result = translate_text(original_text, source_lang, target_lang)
                if result["success"]:
                    translated_text = result["translated_text"]
                    para.text = translated_text
                    translated_count += 1
                else:
                    print(f"DEBUG: Translation failed for para {i}: {result.get('error')}", file=sys.stderr)

        # Translate tables
        for t_idx, table in enumerate(doc.tables):
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        if para.text.strip():
                            original_text = para.text
                            result = translate_text(original_text, source_lang, target_lang)
                            if result["success"]:
                                para.text = result["translated_text"]
                                translated_count += 1
        
        print(f"DEBUG: Total translated segments: {translated_count}", file=sys.stderr)
        
        if translated_count == 0 and (para_count > 0 or table_count > 0):
             print(f"WARNING: Content found but nothing translated (maybe empty text?)", file=sys.stderr)
        elif para_count == 0 and table_count == 0:
             print(f"WARNING: No paragraphs or tables found in DOCX! (Possible scanned PDF or textboxes)", file=sys.stderr)
             
        doc.save(output_path)
        print(f"DEBUG: Saved to {output_path}", file=sys.stderr)
        
        return {
            "success": True,
            "output_path": output_path,
            "file_type": "docx",
            "translated_count": translated_count
        }
        
    except Exception as e:
        print(f"DEBUG: Error in translate_docx_file: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": f"DOCX translation failed: {str(e)}"
        }

def create_simple_translated_docx(text: str, output_path: str, source_lang: str, target_lang: str) -> dict:
    """
    Create a new DOCX with translated text (fallback method).
    """
    try:
        from docx import Document
        doc = Document()
        doc.add_heading('Translated Document (Fallback Layout)', 0)
        
        # Split by double newlines to paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                result = translate_text(para, source_lang, target_lang)
                if result["success"]:
                    doc.add_paragraph(result["translated_text"])
                else:
                    doc.add_paragraph(para) # Keep original if failed
                    
        doc.save(output_path)
        return {"success": True, "output_path": output_path}
    except Exception as e:
        return {"success": False, "error": f"Fallback creation failed: {str(e)}"}

def convert_pdf_to_docx(pdf_path: str, docx_path: str) -> dict:
    """
    Convert PDF to DOCX using pdf2docx.
    """
    try:
        from pdf2docx import Converter
        
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        
        return {"success": True, "output_path": docx_path}
    except Exception as e:
        return {"success": False, "error": f"PDF conversion failed: {str(e)}"}

def translate_document_file(file_path: str, output_path: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate document preserving format (converts PDF to DOCX).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.docx':
        return translate_docx_file(file_path, output_path, source_lang, target_lang)
    
    elif ext == '.pdf':
        # First convert to DOCX
        temp_docx = file_path + ".temp.docx"
        conversion = convert_pdf_to_docx(file_path, temp_docx)
        
        if not conversion["success"]:
            return conversion
            
        # Then translate the DOCX
        if not output_path.endswith('.docx'):
            output_path += '.docx'
            
        result = translate_docx_file(temp_docx, output_path, source_lang, target_lang)
        
        # Check if we successfully translated anything
        if result["success"] and result.get("translated_count", 0) == 0:
            print(f"WARNING: No content translated from converted DOCX. Using fallback extraction.", file=sys.stderr)
            # Fallback: Extract text using pdfplumber and create simple DOCX
            extraction = extract_text_from_pdf(file_path)
            if extraction["success"] and extraction["text"]:
                fallback_result = create_simple_translated_docx(extraction["text"], output_path, source_lang, target_lang)
                # Cleanup temp
                if os.path.exists(temp_docx):
                    os.remove(temp_docx)
                return fallback_result
        
        # Cleanup temp
        if os.path.exists(temp_docx):
            os.remove(temp_docx)
            
        return result
        
    else:
        return {"success": False, "error": "Unsupported file type for format preservation"}

# Quick test when run directly
if __name__ == "__main__":
    print("Document Processor Module")
    print("Supported formats: PDF, DOCX, TXT")
