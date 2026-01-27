"""
Document Processor Module

Extracts text from PDF and DOCX files for translation.
"""

import os
import sys
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
from translator import translate_text, translate_long_text

def translate_paragraph_runs(para, source_lang: str, target_lang: str) -> int:
    """
    Translate paragraph by translating each run individually to preserve formatting.
    Returns count of translated runs.
    """
    translated_count = 0
    runs_with_text = [r for r in para.runs if r.text and r.text.strip()]
    
    if not runs_with_text:
        # No runs with text - paragraph might have text but no runs (common in converted PDFs)
        return 0
    
    for run in runs_with_text:
        # Use translate_long_text for potentially large runs
        result = translate_long_text(run.text, source_lang, target_lang)
        if result["success"]:
            # Preserve the run's text while keeping all formatting (bold, italic, font, etc.)
            run.text = result["translated_text"]
            translated_count += 1
        else:
            print(f"DEBUG: Run translation failed: {result.get('error')}", file=sys.stderr)
    
    return translated_count

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
        
        # Translate paragraphs - use run-level translation to preserve formatting
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                # Translate each run individually to preserve bold, italic, fonts, etc.
                runs_translated = translate_paragraph_runs(para, source_lang, target_lang)
                if runs_translated > 0:
                    translated_count += runs_translated
                    print(f"DEBUG: Para {i} - translated {runs_translated} runs", file=sys.stderr)
                elif para.text.strip():
                    # Fallback: If no runs but text exists, translate whole paragraph
                    print(f"DEBUG: Para {i} - no runs, using paragraph-level translation ({len(para.text)} chars)", file=sys.stderr)
                    result = translate_long_text(para.text, source_lang, target_lang)
                    if result["success"]:
                        para.text = result["translated_text"]
                        translated_count += 1
                    else:
                        print(f"DEBUG: Translation failed for para {i}: {result.get('error')}", file=sys.stderr)

        # Translate tables - use run-level translation to preserve formatting
        for t_idx, table in enumerate(doc.tables):
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        if para.text.strip():
                            # Translate each run individually to preserve formatting
                            runs_translated = translate_paragraph_runs(para, source_lang, target_lang)
                            if runs_translated > 0:
                                translated_count += runs_translated
                            elif para.text.strip():
                                # Fallback for paragraphs without runs
                                result = translate_long_text(para.text, source_lang, target_lang)
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
    Create a professional-looking DOCX with translated text.
    Includes heading detection, bold term preservation, and proper styling.
    """
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.style import WD_STYLE_TYPE
        import re
        
        doc = Document()
        
        # Set up document margins
        for section in doc.sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1.25)
            section.right_margin = Inches(1.25)
        
        # Clean up page markers like "--- Page 1 ---"
        text = re.sub(r'---\s*Page\s*\d+\s*---\n*', '', text)
        
        # Split by double newlines to paragraphs
        paragraphs = text.split('\n\n')
        
        print(f"DEBUG: Creating professional DOCX - {len(paragraphs)} sections", file=sys.stderr)
        
        translated_count = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            # Translate the paragraph
            result = translate_long_text(para, source_lang, target_lang)
            if result["success"]:
                translated_text = result["translated_text"]
                translated_count += 1
            else:
                print(f"DEBUG: Para {i} translation failed: {result.get('error')}", file=sys.stderr)
                translated_text = para  # Keep original if failed
            
            # Detect if this is a heading (short text, likely a title)
            is_heading = (
                len(translated_text) < 100 and 
                '\n' not in translated_text and
                not translated_text.endswith('.') and
                not translated_text.endswith(',')
            )
            
            # Detect centered/special text (often short lines with dashes or special markers)
            is_centered = (
                translated_text.startswith('-') and translated_text.endswith('-') or
                translated_text.startswith('—') and translated_text.endswith('—') or
                'hereinafter' in translated_text.lower() or
                'referred to as' in translated_text.lower() or
                'nachstehend' in para.lower()  # German "hereinafter"
            )
            
            if is_heading:
                # Add as heading
                heading = doc.add_heading(translated_text, level=2)
                heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif is_centered:
                # Add as centered paragraph
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(translated_text)
                run.italic = True
            else:
                # Regular paragraph - detect and bold quoted terms
                p = doc.add_paragraph()
                
                # Find terms in quotes like "PVA" or "Solar Park" and make them bold
                pattern = r'[""„]([^""„"]+)["""]'
                last_end = 0
                
                for match in re.finditer(pattern, translated_text):
                    # Add text before the match
                    if match.start() > last_end:
                        p.add_run(translated_text[last_end:match.start()])
                    
                    # Add the quoted term with quotes and bold
                    quoted_run = p.add_run(f'"{match.group(1)}"')
                    quoted_run.bold = True
                    
                    last_end = match.end()
                
                # Add remaining text
                if last_end < len(translated_text):
                    p.add_run(translated_text[last_end:])
                
                # If no quoted terms were found, the paragraph might be empty
                if last_end == 0:
                    p.add_run(translated_text)
            
            # Add some spacing between paragraphs
            if not is_heading:
                p = doc.paragraphs[-1]
                p.paragraph_format.space_after = Pt(8)
        
        doc.save(output_path)
        print(f"DEBUG: Professional DOCX created with {translated_count} translated sections", file=sys.stderr)
        
        return {"success": True, "output_path": output_path, "translated_count": translated_count}
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
    print(f"DEBUG: translate_document_file called with {file_path}", file=sys.stderr)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.docx':
        return translate_docx_file(file_path, output_path, source_lang, target_lang)
    
    elif ext == '.pdf':
        # For PDFs: Extract text directly and create translated DOCX
        # This is more reliable than pdf2docx conversion
        print(f"DEBUG: Processing PDF file...", file=sys.stderr)
        
        if not output_path.endswith('.docx'):
            output_path += '.docx'
        
        # Step 1: Extract text from PDF using pdfplumber
        extraction = extract_text_from_pdf(file_path)
        
        if not extraction["success"]:
            print(f"DEBUG: PDF text extraction failed: {extraction.get('error')}", file=sys.stderr)
            return {"success": False, "error": f"Failed to extract text from PDF: {extraction.get('error')}"}
        
        text = extraction.get("text", "")
        page_count = extraction.get("page_count", 0)
        
        print(f"DEBUG: Extracted {len(text)} characters from {page_count} pages", file=sys.stderr)
        
        if not text or not text.strip():
            return {"success": False, "error": "No text found in PDF. The PDF might be scanned/image-based."}
        
        # Step 2: Create translated DOCX
        result = create_simple_translated_docx(text, output_path, source_lang, target_lang)
        
        if result["success"]:
            result["page_count"] = page_count
            result["original_chars"] = len(text)
            print(f"DEBUG: PDF translation complete. Output: {output_path}", file=sys.stderr)
        
        return result
        
    else:
        return {"success": False, "error": "Unsupported file type for format preservation"}


def translate_pdf_preserve_layout(file_path: str, output_path: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate PDF while preserving the EXACT original layout.
    Uses BATCH translation for speed - collects all text first, translates in batches.
    
    This creates a translated PDF (not DOCX) that looks like the original.
    """
    try:
        import fitz  # PyMuPDF
        from translator import translate_text
        import requests
        import os
        
        print(f"DEBUG: Starting layout-preserving PDF translation: {file_path}", file=sys.stderr)
        
        # Open the PDF
        doc = fitz.open(file_path)
        
        # Ensure output is PDF
        if not output_path.lower().endswith('.pdf'):
            output_path = output_path.rsplit('.', 1)[0] + '.pdf'
        
        # Step 1: Collect all text spans with their positions
        print(f"DEBUG: Collecting text from {len(doc)} pages...", file=sys.stderr)
        all_spans = []  # List of (page_num, span_info, original_text)
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            
            for block in blocks:
                if block["type"] != 0:  # Skip non-text blocks
                    continue
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        original_text = span.get("text", "").strip()
                        
                        if original_text and len(original_text) >= 2:
                            all_spans.append({
                                "page": page_num,
                                "bbox": span["bbox"],
                                "size": span["size"],
                                "font": span.get("font", "helv"),
                                "color": span.get("color", 0),
                                "original": original_text,
                                "translated": None
                            })
        
        print(f"DEBUG: Found {len(all_spans)} text spans to translate", file=sys.stderr)
        
        if not all_spans:
            return {"success": False, "error": "No text found in PDF"}
        
        # Step 2: Batch translate all texts
        # Azure Translator can handle multiple texts in one request
        batch_size = 25  # Azure allows up to 100, but smaller batches are safer
        translated_count = 0
        
        # Get Azure credentials
        azure_key = os.environ.get("AZURE_TRANSLATOR_KEY", "")
        azure_region = os.environ.get("AZURE_TRANSLATOR_REGION", "")
        azure_endpoint = os.environ.get("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
        
        for i in range(0, len(all_spans), batch_size):
            batch = all_spans[i:i+batch_size]
            texts_to_translate = [{"text": s["original"]} for s in batch]
            
            print(f"DEBUG: Translating batch {i//batch_size + 1}/{(len(all_spans) + batch_size - 1)//batch_size} ({len(batch)} texts)", file=sys.stderr)
            
            # Make batch API call
            try:
                params = {"api-version": "3.0", "to": target_lang}
                if source_lang and source_lang != 'auto':
                    params["from"] = source_lang
                
                headers = {
                    "Ocp-Apim-Subscription-Key": azure_key,
                    "Ocp-Apim-Subscription-Region": azure_region,
                    "Content-type": "application/json"
                }
                
                response = requests.post(
                    f"{azure_endpoint}/translate",
                    params=params,
                    headers=headers,
                    json=texts_to_translate,
                    timeout=60
                )
                
                if response.status_code == 200:
                    results = response.json()
                    for j, result in enumerate(results):
                        if result.get("translations"):
                            batch[j]["translated"] = result["translations"][0]["text"]
                            translated_count += 1
                else:
                    print(f"DEBUG: Batch translation failed: {response.status_code}", file=sys.stderr)
                    # Fall back to individual translation for this batch
                    for span_info in batch:
                        result = translate_text(span_info["original"], source_lang, target_lang)
                        if result["success"]:
                            span_info["translated"] = result["translated_text"]
                            translated_count += 1
                            
            except Exception as batch_error:
                print(f"DEBUG: Batch error: {str(batch_error)}, falling back to individual", file=sys.stderr)
                for span_info in batch:
                    result = translate_text(span_info["original"], source_lang, target_lang)
                    if result["success"]:
                        span_info["translated"] = result["translated_text"]
                        translated_count += 1
        
        # Step 3: Apply translations to PDF
        print(f"DEBUG: Applying {translated_count} translations to PDF...", file=sys.stderr)
        
        for span_info in all_spans:
            if not span_info["translated"]:
                continue
            
            translated_text = span_info["translated"]
            original_text = span_info["original"]
            
            # Skip if same
            if translated_text == original_text:
                continue
            
            page = doc[span_info["page"]]
            bbox = fitz.Rect(span_info["bbox"])
            font_size = span_info["size"]
            
            # Cover original text with white rectangle
            page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1))
            
            # Adjust font size if translated text is longer
            text_length_ratio = len(translated_text) / max(len(original_text), 1)
            adjusted_font_size = font_size
            if text_length_ratio > 1.2:
                adjusted_font_size = font_size / (text_length_ratio * 0.85)
                adjusted_font_size = max(adjusted_font_size, 6)
            
            # Insert translated text
            text_point = fitz.Point(bbox.x0, bbox.y1 - 2)
            
            try:
                page.insert_text(
                    text_point,
                    translated_text,
                    fontsize=adjusted_font_size,
                    fontname="helv"
                )
            except:
                # Fallback
                page.insert_text(text_point, translated_text, fontsize=adjusted_font_size)
        
        # Save the translated PDF
        doc.save(output_path)
        doc.close()
        
        print(f"DEBUG: Layout-preserving translation complete. Translated {translated_count}/{len(all_spans)} spans", file=sys.stderr)
        
        return {
            "success": True,
            "output_path": output_path,
            "translated_count": translated_count,
            "total_blocks": len(all_spans),
            "file_type": "pdf"
        }
        
    except Exception as e:
        print(f"DEBUG: Layout-preserving PDF translation failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"PDF layout translation failed: {str(e)}"
        }


# Quick test when run directly
if __name__ == "__main__":
    print("Document Processor Module")
    print("Supported formats: PDF, DOCX, TXT")

