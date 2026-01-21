"""
Flask REST API for Azure AI Document Translation

Endpoints:
- POST /upload - Upload document and translate
- POST /translate - Translate plain text
- GET /languages - Get supported languages
- GET /health - Health check
"""

import os
import sys
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()

# Import our modules
from translator import translate_text, get_supported_languages
from openai_client import generate_ai_response, summarize_text
from document_processor import extract_text, translate_document_file

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the frontend UI."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Azure AI Document Translation API",
        "version": "1.0.0"
    })

@app.route('/languages', methods=['GET'])
def list_languages():
    """Get list of supported languages for translation."""
    result = get_supported_languages()
    
    if result["success"]:
        # Return simplified language list
        languages = {}
        for code, info in result["languages"].items():
            languages[code] = info.get("name", code)
        
        return jsonify({
            "success": True,
            "languages": languages,
            "count": len(languages)
        })
    else:
        return jsonify({
            "success": False,
            "error": result["error"]
        }), 500

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    """
    Translate plain text.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No JSON data provided"}), 400
    
    text = data.get('text')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'hi')
    should_summarize = data.get('summarize', False)
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400
    
    if not target_lang:
        return jsonify({"success": False, "error": "No target_lang provided"}), 400
    
    response_data = {
        "original_text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    
    # Optional: Summarize first
    if should_summarize:
        summary_result = summarize_text(text, style="concise")
        if summary_result["success"]:
            text = summary_result["response"]
            response_data["summary"] = text
            response_data["tokens_used"] = summary_result["usage"]["total_tokens"]
        else:
            return jsonify({
                "success": False,
                "error": f"Summarization failed: {summary_result['error']}"
            }), 500
    
    # Translate
    result = translate_text(text, source_lang, target_lang)
    
    if result["success"]:
        response_data.update({
            "success": True,
            "translated_text": result["translated_text"]
        })
        return jsonify(response_data)
    else:
        return jsonify({
            "success": False,
            "error": result["error"]
        }), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Serve the translated file."""
    try:
        filename = secure_filename(filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(path):
            return send_file(
                path,
                as_attachment=True,
                download_name=filename.replace('translated_', '').replace(filename[:9], ''), # Try to clean up name roughly
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        else:
            return "File not found or expired.", 404
    except Exception as e:
        return str(e), 500

@app.route('/upload-document', methods=['POST'])
def upload_and_preserve():
    """
    Upload a document, translate it, and return preview + download link.
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File type not allowed"}), 400

    target_lang = request.form.get('target_lang', 'nl')
    source_lang = request.form.get('source_lang', 'en')
    
    filename = secure_filename(file.filename)
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    temp_input = os.path.join(app.config['UPLOAD_FOLDER'], f"in_{unique_id}_{filename}")
    output_filename = f"translated_{unique_id}_{os.path.splitext(filename)[0]}.docx"
    temp_output = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    try:
        file.save(temp_input)
        
        # Translate
        result = translate_document_file(temp_input, temp_output, source_lang, target_lang)
        
        if result["success"]:
            # Extract preview text from the output DOCX
            preview_text = "Preview not available."
            try:
                extraction = extract_text(result["output_path"])
                if extraction["success"]:
                    preview_text = extraction["text"][:1000] + "..." if len(extraction["text"]) > 1000 else extraction["text"]
            except:
                pass

            return jsonify({
                "success": True, 
                "message": "Translation complete",
                "preview_text": preview_text,
                "download_url": f"/download/{output_filename}",
                "original_filename": filename
            })
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

@app.route('/upload', methods=['POST'])
def upload_and_translate():
    """
    Upload a document and translate its content (Text Only).
    """
    # ... (existing code)
    """
    Upload a document and translate its content.
    
    Form Data:
    - file: The document file (PDF, DOCX, or TXT)
    - target_lang: Target language code (e.g., 'hi', 'fr', 'es')
    - source_lang: Source language code (default: 'en')
    - summarize: 'true' to summarize before translating (optional)
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Get parameters
    target_lang = request.form.get('target_lang', 'hi')
    source_lang = request.form.get('source_lang', 'en')
    should_summarize = request.form.get('summarize', 'false').lower() == 'true'
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(temp_path)
        
        # Extract text from document
        extraction_result = extract_text(temp_path)
        
        if not extraction_result["success"]:
            return jsonify({
                "success": False,
                "error": extraction_result["error"]
            }), 500
        
        extracted_text = extraction_result["text"]
        
        if not extracted_text or not extracted_text.strip():
            return jsonify({
                "success": False,
                "error": "No text content found in document"
            }), 400
        
        response_data = {
            "filename": filename,
            "file_type": extraction_result.get("file_type"),
            "extracted_text_length": len(extracted_text),
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        
        text_to_translate = extracted_text
        
        # Optional: Summarize first
        if should_summarize:
            # For long documents, chunk the text
            max_chars = 4000  # Keep under token limits
            if len(extracted_text) > max_chars:
                text_to_summarize = extracted_text[:max_chars]
                response_data["text_truncated"] = True
            else:
                text_to_summarize = extracted_text
            
            summary_result = summarize_text(text_to_summarize, style="concise")
            if summary_result["success"]:
                text_to_translate = summary_result["response"]
                response_data["summary"] = text_to_translate
                response_data["tokens_used"] = summary_result["usage"]["total_tokens"]
            else:
                return jsonify({
                    "success": False,
                    "error": f"Summarization failed: {summary_result['error']}"
                }), 500
        else:
            # For translation without summary, limit text length
            max_chars = 10000
            if len(text_to_translate) > max_chars:
                text_to_translate = text_to_translate[:max_chars]
                response_data["text_truncated"] = True
        
        # Translate
        result = translate_text(text_to_translate, source_lang, target_lang)
        
        if result["success"]:
            response_data.update({
                "success": True,
                "original_text": text_to_translate[:500] + "..." if len(text_to_translate) > 500 else text_to_translate,
                "translated_text": result["translated_text"]
            })
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze', methods=['POST'])
def analyze_document():
    """
    Upload a document and analyze it with Azure OpenAI.
    
    Form Data:
    - file: The document file (PDF, DOCX, or TXT)
    - prompt: Custom prompt for analysis (optional)
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400
    
    custom_prompt = request.form.get('prompt', '')
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(temp_path)
        
        extraction_result = extract_text(temp_path)
        
        if not extraction_result["success"]:
            return jsonify({
                "success": False,
                "error": extraction_result["error"]
            }), 500
        
        extracted_text = extraction_result["text"]
        
        # Limit text for API
        max_chars = 4000
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars]
        
        # Build prompt
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nDocument content:\n{extracted_text}"
        else:
            prompt = f"Analyze and summarize the following document:\n\n{extracted_text}"
        
        result = generate_ai_response(prompt)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "filename": filename,
                "analysis": result["response"],
                "tokens_used": result["usage"]["total_tokens"]
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB."
    }), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Azure AI Document Translation API")
    print("=" * 50)
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  GET  /languages  - List supported languages")
    print("  POST /translate  - Translate text")
    print("  POST /upload     - Upload & translate document")
    print("  POST /analyze    - Analyze document with AI")
    print("\nStarting server on http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
