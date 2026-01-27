"""
Azure AI Translator Integration Module

This module provides a reusable function to translate text using
Azure Cognitive Services Translator API.

Authentication: API Key (Ocp-Apim-Subscription-Key + Region header)
Region: Central India
"""

import os
import requests
import uuid
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    api_key: Optional[str] = None,
    region: Optional[str] = None,
    endpoint: Optional[str] = None
) -> dict:
    """
    Translate text from source language to target language using Azure Translator.
    
    Args:
        text: The text to translate
        source_lang: Source language code (e.g., 'en', 'hi', 'fr')
        target_lang: Target language code (e.g., 'hi', 'en', 'es')
        api_key: Optional API key (defaults to env variable)
        region: Optional region (defaults to env variable)
        endpoint: Optional endpoint (defaults to env variable)
    
    Returns:
        dict containing:
            - success: bool indicating if translation succeeded
            - original_text: the input text
            - translated_text: the translated text (if successful)
            - source_language: detected/specified source language
            - target_language: target language
            - error: error message (if failed)
    
    Example:
        >>> result = translate_text("Hello, world!", "en", "hi")
        >>> print(result['translated_text'])
    """
    
    # Get configuration from environment or parameters
    api_key = api_key or os.getenv("AZURE_TRANSLATOR_KEY")
    region = region or os.getenv("AZURE_TRANSLATOR_REGION", "centralindia")
    endpoint = endpoint or os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com/")
    
    # Validate configuration
    if not api_key:
        return {
            "success": False,
            "error": "Missing AZURE_TRANSLATOR_KEY. Please set it in .env file.",
            "original_text": text,
            "translated_text": None,
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    # Build the API URL
    path = "/translate"
    api_version = "3.0"
    url = f"{endpoint.rstrip('/')}{path}"
    
    # Query parameters - omit 'from' to enable auto-detection
    params = {
        "api-version": api_version,
        "to": target_lang
    }
    
    # Only include source language if explicitly specified (not auto-detect)
    if source_lang and source_lang.lower() not in ['auto', 'auto-detect', '']:
        params["from"] = source_lang
    
    # Request headers
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }
    
    # Request body
    body = [{"text": text}]
    
    try:
        # Make the API request
        response = requests.post(url, params=params, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if result and len(result) > 0:
            translation = result[0]
            translated_text = translation["translations"][0]["text"]
            
            return {
                "success": True,
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "detected_language": translation.get("detectedLanguage", {}).get("language"),
                "error": None
            }
        else:
            return {
                "success": False,
                "error": "Empty response from Translator API",
                "original_text": text,
                "translated_text": None,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e.response.status_code}"
        try:
            error_detail = e.response.json()
            if "error" in error_detail:
                error_msg = f"{error_msg} - {error_detail['error'].get('message', '')}"
        except:
            pass
        
        return {
            "success": False,
            "error": error_msg,
            "original_text": text,
            "translated_text": None,
            "source_language": source_lang,
            "target_language": target_lang
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "original_text": text,
            "translated_text": None,
            "source_language": source_lang,
            "target_language": target_lang
        }


def translate_long_text(
    text: str,
    source_lang: str,
    target_lang: str,
    chunk_size: int = 5000
) -> dict:
    """
    Translate long text by splitting into chunks.
    Azure Translator has a ~10,000 character limit per request.
    
    Args:
        text: The long text to translate
        source_lang: Source language code (use 'auto' for auto-detection)
        target_lang: Target language code
        chunk_size: Maximum characters per chunk (default 5000 for safety)
    
    Returns:
        dict with translated text or error
    """
    if not text or not text.strip():
        return {
            "success": False,
            "error": "No text provided",
            "translated_text": None
        }
    
    # If text is short enough, translate directly
    if len(text) <= chunk_size:
        return translate_text(text, source_lang, target_lang)
    
    # Split text into chunks at paragraph boundaries
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk
        if len(current_chunk) + len(para) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
        else:
            current_chunk += para + "\n"
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"DEBUG: Translating {len(chunks)} chunks for long document...", file=__import__('sys').stderr)
    
    # Translate each chunk
    translated_chunks = []
    detected_lang = None
    
    for i, chunk in enumerate(chunks):
        print(f"DEBUG: Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", file=__import__('sys').stderr)
        result = translate_text(chunk, source_lang, target_lang)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed at chunk {i+1}: {result['error']}",
                "translated_text": None,
                "chunks_completed": i
            }
        
        translated_chunks.append(result["translated_text"])
        
        # Capture detected language from first chunk
        if i == 0 and result.get("detected_language"):
            detected_lang = result["detected_language"]
    
    # Combine all translated chunks
    full_translation = "\n".join(translated_chunks)
    
    return {
        "success": True,
        "original_text": text[:500] + "..." if len(text) > 500 else text,
        "translated_text": full_translation,
        "source_language": source_lang,
        "target_language": target_lang,
        "detected_language": detected_lang,
        "chunks_translated": len(chunks),
        "error": None
    }


def get_supported_languages() -> dict:
    """
    Get list of supported languages from Azure Translator.
    
    Returns:
        dict containing supported languages for translation
    """
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com/")
    url = f"{endpoint.rstrip('/')}/languages?api-version=3.0&scope=translation"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return {"success": True, "languages": response.json().get("translation", {})}
    except Exception as e:
        return {"success": False, "error": str(e), "languages": {}}


# Quick test when run directly
if __name__ == "__main__":
    print("Testing Azure Translator...")
    print("-" * 50)
    
    # Test translation
    result = translate_text("Hello, how are you today?", "en", "hi")
    
    if result["success"]:
        print(f"✓ Original ({result['source_language']}): {result['original_text']}")
        print(f"✓ Translated ({result['target_language']}): {result['translated_text']}")
    else:
        print(f"✗ Translation failed: {result['error']}")
