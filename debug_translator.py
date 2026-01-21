from translator import translate_text
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Translation...")
try:
    res = translate_text("Hello world", "en", "es")
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")
