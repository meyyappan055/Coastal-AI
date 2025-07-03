import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = BASE_DIR / "documents" / "reports" / "report1.pdf"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
COHERE_KEY = os.getenv("COHERE_KEY")

if not (GOOGLE_API_KEY and LANGSMITH_API_KEY and COHERE_KEY):
    raise EnvironmentError("Missing environment variables.")