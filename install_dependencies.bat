@echo off
echo Installing Python dependencies for the MOSDAC Scraper & QA App...
echo.

REM Install main Python libraries
pip install ^
requests ^
beautifulsoup4 ^
streamlit ^
langchain ^
langchain-community ^
langchain-groq ^
sentence-transformers ^
huggingface-hub ^
docx2txt ^
PyMuPDF ^
faiss-cpu ^
spacy ^
scikit-learn ^
matplotlib ^
networkx ^
pandas ^
python-dotenv

echo.
echo Downloading spaCy language model (en_core_web_sm)...
python -m spacy download en_core_web_sm

echo.
echo ✅ All dependencies installed successfully.
pause
https://github.com/smtdhakad/FINAL.git