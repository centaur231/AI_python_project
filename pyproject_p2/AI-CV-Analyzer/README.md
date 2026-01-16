# Smart AI Resume Analyzer

A Python-based application built with Streamlit that helps you analyze and optimize your resume using AI-powered insights, and build professional resumes from scratch.

## Features

### 🔍 Resume Analyzer
- **Standard Analyzer**: Fast, rule-based analysis that checks ATS compatibility, keyword matching, formatting, and essential sections
- **AI Analyzer**: Deep, contextual analysis powered by Google Gemini AI that provides detailed feedback and personalized recommendations
- Supports PDF and DOCX file formats
- Role-specific analysis based on job categories

### 📝 Resume Builder
- Create professional resumes from scratch
- Multiple templates: Modern, Professional, Minimal, Creative
- Customizable sections: Personal info, Experience, Education, Projects, Skills
- Export as Word document (.docx)

## How It Works

### Resume Analysis
1. Upload your resume (PDF or DOCX)
2. Select a job category and specific role
3. Choose between Standard or AI Analyzer
4. Get instant feedback on:
   - ATS compatibility score
   - Keyword matching
   - Missing skills
   - Formatting suggestions
   - Section completeness
   - (AI Analyzer) Detailed contextual feedback and recommendations

### Resume Building
1. Fill out your personal information
2. Add work experience entries
3. Add education details
4. Add projects and skills
5. Select a template
6. Generate and download your resume

## Installation

### Prerequisites
- Python 3.7 or higher
- Windows operating system
- pip package manager

### Setup Steps

1. **Clone or download the repository:**
   ```bash
   cd AI_python_project\pyproject_p2\AI-CV-Analyzer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Gemini API (for AI Analyzer):**
   
   Create a `.env` file in the `utils/` directory:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key
   ```
   
   Get your API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)
   
   > **Note**: The Standard Analyzer works without an API key. Only the AI Analyzer requires the Gemini API key.

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   
   Or use the Windows startup script:
   ```bash
   startup.bat
   ```

## Project Structure

```
Smart-AI-Resume-Analyzer/
├── app.py                      # Main application file
├── config/                     # Configuration files
│   ├── courses.py              # Course recommendations
│   └── job_roles.py            # Job role definitions
├── utils/                      # Core functionality
│   ├── ai_resume_analyzer.py   # AI analysis (Gemini-powered)
│   ├── resume_analyzer.py      # Standard analysis (rule-based)
│   └── resume_builder.py       # Resume document generator
├── ui_components.py            # Reusable UI components
├── style/
│   └── style.css               # Custom styling
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Components

- **app.py**: Main application orchestrating all features
- **resume_analyzer.py**: Rule-based analysis using pattern matching
- **ai_resume_analyzer.py**: AI-powered analysis using Google Gemini
- **resume_builder.py**: Generates Word documents from user input
- **job_roles.py**: Defines job categories and required skills
- **courses.py**: Provides learning resource recommendations

## Standard Analyzer vs AI Analyzer

**Standard Analyzer:**
- Fast, local processing
- No API key required
- Rule-based keyword matching
- Formatting and section checks
- Best for quick ATS optimization

**AI Analyzer:**
- Deep contextual understanding
- Requires Google Gemini API key
- Natural language feedback
- Detailed recommendations
- Can analyze against specific job descriptions
- Best for comprehensive resume review

## Important Notes

- **No Database**: The application runs entirely in memory. No data is stored or persisted.
- **Local Operation**: Designed for local use on Windows. All processing happens during your session.
- **API Key**: Only needed for AI Analyzer functionality. Standard Analyzer works offline.

## Troubleshooting

- **PDF extraction fails**: Ensure the PDF contains actual text (not just images). For scanned PDFs, OCR support is available if Tesseract is installed.
- **API key errors**: Verify your `.env` file is in the `utils/` directory with the correct format.
- **Missing dependencies**: Run `pip install -r requirements.txt` again.

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:
- streamlit
- google-generativeai
- python-docx
- pdfplumber
- plotly
