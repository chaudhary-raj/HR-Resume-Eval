# 🧑‍💼 AI HR Evaluator — Batch Candidate Screening System

An intelligent, AI-powered HR screening platform that automates early-stage candidate evaluation using **Gemini 2.5 Flash**, **LangChain**, and **RAG (Retrieval-Augmented Generation)**. Upload job descriptions and batch resumes to receive structured, rubric-based candidate rankings with a Human-in-the-Loop (HITL) override panel for final HR decisions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📝 JD Validation** | AI validates whether input is a genuine job description (accepts full JDs, short descriptions, or even job titles) |
| **📄 Multi-Format Parsing** | Extracts text from **PDF**, **DOCX**, **JSON** (LinkedIn exports), and **TXT** resumes |
| **🧠 Structured Extraction** | Uses Pydantic models to extract candidate name, experience, skills, education, and project tech stacks |
| **📊 Rubric-Based Scoring** | Weighted evaluation: Skills (30%), Experience (25%), Projects (20%), Education (15%), Communication (10%) |
| **🔍 RAG Context** | Retrieves historical candidate context from a local **ChromaDB** vector store for smarter comparisons |
| **⚖️ HITL Override** | HR Admin panel to audit, override AI decisions, and add mandatory justification reasons |
| **📥 CSV Export** | Download the full ranked shortlist as a CSV report |
| **🎨 Modern UI** | Glassmorphism design with custom Streamlit CSS, responsive layout, and status-aware color theming |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Job Description │────▶│  Gemini 2.5 Flash │────▶│  ParsedJD       │
│  (Text/Title)   │     │  (Pydantic Output)│     │  (Structured)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Resume Files   │────▶│  Text Extraction │────▶│  ResumeData     │
│ (PDF/DOCX/JSON) │     │  (PyMuPDF/docx)  │     │  (Structured)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                    ┌──────────────────────┐
                    │   RAG Vector Store   │
                    │   (ChromaDB + BGE)   │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  CandidateEvaluation│
                    │  (Weighted Scoring) │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   HITL Override UI   │
                    │  (Streamlit + CSS)   │
                    └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google AI API Key (for Gemini)
- HuggingFace Token (for BAAI/bge-small-en-v1.5 — optional but recommended)

### 1. Clone & Install

```bash
git clone https://github.com/chaudhary-raj/HR-Resume-Eval
cd HR-Resume-Eval

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for gated embedding models
```

### 3. Build the Vector Store

Before running the app, create the ChromaDB from your historical resume CSV:

```bash
python create_vector_store.py
```

**Expected output:**
```
Initializing embedding model...
Loading data from AI_Resume_Screening.csv...
Creating and persisting Chroma vector store...
✅ Vector store successfully created and persisted at './candidate_vector_db'
```

> **Note:** Ensure `AI_Resume_Screening.csv` exists in the root folder. This file should contain historical candidate data for RAG context retrieval.

### 4. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
ai-hr-evaluator/
│
├── app.py                      # Main Streamlit application
├── create_vector_store.py      # Script to build ChromaDB from CSV
├── style.css                   # Custom Streamlit UI theme (glassmorphism)
├── AI_Resume_Screening.csv     # Historical data for vector store
├── candidate_vector_db/        # Auto-generated Chroma persistence folder
│
├── .env                        # API keys (not tracked in git)
├── .gitignore
├── requirements.txt
└── README.md                   # This file
```

---

## 🧩 Pydantic Data Models

The system uses strict structured output schemas for reliable AI extraction:

### `ParsedJD` — Job Description Parser
```python
is_valid: bool              # Accepts titles, short descriptions, full JDs
invalid_reason: str         # Explanation if rejected
required_skills: List[str]  # Technical skills extracted
minimum_experience_years: float
required_education: str
job_title: str
```

### `ResumeData` — Candidate Profile
```python
candidate_name: str
total_experience_years: float
key_skills: List[str]
education_level: str
projects: List[ProjectDetail]   # Name, summary, and tech stack per project
```

### `CandidateEvaluation` — Scoring Output
```python
skills_match: DimensionScore        # 0-10 with justification
experience_relevance: DimensionScore
education_certs: DimensionScore
project_portfolio: DimensionScore
communication_quality: DimensionScore
hire_recommendation: Literal["Hire", "No-Hire", "Hold"]
```

---

## 📖 Usage Guide

### Step 1: Input Job Description
Paste the target job description in the text area. The AI will validate it before processing. Even short job titles like *"Software Developer"* are now accepted (with inferred or empty fields).

### Step 2: Upload Resumes
Upload one or more files:
- **PDF** — Standard resumes
- **DOCX** — Word documents
- **JSON** — LinkedIn/profile exports
- **TXT** — Plain text resumes

Click **🚀 Run Batch Analysis**.

### Step 3: Review Ranked Report
View the sorted dataframe showing:
- Weighted Score (out of 10)
- AI Recommendation (Hire / Hold / No-Hire)
- Individual dimension scores
- Extracted projects & tech stacks
- AI justification summary

Download the full report as CSV.

### Step 4: Human-in-the-Loop Override
In the **⚖️ HITL Panel** (amber-themed admin section):
1. Select a candidate from the dropdown
2. Choose new status: **Hire**, **Hold**, or **No-Hire**
3. Provide a mandatory override reason
4. Click **Apply Manual Override**

The candidate's status updates to `OVERRIDE: <Status>` with the HR reason logged.

---

## 🎨 UI Customization

The app includes a custom `style.css` with:
- **Glassmorphism cards** with backdrop blur
- **Gradient buttons** (indigo primary, emerald download, amber HITL)
- **Accessible focus rings** on inputs
- **Status-aware alerts** (left-border accent style)
- **Smooth animations** and hover effects

To modify the theme, edit `style.css` and reload the app.

---

## ⚙️ Scoring Rubric

| Dimension | Weight | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 |
|-----------|--------|-----|-----|-----|-----|------|
| **Skills Match** (30%) | < 20% overlap | 20–40% overlap | 40–60% overlap | 60–80% overlap | > 80% overlap |
| **Experience** (25%) | Unrelated | Slightly adjacent (internship, tangential) | Relevant but junior | Good match & solid seniority | Exact domain + required seniority |
| **Education** (15%) | Far below minimum | Partially meets (wrong field or level) | Meets minimum requirement | Meets + relevant certifications | Exceeds significantly (advanced degree, top-tier) |
| **Projects** (20%) | No evidence | Weak / academic-only, unrelated stack | Generic projects, some tech overlap | Relevant with decent stack alignment | Strong production-grade, exact stack match |
| **Communication** (10%) | Poor grammar, unclear | Below average, awkward phrasing | Adequate, minor issues | Good structure, clarity, professional tone | Crisp, impactful, polished writing |

**Final Score** = Weighted sum of all dimensions (1–10 scale)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Invalid Job Description` error | The JD validator rejects completely unrelated text. Ensure input relates to a job role, skills, or hiring. Short titles are now accepted. |
| `candidate_vector_db` not found | Run `python create_vector_store.py` first. Ensure `AI_Resume_Screening.csv` exists. |
| PyMuPDF import error | Install with `pip install PyMuPDF` (provides `fitz`) |
| Gemini API errors | Verify `GOOGLE_API_KEY` in `.env`. Check quota/limitations on your Google AI Studio account. |
| HuggingFace model download fails | Add `HF_TOKEN` to `.env` if using a gated model, or check internet connection. |
| CSS not loading | Ensure `style.css` is in the same directory as `app.py` and `load_local_css()` path is correct. |

---

## 🛡️ Privacy & Security

- **Local Processing:** Resume text is processed in-memory and never stored persistently unless exported.
- **Vector Store:** ChromaDB runs locally with no external vector database required.
- **API Keys:** Kept in `.env` — never commit this file.
- **HITL Audit Trail:** All manual overrides are logged with timestamps and reasons in the session state and CSV export.

---

## 📦 Requirements

```text
streamlit
pandas
pydantic
python-docx
PyMuPDF
langchain
langchain-community
langchain-google-genai
chromadb
sentence-transformers
dotenv
```

Or simply install via:
```bash
pip install -r requirements.txt
```

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

## 🙋‍♂️ Support

For issues or feature requests, please open an issue on GitHub or contact the HR Tech team.

**Built with ❤️ using Gemini 2.5 Flash, LangChain, and Streamlit.**
