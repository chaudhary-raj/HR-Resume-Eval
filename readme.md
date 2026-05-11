# 🧑‍💼 AI-Powered HR Candidate Evaluator (RAG System)

An advanced, AI-driven Retrieval-Augmented Generation (RAG) application designed to automate the resume screening process. This tool extracts candidate data from PDFs using MuPDF, retrieves historical hiring context, and evaluates candidates against a strict 5-dimension rubric—while keeping the HR manager in control with a Human-in-the-Loop (HITL) override system.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your OpenAI Key in `.env`
3. Build the Database: `python create_vector_store.py`
4. Launch the App: `streamlit run app.py`

---

## 📜 Mandatory Disclosures & Architecture

### 1. LLM Model Chosen
The application utilizes OpenAI's **GPT-4o-mini**. 
**Justification:** This model provides an optimal balance of processing speed and accuracy for batch resume evaluation. GPT-4o-mini excels at "Structured Outputs" (Function Calling), strictly adhering to our Pydantic JSON schemas and mandatory 0-10 integer scoring rubric without text hallucination.

### 2. Agent Framework
The application orchestrates logic via **LangChain**.
**Justification:** LangChain powers the RAG pipeline by providing robust, native wrappers for `PyMuPDFLoader` (PDF extraction), `Chroma` (local vector database), and `ChatPromptTemplate` (clean separation of system instructions and user variables).

### 3. Security & Risk Mitigations
* **Prompt Injection Mitigation:** We enforce strict schema isolation. The "System Prompt" holds the immutable scoring rubric, while candidate resumes are passed strictly as unstructured human data. Using `.with_structured_output()`, the LLM is programmatically barred from outputting executable code or altering the scoring mechanics, neutralizing malicious injection attempts embedded in resumes.
* **Data Privacy:** All PDF handling is ephemeral. Uploads are written to hidden temporary file paths via Python's `tempfile` library and permanently deleted via a `finally: os.remove()` execution block the millisecond text extraction completes. Furthermore, the Chroma Vector DB operates 100% locally on the host machine.
* **Credential Handling:** Zero API keys are hardcoded. Credentials reside exclusively in a local `.env` file accessed via `python-dotenv`. A `.gitignore` policy explicitly prohibits environment variables from being pushed to source control.