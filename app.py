import os
import tempfile
import streamlit as st
import pandas as pd
from typing import List, Literal
from pydantic import BaseModel, Field

# Loading env
from dotenv import load_dotenv 
load_dotenv()

# LangChain & Document Loaders
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. PYDANTIC SCHEMAS (STRUCTURED DATA)
# ==========================================

class ParsedJD(BaseModel):
    required_skills: List[str] = Field(description="List of mandatory and nice-to-have technical skills")
    minimum_experience_years: float = Field(description="Minimum years of experience required")
    required_education: str = Field(description="Minimum education level required")

class ResumeData(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    total_experience_years: float = Field(description="Total years of professional experience")
    key_skills: List[str] = Field(description="List of technical and soft skills")
    education_level: str = Field(description="Highest degree achieved")

class DimensionScore(BaseModel):
    score: int = Field(ge=0, le=10, description="Score from 0 to 10")
    justification: str = Field(description="A crisp, one-line justification")

class CandidateEvaluation(BaseModel):
    skills_match: DimensionScore
    experience_relevance: DimensionScore
    education_certs: DimensionScore
    project_portfolio: DimensionScore
    communication_quality: DimensionScore
    hire_recommendation: Literal["Hire", "No-Hire", "Hold"]

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Updated from gemini-1.5-flash-latest
    temperature=0.7, 
    # max_retries=2, etc.
)
# ==========================================
# 2. CORE BACKEND LOGIC
# ==========================================

@st.cache_resource
def load_vector_store():
    """Loads the pre-existing Chroma DB created by create_vector_db.py"""
    persist_dir = "./candidate_vector_db"
    
    # Embedding token parameter
    hf_token = os.getenv("HF_TOKEN")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu', 'token': hf_token}
    )
    
    # Initialize Chroma connected to the existing persistent directory
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return vector_store

def parse_job_description(jd_text: str) -> ParsedJD:
    parser_llm = llm.with_structured_output(ParsedJD)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract core requirements from the following Job Description."),
        ("human", "{jd_text}")
    ])
    return (prompt | parser_llm).invoke({"jd_text": jd_text})

def extract_resume_data(pdf_path: str) -> ResumeData:
    pages = PyMuPDFLoader(pdf_path).load()
    resume_text = "\n".join([page.page_content for page in pages])
    extractor_llm = llm.with_structured_output(ResumeData)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract candidate's core information from the resume text."),
        ("human", "{resume_text}")
    ])
    return (prompt | extractor_llm).invoke({"resume_text": resume_text})

def evaluate_candidate(resume_data: ResumeData, parsed_jd: ParsedJD, vector_store: Chroma) -> dict:
    # Similarity search happens HERE, strictly after JD parsing and Resume extraction
    search_query = f"Skills: {', '.join(parsed_jd.required_skills)} matching candidate skills: {', '.join(resume_data.key_skills)}"
    
    try:
        docs = vector_store.similarity_search(search_query, k=2)
        context_text = "\n".join([doc.page_content for doc in docs])
    except Exception:
        context_text = "No historical context found or DB missing."

    rubric = """
    Scoring Rubric (Scale of 0 to 10):
    1. Skills Match (30%): 0 (<30%), 5 (50-70%), 10 (>85%)
    2. Experience (25%): 0 (Unrelated), 5 (Adjacent), 10 (Exact domain & seniority)
    3. Education (15%): 0 (Does not meet), 5 (Meets min), 10 (Exceeds)
    4. Projects (20%): 0 (No evidence), 5 (Generic), 10 (Strong relevant)
    5. Communication (10%): 0 (Poor grammar), 5 (Adequate), 10 (Crisp, impactful)
    """

    evaluator_llm = llm.with_structured_output(CandidateEvaluation)
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate the candidate against the PARSED Job Description using the strictly defined Rubric."),
        ("human", "Parsed JD:\n{parsed_jd}\nCandidate Data:\n{candidate_json}\nContext:\n{context}\n{rubric}")
    ])
    
    e: CandidateEvaluation = (eval_prompt | evaluator_llm).invoke({
        "parsed_jd": parsed_jd.model_dump_json(),
        "candidate_json": resume_data.model_dump_json(),
        "context": context_text,
        "rubric": rubric
    })
    
    weighted_total = (e.skills_match.score*0.30) + (e.experience_relevance.score*0.25) + \
                     (e.education_certs.score*0.15) + (e.project_portfolio.score*0.20) + (e.communication_quality.score*0.10)
    
    return {
        "Name": resume_data.candidate_name,
        "Weighted Score": round(weighted_total, 2),
        "Status": e.hire_recommendation,
        "Skills (30%)": e.skills_match.score,
        "Exp (25%)": e.experience_relevance.score,
        "Edu (15%)": e.education_certs.score,
        "Proj (20%)": e.project_portfolio.score,
        "Comm (10%)": e.communication_quality.score,
        "AI Justification Summary": f"Skills: {e.skills_match.justification} | Exp: {e.experience_relevance.justification}",
        "HR Override Reason": "None"
    }

# ==========================================
# 3. STREAMLIT APP & HUMAN-IN-THE-LOOP UI
# ==========================================

st.set_page_config(page_title="AI HR Evaluator", layout="wide")
st.title("🧑‍💼 Batch AI Candidate Evaluator (with HITL Hook)")

# Initialize Session State
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# Load DB (Will fail gracefully during search if 'create_vector_db.py' wasn't run)
vector_store = load_vector_store()

# --- STEP 1: UPLOAD & PROCESSING ---
with st.expander("📝 Step 1: Upload Job Description & Resumes", expanded=True):
    job_description_text = st.text_area("Job Description", height=150)
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Analysis", type="primary"):
        if job_description_text and uploaded_files:
            st.session_state.evaluations = [] # Reset prior runs
            with st.spinner("Parsing Job Description..."):
                parsed_jd = parse_job_description(job_description_text)
            
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    resume_data = extract_resume_data(tmp_path)
                    # DB Similarity query run inside this function
                    result_dict = evaluate_candidate(resume_data, parsed_jd, vector_store)
                    st.session_state.evaluations.append(result_dict)
                except Exception as ex:
                    st.error(f"Error processing {uploaded_file.name}: {ex}")
                finally:
                    os.remove(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("Batch Processing Complete!")
        else:
            st.error("Please provide JD and at least one PDF.")

# --- STEP 2: RANKED REPORT & EXPORT ---
if st.session_state.evaluations:
    st.markdown("---")
    st.subheader("📊 Step 2: Ranked Shortlist Report")
    
    df_results = pd.DataFrame(st.session_state.evaluations)
    df_results = df_results.sort_values(by="Weighted Score", ascending=False).reset_index(drop=True)
    
    st.dataframe(df_results, use_container_width=True)
    
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Report (CSV)", data=csv_data, file_name='shortlist_report.csv', mime='text/csv')

# --- STEP 3: HUMAN IN THE LOOP (HITL) HOOK ---
if st.session_state.evaluations:
    st.markdown("---")
    st.subheader("⚖️ Step 3: Human-in-the-Loop Override")
    st.markdown("Disagree with the AI? Audit and override a candidate's final status below.")
    
    hitl_col1, hitl_col2, hitl_col3 = st.columns([2, 1, 3])
    
    candidate_names = [eval["Name"] for eval in st.session_state.evaluations]
    
    with hitl_col1:
        selected_cand = st.selectbox("Select Candidate to Audit", candidate_names)
    with hitl_col2:
        new_status = st.selectbox("Update Status To", ["Hire", "Hold", "No-Hire"])
    with hitl_col3:
        override_reason = st.text_input("Reason for Override (Required)", placeholder="e.g., Internal referral, strong portfolio interview...")
        
    if st.button("Apply Manual Override"):
        if override_reason.strip() == "":
            st.warning("Please provide a reason for overriding the AI.")
        else:
            for record in st.session_state.evaluations:
                if record["Name"] == selected_cand:
                    record["Status"] = f"OVERRIDE: {new_status}"
                    record["HR Override Reason"] = override_reason
                    st.success(f"Successfully updated {selected_cand}'s status to {new_status}.")
                    st.rerun()