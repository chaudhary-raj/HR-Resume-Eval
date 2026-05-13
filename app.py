import os
import json
from io import BytesIO
import streamlit as st
import pandas as pd
from typing import List, Literal
from pydantic import BaseModel, Field

# Ensure you have installed: pip install python-docx PyMuPDF
import docx
import fitz  # PyMuPDF

# Loading env
from dotenv import load_dotenv 
load_dotenv()

# LangChain & Document Loaders
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 0. PAGE CONFIG & CUSTOM CSS INJECTION
# ==========================================
st.set_page_config(page_title="AI HR Evaluator", page_icon="🧑‍💼", layout="wide")

def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS file
load_local_css("style.css")

# ==========================================
# 1. PYDANTIC SCHEMAS (STRUCTURED DATA)
# ==========================================

class ParsedJD(BaseModel):
    is_valid: bool = Field(description="True if the text relates to a job role, hiring, or skills. "
                    "Accept job titles, short descriptions, or full JDs. "
                    "Only reject completely unrelated text.")
    invalid_reason: str = Field(default="", description="Reason if invalid.")
    required_skills: List[str] = Field(default_factory=list,description="Technical skills mentioned. For job titles only, infer common skills or leave empty.")
    minimum_experience_years: float = Field(default=0)
    required_education: str = Field(default="Not Specified")
    job_title: str = Field(default="", description="Job title if identifiable.")  # Optional: add this
class ProjectDetail(BaseModel):
    project_name: str = Field(description="The name or title of the project")
    summary: str = Field(description="Brief, one-sentence summary of what the project does")
    tech_stack: List[str] = Field(description="List of programming languages, frameworks, libraries, and tools used in this specific project")

class ResumeData(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    total_experience_years: float = Field(description="Total years of professional experience")
    key_skills: List[str] = Field(description="List of technical and soft skills")
    education_level: str = Field(description="Highest degree achieved")
    projects: List[ProjectDetail] = Field(description="List of key projects completed by the candidate including their tech stacks")

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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7, 
)

# ==========================================
# 2. CORE BACKEND LOGIC
# ==========================================

@st.cache_resource
def load_vector_store():
    """Loads the pre-existing Chroma DB created by create_vector_db.py"""
    persist_dir = "./candidate_vector_db"
    hf_token = os.getenv("HF_TOKEN")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu', 'token': hf_token}
    )
    
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return vector_store

def parse_job_description(jd_text: str) -> ParsedJD:
    parser_llm = llm.with_structured_output(ParsedJD)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following text. Determine if it is a valid Job Description. If it is, extract the core requirements. If it is not relevant, flag it as invalid and provide a reason."),
        ("human", "{jd_text}")
    ])
    return (prompt | parser_llm).invoke({"jd_text": jd_text})

def read_uploaded_file(uploaded_file) -> str:
    """Extracts text from PDF, DOCX, JSON (LinkedIn export), or TXT in-memory."""
    ext = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif ext == "docx":
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == "json":
            data = json.load(uploaded_file)
            text = json.dumps(data, indent=2)
        else:
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        raise ValueError(f"Could not parse file '{uploaded_file.name}': {e}")
    
    return text

def extract_resume_data(resume_text: str) -> ResumeData:
    extractor_llm = llm.with_structured_output(ResumeData)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract candidate's core information, summarize their key projects, and explicitly list the tech stack used for each project from the provided resume/profile text."),
        ("human", "{resume_text}")
    ])
    return (prompt | extractor_llm).invoke({"resume_text": resume_text})

def evaluate_candidate(resume_data: ResumeData, parsed_jd: ParsedJD, vector_store: Chroma) -> dict:
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
    4. Projects (20%): 0 (No evidence), 5 (Generic), 10 (Strong relevant - check project tech stacks against JD)
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
    
    if resume_data.projects:
        formatted_projects = " | ".join(
            [f"{p.project_name}: {p.summary} [Stack: {', '.join(p.tech_stack)}]" for p in resume_data.projects]
        )
    else:
        formatted_projects = "No projects listed"
    
    return {
        "Name": resume_data.candidate_name,
        "Weighted Score": round(weighted_total, 2),
        "Status": e.hire_recommendation,
        "Skills (30%)": e.skills_match.score,
        "Exp (25%)": e.experience_relevance.score,
        "Edu (15%)": e.education_certs.score,
        "Proj (20%)": e.project_portfolio.score,
        "Comm (10%)": e.communication_quality.score,
        "Extracted Projects & Stack": formatted_projects,
        "AI Justification Summary": f"Skills: {e.skills_match.justification} | Exp: {e.experience_relevance.justification}",
        "HR Override Reason": "None"
    }

# ==========================================
# 3. STREAMLIT APP & HUMAN-IN-THE-LOOP UI
# ==========================================

# --- SIDEBAR FOR INSTRUCTIONS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135690.png", width=100)
    st.title("Admin Guide")
    st.info("""
    **How to use:**
    1. Paste the complete Job Description.
    2. Upload batch resumes (PDF, DOCX) or scraped LinkedIn profiles (JSON, TXT).
    3. The AI will parse the JD, validate it, extract candidate data (including project stacks), and rank them based on a strict rubric.
    4. Download the CSV or apply manual HR overrides if necessary.
    """)
    st.markdown("---")
    st.caption("Powered by Gemini-2.5-Flash & LangChain")

# --- MAIN DASHBOARD HEADER ---
st.title("🧑‍💼 Batch AI Candidate Evaluator")
st.markdown("Automate early-stage screening with structured AI extraction and rubric-based scoring.")

# Initialize Session State
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

vector_store = load_vector_store()

# --- STEP 1: UPLOAD & PROCESSING ---
with st.container():
    st.subheader("📝 Input Job & Candidate Data")
    st.markdown("Provide the requirements and the candidate documents to begin the batch analysis.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        job_description_text = st.text_area(
            "Target Job Description", 
            height=200, 
            placeholder="Paste the full job description here. The AI will validate it before running.",
            help="The AI uses this to build a dynamic scoring baseline."
        )
        
    with col2:
        uploaded_files = st.file_uploader(
            "Upload Resumes/LinkedIn Profiles", 
            type=["pdf", "docx", "json", "txt"], 
            accept_multiple_files=True,
            help="Supports standard resumes (PDF/DOCX) or structured data exports (JSON/TXT)."
        )
    
    if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
        if job_description_text and uploaded_files:
            st.session_state.evaluations = [] 
            
            with st.spinner("Analyzing and validating Job Description..."):
                parsed_jd = parse_job_description(job_description_text)
            
            # --- JD VALIDATION CHECK ---
            if not parsed_jd.is_valid:
                st.error(f"🚨 **Invalid Job Description Detected:** {parsed_jd.invalid_reason}")
                st.warning("Please update the text area with a genuine job description and try again.")
            else:
                st.success("✅ Job Description is valid! Extracting and scoring candidates...")
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        resume_text = read_uploaded_file(uploaded_file)
                        resume_data = extract_resume_data(resume_text)
                        result_dict = evaluate_candidate(resume_data, parsed_jd, vector_store)
                        st.session_state.evaluations.append(result_dict)
                    except Exception as ex:
                        st.error(f"Error processing {uploaded_file.name}: {ex}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success("🎉 Batch Processing Complete! See the results below.")
        else:
            st.warning("⚠️ Please provide a Job Description and at least one file to process.")

# --- STEP 2: RANKED REPORT & EXPORT ---
if st.session_state.evaluations:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.subheader("📊 Ranked Shortlist Report")
        
        # Prepare Data
        df_results = pd.DataFrame(st.session_state.evaluations)
        df_results = df_results.sort_values(by="Weighted Score", ascending=False).reset_index(drop=True)
        
        # Display Quick Metrics
        met_col1, met_col2, met_col3 = st.columns(3)
        met_col1.metric("Total Candidates", len(df_results))
        met_col2.metric("Highest Score", f"{df_results['Weighted Score'].max()}/10")
        met_col3.metric("Recommended Hires", len(df_results[df_results['Status'] == 'Hire']))
        
        # Display Table
        st.dataframe(df_results, use_container_width=True, height=400)
        
        # Export
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)", 
            data=csv_data, 
            file_name='shortlist_report.csv', 
            mime='text/csv',
            type="primary"
        )

# --- STEP 3: HUMAN IN THE LOOP (HITL) HOOK ---
if st.session_state.evaluations:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 👇 Open wrapper
    st.markdown('<div class="hitl-panel">', unsafe_allow_html=True)
    
    with st.container():
        st.subheader("⚖️ Human-in-the-Loop (Override)")
        st.markdown("As the HR Admin, you have the final say. Audit a candidate and override the AI's status if needed.")
        
        hitl_col1, hitl_col2, hitl_col3 = st.columns([2, 1, 3])
        candidate_names = [eval["Name"] for eval in st.session_state.evaluations]
        
        with hitl_col1:
            selected_cand = st.selectbox("Select Candidate to Audit", candidate_names)
        with hitl_col2:
            new_status = st.selectbox("Update Status To", ["Hire", "Hold", "No-Hire"])
        with hitl_col3:
            override_reason = st.text_input(
                "Reason for Override (Required)", 
                placeholder="e.g., Internal referral, great portfolio..."
            )
            
        if st.button("Apply Manual Override", icon="✍️"):
            if override_reason.strip() == "":
                st.warning("⚠️ Please provide a clear justification for overriding the AI decision.")
            else:
                for record in st.session_state.evaluations:
                    if record["Name"] == selected_cand:
                        record["Status"] = f"OVERRIDE: {new_status}"
                        record["HR Override Reason"] = override_reason
                        st.success(f"✅ Successfully updated {selected_cand}'s status to **{new_status}**.")
                        st.rerun()
    
    # 👇 Close wrapper
    st.markdown('</div>', unsafe_allow_html=True)