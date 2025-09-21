import streamlit as st
import os
import sys
import PyPDF2
import docx
import hashlib
import json
from config import OPENAI_API_KEY
import pandas as pd

# Add the project root to the Python path to access the 'models' module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.openai_utils import format_job_description, generate_job_title
from models.resume_scoring import score_resume
from utils.migration import migrate_resumes_to_scores

# Set page config
st.set_page_config(page_title="Job Application Portal", page_icon="🤖", layout="wide")

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JD_PATH = os.path.join(PROJECT_ROOT, "sample_jd")
RESUMES_PATH = os.path.join(PROJECT_ROOT, "resumes")
UI_STREAMLIT_PATH = os.path.join(PROJECT_ROOT, "ui_streamlit")
HASHES_PATH = os.path.join(PROJECT_ROOT, "jd_hashes.txt")
SCORES_PATH = os.path.join(PROJECT_ROOT, "scores")

# Ensure directories and hash file exist
os.makedirs(SAMPLE_JD_PATH, exist_ok=True)
os.makedirs(RESUMES_PATH, exist_ok=True)
os.makedirs(SCORES_PATH, exist_ok=True)
if not os.path.exists(HASHES_PATH):
    with open(HASHES_PATH, "w") as f:
        pass  # Create empty file

# --- Migration: move any existing scoring_results.json from resumes/ to scores/ ---
migration_summary = migrate_resumes_to_scores(RESUMES_PATH, SCORES_PATH)

# --- Load CSS ---
def local_css(file_name):
    path = os.path.join(UI_STREAMLIT_PATH, "css", file_name)
    with open(path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
local_css("job_card.css")

# --- Helper functions ---
def extract_text(file_path):
    # ... (existing text extraction functions remain the same)
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading TXT: {e}"

def get_existing_hashes():
    with open(HASHES_PATH, "r") as f:
        return set(line.strip() for line in f)

def add_new_hash(file_hash):
    with open(HASHES_PATH, "a") as f:
        f.write(file_hash + "\n")

# --- HR Sidebar ---
st.sidebar.title("HR Department 🏢")
hr_password = st.sidebar.text_input("Enter HR Password", type="password", key="hr_password")

def process_jd_upload():
    """Callback to handle and validate the job description upload."""
    if st.session_state.jd_uploader_key is not None:
        uploaded_jd = st.session_state.jd_uploader_key
        
        # Calculate hash of the uploaded file
        file_bytes = uploaded_jd.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Check for duplicates
        if file_hash in get_existing_hashes():
            st.session_state.upload_error = "This job description has already been uploaded."
            st.session_state.last_uploaded_jd = None
            return

        # If not a duplicate, proceed with saving
        existing_files = [f for f in os.listdir(SAMPLE_JD_PATH) if f.startswith("job_") and f.endswith(".pdf")]
        next_job_number = len(existing_files) + 1
        new_filename = f"job_{next_job_number}.pdf"
        save_path = os.path.join(SAMPLE_JD_PATH, new_filename)

        with open(save_path, "wb") as f:
            f.write(file_bytes)
        
        # Add the new hash to our records
        add_new_hash(file_hash)
        
        st.session_state.last_uploaded_jd = new_filename
        st.session_state.upload_error = None

if hr_password == "hr_pass":
    st.sidebar.success("HR Authenticated")
    st.sidebar.header("Upload Job Description 📝")
    
    st.sidebar.file_uploader(
        "Upload a Job Description (PDF only)", 
        type=["pdf"],
        key="jd_uploader_key",
        on_change=process_jd_upload
    )

    # Display success or error messages
    if "last_uploaded_jd" in st.session_state and st.session_state.last_uploaded_jd:
        st.sidebar.success(f"Uploaded '{st.session_state.last_uploaded_jd}'!")
        st.session_state.last_uploaded_jd = None
    
    if "upload_error" in st.session_state and st.session_state.upload_error:
        st.sidebar.error(st.session_state.upload_error)
        st.session_state.upload_error = None

    # Admin message about scores storage and migration status
    try:
        moved = migration_summary.get("moved", [])
        merged = migration_summary.get("merged", [])
        errors = migration_summary.get("errors", [])
        st.sidebar.markdown("---")
        st.sidebar.subheader("Scores storage")
        st.sidebar.info(f"Scoring files are stored under: {SCORES_PATH}")
        if moved or merged or errors:
            st.sidebar.write(f"Migration run on startup:")
            if moved:
                st.sidebar.success(f"Moved scoring JSONs for: {', '.join(moved)}")
            if merged:
                st.sidebar.success(f"Merged scoring JSONs for: {', '.join(merged)}")
            if errors:
                st.sidebar.error(f"Migration errors for: {', '.join(e[0] for e in errors)}")
        else:
            st.sidebar.write("No legacy scoring files were found in resumes/ to migrate.")
    except Exception:
        # don't break HR UI if sidebar message fails
        pass

# --- Job Data Loading ---
@st.cache_data(ttl=3600)
def load_jobs():
    job_files = [f for f in os.listdir(SAMPLE_JD_PATH) if f.endswith(('.pdf', '.docx', '.txt'))]
    jobs = []
    for job_file in sorted(job_files):
        file_path = os.path.join(SAMPLE_JD_PATH, job_file)
        raw_text = extract_text(file_path)
        if raw_text:
            # Generate a smart title using OpenAI
            job_title = generate_job_title(raw_text)
            jobs.append({"file": job_file, "text": raw_text, "title": job_title})
    return jobs

# --- UI Rendering ---
def display_job_details(job):
    # Use the AI-generated title for display
    job_title_display = job["title"]
    job_file_id = os.path.splitext(job["file"])[0] # e.g., 'job_1'
    
    st.header(f"Job Details: {job_title_display}")

    if st.button("← Back to All Jobs"):
        st.session_state.selected_job = None
        # Clear any leftover success messages
        if 'resume_upload_success' in st.session_state:
            del st.session_state.resume_upload_success
        st.rerun()

    with st.spinner("Formatting job description..."):
        formatted_description = format_job_description(job["text"])
        st.markdown(formatted_description, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Apply for this role")

    # Callback to handle resume saving and scoring
    def process_resume_upload():
        uploader_key = f"resume_{job_file_id}"
        uploaded_resume = st.session_state.get(uploader_key)
        
        if uploaded_resume is not None:
            # initial debug log to confirm handler execution
            job_resume_dir = os.path.join(RESUMES_PATH, job_file_id)
            os.makedirs(job_resume_dir, exist_ok=True)
            log_file_path = os.path.join(job_resume_dir, "score_log.txt")
            try:
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write("HANDLER STARTED\n")
            except Exception:
                pass
            job_resume_dir = os.path.join(RESUMES_PATH, job_file_id)
            os.makedirs(job_resume_dir, exist_ok=True)

            file_extension = os.path.splitext(uploaded_resume.name)[1]
            existing_resumes = [f for f in os.listdir(job_resume_dir) if f.startswith(f"{job_file_id}resume")]
            next_resume_number = len(existing_resumes) + 1
            
            new_resume_filename = f"{job_file_id}resume{next_resume_number}{file_extension}"
            resume_path = os.path.join(job_resume_dir, new_resume_filename)
            
            with open(resume_path, "wb") as f:
                f.write(uploaded_resume.getbuffer())
            
            # --- Scoring Logic ---
            # We'll always attempt scoring, but ensure any error is caught and the JSON file is created.
            # Primary storage for scored results is in the scores/<job_id>/ folder.
            scores_job_dir = os.path.join(SCORES_PATH, job_file_id)
            os.makedirs(scores_job_dir, exist_ok=True)
            scores_file_path = os.path.join(scores_job_dir, "scoring_results.json")
            log_file_path = os.path.join(scores_job_dir, "score_log.txt")
            all_scores = []
            if os.path.exists(scores_file_path):
                try:
                    with open(scores_file_path, "r") as f:
                        all_scores = json.load(f)
                except Exception:
                    # corrupted JSON? start fresh and log
                    all_scores = []

            try:
                st.session_state.resume_upload_success = f"Successfully applied! Scoring in progress..."

                # Score the resume (may raise)
                score_report = score_resume(job["text"], resume_path, openai_api_key=OPENAI_API_KEY)

                applicant_data = {
                    "resume_file": new_resume_filename,
                    "score": score_report.get("total_score", 0),
                    "verdict": score_report.get("verdict", "Unknown"),
                    "details": score_report.get("breakdown", {})
                }

                all_scores.append(applicant_data)
                all_scores_sorted = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)

                with open(scores_file_path, "w", encoding="utf-8") as f:
                    json.dump(all_scores_sorted, f, indent=4)

                # Write a tiny marker file pointing to the last JSON write (helps debugging)
                try:
                    with open(os.path.join(scores_job_dir, "last_score_location.txt"), "w", encoding="utf-8") as lf:
                        lf.write(os.path.abspath(scores_file_path))
                except Exception:
                    pass

                # log success
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write(f"SCORED: {new_resume_filename} -> {applicant_data['score']}\n")

                st.session_state.resume_upload_success = f"Successfully applied for {job_title_display}! Your resume has been scored and submitted."

            except Exception as e:
                # On scoring failure, record minimal entry so JSON exists and log error
                error_entry = {
                    "resume_file": new_resume_filename,
                    "score": 0,
                    "verdict": "Error",
                    "error": str(e)
                }
                all_scores.append(error_entry)
                all_scores_sorted = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)
                try:
                    with open(scores_file_path, "w", encoding="utf-8") as f:
                        json.dump(all_scores_sorted, f, indent=4)
                except Exception:
                    # If writing fails, attempt to write the debug marker and a failure log
                    try:
                        with open(os.path.join(scores_job_dir, "last_score_location.txt"), "w", encoding="utf-8") as lf:
                            lf.write(os.path.abspath(scores_file_path))
                    except Exception:
                        pass
                    try:
                        with open(log_file_path, "a", encoding="utf-8") as lf:
                            lf.write(f"FAILED TO WRITE JSON for {new_resume_filename}\n")
                    except Exception:
                        pass

                try:
                    with open(log_file_path, "a", encoding="utf-8") as lf:
                        lf.write(f"ERROR scoring {new_resume_filename}: {e}\n")
                except Exception:
                    pass

                st.session_state.resume_upload_success = f"Application submitted, but scoring failed (logged)."

    st.file_uploader(
        "Upload Your Resume", 
        type=["pdf", "docx"], 
        key=f"resume_{job_file_id}",
        on_change=process_resume_upload
    )

    # Display success message if the flag is set
    if st.session_state.get('resume_upload_success'):
        st.success(st.session_state.resume_upload_success)
        # We can clear the flag if we only want to show it once per upload action
        # For now, it persists until the user navigates away.
    if hr_password == "hr_pass":
        scores_file_path = os.path.join(SCORES_PATH, job_file_id, "scoring_results.json")

        if os.path.exists(scores_file_path):
            with open(scores_file_path, "r") as f:
                results = json.load(f)

            if results:
                st.subheader("Resume Scoring Results (Table)")
                
                # Convert JSON to DataFrame
                df = pd.DataFrame(results)
                
                # Flatten 'details' dictionary into columns
                if 'details' in df.columns:
                    df_table = df[['resume_file', 'score', 'verdict']]

                    

                st.dataframe(df_table)  # Interactive table with sorting
            else:
                st.info("No resumes have been scored yet for this job.")

            if results:
    # Show dropdown for HR to select a student
                
                resume_options = [r['resume_file'] for r in results]
                selected_resume_file = st.selectbox("Select a student resume to generate feedback", resume_options)

                if selected_resume_file:
                    # Get the selected resume details
                    selected_resume = next(r for r in results if r['resume_file'] == selected_resume_file)

                    # Generate feedback using OpenAI API
                    import openai

                    def generate_feedback(details_dict):
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        prompt = f"""
                        You are an HR assistant. A student's resume has been scored with the following breakdown:
                        {details_dict}

                        Write a polite, constructive email to the student explaining why they were not selected
                        and what areas they could improve.
                        """
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7   
                            )
                            feedback_text = response.choices[0].message.content
                            return feedback_text
                        except Exception as e:
                            return f"Error generating feedback: {e}"

                    if st.button("Generate Feedback"):
                        feedback = generate_feedback(selected_resume['details'])
                        st.text_area("Feedback Preview", feedback, height=200)

                        # Optional: send email (requires SMTP setup)
                        st.subheader("Send Email to Student")
                        recipient_email = st.text_input("Student Email Address")

                        if st.button("Send Email"):
                            import smtplib
                            from email.mime.text import MIMEText
                            from email.mime.multipart import MIMEMultipart

                            try:
                                sender_email = "your_email@example.com"
                                sender_password = "your_email_password"  # Use app password for Gmail

                                msg = MIMEMultipart()
                                msg['From'] = sender_email
                                msg['To'] = recipient_email
                                msg['Subject'] = "Job Application Feedback"

                                msg.attach(MIMEText(feedback, 'plain'))

                                # Connect to SMTP server (Gmail example)
                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                server.starttls()
                                server.login(sender_email, sender_password)
                                server.send_message(msg)
                                server.quit()

                                st.success(f"Feedback email sent to {recipient_email}!")
                            except Exception as e:
                                st.error(f"Failed to send email: {e}")


def display_job_grid(jobs):
    st.header("Available Job Roles 📋")
    
    cols = st.columns(4)
    for i, job in enumerate(jobs):
        # Use the AI-generated title for the card
        job_title_display = job["title"]
        with cols[i % 4]:
            st.markdown(f"""
            <div class="job-card-container">
                <h3 class="job-card-title">{job_title_display}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Details", key=f"details_{i}", use_container_width=True):
                st.session_state.selected_job = job
                st.rerun()

# --- Main App Logic ---
def main():
    st.title("Job Application Portal 📄")

    if "selected_job" not in st.session_state:
        st.session_state.selected_job = None

    jobs = load_jobs()

    if st.session_state.selected_job:
        display_job_details(st.session_state.selected_job)
    else:
        if not jobs:
            st.info("No job roles available at the moment. HR can upload new jobs via the sidebar.")
        else:
            display_job_grid(jobs)

if _name_ == "_main_":
    main()