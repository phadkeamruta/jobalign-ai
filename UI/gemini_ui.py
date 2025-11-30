# ui/gemini_ui.py
import os
import time
import streamlit as st
import google.generativeai as genai

# -------------------------
# Configuration
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # We'll still let the app start so the user sees an error in the UI
    genai_configured = False
else:
    genai.configure(api_key=GEMINI_API_KEY)
    genai_configured = True

# -------------------------
# Helper functions
# -------------------------
def get_prompt(job_description: str, resume_text: str) -> str:
    """Construct the prompt passed to Gemini (same logic as your CLI agent)."""
    return f"""
You are an expert ATS Resume Optimization Agent.

Your tasks:
1. Compare the resume with the job description provided by the user.
2. Provide the following:
   - Resume to Job match percentage (0–100)
   - Missing technical skills / keywords
   - Weak areas in the resume
   - Suggested improvements
   - Recommended bullet points tailored to the job requirements
   - Improved professional summary
   - Final optimized resume text

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide your analysis now.
    """

def analyze_with_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini model with retry/backoff logic and return the textual result."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # Robust check for text
            if not response or not hasattr(response, "text") or not response.text:
                raise RuntimeError("Received an empty or invalid response from Gemini API.")
            return response.text

        except Exception as e:
            error_msg = str(e)
            # Rate limit handling
            if any(x in error_msg.lower() for x in ["429", "rate_limit", "quota", "resource_exhausted"]):
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Rate limit / quota exceeded after {max_retries} attempts. Error: {error_msg}")
            # Authentication errors
            elif any(x in error_msg.lower() for x in ["api key", "authentication", "401", "403", "invalid_argument"]):
                raise RuntimeError(f"API authentication error: {error_msg}")
            else:
                # Other transient errors: retry a couple of times
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
                    continue
                raise RuntimeError(f"Failed to call Gemini API: {error_msg}")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="JobAlign AI — Gemini Resume Analyzer", layout="wide")

st.title("JobAlign AI — Gemini Resume Analyzer")
st.markdown(
    "Paste a **Job Description** and a **Resume** below, then click **Analyze**. "
    "This UI calls Google Gemini (via `google.generativeai`) and shows the model's analysis."
)

with st.expander("Instructions", expanded=False):
    st.write(
        "• Make sure the environment variable `GEMINI_API_KEY` is set where you run Streamlit.\n\n"
        "• Press Enter twice (or leave an empty line) after pasting a block of text in the text areas.\n\n"
        "• The analysis may take a few seconds depending on API latency and retry behavior."
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    job_description = st.text_area("Paste the full job description here", height=300, placeholder="Paste job description...")

with col2:
    st.subheader("Resume Text")
    resume_text = st.text_area("Paste the full resume text here", height=300, placeholder="Paste resume text...")

st.markdown("---")

# Options row
opt_col1, opt_col2, opt_col3 = st.columns([1, 1, 2])
with opt_col1:
    max_retries = st.number_input("Max Retries", min_value=1, max_value=6, value=3, step=1)
with opt_col2:
    show_prompt = st.checkbox("Show prompt sent to Gemini", value=False)
with opt_col3:
    st.write("")  # spacer
    analyze_btn = st.button("Analyze with Gemini", type="primary")

# Result area
result_area = st.empty()

if analyze_btn:
    # Basic input validation
    if not genai_configured:
        st.error("GEMINI_API_KEY not found. Please set the GEMINI_API_KEY environment variable before running this app.")
    elif not job_description or not job_description.strip():
        st.warning("Please paste a job description before analyzing.")
    elif not resume_text or not resume_text.strip():
        st.warning("Please paste a resume text before analyzing.")
    else:
        prompt = get_prompt(job_description, resume_text)
        if show_prompt:
            st.subheader("Generated Prompt")
            st.code(prompt[:5000] + ("..." if len(prompt) > 5000 else ""), language="text")

        with st.spinner("Analyzing resume with Gemini AI..."):
            try:
                output = analyze_with_gemini(prompt, max_retries=int(max_retries))
                # Display results
                result_area.subheader("🧾 Gemini Analysis Output")
                result_area.text_area("Raw AI Output", output, height=400)
            except Exception as e:
                result_area.error(f"Error during analysis: {str(e)}")

st.markdown("---")
st.caption("JobAlign AI — Gemini UI • Ensure GEMINI_API_KEY is set in the environment where Streamlit runs.")
