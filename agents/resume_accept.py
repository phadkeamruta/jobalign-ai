import os
import time
import google.generativeai as genai

# Configure Google AI API with key from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running this script.")

genai.configure(api_key=api_key)


def get_multiline_input(prompt: str) -> str:
    """
    Get multi-line input from user (press Enter twice to finish).
    
    :param prompt: Prompt to display
    :return: User input as string
    """
    print(prompt)
    lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                lines.append(line)
        except EOFError:
            # Handle when input ends (e.g., piped input)
            break
    
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("Input cannot be empty")
    return text


def analyze_resume_with_inputs(max_retries: int = 3):
    """
    Analyze resume against job description using Google Gemini AI.
    
    :param max_retries: Maximum number of retry attempts for API calls
    """
    print("\n" + "="*60)
    print("📄 Resume & Job Description Analyzer (Gemini AI)")
    print("="*60 + "\n")

    # --- USER INPUTS ---
    try:
        job_description = get_multiline_input("Paste the JOB DESCRIPTION below (press Enter twice when done):\n")
    except ValueError as e:
        raise ValueError(f"Job description error: {str(e)}")
    
    print("\n" + "-"*60 + "\n")

    try:
        resume_text = get_multiline_input("Paste the RESUME text below (press Enter twice when done):\n")
    except ValueError as e:
        raise ValueError(f"Resume text error: {str(e)}")
    
    print("\n" + "-"*60 + "\n")

    # --- AI PROMPT ---
    prompt = f"""
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

    # --- CALL GEMINI MODEL WITH RETRY LOGIC ---
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    for attempt in range(max_retries):
        try:
            print("⏳ Analyzing resume with Gemini AI...\n")
            response = model.generate_content(prompt)
            
            # Check if response was successful
            if not response or not hasattr(response, 'text'):
                raise RuntimeError("Invalid response from Gemini API")
            
            print("\n" + "="*60)
            print("✅ AI ANALYSIS RESULT")
            print("="*60 + "\n")
            print(response.text)
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for rate limit errors
            if any(x in error_msg for x in ["429", "rate_limit", "quota", "RESOURCE_EXHAUSTED"]):
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts.")
                    print(f"   Error: {error_msg}")
                    raise
            # Check for API key errors
            elif any(x in error_msg for x in ["API key", "authentication", "401", "403", "INVALID_ARGUMENT"]):
                print(f"❌ API Authentication Error: {error_msg}")
                print("   Please check your GEMINI_API_KEY environment variable.")
                raise
            # Other errors
            else:
                print(f"❌ Error during analysis: {error_msg}")
                raise


# Run the agent
if __name__ == "__main__":
    try:
        analyze_resume_with_inputs()
    except ValueError as e:
        print(f"❌ Input Error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
