import os
import time
import json
from openai import OpenAI, RateLimitError

# Initialize OpenAI client with API key from environment
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

client = OpenAI(api_key=openai_key)


def analyze_resume(job_description: str, resume_text: str, max_retries: int = 3) -> str:
    """
    AI Agent to compare resume with job description and suggest improvements.
    
    :param job_description: str - Job description text
    :param resume_text: str - Resume text to analyze
    :param max_retries: int - Maximum retry attempts for rate limiting
    :return: str - Analysis result from AI
    :raises ValueError: if inputs are empty
    :raises RateLimitError: if rate limit exceeded after retries
    """
    
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty")
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty")

    prompt = f"""
You are an expert Resume Optimization Agent.

Your tasks:
1. Compare the resume with the job description.
2. Provide:
   - Match percentage (0–100)
   - Missing keywords / skills
   - Weak areas in the resume
   - Rewrite or improve bullet points to match the job
   - Suggest an improved summary section
   - Provide final tailored version of the resume content

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Begin your analysis now.
    """

    # Retry logic for rate limiting
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Changed from gpt-4.1 for cost savings
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            return response.choices[0].message.content

        except RateLimitError as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
            if attempt < max_retries - 1:
                print(f"⚠️  Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"❌ Rate limit error after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            print(f"❌ Error analyzing resume: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    job_description = """ 
    We are looking for a Python Automation Engineer with:
    - Playwright or Selenium experience
    - Robot Framework knowledge
    - API automation testing
    - GitHub / CI-CD experience
    - Cloud experience (AWS or Azure)
    """

    resume_text = """
    Experienced QA Engineer with background in manual and automation testing.
    Worked with Python and some automation tools.
    Familiarity with cloud environments.
    """

    try:
        print("📊 Analyzing resume against job description...\n")
        result = analyze_resume(job_description, resume_text)
        print("\n=== ANALYSIS RESULT ===\n")
        print(result)
    except ValueError as e:
        print(f"❌ Input Error: {str(e)}")
    except RateLimitError:
        print("❌ API rate limit exceeded. Please try again later or upgrade your OpenAI account.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
