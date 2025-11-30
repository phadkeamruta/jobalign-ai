import os
import json
import re
import time
from pathlib import Path
from openai import OpenAI, RateLimitError

# Initialize OpenAI client with API key from environment
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

client = OpenAI(api_key=openai_key)

# Define resume storage directory
RESUME_DIR = os.path.join(os.path.dirname(__file__), '../data/resumes')
os.makedirs(RESUME_DIR, exist_ok=True)



class ResumeParserAgent:
    """
    Extracts structured information from a resume using LLMs.
    Outputs clean JSON that other agents (Analyzer, Rewriter, Matcher)
    can directly consume.
    """

    def __init__(self, model="gpt-4.1"):
        self.model = model

    def get_resume_from_file(self, filepath: str) -> str:
        """
        Load resume content from a file (supports .txt, .pdf as text).
        
        :param filepath: path to resume file
        :return: resume text content
        :raises FileNotFoundError: if file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Resume file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading resume file: {str(e)}")

    def get_resume_from_user_input(self) -> str:
        """
        Get resume content from user via command line input.
        User can paste resume text directly.
        
        :return: resume text content
        """
        print("\n" + "="*60)
        print("📄 RESUME INPUT")
        print("="*60)
        print("Paste your resume text below. When done, press Enter twice:")
        print("="*60 + "\n")
        
        lines = []
        empty_lines = 0
        
        try:
            while empty_lines < 2:
                line = input()
                if line.strip() == "":
                    empty_lines += 1
                else:
                    empty_lines = 0
                    lines.append(line)
            
            resume_text = "\n".join(lines).strip()
            if not resume_text:
                raise ValueError("Resume content cannot be empty")
            return resume_text
        except KeyboardInterrupt:
            raise ValueError("Resume input cancelled by user")

    def save_resume_to_file(self, resume_text: str, filename: str) -> str:
        """
        Save resume text to a file for future reference.
        
        :param resume_text: resume content to save
        :param filename: name of the file (without path)
        :return: full path to saved file
        """
        filepath = os.path.join(RESUME_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(resume_text)
            print(f"✅ Resume saved to: {filepath}")
            return filepath
        except Exception as e:
            raise RuntimeError(f"Error saving resume: {str(e)}")

    def list_saved_resumes(self) -> list:
        """
        List all saved resumes in the data/resumes directory.
        
        :return: list of filenames
        """
        try:
            files = os.listdir(RESUME_DIR)
            resume_files = [f for f in files if f.endswith(('.txt', '.pdf'))]
            return sorted(resume_files)
        except Exception as e:
            print(f"Error listing resumes: {str(e)}")
            return []

    def get_resume_from_list(self) -> str:
        """
        Display saved resumes and let user choose one.
        
        :return: resume text content
        :raises ValueError: if no resumes available or invalid selection
        """
        saved_resumes = self.list_saved_resumes()
        
        if not saved_resumes:
            raise ValueError("No saved resumes found in " + RESUME_DIR)
        
        print("\n" + "="*60)
        print("📋 SAVED RESUMES")
        print("="*60)
        for i, resume in enumerate(saved_resumes, 1):
            print(f"{i}. {resume}")
        print("="*60)
        
        try:
            choice = int(input("Select resume number (or 0 to go back): "))
            if choice == 0:
                return ""
            if 1 <= choice <= len(saved_resumes):
                filepath = os.path.join(RESUME_DIR, saved_resumes[choice - 1])
                return self.get_resume_from_file(filepath)
            else:
                raise ValueError("Invalid selection")
        except ValueError as e:
            raise ValueError(f"Invalid input: {str(e)}")

    def parse_resume(self, resume_text: str, max_retries: int = 3) -> dict:
        """
        Takes raw resume text and returns structured data.
        Includes retry logic for rate limiting.

        :param resume_text: string of full resume content
        :param max_retries: maximum number of retry attempts
        :return: dict with structured fields
        """
        prompt = f"""
        You are a Resume Parsing AI Agent.

        Extract clean, structured JSON from the following resume text.
        Do NOT add explanations. JSON only.

        RESUME TEXT:
        {resume_text}

        Return JSON in the following format:

        {{
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "summary": "",
            "skills": [],
            "experience": [
                {{
                    "job_title": "",
                    "company": "",
                    "location": "",
                    "start_date": "",
                    "end_date": "",
                    "description": []
                }}
            ],
            "education": [
                {{
                    "degree": "",
                    "school": "",
                    "year": ""
                }}
            ],
            "certifications": [],
            "projects": [],
            "ats_keywords": []
        }}
        """

        # Retry logic for rate limiting
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                parsed_output = response.choices[0].message.content

                # The response is JSON text. Convert to Python dict safely.
                try:
                    return json.loads(parsed_output)
                except json.JSONDecodeError:
                    # In case the model returns extra text, attempt cleanup.
                    cleaned_json = self._extract_json(parsed_output)
                    return json.loads(cleaned_json)

            except RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rate limit error after {max_retries} attempts: {str(e)}")
                    raise
            except Exception as e:
                print(f"❌ Error parsing resume: {str(e)}")
                raise

    def _extract_json(self, text: str) -> str:
        """
        Helper: Extract JSON substring from messy LLM output.
        """
        import re
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No valid JSON found in LLM response.")


# ---------- Example Usage ----------
if __name__ == "__main__":
    agent = ResumeParserAgent()
    
    # Example 1: Use sample resume
    print("\n" + "="*60)
    print("EXAMPLE 1: Parsing Sample Resume")
    print("="*60 + "\n")
    
    sample_resume = """
    John Doe
    Email: john.doe@email.com
    Phone: 555-123-4567
    Location: Austin, TX

    Summary:
    Experienced QA Engineer with 5+ years in automation (Playwright, Selenium, Robot Framework).

    Skills: Python, Playwright, Selenium, Robot Framework, API Testing, CI/CD

    Experience:
    QA Automation Engineer, ABC Corp, Austin TX (2020 - Present)
    - Developed automation scripts using Playwright
    - Improved test coverage by 40%

    Education:
    B.S. in Computer Science, Texas State University, 2018
    """
    
    result = agent.parse_resume(sample_resume)
    print("Parsed Resume:")
    print(json.dumps(result, indent=2))
    
    # Save for future use
    agent.save_resume_to_file(sample_resume, "sample_resume.txt")
    
    # Example 2: Interactive menu for getting resume from user
    print("\n" + "="*60)
    print("EXAMPLE 2: Interactive Resume Input")
    print("="*60)
    
    def interactive_resume_menu():
        """Interactive menu to get resume from user"""
        while True:
            print("\n" + "="*60)
            print("RESUME INPUT OPTIONS")
            print("="*60)
            print("1. Paste resume text directly")
            print("2. Load from saved resume file")
            print("3. Exit")
            print("="*60)
            
            choice = input("Select option (1-3): ").strip()
            
            if choice == "1":
                try:
                    resume_text = agent.get_resume_from_user_input()
                    
                    # Ask if user wants to save
                    save_choice = input("\nSave this resume? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        filename = input("Enter filename (e.g., my_resume.txt): ").strip()
                        agent.save_resume_to_file(resume_text, filename)
                    
                    # Parse the resume
                    print("\n⏳ Parsing resume...")
                    result = agent.parse_resume(resume_text)
                    print("\n✅ Parsed Resume:")
                    print(json.dumps(result, indent=2))
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
            
            elif choice == "2":
                try:
                    resume_text = agent.get_resume_from_list()
                    if resume_text:
                        print("\n⏳ Parsing resume...")
                        result = agent.parse_resume(resume_text)
                        print("\n✅ Parsed Resume:")
                        print(json.dumps(result, indent=2))
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
            
            elif choice == "3":
                print("Exiting...")
                break
            
            else:
                print("Invalid option. Please try again.")
    
    # Uncomment to run interactive mode:
    # interactive_resume_menu()
