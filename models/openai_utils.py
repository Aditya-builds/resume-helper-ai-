import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def format_job_description(text):
    """
    Uses OpenAI's GPT model to format a raw job description text.
    """
    prompt = f"""
    Please reformat the following job description into a clean, professional, and easy-to-read layout. 
    Use markdown for formatting, including headers, bullet points for responsibilities and qualifications, and bold text for emphasis.
    The output should be ready to be displayed directly to a job applicant.

    Raw Text:
    ---
    {text}
    ---
    Formatted Job Description:
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR content formatter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred with the OpenAI API: {e}"

def generate_job_title(text):
    """
    Uses OpenAI's GPT model to generate a concise job title from raw text.
    """
    prompt = f"""
    Based on the following job description text, please generate a concise and professional job title.
    The title should be no more than 4-5 words.

    Raw Text:
    ---
    {text[:1500]} 
    ---
    Job Title:
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR professional who creates concise job titles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20,
        )
        # Clean up the response to ensure it's just the title
        title = response.choices[0].message.content.strip()
        # Remove potential quotes or extra phrases
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        return title
    except Exception as e:
        return "Job Title Generation Error"
