import os
from dotenv import load_dotenv
load_dotenv()

def _get_llm():
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing in .env")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0.2
    )

def create_study_plan(profile, exam_date, hours_per_day, days):
    subjects = profile.get("subjects", [])
    subject_text = ", ".join(subjects) if subjects else "Subjects not provided"

    prompt = f"""
Create a personalized college examination study plan.

Student:
Year: {profile.get("year", "Not provided")}
Branch: {profile.get("branch", "Not provided")}
Semester: {profile.get("semester", "Not provided")}
Subjects: {subject_text}
Exam date: {exam_date}
Available study time per day: {hours_per_day} hours
Plan duration: {days} days

Requirements:
- Divide study time realistically.
- Cover all subjects.
- Include revision.
- Include practice questions/tests.
- Do not invent syllabus topics.
- Use a clear day-by-day table.
- End with practical study tips.
"""
    return _get_llm().invoke(prompt).content
