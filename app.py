import streamlit as st
from dotenv import load_dotenv
from modules.rag import build_vectorstore, ask_rag, summarize_document
from modules.graph import run_academic_graph
from modules.planner import create_study_plan
from modules.tools import calculate_expression
from pathlib import Path
load_dotenv()

def load_css():
    css_file = Path("style.css")
    if css_file.exists():
        st.markdown(
            f"<style>{css_file.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )

load_css()
st.set_page_config(page_title="AI College Academic Assistant", page_icon="🎓", layout="wide")

st.title("AI-Based College Academic Assistant")
st.caption("RAG + LangChain + LangGraph + Streamlit")

with st.sidebar:
    st.header("Student Profile")
    year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
    branch = st.text_input("Branch", "Computer Science and Engineering")
    semester = st.text_input("Semester", "5")
    subjects_text = st.text_area(
        "Subjects (one per line)",
        "Machine Learning\nComputer Networks\nDatabase Management Systems\nSoftware Engineering\nWeb Technology"
    )

    st.divider()
    st.header("Knowledge Base")
    if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Loading documents and creating embeddings..."):
            result = build_vectorstore()
        st.success(result)

tabs = st.tabs(["Academic Assistant", "Study Planner", "Summarizer", "Calculator", "Basic LLM vs RAG"])

with tabs[0]:
    st.subheader("Ask the Academic Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about syllabus, exams, internships, placements, etc.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = "\n".join(
            f'{m["role"]}: {m["content"]}' for m in st.session_state.messages[-8:]
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_academic_graph(
                        question=question,
                        chat_history=history,
                        student_profile={
                            "year": year,
                            "branch": branch,
                            "semester": semester,
                            "subjects": [s.strip() for s in subjects_text.splitlines() if s.strip()]
                        }
                    )
                    answer = result.get("final_answer", "No answer generated.")
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

with tabs[1]:
    st.subheader("Personalized Study Planner")

    subjects = [s.strip() for s in subjects_text.splitlines() if s.strip()]
    exam_date = st.date_input("Examination Date")
    hours = st.number_input("Available study hours per day", min_value=1.0, max_value=16.0, value=3.0, step=0.5)
    plan_days = st.number_input("Number of days", min_value=1, max_value=60, value=15)

    if st.button("Generate Study Plan"):
        profile = {
            "year": year,
            "branch": branch,
            "semester": semester,
            "subjects": subjects
        }
        with st.spinner("Creating personalized plan..."):
            try:
                plan = create_study_plan(profile, str(exam_date), hours, int(plan_days))
                st.markdown(plan)
            except Exception as e:
                st.error(str(e))

with tabs[2]:
    st.subheader("Document Summarizer")
    uploaded = st.file_uploader("Upload a TXT document", type=["txt"])
    if uploaded:
        content = uploaded.read().decode("utf-8", errors="ignore")
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                try:
                    st.markdown(summarize_document(content))
                except Exception as e:
                    st.error(str(e))

with tabs[3]:
    st.subheader("Calculator Tool")
    expression = st.text_input("Enter an expression", "25 * 4 + 10")
    if st.button("Calculate"):
        try:
            st.success(f"Result: {calculate_expression(expression)}")
        except Exception as e:
            st.error(str(e))

with tabs[4]:
    st.subheader("Basic LLM vs RAG")
    st.write("Use this section to demonstrate the difference between a general LLM answer and a college-document-grounded answer.")
    comparison_question = st.text_input("Question for comparison", "What are the internship requirements?")
    if st.button("Compare"):
        try:
            basic, rag = ask_rag(comparison_question, comparison_only=True)
            st.markdown("### Basic LLM Answer")
            st.write(basic)
            st.markdown("### RAG Answer")
            st.write(rag)
        except Exception as e:
            st.error(str(e))
