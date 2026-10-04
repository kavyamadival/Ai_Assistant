# AI-Based College Academic Assistant

This is a Streamlit-based academic assistant implementing:

- LLM
- RAG
- LangChain
- LangGraph
- FAISS vector database
- Conversational context
- Personalized study planner
- Document summarization
- Calculator tool
- Basic LLM vs RAG comparison
- Unknown-information handling

## 1. Open the project

Open this folder in VS Code.

## 2. Create a virtual environment

Windows:

python -m venv venv

Activate:

venv\Scripts\activate

## 3. Install packages

pip install -r requirements.txt

## 4. Configure the API key

Copy:

.env.example

to:

.env

Then put your Groq API key in:

GROQ_API_KEY=your_key_here

## 5. Add your college data

Fill the TXT files inside:

data/

Recommended information:
- 3rd-year CSE syllabus
- subjects
- academic regulations
- examination rules
- internship rules
- placement information
- FAQs
- study resources

You can add additional .txt files too.

## 6. Run

streamlit run app.py

## 7. Build the knowledge base

Open the application and click:

Build / Rebuild Knowledge Base

Do this again whenever you change the documents.

## Main workflow

Question
 -> Question Analysis
 -> Information Retrieval
 -> Response Generation
 -> Response Review
 -> Final Answer

Complex study-planning requests are routed through the LangGraph planner path.

## Notes

The first embedding model download can take some time because
sentence-transformers/all-MiniLM-L6-v2 is downloaded locally.

For college-specific questions, the assistant is instructed not to invent
information that is absent from the knowledge base.
