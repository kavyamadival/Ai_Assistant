from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from modules.rag import retrieve_context, generate_rag_answer
from modules.planner import create_study_plan
from modules.tools import calculate_expression

class AcademicState(TypedDict, total=False):
    question: str
    chat_history: str
    student_profile: Dict[str, Any]
    intent: str
    context: str
    tool_result: str
    draft_answer: str
    final_answer: str

def analyze_question(state):
    q = state["question"].lower()

    if any(x in q for x in ["study plan", "study schedule", "timetable", "prepare for exam"]):
        intent = "study_plan"
    elif any(x in q for x in ["calculate", "how many days", "percentage", "average"]):
        intent = "calculator"
    else:
        intent = "academic_rag"

    return {"intent": intent}

def retrieve_information(state):
    return {"context": retrieve_context(state["question"])}

def generate_response(state):
    answer = generate_rag_answer(
        state["question"],
        state.get("context", ""),
        state.get("chat_history", "")
    )
    return {"draft_answer": answer}

def run_tool(state):
    q = state["question"]
    return {"tool_result": "This request may require the calculator/tool. Use the Calculator tab for direct expressions."}

def plan_response(state):
    profile = state.get("student_profile", {})
    subjects = profile.get("subjects", [])
    if not subjects:
        subjects = ["Subject 1", "Subject 2", "Subject 3"]

    plan = create_study_plan(
        profile,
        "the examination date provided by the student",
        3,
        7
    )
    return {"draft_answer": plan}

def review_response(state):
    answer = state.get("draft_answer", "")
    if not answer:
        answer = "Sorry, I could not generate a response."
    return {"final_answer": answer}

def _route(state):
    if state["intent"] == "study_plan":
        return "plan"
    if state["intent"] == "calculator":
        return "tool"
    return "retrieve"

def build_graph():
    graph = StateGraph(AcademicState)

    graph.add_node("analyze", analyze_question)
    graph.add_node("retrieve", retrieve_information)
    graph.add_node("generate", generate_response)
    graph.add_node("tool", run_tool)
    graph.add_node("plan", plan_response)
    graph.add_node("review", review_response)

    graph.set_entry_point("analyze")
    graph.add_conditional_edges(
        "analyze",
        _route,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "plan": "plan"
        }
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "review")
    graph.add_edge("tool", "review")
    graph.add_edge("plan", "review")
    graph.add_edge("review", END)

    return graph.compile()

def run_academic_graph(question, chat_history="", student_profile=None):
    app = build_graph()
    return app.invoke({
        "question": question,
        "chat_history": chat_history,
        "student_profile": student_profile or {}
    })
