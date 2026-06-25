from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.executor.executor_agent import executor
from agents.planner.planner_agent import planner
from agents.reviewer.reviewer_agent import reviewer


class WorkflowState(TypedDict, total=False):
    task: str
    plan: list[str]
    current_step_index: int
    current_step: str
    current_output: str
    current_feedback: str
    current_review: dict[str, Any]
    revision_count: int
    max_revisions: int
    completed_steps: list[dict[str, Any]]
    execution_history: list[dict[str, Any]]
    review_history: list[dict[str, Any]]
    final_result: str
    status: str


def _defaults(state: WorkflowState) -> WorkflowState:
    return {
        "task": state.get("task", ""),
        "plan": state.get("plan", []),
        "current_step_index": state.get("current_step_index", 0),
        "current_step": state.get("current_step", ""),
        "current_output": state.get("current_output", ""),
        "current_feedback": state.get("current_feedback", ""),
        "current_review": state.get("current_review", {}),
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 2),
        "completed_steps": state.get("completed_steps", []),
        "execution_history": state.get("execution_history", []),
        "review_history": state.get("review_history", []),
        "final_result": state.get("final_result", ""),
        "status": state.get("status", "running"),
    }


def plan_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    plan_result = planner(state["task"])
    steps = plan_result.get("steps", [])

    if not steps:
        return {
            "plan": [],
            "status": f"failed: planner returned no steps. error={plan_result.get('error', 'unknown')}",
            "final_result": "",
        }

    return {
        "plan": steps,
        "current_step_index": 0,
        "current_step": steps[0],
        "revision_count": 0,
        "current_feedback": "",
        "status": "planned",
    }


def execute_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    output = executor(
        step=state["current_step"],
        task=state["task"],
        previous_output=state["current_output"],
        feedback=state["current_feedback"],
    )
    attempt_number = state["revision_count"] + 1
    execution_entry = {
        "step_index": state["current_step_index"],
        "step": state["current_step"],
        "attempt": attempt_number,
        "feedback_used": state["current_feedback"],
        "output": output,
    }

    return {
        "current_output": output,
        "execution_history": state["execution_history"] + [execution_entry],
        "status": "executed",
    }


def review_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    review_result = reviewer(state["current_step"], state["current_output"])
    review_entry = {
        "step_index": state["current_step_index"],
        "step": state["current_step"],
        "attempt": state["revision_count"] + 1,
        "review": review_result,
    }

    return {
        "current_review": review_result,
        "current_feedback": review_result.get("feedback", ""),
        "review_history": state["review_history"] + [review_entry],
        "status": "reviewed",
    }


def retry_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    return {
        "revision_count": state["revision_count"] + 1,
        "status": "retrying",
    }


def advance_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    completed_entry = {
        "step_index": state["current_step_index"],
        "step": state["current_step"],
        "output": state["current_output"],
        "review": state["current_review"],
    }
    completed_steps = state["completed_steps"] + [completed_entry]
    next_index = state["current_step_index"] + 1
    has_next_step = next_index < len(state["plan"])

    return {
        "completed_steps": completed_steps,
        "current_step_index": next_index,
        "current_step": state["plan"][next_index] if has_next_step else "",
        "current_output": "",
        "current_feedback": "",
        "current_review": {},
        "revision_count": 0,
        "status": "advanced",
    }


def finalize_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    lines = [f"Task: {state['task']}", ""]

    for item in state["completed_steps"]:
        lines.append(f"Step {item['step_index'] + 1}: {item['step']}")
        lines.append(item["output"])
        lines.append("")

    if state["status"].startswith("failed"):
        lines.append(f"Workflow ended with status: {state['status']}")
    else:
        lines.append("Workflow completed successfully.")

    return {
        "final_result": "\n".join(lines).strip(),
        "status": state["status"] if state["status"].startswith("failed") else "completed",
    }


def route_after_review(state: WorkflowState) -> str:
    state = _defaults(state)
    approved = bool(state["current_review"].get("approved"))
    is_last_step = state["current_step_index"] >= len(state["plan"]) - 1

    if approved and is_last_step:
        return "finalize"
    if approved:
        return "advance"
    if state["revision_count"] >= state["max_revisions"]:
        return "fail"
    return "retry"


def route_after_advance(state: WorkflowState) -> str:
    state = _defaults(state)
    if state["current_step"]:
        return "execute"
    return "finalize"


def fail_node(state: WorkflowState) -> WorkflowState:
    state = _defaults(state)
    review = state["current_review"]
    return {
        "status": (
            "failed: reviewer did not approve the step within the revision limit. "
            f"last_feedback={review.get('feedback', 'none')}"
        )
    }


def build_workflow(checkpointer: InMemorySaver | None = None):
    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_node("retry", retry_node)
    graph.add_node("advance", advance_node)
    graph.add_node("fail", fail_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "retry": "retry",
            "advance": "advance",
            "finalize": "finalize",
            "fail": "fail",
        },
    )
    graph.add_edge("retry", "execute")
    graph.add_conditional_edges(
        "advance",
        route_after_advance,
        {
            "execute": "execute",
            "finalize": "finalize",
        },
    )
    graph.add_edge("fail", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


CHECKPOINTER = InMemorySaver()
APP = build_workflow(checkpointer=CHECKPOINTER)


def run_task(task: str, thread_id: str = "demo-thread", max_revisions: int = 2):
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: WorkflowState = {
        "task": task,
        "max_revisions": max_revisions,
    }
    return APP.invoke(initial_state, config=config)


def get_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return APP.get_state(config)


if __name__ == "__main__":
    result = run_task(
        task="Write a short blog on microservices",
        thread_id="microservices-demo",
        max_revisions=2,
    )
    print(result["final_result"])
