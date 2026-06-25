from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["task", "step", "previous_output", "feedback"],
    template="""You are the executor agent in a multi-agent workflow.
Your job is to complete the current step for the task.

Task: {task}
Step: {step}
Previous output: {previous_output}
Reviewer feedback: {feedback}

If reviewer feedback is provided, revise the output to address it directly.
Return only the improved step result, with no meta commentary."""
)


def executor(
    step: str,
    task: str = "",
    previous_output: str = "",
    feedback: str = "",
) -> str:
    """
    Execute a given step and return the result.

    Args:
        step: The step to execute
        task: The parent task for additional context
        previous_output: Previous attempt for the same step
        feedback: Reviewer feedback to incorporate in the new attempt

    Returns:
        A string containing the execution result
    """
    try:
        # Create the chain using LangChain's pipe operator
        chain = prompt_template | llm

        # Execute the chain
        result = chain.invoke(
            {
                "task": task or "No overall task provided.",
                "step": step,
                "previous_output": previous_output or "No previous output.",
                "feedback": feedback or "No reviewer feedback yet.",
            }
        )
        return result.content
    except Exception as e:
        print(f"Error executing step: {e}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    result = executor(
        "Write the introduction section for the microservices blog",
        task="Write a blog on microservices",
    )
    print(result)
