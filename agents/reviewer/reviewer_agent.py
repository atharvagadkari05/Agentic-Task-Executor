import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize the LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["step", "output"],
    template="""You are a strict evaluator in a planner-executor-reviewer loop.
Evaluate whether the output is good enough to accept for the current step.

Step: {step}
Output: {output}

Evaluate based on:
- Accuracy (0-10): How correct is the output?
- Completeness (0-10): Does it address all aspects of the step?
- Clarity (0-10): Is the output clear and well-structured?

Set approved to true only if the output is solid enough to move to the next step.
Feedback should be concrete and actionable so the executor can revise the step.

Respond with valid JSON in this format:
{{
    "accuracy": <number>,
    "completeness": <number>,
    "clarity": <number>,
    "approved": <true/false>,
    "feedback": "<constructive feedback>"
}}

Only return the JSON, no additional text."""
)

# Initialize the JSON output parser
output_parser = JsonOutputParser()


def reviewer(step: str, output: str) -> dict:
    """
    Review and evaluate the output of a given step.

    Args:
        step: The step that was executed
        output: The output/result from the step

    Returns:
        A dictionary with accuracy, completeness, clarity scores and approval status
    """
    try:
        # Create the chain using LangChain's pipe operator
        chain = prompt_template | llm | output_parser

        # Execute the chain
        result = chain.invoke({"step": step, "output": output})
        return result
    except Exception as e:
        print(f"Error reviewing output: {e}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "approved": False,
            "error": str(e),
            "feedback": "Failed to evaluate output"
        }


if __name__ == "__main__":
    # Test the reviewer
    test_step = "Write the introduction section for the microservices blog"
    test_output = "Microservices architecture is a modern approach to building applications..."
    
    result = reviewer(test_step, test_output)
    print(json.dumps(result, indent=2))
