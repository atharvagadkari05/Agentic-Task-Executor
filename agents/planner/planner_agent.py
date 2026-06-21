import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize the LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["task"],
    template="""Break the following task into clear, actionable steps.

Task: {task}

Respond with valid JSON in this format:
{{
    "steps": ["step1", "step2", "step3"]
}}

Only return the JSON, no additional text."""
)

# Initialize the JSON output parser
output_parser = JsonOutputParser()


def planner(task: str) -> dict:
    """
    Break down a task into clear steps using the LLM.

    Args:
        task: The task to break down into steps

    Returns:
        A dictionary with 'steps' key containing a list of action steps
    """
    try:
        # Create the chain using LangChain's pipe operator
        chain = prompt_template | llm | output_parser

        # Execute the chain
        result = chain.invoke({"task": task})
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {"steps": [], "error": str(e)}


if __name__ == "__main__":
    result = planner("Write a blog on microservices")
    print(json.dumps(result, indent=2))