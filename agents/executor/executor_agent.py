from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["step"],
    template="""Execute the following step and provide a clear, concise result.

Step: {step}

Provide a direct result without additional explanation."""
)


def executor(step: str) -> str:
    """
    Execute a given step and return the result.

    Args:
        step: The step to execute

    Returns:
        A string containing the execution result
    """
    try:
        # Create the chain using LangChain's pipe operator
        chain = prompt_template | llm

        # Execute the chain
        result = chain.invoke({"step": step})
        return result.content
    except Exception as e:
        print(f"Error executing step: {e}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    result = executor("Write the introduction section for the microservices blog")
    print(result)