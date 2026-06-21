from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

response = llm.invoke("Explain REST API in simple terms")

print(response.content)