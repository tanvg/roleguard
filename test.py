from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")
response = llm.invoke("say hello in one sentence")
print(response.content)
