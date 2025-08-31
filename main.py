from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("API Key Loaded:", bool(OPENAI_API_KEY))

# Load documents
data_file = "docs"
documents = []
for file in os.listdir(data_file):
    try:
        file_path = os.path.join(data_file, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            print(f"Skipped {file}, can't upload this file.")
            continue

        data = loader.load()
        documents.extend(data)
        print(f"Loaded {len(data)} documents from {file}")

    except Exception as e:
        print(f"Error loading {file}: {e}")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=45,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(len(chunks), "chunks created")

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#store Embeddings in Vector DB(FAISS)
vectorstore = FAISS.from_documents(chunks, embeddings)

#retrieve relevant chunks only
retriever = vectorstore.as_retriever()

# Chat Memory
memory = ConversationBufferWindowMemory(
    k=5,  # remembers last 5 interactions
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# ChatPromptTemplate for structured prompt engineering
system_template = """
You are an AI Assistant answering questions based ONLY on the provided documents.

Guidelines:
- Your tone must be professional and helpful.
- Only answer using the document context.
- If the answer is not in the documents, say:
  "I can't answer this based on the provided documents."
- Be concise and to the point.

Context:
{context}
"""
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{question}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Conversational QA Chain (with memory + custom prompt)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    return_source_documents=True,
)

# Main loop
def main():
    print("\nInteractive Document QA System. Type 'exit' to quit.\n")
    while True:
        query = input("Enter Your Query: ")
        if query.lower() in ["exit", "stop"]:
            print("Exiting the Program\nGood Bye")
            break
        else:
            response = qa_chain.invoke({"question": query})
            print("\nAnswer:", response["answer"])
            print("\nSources:")
            for doc in response["source_documents"]:
                print("-", doc.metadata.get("source", "Unknown"))

if __name__ == "__main__":
    main()
