import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def load_documents():
    """Load all PDFs from resources folder"""
    documents = []
    resources_dir = "./resources"

    if not os.path.exists(resources_dir):
        print(f"Error: {resources_dir} folder not found")
        return documents

    pdf_files = [f for f in os.listdir(resources_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(resources_dir, pdf_file)
        print(f"Loading {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents

def setup_vectorstore(documents):
    """Split documents, embed, and store in Chroma"""
    print("\nSplitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

def setup_rag_chain(vectorstore):
    """Create RAG chain with llama.cpp server"""
    print("\nInitializing LLM (llama-server)...")
    llm = ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="none",
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    template = """Answer the question based on the following context from the documents:

{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain, retriever

def ask_question(rag_chain, retriever, question):
    """Query the RAG pipeline"""
    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer = rag_chain.invoke(question)
    print(f"Answer: {answer.content}")

    docs = retriever.invoke(question)
    print("\nSource documents:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"  [{i}] {source}")
        print(f"      {doc.page_content[:100]}...")

def main():
    # Check if vectorstore exists
    if os.path.exists("./chroma_db"):
        print("Using existing Chroma database...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("Building new RAG pipeline...")
        documents = load_documents()
        if not documents:
            print("No documents loaded. Exiting.")
            return

        vectorstore = setup_vectorstore(documents)

    rag_chain, retriever = setup_rag_chain(vectorstore)

    print("\n" + "="*50)
    print("RAG Pipeline Ready!")
    print("="*50)
    print("Type 'quit' to exit\n")

    # Interactive loop
    while True:
        try:
            question = input("Ask a question: ").strip()
            if question.lower() == "quit":
                print("Exiting...")
                break
            if question:
                ask_question(rag_chain, retriever, question)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
