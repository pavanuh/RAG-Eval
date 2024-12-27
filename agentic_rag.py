
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate

# --- Step 1: Load Data ---
# Load the test_data (links and texts) and single_passage_answers (questions and answers)
links_file = "test_data.csv"
qa_file = "single_passage_answer_questions.csv"

test_data = pd.read_csv(links_file, encoding="ISO-8859-1").dropna().reset_index(drop=True)
qa_data = pd.read_csv(qa_file)
qa_data = qa_data[qa_data['document_index'].isin(test_data['index'])]

# --- Step 2: Chunk Text Using a Single Splitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Prepare chunks with associated metadata
documents, metadata = [], []
for _, row in test_data.iterrows():
    chunks = splitter.split_text(row['text'])
    for chunk in chunks:
        documents.append(chunk)
        metadata.append({"index": row['index'], "source_url": row['source_url']})

# --- Step 3: Create FAISS Index ---
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_texts(documents, embeddings, metadatas=metadata)
retriever = faiss_index.as_retriever()

# --- Step 4: Define Agent Tools ---
# Retrieval Tool
def retrieval_tool(question: str) -> str:
    """
    Retrieve relevant context based on the user's question.
    """
    context_docs = retriever.get_relevant_documents(question)
    return "\n".join([doc.page_content for doc in context_docs])

# Tools list
tools = [
    Tool(
        name="DocumentRetriever",
        func=retrieval_tool,
        description="Retrieve relevant context for the user's question."
    )
]

# --- Step 5: Define the Agent ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Step 6: Generate Agentic RAG Answers ---
results = []

for _, row in qa_data.iterrows():
    question = row['question']
    actual_answer = row['answer']
    document_index = row['document_index']
    source_url = test_data[test_data['index'] == document_index]['source_url'].values[0]

    # Run Agent to answer the question
    try:
        response = agent.run(f"Question: {question}")
    except Exception as e:
        response = f"Error: {str(e)}"

    # Store results
    results.append({
        "index": document_index,
        "link": source_url,
        "question": question,
        "agentic_rag_answer": response.strip(),
        "actual_answer": actual_answer
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
output_file = "results/agentic_rag_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Agentic RAG results saved to {output_file}")
