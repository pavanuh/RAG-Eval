
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from itertools import chain

# --- Step 1: Load Data ---
# Load the test_data (links and texts) and single_passage_answers (questions and answers)
links_file = "test_data.csv"
qa_file = "single_passage_answer_questions.csv"

test_data = pd.read_csv(links_file, encoding="ISO-8859-1").dropna().reset_index(drop=True)
qa_data = pd.read_csv(qa_file)
qa_data = qa_data[qa_data['document_index'].isin(test_data['index'])]

# --- Step 2: Chunk Text Dynamically ---
# Initialize Recursive Text Splitter
splitters = [
    RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30),
    RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
]

# Function to create retriever for a specific document and splitter
def create_retriever_for_document(doc_text, doc_metadata, splitter):
    """
    Creates a FAISS retriever for a specific document using a specific text splitter.
    """
    chunks = splitter.split_text(doc_text)
    metadata = [{"source": doc_metadata} for _ in chunks]
    faiss_index = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
    return faiss_index.as_retriever()

# --- Step 3: Self-RAG Process ---
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Prompt Templates
initial_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question:
    {context}

    Question: {question}
    Answer:
    """
)

self_assess_prompt_template = PromptTemplate(
    input_variables=["context", "question", "previous_answer"],
    template="""
    The previous answer was:
    {previous_answer}

    Use the following context to determine if the answer is correct and complete:
    {context}

    If the previous answer is correct, confirm it. If not, refine and improve the answer.

    Question: {question}
    Improved Answer:
    """
)

def self_rag(question, doc_text, doc_metadata):
    """
    Perform Self-RAG with dynamic retrievers for the given question and document.
    """
    retrievers = [create_retriever_for_document(doc_text, doc_metadata, splitter) for splitter in splitters]

    # Step 1: Initial Retrieval and Response
    initial_docs = list(chain.from_iterable([retriever.get_relevant_documents(question) for retriever in retrievers]))
    unique_initial_docs = {doc.page_content: doc for doc in initial_docs}
    initial_context = "\n".join([doc.page_content for doc in unique_initial_docs.values()])

    initial_chain = LLMChain(llm=llm, prompt=initial_prompt_template)
    initial_answer = initial_chain.run({"context": initial_context, "question": question}).strip()

    # Step 2: Self-Assess and Iterate
    iteration = 0
    max_iterations = 3
    current_answer = initial_answer

    while iteration < max_iterations:
        iteration += 1
        assess_docs = list(chain.from_iterable([retriever.get_relevant_documents(question) for retriever in retrievers]))
        unique_assess_docs = {doc.page_content: doc for doc in assess_docs}
        assess_context = "\n".join([doc.page_content for doc in unique_assess_docs.values()])

        assess_chain = LLMChain(llm=llm, prompt=self_assess_prompt_template)
        improved_answer = assess_chain.run({
            "context": assess_context,
            "question": question,
            "previous_answer": current_answer
        }).strip()

        if improved_answer == current_answer:
            break  # Stop iteration if no improvement
        current_answer = improved_answer

    return current_answer

# --- Step 4: Generate Self-RAG Answers ---
results = []

for _, row in qa_data.iterrows():
    question = row['question']
    actual_answer = row['answer']
    document_index = row['document_index']

    # Get the document text
    doc_row = test_data[test_data['index'] == document_index].iloc[0]
    doc_text = doc_row['text']
    source_url = doc_row['source_url']

    # Run Self-RAG
    self_rag_answer = self_rag(question, doc_text, source_url)

    # Store results
    results.append({
        "index": document_index,
        "link": source_url,
        "question": question,
        "self_rag_answer": self_rag_answer,
        "actual_answer": actual_answer
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
output_file = "results/self_rag_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Self-RAG results saved to {output_file}")
