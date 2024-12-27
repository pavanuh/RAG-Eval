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

# --- Step 3: Fusion Context from Dynamic Retrievers ---
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

def fuse_contexts(question, doc_text, doc_metadata):
    """
    Fuse contexts from multiple dynamic retrievers for a single document.
    """
    retrievers = [create_retriever_for_document(doc_text, doc_metadata, splitter) for splitter in splitters]
    
    docs = list(chain.from_iterable([retriever.get_relevant_documents(question) for retriever in retrievers]))
    unique_docs = {doc.page_content: doc for doc in docs}  # Deduplicate by content
    return "\n".join([doc.page_content for doc in unique_docs.values()])

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question:
    {context}

    Question: {question}
    Answer:
    """
)

qa_chain = LLMChain(llm=llm, prompt=prompt_template)

# --- Step 4: Generate RAG Fusion Answers Dynamically ---
results = []

for _, row in qa_data.iterrows():
    question = row['question']
    actual_answer = row['answer']
    document_index = row['document_index']

    # Get the document text
    doc_row = test_data[test_data['index'] == document_index].iloc[0]
    doc_text = doc_row['text']
    source_url = doc_row['source_url']

    # Fuse contexts dynamically from three retrievers
    fused_context = fuse_contexts(question, doc_text, source_url)

    # Generate answer using fused context
    openai_answer = qa_chain.run({"context": fused_context, "question": question})

    # Store results
    results.append({
        "index": document_index,
        "link": source_url,
        "question": question,
        "openai_answer": openai_answer.strip(),
        "actual_answer": actual_answer
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
output_file = "results/rag_fusion_results.csv"
results_df.to_csv(output_file, index=False)

print(f"RAG Fusion results saved to {output_file}")
