import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

links_file = "test_data.csv"
qa_file = "single_passage_answer_questions.csv"

test_data = pd.read_csv(links_file, encoding="ISO-8859-1")
qa_data = pd.read_csv(qa_file)
test_data = test_data.dropna(how='any').reset_index(drop=True)

# Filter test_data for the first 3 links
# test_data = test_data.iloc[:3]
qa_data = qa_data[qa_data['document_index'].isin(test_data['index'])]

# --- Step 2: Initialize Components ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n", "\s", ""]
)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

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

# --- Step 3: Define Dynamic QA Chain Creation ---
def create_retriever_for_document(doc_text, doc_metadata):
    """
    Creates a FAISS retriever for a specific document.
    """
    # Split the document text
    chunks = text_splitter.split_text(doc_text)
    metadata = [{"source": doc_metadata} for _ in chunks]
    
    # Create FAISS index for the specific document
    faiss_index = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
    return faiss_index.as_retriever()

def generate_answer_for_question(question, doc_text, doc_metadata):
    """
    Dynamically loads a retriever for the document and generates an answer.
    """
    retriever = create_retriever_for_document(doc_text, doc_metadata)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return qa_chain.run(question)

# --- Step 4: Generate Answers Dynamically ---
results = []

for _, row in qa_data.iterrows():
    question = row['question']
    actual_answer = row['answer']
    document_index = row['document_index']
    
    # Find the corresponding document
    doc_row = test_data[test_data['index'] == document_index].iloc[0]
    doc_text = doc_row['text']
    source_url = doc_row['source_url']
    
    # Dynamically create retriever and generate the answer
    openai_answer = generate_answer_for_question(question, doc_text, source_url)
    
    # Store results
    results.append({
        "index": document_index,
        "link": source_url,
        "question": question,
        "openai_answer": openai_answer.strip(),
        "actual_answer": actual_answer
    })

# --- Step 5: Save Results ---
results_df = pd.DataFrame(results)
output_file = "results/basic_rag_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
