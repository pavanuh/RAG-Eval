import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Load the test data
links_file = "test_data.csv"
qa_file = "single_passage_answer_questions.csv"

test_data = pd.read_csv(links_file, encoding="ISO-8859-1").dropna().reset_index(drop=True)
qa_data = pd.read_csv(qa_file)
qa_data = qa_data[qa_data['document_index'].isin(test_data['index'])]

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n", "\s", ""])
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# Corrective Prompt Template
corrective_prompt_template = PromptTemplate(
    input_variables=["context", "question", "previous_answer"],
    template="""
    Use the following context to answer the question:
    {context}

    The initial answer provided was: {previous_answer}

    If the initial answer is correct, confirm it. If it is incorrect or incomplete, provide a corrected and improved answer.

    Question: {question}
    Corrected Answer:
    """
)

# Create retriever for each document
def create_retriever_for_document(doc_text, doc_metadata):
    """
    Creates a FAISS retriever for a specific document.
    """
    chunks = text_splitter.split_text(doc_text)
    metadata = [{"source": doc_metadata} for _ in chunks]
    faiss_index = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
    return faiss_index.as_retriever()

def generate_corrective_answer(question, doc_text, doc_metadata):
    """
    Generate initial and corrected answers for a question.
    """
    retriever = create_retriever_for_document(doc_text, doc_metadata)
    context_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in context_docs])

    # Step 1: Generate initial answer
    initial_prompt = f"Use the context to answer: {context}\nQuestion: {question}\nAnswer:"
    initial_answer = llm.predict(initial_prompt).strip()

    # Step 2: Generate corrective answer
    corrected_prompt = corrective_prompt_template.format(
        context=context, question=question, previous_answer=initial_answer
    )
    corrected_answer = llm.predict(corrected_prompt).strip()

    return initial_answer, corrected_answer

# Process each QA pair
results = []
for _, row in qa_data.iterrows():
    question = row['question']
    actual_answer = row['answer']
    document_index = row['document_index']

    # Get the document text
    doc_row = test_data[test_data['index'] == document_index].iloc[0]
    doc_text = doc_row['text']
    source_url = doc_row['source_url']

    # Generate answers
    initial_answer, corrected_answer = generate_corrective_answer(question, doc_text, source_url)

    # Append results
    results.append({
        "index": document_index,
        "link": source_url,
        "question": question,
        "initial_answer": initial_answer,
        "corrected_answer": corrected_answer,
        "actual_answer": actual_answer
    })

# Save results to CSV
results_df = pd.DataFrame(results)
output_file = "results/corrective_rag_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Corrective RAG results saved to {output_file}")
