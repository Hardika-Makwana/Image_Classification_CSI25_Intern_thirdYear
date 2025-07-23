import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import torch

# 1 Load embedder (DistilBERT) for doc & query embeddings
from transformers import AutoTokenizer, AutoModel

embed_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embed_model = AutoModel.from_pretrained("distilbert-base-uncased")

#  Load generator (Flan-T5)
generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Example docs
docs = [
    "Applicant must have a stable income and employment.",
    "High loan amount may need collateral.",
    "Applicants with bad credit history may be rejected.",
    "Good credit score increases chances of approval.",
    "Previous loan defaults reduce chances.",
    "Applicants with high income can get larger loans approved."
]

# Embedder function
def embed(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Build FAISS index
embeddings = np.vstack([embed(doc) for doc in docs])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Streamlit UI
st.title("Loan Approval RAG Chatbot")

query = st.text_input("Ask a question:")

if query:
    # Embed query
    q_emb = embed(query).reshape(1, -1)

    # Retrieve
    D, I = index.search(q_emb, k=3)
    retrieved_docs = [docs[idx] for idx in I[0]]
    context = "\n".join(retrieved_docs)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate answer
    inputs = generator_tokenizer(prompt, return_tensors="pt")
    outputs = generator_model.generate(**inputs, max_length=128)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write("###  Answer:")
    st.write(answer)
