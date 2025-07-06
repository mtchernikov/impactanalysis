
import streamlit as st
import json
import openai
import faiss
import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Safety Impact Analysis with GPT-4o", layout="wide")
st.title("🔍 Safety Impact Analysis using RAG + GPT-4o")

# Load structured safety artifact map
artifact_json = st.file_uploader("📤 Upload your safety_artifacts.json", type="json")

# Load documents as plain text (e.g., HARA, FSC, TSC)
doc_files = st.file_uploader("📤 Upload artifact documents (txt, md, etc.)", type=["txt", "md"], accept_multiple_files=True)

# Input change request
change_input = st.text_area("✏️ Describe the change request", height=150)

# Configure embedding and OpenAI
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Replaceable by OpenAI embeddings
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔑 Enter OpenAI API Key", type="password")

if st.button("🔍 Analyze Impact") and change_input and artifact_json and doc_files:
    # Load structured artifacts
    artifacts = json.load(artifact_json)
    artifact_map = {a["ID"]: a for a in artifacts}

    # Prepare document index
    texts, metadata = [], []
    for file in doc_files:
        text = file.read().decode("utf-8")
        texts.append(text)
        metadata.append({"filename": file.name})

    # Embed documents
    doc_embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(doc_embeddings[0]))
    index.add(np.array(doc_embeddings))

    # Embed change request
    query_vec = embedding_model.encode([change_input])[0]
    D, I = index.search(np.array([query_vec]), k=5)

    # Retrieve top-k matches
    retrieved_texts = [texts[i] for i in I[0]]
    retrieved_context = "\n\n".join(retrieved_texts)

    # Combine with structured artifact info
    context_snippets = "\n".join([
        f"Artifact: {a['ID']}\nName: {a['Name']}\nInputs: {a['Inputs (ISO Ref)']}\nOutputs: {a['Outputs (ISO Ref)']}\nContent: {a['Content (ISO Ref)']}"
        for a in artifacts if any(term in a['Name'].lower() for term in change_input.lower().split())
    ])

    full_prompt = f"""
You are an expert in ISO 26262 safety engineering. A change has been proposed:
"""
    full_prompt += f"\n---\n{change_input}\n---\n"
    full_prompt += f"Relevant retrieved context from safety documents:\n{retrieved_context}\n"
    full_prompt += f"Structured safety artifact map:\n{context_snippets}\n"
    full_prompt += """
Based on this, list:
1. Safety artifacts potentially impacted by this change.
2. What needs to be re-validated or updated.
3. Any recommended actions to preserve safety compliance.
"""

    # Call GPT-4o
    with st.spinner("Asking GPT-4o..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in safety and impact analysis."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2
        )
        st.markdown("### 🧠 GPT-4o Impact Analysis Result")
        st.write(response.choices[0].message.content)
else:
    st.info("Please upload the JSON, documents and provide a change request to start analysis.")
