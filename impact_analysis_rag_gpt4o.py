
import streamlit as st
import json
import openai
import faiss
import numpy as np

st.set_page_config(page_title="Safety Impact Analysis with GPT-4o", layout="wide")
st.title("🔍 Safety Impact Analysis using RAG + GPT-4o")

# Upload safety artifact structure
artifact_json = st.file_uploader("📤 Upload your safety_artifacts.json", type="json")

# Upload plain text documents (HARA, FSC, etc.)
doc_files = st.file_uploader("📤 Upload artifact documents (txt, md, etc.)", type=["txt", "md"], accept_multiple_files=True)

# Input change description
change_input = st.text_area("✏️ Describe the change request", height=150)

# OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔑 Enter OpenAI API Key", type="password")

# Embed using OpenAI API
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

if st.button("🔍 Analyze Impact") and change_input and artifact_json and doc_files:
    # Load structured artifacts
    artifacts = json.load(artifact_json)

    # Prepare document index
    texts, metadata = [], []
    for file in doc_files:
        text = file.read().decode("utf-8")
        texts.append(text)
        metadata.append({"filename": file.name})

    # Create vector index using OpenAI embeddings
    doc_embeddings = [get_embedding(t) for t in texts]
    index = faiss.IndexFlatL2(len(doc_embeddings[0]))
    index.add(np.array(doc_embeddings))

    # Embed query
    query_vec = get_embedding(change_input)
    D, I = index.search(np.array([query_vec]), k=5)
    retrieved_texts = [texts[i] for i in I[0]]
    retrieved_context = "\n\n".join(retrieved_texts)

    # Build structured context
    context_snippets = "\n".join([
        f"Artifact: {a['ID']}\nName: {a['Name']}\nInputs: {a['Inputs (ISO Ref)']}\nOutputs: {a['Outputs (ISO Ref)']}\nContent: {a['Content (ISO Ref)']}"
        for a in artifacts if any(term in a['Name'].lower() for term in change_input.lower().split())
    ])

    # Compose prompt
    full_prompt = f"""
You are an expert in ISO 26262 safety engineering. A change has been proposed:

---
{change_input}
---

Relevant retrieved context from safety documents:
{retrieved_context}

Structured safety artifact map:
{context_snippets}

Based on this, list:
1. Safety artifacts potentially impacted by this change.
2. What needs to be re-validated or updated.
3. Any recommended actions to preserve safety compliance.
"""

    with st.spinner("🔍 GPT-4o is analyzing..."):
        response = openai.chat.completions.create(
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
