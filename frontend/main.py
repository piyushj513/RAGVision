from io import BytesIO
import uuid
import requests
import streamlit as st

# Page setup
st.set_page_config(page_title="RAGVision")
st.title("🧠 RAGVision")
st.caption("Upload an image or PDF to ask about its content / or just chat like a regular assistant.")

# Custom CSS for chat-style spinner
st.markdown(
    """
<style>
.loading-container {
    display: flex;
    justify-content: left;
    align-items: center;
    margin-bottom: 15px;
}
.loading-dot {
    width: 10px;
    height: 10px;
    background-color: #4a5568;
    border-radius: 50%;
    margin: 0 2px;
    animation: pulse 1.5s infinite ease-in-out;
}
.loading-dot:nth-child(2) { animation-delay: 0.5s; }
.loading-dot:nth-child(3) { animation-delay: 1s; }
@keyframes pulse {
    0%, 100% { transform: scale(0.8); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
}
</style>
""",
    unsafe_allow_html=True,
)

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

with st.sidebar:
    st.header("About RAGVision")
    st.markdown(
        """
        RAGVision uses Retrieval-Augmented Generation (RAG) to answer questions based on your uploaded documents.
        
        ### Supported file types:
        - PDF documents
        - Images (PNG, JPG)
        
        ### Tips for best results:
        - Upload clear, high-resolution documents
        - Use extractable pdfs or upload them as images
        - Ask specific questions about document content
        - For complex documents, ask one question at a time
        """
    )

    st.divider()

    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.uploaded_files = None
        st.rerun()


# Upload file
uploaded_files = st.file_uploader(
    "Upload a PDF or Image",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg"],
)

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload logic
if uploaded_files and uploaded_files != st.session_state.uploaded_files:
    st.session_state.uploaded_files = uploaded_files

    assistant_box = st.chat_message("assistant").empty()
    with assistant_box:
        with st.empty():
            # Animated spinner
            st.markdown(
                """
                <div class="loading-container">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            """,
                unsafe_allow_html=True,
            )
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    files=[
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ],
                    data={"session_id": st.session_state.session_id},
                )
                if response.status_code == 200:
                    st.toast("✅ File uploaded and indexed!")
                else:
                    st.toast("❌ Failed to process the file!")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.json(),
                    }
                )
                assistant_box.write(response.json())
            except Exception as e:
                print(f"Upload error: {e}")
                st.error("🚨 Error while uploading file.")


if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    assistant_box = st.chat_message("assistant").empty()

    try:
        with assistant_box:
            with st.empty():
                # Animated spinner
                st.markdown(
                    """
                    <div class="loading-container">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            # Stream response
            with requests.post(
                "http://localhost:8000/chat",
                data={
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
            ) as stream_response:
                for chunk in stream_response.iter_content(chunk_size=1024):
                    response_text += chunk.decode("utf-8")
                    assistant_box.write(response_text)

    except Exception as e:
        print(f"AI response error: {e}")
        response_text = "⚠️ Failed to get a response from the assistant."

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    assistant_box.write(response_text)
