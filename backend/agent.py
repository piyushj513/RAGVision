import asyncio
from io import BytesIO
import os
from PyPDF2 import PdfReader
from PIL import Image
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput
from llama_index.core.workflow import Context
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD_PATH")


# Load environment variables
load_dotenv()

# Configure global settings for embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = AzureOpenAI(
    model=os.getenv("AZURE_OPENAI_MODEL"),
    engine=os.getenv("AZURE_OPENAI_ENGINE"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Create agent
workflow = AgentWorkflow.from_tools_or_functions(
    [],
    system_prompt=(
        "You are RAGVision, an intelligent assistant capable of answering questions "
        "based on user queries, uploaded documents, and images. Your core capabilities include:\n"
        "- Analyzing and extracting text from PDFs, images (via OCR), and plain text files.\n"
        "- Answering questions by searching through the uploaded content when available.\n"
        "- Providing helpful and accurate responses even when no documents are provided, using general knowledge.\n"
        "- Supporting multi-session interactions with context awareness.\n"
        "Always explain your answers clearly and concisely. If documents are available, refer to them explicitly."
    ),
)


# Dictionary to store contexts & index for different sessions
session_contexts = {}
session_indexes = {}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf = PdfReader(BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
        return text or "PDF appears to contain no extractable text."
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using OCR."""
    try:
        image = Image.open(BytesIO(file_bytes))
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print(f"Image extraction error: {e}")
        return ""


def get_file_text(file_bytes: bytes, content_type: str) -> str:
    """Extract text from files based on content type."""
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif content_type.startswith("image/"):
        return extract_text_from_image(file_bytes)
    elif content_type == "text/plain":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        print(f"Unsupported file type: {content_type}")
        return ""


def index_uploaded_files(files, session_id):
    """Processes uploaded files in-memory and stores session-specific indexes."""
    if not files:
        print("No files provided to index")
        return False

    print(f"Processing files for session {session_id}")

    temp_docs = []

    # Handle single file or list of files
    file_list = [files] if not isinstance(files, list) else files

    for file in file_list:
        if not file:
            continue

        try:
            print(f"Processing file: {file.filename}, type: {file.content_type}")
            # Reset file pointer to beginning
            file.file.seek(0)
            file_bytes = file.file.read()
            content_type = file.content_type
            text = get_file_text(file_bytes, content_type)

            if text:
                print(f"Extracted text length: {len(text)}")
                doc = Document(text=text, metadata={"filename": file.filename})
                temp_docs.append(doc)
            else:
                print(f"No text extracted from file: {file.filename}")
        except Exception as e:
            print(
                f"Error processing file {file.filename if hasattr(file, 'filename') else 'unknown'}: {str(e)}"
            )

    if temp_docs:
        print(f"Creating index from {len(temp_docs)} documents")
        new_index = VectorStoreIndex.from_documents(temp_docs)
        session_indexes[session_id] = new_index
        return True
    else:
        print("No documents were extracted from files")
        return False


async def agent_response_stream(query, session_id, files=None):
    """Handles query processing based on regular chatbot or uploaded files."""
    # Ensure the session has a context
    if session_id not in session_contexts:
        session_contexts[session_id] = Context(workflow)

    ctx = session_contexts[session_id]

    # Check if there's an index for this session
    has_index = (
        session_id in session_indexes and session_indexes[session_id] is not None
    )

    # If there's an index, use it for queries
    if has_index:
        print(f"Using document index for session {session_id}")
        try:
            file_index = session_indexes[session_id]
            query_engine = file_index.as_query_engine(streaming=True)
            streaming_response = query_engine.query(query)
            for text in streaming_response.response_gen:
                yield text
        except Exception as e:
            print(f"Error using document index: {str(e)}")
            yield f"Error retrieving information from documents: {str(e)}. Falling back to regular chatbot.\n"
            # Fall back to regular chatbot if document query fails
            handler = workflow.run(user_msg=query, ctx=ctx)
            async for event in handler.stream_events():
                if (
                    isinstance(event, AgentOutput)
                    and event.response
                    and hasattr(event.response, "content")
                ):
                    yield event.response.content
    else:
        # Default: Use regular chatbot workflow
        print(f"Using regular chatbot for session {session_id} (no index available)")
        handler = workflow.run(user_msg=query, ctx=ctx)
        async for event in handler.stream_events():
            if (
                isinstance(event, AgentOutput)
                and event.response
                and hasattr(event.response, "content")
            ):
                yield event.response.content
            await asyncio.sleep(0)


# Main function that can be called from the API
async def main(query, session_id, files=None):
    """Main function to handle chat responses."""
    print(
        f"Main function called with query: '{query}', session: {session_id}, files: {files is not None}"
    )
    async for chunk in agent_response_stream(query, session_id, files):
        yield chunk
