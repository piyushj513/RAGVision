from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.agent import main, index_uploaded_files

# Setup app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_plain_text(input_text: str) -> bool:
    """Check if the input is simple text and not structured like JSON, SQL, etc."""
    patterns = {
        "JSON": r"^\s*[\{\[][\s\S]*[\}\]]\s*$",
        "XML": r"<[^>]+>.*<\/[^>]+>",
        "SQL": r"\b(SELECT|UPDATE|DELETE|INSERT)\b",
        "YAML": r"^\s*[a-zA-Z0-9_-]+\s*:\s*[^\n]+$",
        "HTML": r"<(html|head|body|div|span|img)[^>]*>",
        "Markdown Table": r"^\|(.+)\|$",
        "Base64": r"^[A-Za-z0-9+/=]{10,}={0,2}$",
    }

    return not any(re.search(pattern, input_text) for pattern in patterns.values())

@app.post("/chat")
async def chat(
    session_id: Annotated[str, Form()],
    query: Annotated[str, Form()] = "",
    files: Annotated[List[UploadFile] | None, File()] = None,
):
    try:
        # Process uploaded files (if any)
        if files:
            result = index_uploaded_files(files, session_id)
            if result:
                return JSONResponse(
                    "Your document has been processed! Ask me anything about it. 🤓",
                    status_code=200,
                )
            else:
                return JSONResponse(
                    "❌ Failed to process files!",
                    status_code=400,
                )

        # Validate input
        if not is_plain_text(query):
            return StreamingResponse(
                (chunk for chunk in ["⚠️ Unsupported input format. Please enter a clear text question."]),
                media_type="text/plain",
            )

        # Generate response from model
        return StreamingResponse(
            main(query=query, session_id=session_id, files=files),
            media_type="text/plain",
        )

    except Exception as e:
        return StreamingResponse(
            (chunk for chunk in [f"🚨 An unexpected error occurred: {e}"]),
            status_code=500,
            media_type="text/plain",
        )
