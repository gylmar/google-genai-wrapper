#!/usr/bin/env python3
"""
Script to call Google Generative AI models (Gemini, Gemma, etc.) using the Google Generative AI SDK.

Supports all Google Generative AI models including Gemini and Gemma model families.

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY  # Required for Gemini Developer API (unless --api-key is passed)
    GEMINI_MODEL                      # Optional default model name (e.g. gemini-2.5-flash, gemma-3-27b-it)
    GOOGLE_CLOUD_PROJECT              # Required for Vertex AI mode (--vertexai)
    GOOGLE_CLOUD_LOCATION             # Optional for Vertex AI (defaults to us-central1)

Usage:
    python call_genai.py "Your prompt here"              # Uses GEMINI_MODEL env var if set
    python call_genai.py "Your prompt" --model gemini-2.5-flash
    python call_genai.py "Your prompt" --model gemma-3-27b-it
    python call_genai.py "Your prompt" --api-key YOUR_API_KEY
    python call_genai.py "Your prompt" --stream
    python call_genai.py "What's in this image?" --image photo.jpg
    python call_genai.py "Summarize this document" --file document.pdf
    python call_genai.py "Compare these images" --image img1.jpg --image img2.png
    python call_genai.py "Hello" --conversation-id my-chat  # Start/resume conversation
    python call_genai.py "Continue our chat" --conversation-id my-chat  # Continue thread
    python call_genai.py --list-models  # Discover available models
"""

import argparse
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Union

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Error: google-genai package not found. Please install it with: pip install google-genai")
    sys.exit(1)

# Try to import PIL for image support
try:
    import PIL.Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Directory for storing conversation history
CONVERSATIONS_DIR = Path.home() / ".genai_conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)


def get_client(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """
    Initialize and return a Google Generative AI client.
    
    Args:
        api_key: API key for Gemini Developer API (or use GEMINI_API_KEY env var)
        vertexai: Whether to use Vertex AI API instead of Gemini Developer API
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI)
    
    Returns:
        Initialized client
    """
    if vertexai:
        return genai.Client(
            vertexai=True,
            project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    else:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Provide --api-key or set GEMINI_API_KEY environment variable."
            )
        return genai.Client(api_key=api_key)


def list_available_models(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    filter_pattern: Optional[str] = None,
) -> List[str]:
    """
    List all available models from the API.
    
    Args:
        api_key: API key for Gemini Developer API (or use GEMINI_API_KEY env var)
        vertexai: Whether to use Vertex AI API instead of Gemini Developer API
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI)
        filter_pattern: Optional pattern to filter model names (e.g., 'gemma', 'gemini'). If None, shows all models.
    
    Returns:
        List of available model names
    """
    client = get_client(api_key, vertexai, project, location)
    
    try:
        models = []
        pager = client.models.list(config={'page_size': 100, 'query_base': True})

        def _iter_models(entry):
            """Normalize pager output into an iterable of model objects."""
            if hasattr(entry, "models") and entry.models:
                return entry.models
            if isinstance(entry, list):
                return entry
            return [entry]

        for entry in pager:
            for model in _iter_models(entry):
                if not model or not hasattr(model, "name") or not model.name:
                    continue

                model_name = model.name
                if '/' in model_name:
                    model_name = model_name.split('/')[-1]

                if not filter_pattern or filter_pattern.lower() in model_name.lower():
                    models.append(model_name)

        return sorted(set(models))
    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        raise


def prepare_contents(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
) -> List[Union[str, genai_types.Part]]:
    """
    Prepare contents for the API call, including text, images, and files.
    
    Args:
        prompt: The text prompt
        image_paths: List of image file paths to include
        file_paths: List of document file paths to include
    
    Returns:
        List of content parts (text and/or media)
    """
    contents = []
    
    # Add images and files first (if any)
    if image_paths:
        for image_path in image_paths:
            img_path = Path(image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if PIL_AVAILABLE:
                try:
                    # Try to load as PIL Image (handles most image formats)
                    img = PIL.Image.open(img_path)
                    contents.append(genai_types.Part(img))
                    continue
                except Exception:
                    pass
            
            # Fallback to reading as bytes
            with open(img_path, 'rb') as f:
                image_data = f.read()
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(img_path))
            if not mime_type or not mime_type.startswith('image/'):
                # Try to infer from extension
                ext = img_path.suffix.lower()
                mime_map = {
                    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.png': 'image/png', '.gif': 'image/gif',
                    '.webp': 'image/webp', '.bmp': 'image/bmp'
                }
                mime_type = mime_map.get(ext, 'image/jpeg')
            
            contents.append(genai_types.Part.from_bytes(data=image_data, mime_type=mime_type))
    
    if file_paths:
        for file_path in file_paths:
            file_p = Path(file_path)
            if not file_p.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(file_p))
            if not mime_type:
                # Common document MIME types
                ext = file_p.suffix.lower()
                mime_map = {
                    '.pdf': 'application/pdf',
                    '.txt': 'text/plain',
                    '.md': 'text/markdown',
                    '.csv': 'text/csv',
                    '.json': 'application/json',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                }
                mime_type = mime_map.get(ext, 'application/octet-stream')
            
            # Read file as bytes
            with open(file_p, 'rb') as f:
                file_data = f.read()
            
            contents.append(genai_types.Part.from_bytes(data=file_data, mime_type=mime_type))
    
    # Add text prompt last
    if prompt:
        contents.append(prompt)
    
    return contents


def save_conversation(conversation_id: str, history: List[genai_types.Content], model: str) -> None:
    """Save conversation history to a file."""
    conversation_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    data = {
        "conversation_id": conversation_id,
        "model": model,
        "history": [content.model_dump() for content in history]
    }
    with open(conversation_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_conversation(conversation_id: str) -> Optional[dict]:
    """Load conversation history from a file."""
    conversation_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if not conversation_file.exists():
        return None
    
    try:
        with open(conversation_file, 'r') as f:
            data = json.load(f)
        # Convert history dicts back to Content objects
        if "history" in data:
            data["history"] = [genai_types.Content.model_validate(h) for h in data["history"]]
        return data
    except Exception as e:
        print(f"Warning: Could not load conversation {conversation_id}: {e}", file=sys.stderr)
        return None


def call_genai(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    stream: bool = False,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Call a Google Generative AI model (Gemini, Gemma, etc.) with the given prompt, optionally including images and documents.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model name to use (e.g., 'gemini-2.5-flash', 'gemma-3-27b-it')
        api_key: API key for Gemini Developer API (or use GEMINI_API_KEY env var)
        stream: Whether to stream the response
        vertexai: Whether to use Vertex AI API instead of Gemini Developer API
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI)
        image_paths: List of image file paths to include
        file_paths: List of document file paths to include
        conversation_id: Optional conversation ID to maintain thread context
    
    Returns:
        Tuple of (response_text, conversation_id)
    """
    client = get_client(api_key, vertexai, project, location)
    
    # Prepare contents (text + images + files)
    contents = prepare_contents(prompt, image_paths, file_paths)
    
    # Use chat API if conversation_id is provided, otherwise use direct generate_content
    use_chat = conversation_id is not None
    
    if use_chat:
        # Load existing conversation or create new one
        conversation_data = load_conversation(conversation_id) if conversation_id else None
        
        if conversation_data:
            # Resume existing conversation
            history = conversation_data.get("history", [])
            # Use the model from the conversation if it was saved
            if "model" in conversation_data:
                model = conversation_data["model"]
        else:
            # New conversation
            history = []
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
        
        # Create chat session
        chat = client.chats.create(model=model, history=history)
        
        try:
            if stream:
                # Stream the response
                print("Response (streaming):\n")
                full_response = ""
                for chunk in chat.send_message_stream(contents):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        full_response += chunk.text
                print("\n")  # New line after streaming
                
                # Save conversation history
                save_conversation(conversation_id, chat.get_history(curated=True), model)
                return full_response, conversation_id
            else:
                # Get complete response
                response = chat.send_message(contents)
                response_text = response.text
                
                # Save conversation history
                save_conversation(conversation_id, chat.get_history(curated=True), model)
                return response_text, conversation_id
        except Exception as e:
            print(f"Error calling model: {e}", file=sys.stderr)
            raise
    else:
        # Direct API call (no conversation context)
        try:
            if stream:
                # Stream the response
                print("Response (streaming):\n")
                full_response = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        full_response += chunk.text
                print("\n")  # New line after streaming
                return full_response, None
            else:
                # Get complete response
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
                return response.text, None
        except Exception as e:
            print(f"Error calling model: {e}", file=sys.stderr)
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Call Google Generative AI models (Gemini, Gemma, etc.) using Google Generative AI SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  GEMINI_API_KEY or GOOGLE_API_KEY  Required for Gemini Developer API (unless --api-key is passed)
  GEMINI_MODEL                      Optional default model name (e.g. gemini-2.5-flash, gemma-3-27b-it)
  GOOGLE_CLOUD_PROJECT              Required for Vertex AI mode (--vertexai)
  GOOGLE_CLOUD_LOCATION             Optional for Vertex AI (defaults to us-central1)

Examples:
  python call_genai.py "Explain quantum computing"  # Uses GEMINI_MODEL env var if set
  python call_genai.py "Write a poem" --model gemini-2.5-flash
  python call_genai.py "Write a poem" --model gemma-3-27b-it
  python call_genai.py "Hello" --stream
  python call_genai.py "What's in this image?" --image photo.jpg
  python call_genai.py "Summarize this" --file document.pdf
  python call_genai.py "Compare these" --image img1.jpg --image img2.png --file doc.pdf
  python call_genai.py "Hello" --conversation-id my-chat  # Start conversation
  python call_genai.py "Continue" --conversation-id my-chat  # Continue thread
  python call_genai.py --list-models  # Discover available models
  python call_genai.py --list-models --filter gemma  # Filter by pattern
        """,
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The text prompt to send to the model",
    )
    
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("GEMINI_MODEL"),
        help="The model to use (e.g., 'gemini-2.5-flash', 'gemma-3-27b-it'). Defaults to GEMINI_MODEL env var if set. Use --list-models to see available models.",
    )
    
    parser.add_argument(
        "--api-key",
        "-k",
        help="API key for Gemini Developer API (or set GEMINI_API_KEY env var)",
    )
    
    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Stream the response as it's generated",
    )
    
    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="Use Vertex AI API instead of Gemini Developer API",
    )
    
    parser.add_argument(
        "--project",
        help="Google Cloud project ID (required for Vertex AI)",
    )
    
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud location (for Vertex AI, default: us-central1)",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models from the API and exit",
    )
    
    parser.add_argument(
        "--filter",
        help="Filter model names by pattern when using --list-models (e.g., 'gemma', 'gemini')",
    )
    
    parser.add_argument(
        "--conversation-id",
        "--convo-id",
        dest="conversation_id",
        help="Conversation ID to maintain thread context. If not provided, a new ID will be generated. Use the same ID to continue a conversation.",
    )
    
    parser.add_argument(
        "--image",
        "-i",
        action="append",
        dest="image_paths",
        help="Path to an image file to include (can be used multiple times for multiple images). Supports: JPEG, PNG, GIF, WebP, BMP",
    )
    
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        dest="file_paths",
        help="Path to a document file to include (can be used multiple times for multiple files). Supports: PDF, TXT, MD, CSV, JSON, DOCX, DOC",
    )
    
    args = parser.parse_args()
    
    # Handle list models command
    if args.list_models:
        try:
            models = list_available_models(
                api_key=args.api_key,
                vertexai=args.vertexai,
                project=args.project,
                location=args.location,
                filter_pattern=args.filter,
            )
            
            if args.filter:
                print(f"Available models (filtered by '{args.filter}'):")
            else:
                print("All available models:")
            
            if models:
                for model in models:
                    print(f"  - {model}")
            else:
                print("No models found. Make sure your API key is set correctly.")
        except Exception as e:
            print(f"Error listing models: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Require prompt if not listing models
    if not args.prompt:
        parser.error("prompt is required unless using --list-models")
    
    # Require model name if not provided
    if not args.model:
        parser.error("--model is required or set GEMINI_MODEL environment variable. Use --list-models to discover available models.")
    
    try:
        response, conversation_id = call_genai(
            prompt=args.prompt,
            model=args.model,
            api_key=args.api_key,
            stream=args.stream,
            vertexai=args.vertexai,
            project=args.project,
            location=args.location,
            image_paths=args.image_paths,
            file_paths=args.file_paths,
            conversation_id=args.conversation_id,
        )
        
        if not args.stream:
            print("\nResponse:\n")
            print(response)
        
        # Output conversation ID for reference
        if conversation_id:
            print(f"\n[Conversation ID: {conversation_id}]")
            print(f"To continue this conversation, use: --conversation-id {conversation_id}")
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
