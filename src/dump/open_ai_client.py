import os
import time
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class ChannelInsightsClient:
    """
    Client to interact with OpenAI for YouTube channel insights.
    Generates structured metadata and embeddings.
    """

    def __init__(
        self,
        chat_model: str = "gpt-4.1",
        embeddings_model: str = "text-embedding-3-large",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        template: Optional[Dict[str, Any]] = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.template = template or {}  # <-- pass template here

    def _extract_text(self, response: Any) -> str:
        """Extract text from OpenAI Responses API output."""
        # Note: The response structure might depend on the OpenAI SDK version.
        # This implementation follows the user's provided pattern.
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        text_output = ""
        # Accessing choices and message content as per standard OpenAI SDK
        if hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "message"):
                text_output = response.choices[0].message.content or ""
        
        # Fallback to user's provided nested structure if it exists
        if not text_output:
            for item in getattr(response, "output", []):
                for content in item.get("content", []):
                    text_output += content.get("text", "")
        
        return text_output.strip() or "No text found in response."

    def describe_channel(
        self,
        channel_name: str,
        context: Optional[str] = None,
        structured: bool = True,  # always True now
    ) -> str:
        """Generate a structured JSON description using the template."""
        system_prompt = "You are a helpful assistant that summarizes information about YouTube channels."

        template_str = json.dumps(self.template, indent=4) if self.template else "{}"
        user_prompt = (
            f"Create a structured JSON description for the YouTube channel '{channel_name}'. "
            f"Use the following JSON structure exactly, filling in all fields with factual information:\n\n{template_str} "
            "Exclude any adult, violent, spammy, or inappropriate content."
        )

        if context:
            user_prompt += f"\n\nRelevant context: {context}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # User's provided code uses self.client.responses.create
                # However, standard OpenAI chat completions use self.client.chat.completions.create
                # I'll try to support both or stick to user's provided code if it's a custom/special client
                if hasattr(self.client, "responses"):
                    response = self.client.responses.create(
                        model=self.chat_model,
                        input=messages,
                        temperature=0.4,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=0.4,
                    )
                
                summary = self._extract_text(response)
                if summary:
                    return " ".join(summary.split())
            except Exception as exc:
                last_error = exc
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Failed to describe channel after {self.max_retries} retries: {last_error}"
        )

    def describe_channel_json(self, channel_name: str, context: Optional[str] = None) -> str:
        """Generate structured JSON metadata for the channel."""
        raw_text = self.describe_channel(channel_name, context=context, structured=True)
        try:
            # Try to find JSON block if raw_text contains preamble
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            if json_start != -1 and json_end != 0:
                json_part = raw_text[json_start:json_end]
                json.loads(json_part)
                return json_part
            
            json.loads(raw_text)
            return raw_text
        except json.JSONDecodeError:
            fallback = self.template.copy() if self.template else {"overview": raw_text}
            fallback["overview"] = raw_text
            return json.dumps(fallback, ensure_ascii=False, indent=4)

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for the provided text."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text must be non-empty for embedding generation.")

        response = self.client.embeddings.create(model=self.embeddings_model, input=cleaned)
        
        return response.data[0].embedding

    def close(self) -> None:
        """Close OpenAI client if possible."""
        try:
            self.client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()


class ChannelSemanticClient:
    """Client for generating semantic embeddings and search text."""
    
    def __init__(self, insights_client: Optional[ChannelInsightsClient] = None):
        self.insights_client = insights_client or ChannelInsightsClient()

    def generate_channel_embedding(self, channel_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate full semantic text for search."""
        meta = channel_payload.get("channel_metadata", {})
        structured = channel_payload.get("channel_description_structured", {})
        
        # Combine fields into a searchable blob
        semantic_parts = [
            f"Title: {meta.get('title', '')}",
            f"Description: {meta.get('description', '')}",
            f"Topics: {', '.join(structured.get('content_topics', []))}",
            f"Value Proposition: {structured.get('value_proposition', '')}",
            f"Tone: {structured.get('tone_style', '')}",
        ]
        semantic_text = " ".join(semantic_parts)
        
        channel_payload["semantic_text"] = semantic_text
        return channel_payload

    def close(self):
        self.insights_client.close()
