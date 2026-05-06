"""
Shared Gemini API client used by all three pipelines.
Handles API calls, extracts token usage metadata, and retries on rate limits.
"""

import time
import re
from google import genai
from google.genai import types
from google.genai.errors import ClientError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_client():
    """Create and return a Gemini client."""
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate(prompt: str, system_instruction: str = None, max_retries: int = 5) -> dict:
    """
    Send a prompt to Gemini and return the response with token metadata.
    Automatically retries on rate limit (429) errors.

    Args:
        prompt: The user query (with or without context)
        system_instruction: Optional system prompt
        max_retries: Max number of retries on rate limit errors

    Returns:
        dict with keys:
            - answer: str (the LLM response text)
            - input_tokens: int (prompt tokens used)
            - output_tokens: int (completion tokens used)
            - total_tokens: int (sum of input + output)
    """
    client = get_client()

    generate_config = types.GenerateContentConfig()
    if system_instruction:
        generate_config.system_instruction = system_instruction

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=generate_config,
            )

            # Extract token usage from response metadata
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0

            return {
                "answer": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Parse retry delay from error message if available
                wait_time = 60  # default wait
                match = re.search(r'retryDelay.*?(\d+)', str(e))
                if match:
                    wait_time = int(match.group(1)) + 5  # add buffer

                print(f"  Rate limited. Waiting {wait_time}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise non-rate-limit errors

    raise Exception(f"Failed after {max_retries} retries due to rate limiting. "
                    f"Try again in a few minutes.")
