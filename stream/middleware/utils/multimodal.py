"""
Multimodal message utilities for STREAM.

This module provides helper functions for working with OpenAI-format
multimodal messages — messages that can contain both text and images.

BACKGROUND: THE OPENAI VISION MESSAGE FORMAT
=============================================

In a text-only conversation, each message looks like this:

    {"role": "user", "content": "What is Python?"}

The "content" field is a simple string. But when images are involved,
the OpenAI vision format changes "content" to a LIST of content blocks:

    {"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    ]}

Each block has a "type" field:
  - "text"      → contains the text portion in a "text" field
  - "image_url" → contains the image data in an "image_url.url" field
                   (the "url" is a base64 data URL, not an external URL)

STREAM uses base64 data URLs (not external URLs) because:
  - Works offline (desktop mode)
  - No CORS issues
  - HPC compute nodes may not have outbound internet access
  - Images are self-contained in the message payload

WHY THIS MODULE EXISTS
======================

Multiple parts of the STREAM backend need to handle multimodal content:
  - chat.py: Validates and parses incoming messages
  - token_estimator.py: Counts tokens (must skip base64 data)
  - complexity_judge.py: Extracts text for the complexity judge
  - query_router.py: Checks if images are present for routing decisions
  - globus_compute_client.py: Validates payload size (base64 images are large)

By putting these helpers in a shared utility module, we avoid code
duplication and prevent circular imports between these modules.
"""


def extract_text_content(content: str | list[dict]) -> str:
    """
    Extract the text portion from a message's content field.

    This handles both formats:
      - String content (text-only): returns the string as-is
      - List content (multimodal): extracts and joins all text blocks

    Args:
        content: Either a plain string or a list of content blocks
                 in OpenAI vision format.

    Returns:
        The text content as a single string. If the content is a list,
        all text blocks are joined with spaces.

    Examples:
        >>> extract_text_content("What is Python?")
        'What is Python?'

        >>> extract_text_content([
        ...     {"type": "text", "text": "What is in this image?"},
        ...     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ... ])
        'What is in this image?'

        >>> extract_text_content([
        ...     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ... ])
        ''
    """
    # Simple case: content is already a plain string (text-only message)
    if isinstance(content, str):
        return content

    # Multimodal case: content is a list of typed blocks.
    # We only care about "text" blocks — image blocks don't contribute
    # to the text that gets sent to the complexity judge or logged.
    return " ".join(block.get("text", "") for block in content if block.get("type") == "text")


def has_images(messages: list[dict]) -> bool:
    """
    Check if ANY message in the conversation contains image content.

    This scans all messages (not just the latest one) because some models
    need to know if images appeared anywhere in the conversation history,
    not just in the current turn.

    Args:
        messages: List of message dictionaries, each with "role" and "content".

    Returns:
        True if at least one message contains an image_url content block.

    Examples:
        >>> has_images([{"role": "user", "content": "Hello"}])
        False

        >>> has_images([{"role": "user", "content": [
        ...     {"type": "text", "text": "What is this?"},
        ...     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ... ]}])
        True

    Performance note:
        This short-circuits on the first image found, so it's efficient
        even with long conversation histories.
    """
    for msg in messages:
        content = msg.get("content", "")

        # Only list-type content can contain images.
        # String content is always text-only.
        if isinstance(content, list) and any(block.get("type") == "image_url" for block in content):
            return True

    return False


def strip_old_images(messages: list[dict]) -> list[dict]:
    """
    Remove images from all messages EXCEPT the latest user message.

    WHY:
    Globus Compute has an 8 MB payload limit. A long conversation with
    multiple image-bearing messages can easily exceed this. By stripping
    images from older messages, we keep the payload small while preserving:
      - All text content (including the model's previous descriptions of images)
      - Images in the CURRENT user message (what the model needs to process)

    The model's prior text responses about old images provide sufficient
    context for follow-up questions. If the user asks "what about that first
    image?", the model can reference its own earlier description.

    For messages with mixed content (text + images), images are removed but
    text blocks are preserved. If ALL blocks were images (no text), the
    content becomes "(image)" as a placeholder.

    This function returns a NEW list — the original messages are not modified.

    Args:
        messages: Full conversation history in OpenAI format.

    Returns:
        A copy of messages with images stripped from all but the last user message.
    """
    if not messages:
        return messages

    # Find the index of the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    result = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")

        # Keep the last user message intact (images included)
        if i == last_user_idx:
            result.append(msg)
            continue

        # For other messages, strip images if content is a list
        if isinstance(content, list):
            text_blocks = [b for b in content if b.get("type") != "image_url"]
            if text_blocks:
                result.append({**msg, "content": text_blocks})
            else:
                result.append({**msg, "content": "(image)"})
        else:
            result.append(msg)

    return result


def count_images(messages: list[dict]) -> int:
    """
    Count the total number of images across all messages.

    This is used by:
      - token_estimator.py: to estimate image token costs
      - globus_compute_client.py: to validate payload size

    Args:
        messages: List of message dictionaries.

    Returns:
        Total number of image_url blocks across all messages.

    Example:
        >>> count_images([
        ...     {"role": "user", "content": [
        ...         {"type": "text", "text": "Compare these two images"},
        ...         {"type": "image_url", "image_url": {"url": "data:..."}},
        ...         {"type": "image_url", "image_url": {"url": "data:..."}}
        ...     ]}
        ... ])
        2
    """
    total = 0

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            total += sum(1 for block in content if block.get("type") == "image_url")

    return total
