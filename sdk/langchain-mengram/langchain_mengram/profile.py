"""Mengram Cognitive Profile utilities for LangChain."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_mengram_profile(
    api_key: Optional[str] = None,
    user_id: str = "default",
    api_url: str = "https://mengram.io",
    force: bool = False,
) -> str:
    """Get a Cognitive Profile as a system prompt string.

    One API call generates a ready-to-use system prompt from all three
    memory types. Cached for 1 hour on the server side.

    Args:
        api_key: Mengram API key. Falls back to MENGRAM_API_KEY env var.
        user_id: User to generate profile for.
        api_url: Mengram API base URL.
        force: Force regeneration (ignore cache).

    Returns:
        System prompt string, or empty string if no data.
    """
    from mengram import Mengram

    client = Mengram(api_key=api_key, base_url=api_url)
    try:
        profile = client.get_profile(user_id, force=force)
        return profile.get("system_prompt", "")
    except Exception as e:
        logger.warning("Failed to get Mengram profile: %s", e)
        return ""
