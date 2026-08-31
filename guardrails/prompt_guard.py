"""Content safety checker using Meta's Llama Prompt Guard 2."""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment,misc]


@dataclass
class PromptGuardResult:
    """Result from prompt guard classification."""
    is_safe: bool
    label: str  # "benign" or "malicious"
    confidence: float
    error: Optional[str] = None


async def check_content_safety(
    content: str,
    api_key: Optional[str] = None,
    model: str = "meta-llama/llama-prompt-guard-2-86m",
    threshold: float = 0.5,
) -> PromptGuardResult:
    """Check content for prompt injection/jailbreak attacks.
    
    Args:
        content: Text to check for safety
        api_key: Groq API key (if None, reads from GROQ_API_KEY env var)
        model: Model ID to use
        threshold: Threshold for malicious classification
        
    Returns:
        PromptGuardResult with safety classification
    """
    try:
        if Groq is None:
            logger.warning("groq package not installed")
            return PromptGuardResult(is_safe=True, label="benign", confidence=1.0, error="groq not installed")

        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            logger.warning("No Groq API key, skipping prompt guard")
            return PromptGuardResult(is_safe=True, label="benign", confidence=1.0, error="No API key")

        client = Groq(api_key=key)
        
        truncated = content[:2000]
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": truncated}],
        )
        
        response_text = completion.choices[0].message.content.strip().lower()
        
        if "malicious" in response_text:
            label = "malicious"
            is_safe = False
        else:
            label = "benign"
            is_safe = True
        
        confidence = 0.9 if not is_safe else 0.1
        
        return PromptGuardResult(
            is_safe=is_safe and confidence < threshold,
            label=label,
            confidence=confidence,
        )
        
    except Exception as e:
        logger.error("Prompt guard check failed: %s", e)
        return PromptGuardResult(is_safe=True, label="benign", confidence=1.0, error=str(e))


def check_content_safety_sync(content: str, **kwargs) -> PromptGuardResult:
    """Synchronous wrapper for check_content_safety."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(check_content_safety(content, **kwargs))
