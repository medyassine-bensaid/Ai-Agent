"""
grounding.py - Deterministic guardrail for source attribution.

The answer prompt tells the model to cite only sources present in the gathered
information, but models still occasionally "remember" plausible URLs (observed: citing
helm.sh after the web search had failed). Any URL in the final answer that does not
appear in the tool evidence or in the user's own message is replaced, so the agent can
never present a link it did not actually retrieve.
"""

import re

URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"'`|]+")
UNVERIFIED = "[unverified link removed]"


def _clean(url: str) -> str:
    return url.rstrip(".,;:!?*_")


def strip_ungrounded_urls(answer: str, *allowed_texts: str) -> tuple:
    """Return (answer_without_ungrounded_urls, removed_urls).

    A URL is grounded if it appears in one of `allowed_texts` (tool evidence, user
    question), or if it is a prefix/extension of such a URL.
    """
    allowed = [_clean(u) for text in allowed_texts for u in URL_RE.findall(text or "")]
    removed = []

    def replace(match):
        url = _clean(match.group(0))
        trailing = match.group(0)[len(url) :]
        if any(a == url or a.startswith(url) or url.startswith(a) for a in allowed):
            return match.group(0)
        removed.append(url)
        return UNVERIFIED + trailing

    return URL_RE.sub(replace, answer or ""), removed
