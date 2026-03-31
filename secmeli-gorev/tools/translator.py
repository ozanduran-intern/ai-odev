"""
번역 도구
LLM을 활용한 다국어 번역 (10+ 언어 지원)
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import MODEL_NAME, BASE_URL

# 지원 언어 코드 → 언어명 매핑
SUPPORTED_LANGS: dict[str, str] = {
    "ko": "한국어 (Korean)",
    "en": "English",
    "tr": "Türkçe (Turkish)",
    "ja": "日本語 (Japanese)",
    "zh": "中文 (Chinese)",
    "es": "Español (Spanish)",
    "fr": "Français (French)",
    "de": "Deutsch (German)",
    "it": "Italiano (Italian)",
    "pt": "Português (Portuguese)",
    "ar": "العربية (Arabic)",
    "th": "ไทย (Thai)",
    "vi": "Tiếng Việt (Vietnamese)",
    "id": "Bahasa Indonesia",
    "ru": "Русский (Russian)",
}


@tool
def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between languages.
    Supported: ko, en, tr, ja, zh, es, fr, de, it, pt, ar, th, vi, id, ru
    Example: translate('hello', 'en', 'ko')"""
    try:
        src = source_lang.strip().lower()
        tgt = target_lang.strip().lower()

        if src not in SUPPORTED_LANGS:
            supported = ", ".join(f"{k}({v})" for k, v in SUPPORTED_LANGS.items())
            return f"'{source_lang}' is not supported. Supported: {supported}"
        if tgt not in SUPPORTED_LANGS:
            supported = ", ".join(f"{k}({v})" for k, v in SUPPORTED_LANGS.items())
            return f"'{target_lang}' is not supported. Supported: {supported}"
        if src == tgt:
            return f"Source and target language are the same ({SUPPORTED_LANGS[src]})."
        if not text.strip():
            return "No text to translate."

        llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL, temperature=0)

        src_name = SUPPORTED_LANGS[src]
        tgt_name = SUPPORTED_LANGS[tgt]

        messages = [
            SystemMessage(content=(
                f"You are a professional translator. "
                f"Translate the following text from {src_name} to {tgt_name}. "
                f"Return ONLY the translated text. No explanations, no notes."
            )),
            HumanMessage(content=text),
        ]

        response = llm.invoke(messages)
        translated = response.content.strip()

        return (
            f"🌐 Translation ({src_name} → {tgt_name}):\n"
            f"  Original: {text}\n"
            f"  Translated: {translated}"
        )

    except Exception as e:
        return f"Translation error: {str(e)}"
