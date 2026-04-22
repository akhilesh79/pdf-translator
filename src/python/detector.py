from langdetect import detect, LangDetectException


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "auto"
    try:
        return detect(text[:4000])
    except LangDetectException:
        return "auto"
