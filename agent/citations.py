# from dataclasses import dataclass

# @dataclass
# class Citation:
#     source: str
#     competition: str | None
#     year: int | None
#     section: str | None
#     confidence: str

def extract_confidence(text: str) -> str:
    t = text.lower()
    if "shall" in t or "must" in t:
        return "must"
    elif "should" in t:
        return "should"
    elif "may" in t:
        return "may"
    return "unspecified"

def analyze_rule_docs(docs):
    analyzed = []
    for d in docs:
        analyzed.append({"text": d.page_content,
                         "metadata": d.metadata,
                         "confidence": extract_confidence(d.page_content)})
    return analyzed

