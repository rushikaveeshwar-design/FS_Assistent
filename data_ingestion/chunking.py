from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_engg_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n","\n"," ","."]
    )
    return splitter.split_text(text)

def chunk_rule_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n","\n"]
    )
    return splitter.split_text(text)
