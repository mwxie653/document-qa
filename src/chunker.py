"""Split long text into overlapping chunks for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into chunks suitable for embedding and retrieval.

    Separators are ordered from most to least semantic for Chinese + English:
    paragraph breaks → sentence endings → clause markers → spaces.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",     # paragraph
            "\n",       # line break
            "。",       # Chinese period
            "！",       # Chinese exclamation
            "？",       # Chinese question
            "；",       # Chinese semicolon
            ".",        # English period
            "!",        # English exclamation
            "?",        # English question
            ";",        # English semicolon
            " ",        # space (last resort)
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)
