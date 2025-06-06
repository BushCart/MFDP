def text_splitter(text: str, max_length: int = 500, overlap: int = 50, tail = 50) -> list[str]:
    chunks = []
    count = 1
    start = 0
    while start<len(text):
        window = text[start:start+max_length+tail]
        cut_id = max(
            window.rfind("\n\n"),
            window.rfind("."), 
            window.rfind("\n")
                    )
        if cut_id == -1 or cut_id < 50: # я знаю
            chunk_end = start + max_length
        else:
            chunk_end = start + cut_id+1

        if chunk_end <= start:
            chunk_end = len(text)

        buffer_text = text[start:chunk_end].strip()
        if buffer_text:
            chunks.append({
                    "text": buffer_text,
                    "chunk_id": count,
                })       
        count+=1
        next_start = chunk_end - overlap
        start = max(chunk_end, next_start)
  
    return chunks
