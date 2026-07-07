# chunker.py
# ----------------------------
# This file takes raw text from an ICU protocol PDF
# and splits it into smaller "chunks" that respect sentence/word boundaries
# ----------------------------

import re
from src.ingest import extract_text

def chunk_text_smart(text, max_chunk_size=500, overlap=100):
    """
    Splits long text into overlapping chunks that respect sentence boundaries.
    
    Parameters:
    - text (str): full extracted PDF text
    - max_chunk_size (int): maximum size of each chunk in characters
    - overlap (int): approximate number of characters to overlap between chunks
    
    Returns:
    - List[str]: list of text chunks with clean sentence boundaries
    """
    # Clean up the text first
    text = clean_text(text)
    
    # Split into sentences using multiple sentence ending patterns
    sentence_endings = r'[.!?]+[\s\n]+'
    sentences = re.split(sentence_endings, text)
    
    # Remove empty sentences and clean them up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # If adding this sentence would exceed max size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # Save the current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            current_chunk = get_overlap_text(current_chunk, overlap) + " " + sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
        
        i += 1
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle any chunks that are still too long by splitting on words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long chunks by words
            word_chunks = split_by_words(chunk, max_chunk_size, overlap)
            final_chunks.extend(word_chunks)
    
    return final_chunks

def clean_text(text):
    """
    Clean up PDF text to remove common extraction artifacts
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF issues
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add periods between sentences that got merged
    
    # Remove page headers/footers (common patterns)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\d+/\d+/\d+', '', text)  # Remove dates
    
    return text.strip()

def get_overlap_text(chunk, overlap_size):
    """
    Get the last portion of a chunk for overlap with the next chunk
    Tries to end at word boundaries
    """
    if len(chunk) <= overlap_size:
        return chunk
    
    # Get approximately the last overlap_size characters
    overlap_start = len(chunk) - overlap_size
    overlap_text = chunk[overlap_start:]
    
    # Try to start at a word boundary
    space_index = overlap_text.find(' ')
    if space_index > 0:
        overlap_text = overlap_text[space_index:].strip()
    
    return overlap_text

def split_by_words(text, max_size, overlap):
    """
    Split text that's too long by word boundaries
    """
    words = text.split()
    chunks = []
    current_chunk_words = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_size and current_chunk_words:
            # Save current chunk
            chunks.append(' '.join(current_chunk_words))
            
            # Start new chunk with overlap
            overlap_words = get_overlap_words(current_chunk_words, overlap)
            current_chunk_words = overlap_words + [word]
            current_length = sum(len(w) + 1 for w in current_chunk_words)
        else:
            current_chunk_words.append(word)
            current_length += word_length
    
    # Add the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
    
    return chunks

def get_overlap_words(words, overlap_chars):
    """
    Get the last few words that fit within overlap_chars
    """
    overlap_words = []
    current_length = 0
    
    # Work backwards through words
    for word in reversed(words):
        word_length = len(word) + 1
        if current_length + word_length <= overlap_chars:
            overlap_words.insert(0, word)
            current_length += word_length
        else:
            break
    
    return overlap_words

# Keep the old function name for compatibility
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Wrapper to maintain compatibility with existing code
    """
    return chunk_text_smart(text, chunk_size, overlap)

# ----------------------------
# TESTING SECTION
# ----------------------------
if __name__ == "__main__":
    # Test with your ICU protocol
    PDF_PATH = "protocols/UFH-Antibiotic-Protocol.pdf"
    
    print("📄 Testing improved chunker...")
    
    # Step 1: extract full text from PDF
    text = extract_text(PDF_PATH)
    print(f"Total text length: {len(text)} characters")
    
    # Step 2: chunk the text with improved method
    chunks = chunk_text_smart(text, max_chunk_size=500, overlap=100)
    
    # Step 3: show results
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\n" + "="*60)
    print("CHUNK EXAMPLES:")
    print("="*60)
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- CHUNK {i+1} ({len(chunk)} chars) ---")
        print(chunk)
        if i < len(chunks) - 1:
            print(f"\n[Overlap with next chunk: '{chunks[i+1][:50]}...']")
