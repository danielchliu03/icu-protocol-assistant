# embeddings.py
# ----------------------------
# This file creates vector embeddings from text chunks
# Embeddings are numerical representations that capture meaning
# Similar text will have similar embeddings (vectors)
# ----------------------------

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

class EmbeddingManager:
    """
    Handles creating, saving, and loading text embeddings
    Think of this as a translator that converts text to numbers
    that the computer can understand and compare
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding model
        - model_name: The AI model that converts text to vectors
        - 'all-MiniLM-L6-v2' is fast and good for general text
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully!")
    
    def create_embeddings(self, text_chunks):
        """
        Convert a list of text chunks into embeddings (vectors)
        Parameters:
        - text_chunks: List of strings (from your chunker.py)
        Returns:
        - numpy array of embeddings (one vector per chunk)
        """
        print(f"Creating embeddings for {len(text_chunks)} chunks...")
        
        # Convert all chunks to embeddings at once (faster than one-by-one)
        embeddings = self.model.encode(text_chunks, show_progress_bar=True)
        
        print("✅ Embeddings created successfully!")
        return embeddings
    
    def save_embeddings(self, embeddings, text_chunks, save_path="embeddings_data.pkl"):
        """
        Save embeddings and their corresponding text to a file
        This lets you reuse them without recreating every time
        """
        data = {
            'embeddings': embeddings,
            'text_chunks': text_chunks,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Embeddings saved to: {save_path}")
    
    def load_embeddings(self, load_path="embeddings_data.pkl"):
        """
        Load previously saved embeddings from file
        Returns: (embeddings_array, text_chunks_list)
        """
        if not os.path.exists(load_path):
            print(f"❌ No embeddings found at: {load_path}")
            return None, None
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded {len(data['text_chunks'])} embeddings from: {load_path}")
        return data['embeddings'], data['text_chunks']

# ----------------------------
# TESTING SECTION
# ----------------------------
if __name__ == "__main__":
    # Test the embedding system with your ICU protocols
    from chunker import chunk_text
    from ingest import extract_text
    
    # Step 1: Extract and chunk text from PDF
    print("📄 Processing ICU protocol...")
    pdf_path = "protocols/UFH-Antibiotic-Protocol.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    
    # Step 2: Create embeddings
    print("\n🔢 Creating embeddings...")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.create_embeddings(chunks)
    
    # Step 3: Save for later use
    print("\n💾 Saving embeddings...")
    embedding_manager.save_embeddings(embeddings, chunks, "icu_protocol_embeddings.pkl")
    
    # Step 4: Show results
    print(f"\n📊 Results:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Embedding dimensions: {embeddings.shape}")
    print(f"   - First chunk preview: {chunks[0][:100]}...")