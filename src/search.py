# search.py
# ----------------------------
# This file handles searching through your ICU protocol embeddings
# When you ask a question, it finds the most relevant protocol sections
# ----------------------------

import numpy as np
from src.embeddings import EmbeddingManager
from sklearn.metrics.pairwise import cosine_similarity

class ProtocolSearcher:
    """
    Searches through ICU protocol embeddings to find relevant information
    Think of this as a smart search engine for your medical protocols
    """
    
    def __init__(self, embeddings_file="icu_protocol_embeddings.pkl"):
        """
        Initialize the searcher by loading saved embeddings
        - embeddings_file: Path to your saved embeddings
        """
        print("🔍 Loading protocol search system...")
        
        # Load the embedding model (same one used to create embeddings)
        self.embedding_manager = EmbeddingManager()
        
        # Load the saved embeddings and text chunks
        self.embeddings, self.text_chunks = self.embedding_manager.load_embeddings(embeddings_file)
        
        if self.embeddings is None:
            raise Exception(f"❌ Could not load embeddings from {embeddings_file}")
        
        print(f"✅ Loaded {len(self.text_chunks)} protocol sections for search!")
    
    def search(self, question, top_k=3):
        """
        Search for the most relevant protocol sections for a given question
        
        Parameters:
        - question (str): The clinical question you want to ask
        - top_k (int): How many relevant sections to return (default: 3)
        
        Returns:
        - List of tuples: [(similarity_score, text_chunk), ...]
        """
        print(f"🔎 Searching for: '{question}'")
        
        # Step 1: Convert the question into an embedding
        question_embedding = self.embedding_manager.model.encode([question])
        
        # Step 2: Calculate similarity between question and all protocol chunks
        # Cosine similarity measures how "similar" two vectors are (0 = different, 1 = identical)
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        
        # Step 3: Get the indices of the most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Step 4: Return the top results with their similarity scores
        results = []
        for i, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            text_chunk = self.text_chunks[idx]
            results.append((similarity_score, text_chunk))
            print(f"\n📄 Result {i+1} (similarity: {similarity_score:.3f}):")
            print(f"   {text_chunk[:150]}...")
        
        return results
    
    def answer_question(self, question, top_k=3):
        """
        Get the best protocol sections for a clinical question
        This is what you'll use most often in your ICU RAG system
        """
        results = self.search(question, top_k)
        
        # Combine the top results into context for the AI
        context_chunks = [chunk for score, chunk in results if score > 0.1]  # Only include reasonably relevant chunks
        
        if not context_chunks:
            return "❌ No relevant protocol information found for this question."
        
        # Join the relevant chunks together
        context = "\n\n---\n\n".join(context_chunks)
        return context

# ----------------------------
# TESTING SECTION
# ----------------------------
if __name__ == "__main__":
    # Test the search system with clinical questions
    print("🏥 Testing ICU Protocol Search System\n")
    
    # Initialize the searcher
    searcher = ProtocolSearcher()
    
    # Test with some common ICU questions
    test_questions = [
        "What antibiotic should I use for pneumonia?",
        "How long should antibiotic treatment last?",
        "What is the dosing for antibiotics?",
        "When do I need ID approval?"
    ]
    
    for question in test_questions:
        print("\n" + "="*60)
        context = searcher.answer_question(question, top_k=2)
        print(f"\n📋 RELEVANT PROTOCOL INFO:")
        print(context[:500] + "..." if len(context) > 500 else context)
        print("\n" + "="*60)