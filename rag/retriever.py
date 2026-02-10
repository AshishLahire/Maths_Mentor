"""
Bhai, ye retriever module hai.
Iska kaam hai RAG pipeline mein relevant context retrieve karna.
"""

from .vector_store import get_vector_store, initialize_knowledge_base
import re

class Retriever:
    def __init__(self):
        # Initialize knowledge base if not already
        initialize_knowledge_base()
        
        # Query enhancement patterns
        self.math_patterns = {
            'derivative': ['differentiate', 'derivative', 'rate of change'],
            'integral': ['integral', 'integration', 'antiderivative'],
            'probability': ['probability', 'chance', 'likelihood'],
            'algebra': ['solve', 'equation', 'polynomial'],
            'matrix': ['matrix', 'determinant', 'eigenvalue']
        }
    
    def enhance_query(self, query):
        """
        Query ko enhance karo better retrieval ke liye
        """
        enhanced = query.lower()
        
        # Add relevant math terms
        for term, synonyms in self.math_patterns.items():
            if any(syn in enhanced for syn in synonyms):
                if term not in enhanced:
                    enhanced = f"{enhanced} {term}"
        
        # Clean query
        enhanced = re.sub(r'[^\w\s\.\?]', ' ', enhanced)
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()
        
        return enhanced
    
    def retrieve(self, query, top_k=5):
        """
        Main retrieval function
        """
        try:
            # Enhance query
            enhanced_query = self.enhance_query(query)
            
            # Get vector store
            vector_store = get_vector_store()
            
            # Search in vector store
            results = vector_store.search(enhanced_query, top_k=top_k)
            
            # Format results
            formatted_context = []
            for result in results:
                formatted_context.append({
                    'content': result['content'],
                    'score': result['score'],
                    'source': result['metadata'].get('source', 'unknown'),
                    'metadata': result['metadata']
                })
            
            return formatted_context
            
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            # Return empty context instead of crashing
            return []
    
    def get_relevant_formulas(self, query):
        """
        Specific formulas retrieve karo query ke liye
        """
        results = self.retrieve(query, top_k=3)
        
        # Filter for formulas
        formulas = []
        for result in results:
            if any(keyword in result['content'].lower() for keyword in 
                   ['formula', '=', 'derivative', 'integral', 'probability']):
                formulas.append(result['content'])
        
        return formulas

# Global instance
retriever = Retriever()

def retrieve_context(query):
    """Simple function for main app"""
    return retriever.retrieve(query)
