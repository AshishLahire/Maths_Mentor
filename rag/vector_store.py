"""
Bhai, ye vector store setup hai.
Hum ChromaDB use karenge - NEW API wala version.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
import json

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Vector store initialize karo - NEW Chroma API
        """
        # Embedding model (free, local)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Chroma client setup - NEW PERSISTENT CLIENT
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Collection create/load karo
        try:
            self.collection = self.client.get_collection("math_knowledge")
            print("✅ Existing collection loaded")
        except:
            self.collection = self.client.create_collection(
                name="math_knowledge",
                metadata={"description": "JEE Math Knowledge Base"}
            )
            print("✅ New collection created")
        
        self.persist_directory = persist_directory
    
    def embed_text(self, text):
        """
        Text ko embeddings mein convert karo
        """
        # SentenceTransformer se embedding generate karo
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_documents(self, documents, metadatas=None, ids=None):
        """
        Documents ko vector store mein add karo
        """
        if not documents:
            return
        
        # Auto-generate ids if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Auto-generate metadatas if not provided
        if metadatas is None:
            metadatas = [{"type": "math_knowledge"} for _ in documents]
        
        # Generate embeddings
        embeddings = []
        for doc in documents:
            embedding = self.embed_text(doc)
            embeddings.append(embedding)
        
        # Add to collection - NEW API
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(self, query, top_k=5):
        """
        Query ke similar documents dhundho
        """
        try:
            # Query ka embedding banao
            query_embedding = self.embed_text(query)
            
            # Search in vector store - NEW API
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'score': 1 - results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0.9,
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []
    
    def get_all_documents(self):
        """
        Saare documents get karo
        """
        try:
            results = self.collection.get()
            return results
        except:
            return {'documents': []}
    
    def delete_all(self):
        """
        Saara data delete karo (testing ke liye)
        """
        try:
            self.client.delete_collection("math_knowledge")
            self.collection = self.client.create_collection("math_knowledge")
            print("✅ All documents deleted")
        except:
            print("⚠️ Could not delete collection")
    
    def save(self):
        """
        Persist karo disk par
        """
        # Chroma PersistentClient automatically saves
        print(f"✅ Vector store persisted to {self.persist_directory}")

# Global instance - DON'T CREATE HERE, create in function
vector_store = None

def get_vector_store():
    """Lazy initialization of vector store"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store

def initialize_knowledge_base():
    """
    Knowledge base ko initialize karo with default documents
    """
    vector_store = get_vector_store()
    
    # Check if already populated
    existing = vector_store.get_all_documents()
    if len(existing.get('documents', [])) > 0:
        print(f"✅ Knowledge base already has {len(existing['documents'])} documents")
        return
    
    # Create default knowledge documents
    documents = []
    metadatas = []
    ids = []
    
    # Load from knowledge_base directory if exists
    knowledge_dir = "rag/knowledge_base"
    if os.path.exists(knowledge_dir):
        print(f"📁 Loading knowledge from {knowledge_dir}")
        for file_name in os.listdir(knowledge_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(knowledge_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split into chunks (by sections)
                    sections = content.split('\n## ')
                    for i, section in enumerate(sections):
                        if section.strip():
                            # Add section title back
                            if i == 0:
                                chunk = section
                            else:
                                chunk = f"## {section}"
                            
                            documents.append(chunk)
                            metadatas.append({
                                "source": file_name,
                                "section": i,
                                "type": "math_knowledge"
                            })
                            ids.append(f"{file_name}_{i}")
    
    # Add some hardcoded knowledge too
    if not documents:
        print("📝 No knowledge files found, adding default knowledge...")
        default_docs = [
            "Quadratic formula: x = [-b ± √(b² - 4ac)] / 2a",
            "Derivative of x^n is nx^(n-1)",
            "Probability P(A) = n(A) / n(S)",
            "sin²θ + cos²θ = 1",
            "Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)",
            "Integration: ∫x^n dx = x^(n+1)/(n+1) + C",
            "Limit: lim_{x→0} sin x / x = 1",
            "Binomial coefficient: C(n,k) = n! / (k!(n-k)!)",
            "Matrix determinant 2x2: ad - bc",
            "Distance formula: √[(x2-x1)² + (y2-y1)²]",
            "Arithmetic progression nth term: a + (n-1)d",
            "Geometric progression sum: a(1 - r^n)/(1 - r)",
            "Conditional probability: P(A|B) = P(A∩B)/P(B)",
            "Bayes theorem: P(A|B) = [P(B|A)P(A)]/P(B)",
            "Product rule: d/dx(uv) = u'v + uv'",
            "Quotient rule: d/dx(u/v) = (u'v - uv')/v²",
            "Pythagorean theorem: a² + b² = c²",
            "Logarithm property: log(ab) = log(a) + log(b)",
            "Exponential property: e^(a+b) = e^a * e^b",
            "Sum of angles: sin(A+B) = sinA cosB + cosA sinB"
        ]
        
        for i, doc in enumerate(default_docs):
            documents.append(doc)
            metadatas.append({"type": "math_formula", "source": "default"})
            ids.append(f"default_{i}")
    
    # Add to vector store
    vector_store.add_documents(documents, metadatas, ids)
    vector_store.save()
    
    print(f"✅ Initialized knowledge base with {len(documents)} chunks")

if __name__ == "__main__":
    # Initialize and test
    initialize_knowledge_base()
    
    # Test search
    test_query = "How to solve quadratic equation?"
    results = vector_store.search(test_query, top_k=3)
    
    print(f"\n🔍 Test query: {test_query}")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result['content'][:100]}...")
        print(f"Score: {result['score']:.3f}")
