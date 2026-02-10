import json
import os
from datetime import datetime

class MemoryHandler:
    """Simple memory storage for problems and solutions"""
    
    def __init__(self, storage_file="memory_data.json"):
        self.storage_file = storage_file
        self.memory = self._load_memory()
        self.last_id = None
        print("✅ Simple memory handler initialized")
    
    def _load_memory(self):
        """Load memory from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    def store(self, original_input, parsed_problem, solution, verification, explanation):
        """Store a problem and solution"""
        try:
            entry = {
                'id': len(self.memory) + 1,
                'timestamp': datetime.now().isoformat(),
                'original_input': original_input,
                'parsed_problem': parsed_problem,
                'solution': solution,
                'verification': verification,
                'explanation': explanation
            }
            
            self.memory.append(entry)
            self._save_memory()
            self.last_id = entry['id']
            
            print(f"✅ Memory stored: ID {entry['id']}")
            return entry['id']
            
        except Exception as e:
            print(f"❌ Error storing memory: {e}")
            return None
    
    def get_all(self, limit=10):
        """Get all stored memories"""
        return self.memory[-limit:] if self.memory else []
    
    def get_stats(self):
        """Get statistics"""
        total = len(self.memory)
        correct = sum(1 for entry in self.memory 
                     if entry.get('verification', {}).get('is_correct', False))
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct/total if total > 0 else 0
        }
    
    def add_feedback(self, problem_id, is_correct, feedback):
        """Add user feedback"""
        for entry in self.memory:
            if entry.get('id') == problem_id:
                entry['user_feedback'] = {
                    'is_correct': is_correct,
                    'feedback': feedback,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_memory()
                return True
        return False
    
    def find_similar(self, problem_text, limit=3):
        """Find similar problems (simple keyword matching)"""
        if not self.memory:
            return []
        
        problem_words = set(problem_text.lower().split())
        similarities = []
        
        for entry in self.memory[-20:]:  # Check recent entries
            stored_text = entry.get('parsed_problem', {}).get('problem_text', '')
            stored_words = set(stored_text.lower().split())
            
            # Simple word overlap
            overlap = len(problem_words.intersection(stored_words))
            if overlap > 0:
                similarities.append((overlap, entry))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:limit]]
