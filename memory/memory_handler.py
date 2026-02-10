"""
Bhai, ye memory handler hai.
Ye database ko use karke memory operations handle karega.
"""

import json
import datetime
from .database import MemoryDatabase

class MemoryHandler:
    def __init__(self, db_path="math_mentor_memory.db"):
        """
        Memory handler initialize karo
        """
        self.db = MemoryDatabase(db_path)
        self.last_id = None
    
    def store(self, original_input, parsed_problem, solution, verification, explanation):
        """
        Complete problem-solving session store karo - FIXED VERSION
        """
        try:
            # Add timestamp to parsed_problem
            parsed_problem['parsed_at'] = datetime.datetime.now().isoformat()
            
            # Step 1: Store problem
            problem_id = self.db.store_problem(original_input, parsed_problem)
            self.last_id = problem_id
            
            # Step 2: Store solution
            solution_id = self.db.store_solution(problem_id, solution)
            
            # Step 3: Store verification - CREATE COMPLETE VERIFICATION DICT
            # Database expects these exact fields:
            # 1. solution_id (already passed)
            # 2. is_correct (boolean)
            # 3. confidence (float)
            # 4. issues (JSON string of list)
            # 5. warnings (JSON string of list)
            # 6. requires_human_review (boolean)
            
            # Ensure all required fields exist with proper defaults
            complete_verification = {
                'is_correct': bool(verification.get('is_correct', False)),
                'confidence': float(verification.get('confidence', 0.5)),
                'issues': list(verification.get('issues', [])),  # Must be list
                'warnings': list(verification.get('warnings', [])),  # Must be list
                'requires_human_review': bool(verification.get('requires_human_review', False)),
                'suggestions': list(verification.get('suggestions', []))  # Optional field
            }
            
            # Debug print to see what we're storing
            print(f"DEBUG Verification to store:")
            print(f"  is_correct: {complete_verification['is_correct']}")
            print(f"  confidence: {complete_verification['confidence']}")
            print(f"  issues: {complete_verification['issues']}")
            print(f"  warnings: {complete_verification['warnings']}")
            print(f"  requires_human_review: {complete_verification['requires_human_review']}")
            
            # Store verification
            self.db.store_verification(solution_id, complete_verification)
            
            print(f"✅ Memory stored: Problem ID {problem_id}, Solution ID {solution_id}")
            return problem_id
            
        except Exception as e:
            print(f"❌ Memory storage error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_feedback(self, problem_id, is_correct, feedback=""):
        """
        User feedback add karo - FIXED VERSION
        """
        try:
            # Find the latest solution for this problem
            self.db.cursor.execute('''
                SELECT id FROM solutions 
                WHERE problem_id = ? 
                ORDER BY id DESC 
                LIMIT 1
            ''', (problem_id,))
            
            result = self.db.cursor.fetchone()
            if result:
                solution_id = result[0]
                self.db.store_feedback(
                    solution_id, 
                    is_correct, 
                    feedback,
                    given_by="user"
                )
                print(f"✅ Feedback stored for solution ID: {solution_id}")
                return True
            else:
                print(f"⚠️ No solution found for problem ID {problem_id}")
                return False
        
        except Exception as e:
            print(f"❌ Feedback storage error: {e}")
            return False
    
    def find_similar(self, problem_text, limit=3):
        """
        Similar problems find karo
        """
        try:
            similar = self.db.find_similar_problems(problem_text, limit)
            return similar
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []
    
    def get_all(self, limit=20):
        """
        Saare stored problems get karo
        """
        try:
            self.db.cursor.execute('''
                SELECT p.id, p.problem_text, p.topic, p.created_at,
                       s.final_answer
                FROM problems p
                LEFT JOIN solutions s ON p.id = s.problem_id
                ORDER BY p.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            rows = self.db.cursor.fetchall()
            
            problems = []
            for row in rows:
                problems.append({
                    'id': row[0],
                    'problem': row[1],
                    'topic': row[2],
                    'timestamp': row[3],
                    'answer': row[4]
                })
            
            return problems
        
        except Exception as e:
            print(f"❌ Get all error: {e}")
            return []
    
    def get_stats(self):
        """
        Statistics get karo - FIXED FOR STREAMLIT
        """
        try:
            stats = self.db.get_stats()
            
            # Ensure 'total' key exists
            if 'total' not in stats:
                if 'total_problems' in stats:
                    stats['total'] = stats['total_problems']
                else:
                    stats['total'] = 0
            
            # Ensure 'correct' key exists  
            if 'correct' not in stats:
                if 'correct_solutions' in stats:
                    stats['correct'] = stats['correct_solutions']
                else:
                    stats['correct'] = 0
            
            return stats
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            # Return default stats
            return {'total': 0, 'correct': 0, 'total_problems': 0, 'correct_solutions': 0}
    
    def close(self):
        """
        Cleanup karo
        """
        self.db.close()


# ============================================================================
# SIMPLE TEST FUNCTION - Add this to debug
# ============================================================================
def test_memory_storage():
    """Test memory storage independently"""
    print("\n" + "="*60)
    print("TESTING MEMORY STORAGE INDEPENDENTLY")
    print("="*60)
    
    memory = MemoryHandler(":memory:")
    
    # Test storing with COMPLETE verification data
    test_parsed = {
        'problem_text': 'Test problem 2x + 5 = 13',
        'topic': 'algebra',
        'variables': ['x'],
        'constraints': []
    }
    
    test_solution = {
        'solution_steps': ['Step 1: 2x + 5 = 13', 'Step 2: 2x = 8', 'Step 3: x = 4'],
        'final_answer': '4',
        'formulas_used': ['linear equation solving'],
        'method': 'algebra'
    }
    
    # COMPLETE verification dict with ALL required fields
    test_verification = {
        'is_correct': True,
        'confidence': 0.95,
        'issues': [],      # REQUIRED FIELD - MUST BE LIST
        'warnings': [],    # REQUIRED FIELD - MUST BE LIST
        'requires_human_review': False,  # REQUIRED FIELD
        'suggestions': ['Check calculation']  # Optional
    }
    
    # Store the complete test case
    problem_id = memory.store(
        original_input="Test input: 2x + 5 = 13",
        parsed_problem=test_parsed,
        solution=test_solution,
        verification=test_verification,
        explanation={'explanation': 'Test explanation'}
    )
    
    if problem_id:
        print(f"\n✅ SUCCESS: Stored problem ID: {problem_id}")
        
        # Test retrieval
        similar = memory.find_similar("Test problem")
        print(f"Similar problems found: {len(similar)}")
        
        # Test stats
        stats = memory.get_stats()
        print(f"Stats: {stats}")
        
        memory.close()
        return True
    else:
        print(f"\n❌ FAILED: Could not store problem")
        memory.close()
        return False


if __name__ == "__main__":
    # Run independent test
    success = test_memory_storage()
    
    if success:
        print("\n" + "="*60)
        print("✅ MEMORY SYSTEM TEST PASSED!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ MEMORY SYSTEM TEST FAILED")
        print("="*60)
