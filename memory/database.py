"""
Bhai, ye SQLite database setup hai.
Memory store ke liye use karenge.
"""

import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional

class MemoryDatabase:
    def __init__(self, db_path: str = "math_mentor_memory.db"):
        """
        SQLite database initialize karo
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.init_database()
        
    def get_connection(self):
        """Get fresh connection for current thread"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        return conn, cursor
    
    def init_database(self):
        """Database tables create karo - WITH DROP AND RECREATE"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Problems table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_input TEXT,
                problem_text TEXT,
                topic TEXT,
                variables TEXT,
                constraints TEXT,
                parsed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Solutions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER,
                solution_text TEXT,
                final_answer TEXT,
                formulas_used TEXT,
                method TEXT,
                solved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (problem_id) REFERENCES problems (id)
            )
        ''')
        
        # Verification table - DROP OLD AND CREATE NEW
        self.cursor.execute('DROP TABLE IF EXISTS verifications')
        self.cursor.execute('''
            CREATE TABLE verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER,
                is_correct BOOLEAN,
                confidence REAL,
                issues TEXT,
                warnings TEXT,
                requires_human_review BOOLEAN,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (solution_id) REFERENCES solutions (id)
            )
        ''')
        
        # Feedback table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER,
                is_correct BOOLEAN,
                feedback_text TEXT,
                corrected_answer TEXT,
                given_by TEXT DEFAULT 'user',
                given_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (solution_id) REFERENCES solutions (id)
            )
        ''')
        
        self.conn.commit()
        print(f"✅ Database initialized at {self.db_path}")
        print(f"✅ Verification table created fresh")
    
    def store_problem(self, original_input: str, parsed_problem: Dict) -> int:
        """
        Problem store karo database mein
        Returns: problem_id
        """
        # Convert lists to JSON strings
        variables_json = json.dumps(parsed_problem.get('variables', []))
        constraints_json = json.dumps(parsed_problem.get('constraints', []))
        
        # Get timestamp
        parsed_at = parsed_problem.get('parsed_at', datetime.datetime.now().isoformat())
        
        self.cursor.execute('''
            INSERT INTO problems 
            (original_input, problem_text, topic, variables, constraints, parsed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            original_input,
            parsed_problem.get('problem_text', ''),
            parsed_problem.get('topic', 'unknown'),
            variables_json,
            constraints_json,
            parsed_at
        ))
        
        self.conn.commit()
        problem_id = self.cursor.lastrowid
        print(f"✅ Stored problem with ID: {problem_id}")
        return problem_id
    
    def store_solution(self, problem_id: int, solution: Dict) -> int:
        """
        Solution store karo
        Returns: solution_id
        """
        # Convert to JSON strings
        formulas_json = json.dumps(solution.get('formulas_used', []))
        solution_text = json.dumps(solution.get('solution_steps', []))
        final_answer = str(solution.get('final_answer', ''))
        method = solution.get('method', 'unknown')
        
        self.cursor.execute('DROP TABLE IF EXISTS verifications')
        self.cursor.execute('''
            CREATE TABLE verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER,
                is_correct BOOLEAN,
                confidence REAL,
                issues TEXT,
                warnings TEXT,
                requires_human_review BOOLEAN,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (solution_id) REFERENCES solutions (id)
            )
        ''')
        
        self.conn.commit()
        solution_id = self.cursor.lastrowid
        print(f"✅ Stored solution with ID: {solution_id}")
        return solution_id
    
    def store_verification(self, solution_id: int, verification: Dict) -> int:
        """
        Verification store karo - COMPLETELY FIXED
        """
        try:
            # DEBUG: Check verification dict
            print(f"\n🔍 DEBUG store_verification:")
            print(f"  Verification dict keys: {list(verification.keys())}")
            
            # Get values with PROPER defaults
            is_correct = verification.get('is_correct')
            confidence = verification.get('confidence')
            issues = verification.get('issues', [])
            warnings = verification.get('warnings', [])
            requires_human_review = verification.get('requires_human_review')
            
            # Convert to proper types
            if is_correct is None:
                is_correct = False
            else:
                is_correct = bool(is_correct)
                
            if confidence is None:
                confidence = 0.5
            else:
                confidence = float(confidence)
                
            if requires_human_review is None:
                requires_human_review = False
            else:
                requires_human_review = bool(requires_human_review)
            
            # Ensure issues and warnings are lists
            if not isinstance(issues, list):
                issues = []
            if not isinstance(warnings, list):
                warnings = []
            
            # Convert to JSON
            issues_json = json.dumps(issues)
            warnings_json = json.dumps(warnings)
            
            print(f"  Values after processing:")
            print(f"    solution_id: {solution_id} (type: {type(solution_id)})")
            print(f"    is_correct: {is_correct} (type: {type(is_correct)})")
            print(f"    confidence: {confidence} (type: {type(confidence)})")
            print(f"    issues: {issues_json[:50]}... (length: {len(issues_json)})")
            print(f"    warnings: {warnings_json[:50]}... (length: {len(warnings_json)})")
            print(f"    requires_human_review: {requires_human_review} (type: {type(requires_human_review)})")
            
            # Check for None values
            values = [solution_id, is_correct, confidence, issues_json, warnings_json, requires_human_review]
            for i, val in enumerate(values):
                if val is None:
                    print(f"  ❌ ERROR: Value at position {i} is None!")
            
            # Execute with ALL 6 values
            self.cursor.execute('''
                INSERT INTO verifications 
                (solution_id, is_correct, confidence, issues, warnings, requires_human_review)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', values)
            
            self.conn.commit()
            verification_id = self.cursor.lastrowid
            print(f"  ✅ Stored verification with ID: {verification_id}")
            return verification_id
            
        except Exception as e:
            print(f"❌ store_verification ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Check table structure
            print("\n🔍 Checking table structure...")
            try:
                self.cursor.execute("PRAGMA table_info(verifications)")
                columns = self.cursor.fetchall()
                print("Verifications table columns:")
                for col in columns:
                    print(f"  {col[0]}. {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
                print(f"Total columns: {len(columns)}")
            except:
                print("Could not check table structure")
            
            raise e
    
    def store_feedback(self, solution_id: int, is_correct: bool, 
                      feedback_text: str = "", corrected_answer: str = "",
                      given_by: str = "user") -> int:
        """
        User feedback store karo
        """
        self.cursor.execute('''
            INSERT INTO feedback 
            (solution_id, is_correct, feedback_text, corrected_answer, given_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            solution_id,
            is_correct,
            feedback_text,
            corrected_answer,
            given_by
        ))
        
        self.conn.commit()
        feedback_id = self.cursor.lastrowid
        print(f"✅ Stored feedback with ID: {feedback_id}")
        return feedback_id
    
    def find_similar_problems(self, problem_text: str, limit: int = 5) -> List[Dict]:
        """
        Similar problems find karo using text similarity
        """
        # Simple text-based similarity
        search_term = f"%{problem_text[:20]}%"
        
        self.cursor.execute('''
            SELECT id, problem_text, topic, created_at
            FROM problems
            WHERE problem_text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (search_term, limit))
        
        rows = self.cursor.fetchall()
        
        similar_problems = []
        for row in rows:
            similar_problems.append({
                'id': row[0],
                'problem_text': row[1],
                'topic': row[2],
                'created_at': row[3]
            })
        
        return similar_problems
    
    def get_stats(self) -> Dict:
        """
        Overall statistics get karo
        """
        stats = {}
        
        # Total problems
        self.cursor.execute("SELECT COUNT(*) FROM problems")
        stats['total_problems'] = self.cursor.fetchone()[0]
        
        # Correct solutions
        self.cursor.execute('''
            SELECT COUNT(*) FROM verifications WHERE is_correct = 1
        ''')
        stats['correct_solutions'] = self.cursor.fetchone()[0]
        
        # Topics distribution
        self.cursor.execute('''
            SELECT topic, COUNT(*) as count 
            FROM problems 
            GROUP BY topic 
            ORDER BY count DESC
        ''')
        topics_data = self.cursor.fetchall()
        stats['topics'] = {topic: count for topic, count in topics_data} if topics_data else {}
        
        # Average confidence
        self.cursor.execute("SELECT AVG(confidence) FROM verifications")
        avg_conf = self.cursor.fetchone()[0]
        stats['average_confidence'] = float(avg_conf) if avg_conf else 0.0
        
        # Feedback stats
        self.cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1")
        stats['positive_feedback'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 0")
        stats['negative_feedback'] = self.cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Database connection close karo"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

# Test function
def test_database():
    """Test database functionality"""
    print("\n" + "="*60)
    print("TESTING DATABASE")
    print("="*60)
    
    # Use in-memory database for testing
    db = MemoryDatabase(":memory:")
    
    # Test storing a problem
    test_parsed = {
        'problem_text': 'Solve 2x + 5 = 13',
        'topic': 'algebra',
        'variables': ['x'],
        'constraints': [],
        'parsed_at': datetime.datetime.now().isoformat()
    }
    
    problem_id = db.store_problem("Solve 2x + 5 = 13", test_parsed)
    print(f"\n✅ Stored problem with ID: {problem_id}")
    
    # Test storing solution
    test_solution = {
        'solution_steps': ['2x + 5 = 13', '2x = 8', 'x = 4'],
        'final_answer': '4',
        'formulas_used': ['linear equation solving'],
        'method': 'algebraic manipulation'
    }
    
    solution_id = db.store_solution(problem_id, test_solution)
    print(f"✅ Stored solution with ID: {solution_id}")
    
    # Test storing verification - WITH COMPLETE DATA
    test_verification = {
        'is_correct': True,
        'confidence': 0.95,
        'issues': [],           # MUST BE LIST
        'warnings': [],         # MUST BE LIST
        'requires_human_review': False,
        'suggestions': ['Check calculation']
    }
    
    verification_id = db.store_verification(solution_id, test_verification)
    print(f"✅ Stored verification with ID: {verification_id}")
    
    # Test storing feedback
    feedback_id = db.store_feedback(
        solution_id, 
        is_correct=True,
        feedback_text="Good solution!",
        given_by="user"
    )
    print(f"✅ Stored feedback with ID: {feedback_id}")
    
    # Test getting stats
    stats = db.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    # Test finding similar problems
    similar = db.find_similar_problems("Solve equation", limit=3)
    print(f"\n🔍 Similar problems found: {len(similar)}")
    
    db.close()
    
    print("\n" + "="*60)
    print("✅ DATABASE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    return True

# Global instance
memory_db = None

def get_memory_db():
    """Get or create memory database instance"""
    global memory_db
    if memory_db is None:
        memory_db = MemoryDatabase()
    return memory_db

if __name__ == "__main__":
    # Run test
    test_database()
