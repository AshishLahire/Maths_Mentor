"""
Bhai, ye LLM handler hai. Groq API use karega free mein.
"""

import os
from groq import Groq
import json

class GroqHandler:
    def __init__(self, api_key=None):
        """
        Groq LLM initialize karo
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found. Using fallback responses.")
            self.client = None
            self.available = False
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                self.available = True
                print("✅ Groq LLM initialized")
            except Exception as e:
                print(f"❌ Groq initialization failed: {e}")
                self.client = None
                self.available = False
    
    def generate_math_solution(self, problem, context):
        """
        Math problem solve karo LLM se
        """
        if not self.available or not self.client:
            return self.fallback_response(problem, context)
        
        try:
            # Prepare system prompt
            system_prompt = """You are an expert math tutor specializing in JEE-level mathematics.
            Solve the given math problem step by step.
            Provide clear explanations and show all working.
            Format your response with:
            1. Problem understanding
            2. Step-by-step solution
            3. Final answer
            4. Verification steps"""
            
            # Prepare user prompt with context
            user_prompt = f"""
            Math Problem: {problem}
            
            Relevant Knowledge from Database:
            {context}
            
            Please solve this math problem step by step.
            """
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Free model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return self.fallback_response(problem, context)
    
    def generate_explanation(self, solution, verification):
        """
        Student-friendly explanation generate karo
        """
        if not self.available or not self.client:
            return self.fallback_explanation(solution, verification)
        
        try:
            system_prompt = """You are a friendly math tutor explaining concepts to a high school student.
            Make the explanation simple, engaging, and easy to understand.
            Use analogies and real-life examples.
            Break down complex steps into smaller parts."""
            
            user_prompt = f"""
            Solution: {json.dumps(solution, indent=2)}
            
            Verification: {json.dumps(verification, indent=2)}
            
            Please create a student-friendly explanation of this solution.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Groq explanation error: {e}")
            return self.fallback_explanation(solution, verification)
    
    def fallback_response(self, problem, context):
        """Fallback response if LLM not available"""
        return f"""
        **Problem:** {problem}
        
        **Solution Approach:**
        1. Analyze the problem: {problem[:50]}...
        2. Apply relevant formulas from knowledge base
        3. Solve step by step
        
        **Note:** LLM service currently unavailable. Using rule-based solver.
        """
    
    def fallback_explanation(self, solution, verification):
        """Fallback explanation"""
        return f"""
        **Solution Explanation:**
        
        The problem has been solved using mathematical rules and formulas.
        
        Verification Status: {'Correct' if verification.get('is_correct') else 'Needs Review'}
        Confidence: {verification.get('confidence', 0):.2%}
        
        For detailed explanation, please enable LLM integration with Groq API.
        """

# Global instance
groq_handler = GroqHandler()

def get_llm_response(problem, context):
    """Get LLM response for problem"""
    return groq_handler.generate_math_solution(problem, context)

def get_llm_explanation(solution, verification):
    """Get LLM explanation"""
    return groq_handler.generate_explanation(solution, verification)
