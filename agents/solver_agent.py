import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get model from .env or use default
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class SolverAgent:
    """Main solver agent - uses Groq LLM for math problems"""
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.llm_client = None
        self.model = GROQ_MODEL
        
        if use_llm and GROQ_API_KEY:
            try:
                from groq import Groq
                self.llm_client = Groq(api_key=GROQ_API_KEY)
                print(f"✅ Solver Agent initialized with model: {self.model}")
            except ImportError:
                print("⚠️ groq package not installed. Install with: pip install groq")
                self.use_llm = False
            except Exception as e:
                print(f"❌ Error initializing Groq client: {e}")
                self.use_llm = False
        else:
            if not GROQ_API_KEY:
                print("⚠️ GROQ_API_KEY not found in .env file")
            self.use_llm = False
    
    def solve(self, parsed_problem, context=None):
        """
        Solve math problem using LLM or fallback to rules
        """
        if self.use_llm and self.llm_client:
            return self._solve_with_llm(parsed_problem, context)
        else:
            return self._solve_with_rules(parsed_problem)
    
    def _solve_with_llm(self, parsed_problem, context):
        """
        Use Groq LLM to solve the problem
        """
        try:
            # Build prompt
            problem_text = parsed_problem.get('problem_text', '')
            topics = parsed_problem.get('topics', [])
            difficulty = parsed_problem.get('difficulty', 'medium')
            
            prompt = f"""You are an expert JEE-level math tutor. Solve this problem step by step.

PROBLEM: {problem_text}

TOPICS: {', '.join(topics) if topics else 'General Math'}
DIFFICULTY: {difficulty}

INSTRUCTIONS:
1. Provide the final answer clearly at the beginning
2. Show step-by-step solution
3. Explain each step in simple Hindi/English mix
4. Mention key formulas/concepts used
5. End with a summary

IMPORTANT: Format your response with clear headings:
• **Final Answer:** [Your answer]
• **Solution Steps:**
  1. [Step 1]
  2. [Step 2]
• **Explanation:** [Detailed explanation]
• **Key Concepts:** [List concepts]"""
            
            # Add context if available
            if context and len(context) > 0:
                relevant_info = "\n".join([f"- {doc['content'][:200]}..." for doc in context[:2]])
                prompt += f"\n\nRELEVANT CONTEXT:\n{relevant_info}"
            
            # Call Groq API
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            final_answer = self._extract_answer(content)
            
            return {
                'final_answer': content,
                'parsed_answer': final_answer,
                'solution_steps': self._extract_steps(content),
                'confidence': 0.85,
                'method': f'Groq LLM ({self.model})',
                'model_used': self.model,
                'response_raw': content[:500] + "..." if len(content) > 500 else content
            }
            
        except Exception as e:
            print(f"❌ LLM solving error: {e}")
            return self._solve_with_rules(parsed_problem)
    
    def _solve_with_rules(self, parsed_problem):
        """
        Fallback rule-based solver
        """
        problem_text = parsed_problem.get('problem_text', '').lower()
        
        # Simple rule-based solutions
        if 'derivative' in problem_text or 'differentiate' in problem_text:
            return self._solve_derivative(problem_text)
        elif 'integral' in problem_text or 'integrate' in problem_text:
            return self._solve_integral(problem_text)
        elif 'solve' in problem_text and ('=' in problem_text):
            return self._solve_equation(problem_text)
        elif 'probability' in problem_text:
            return self._solve_probability(problem_text)
        else:
            return {
                'final_answer': "I need more advanced AI to solve this problem. Please check your Groq API configuration.",
                'solution_steps': [],
                'confidence': 0.2,
                'method': 'Rule-based fallback',
                'error': 'LLM unavailable'
            }
    
    def _extract_answer(self, content):
        """Extract final answer from LLM response"""
        lines = content.split('\n')
        for line in lines:
            if 'answer' in line.lower() and ':' in line:
                return line.split(':', 1)[1].strip()
        return content[:100] + "..." if len(content) > 100 else content
    
    def _extract_steps(self, content):
        """Extract steps from LLM response"""
        steps = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '•', '-', '*')):
                steps.append(line.strip())
        return steps if steps else [content[:200] + "..."]
    
    # Rule-based solvers
    def _solve_derivative(self, problem_text):
        """Simple derivative solver"""
        match = re.search(r'(\w+)\^(\d+)', problem_text)
        if match:
            base, power = match.groups()
            try:
                power = int(power)
                result = f"{power}*{base}^{power-1}"
                return {
                    'final_answer': f"d/dx({base}^{power}) = {result}",
                    'solution_steps': [
                        f"1. Apply power rule: d/dx(x^n) = n*x^(n-1)",
                        f"2. Here, n = {power}",
                        f"3. Result: {power}*{base}^{power-1}"
                    ],
                    'confidence': 0.7,
                    'method': 'Rule-based (power rule)'
                }
            except:
                pass
        return self._generic_fallback()
    
    def _solve_equation(self, problem_text):
        """Simple equation solver"""
        # Look for pattern like "solve 2x + 5 = 13"
        if 'x' in problem_text:
            return {
                'final_answer': "For equation solving, please use the LLM feature.",
                'solution_steps': ["Install groq and add API key to .env for full functionality"],
                'confidence': 0.3,
                'method': 'Rule-based hint'
            }
        return self._generic_fallback()
    
    def _solve_probability(self, problem_text):
        """Simple probability solver"""
        return {
            'final_answer': "Probability = (Favorable Outcomes) / (Total Outcomes)",
            'solution_steps': [
                "1. Count favorable outcomes",
                "2. Count total possible outcomes",
                "3. Divide: P = favorable/total"
            ],
            'confidence': 0.5,
            'method': 'Rule-based (probability formula)'
        }
    
    def _solve_integral(self, problem_text):
        """Simple integral solver"""
        return {
            'final_answer': "∫x^n dx = x^(n+1)/(n+1) + C",
            'solution_steps': [
                "1. Apply power rule for integration",
                "2. Add 1 to exponent",
                "3. Divide by new exponent",
                "4. Add constant C"
            ],
            'confidence': 0.6,
            'method': 'Rule-based (integration power rule)'
        }
    
    def _generic_fallback(self):
        """Generic fallback response"""
        return {
            'final_answer': "Please configure Groq API for complete solution.",
            'solution_steps': [
                "1. Get API key from https://console.groq.com",
                "2. Add to .env file: GROQ_API_KEY=your_key",
                "3. Install: pip install groq",
                "4. Restart the app"
            ],
            'confidence': 0.2,
            'method': 'Rule-based instruction'
        }


# For backward compatibility
class EnhancedSolverAgent(SolverAgent):
    """Alias for backward compatibility"""
    pass
