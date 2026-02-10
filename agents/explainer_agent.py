import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get model from .env or use default
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ExplainerAgent:
    """Explain solutions in simple terms"""
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.llm_client = None
        self.model = GROQ_MODEL
        
        if use_llm and GROQ_API_KEY:
            try:
                from groq import Groq
                self.llm_client = Groq(api_key=GROQ_API_KEY)
                print(f"✅ Explainer Agent initialized with model: {self.model}")
            except ImportError:
                print("⚠️ groq package not installed")
                self.use_llm = False
            except Exception as e:
                print(f"❌ Error initializing explainer: {e}")
                self.use_llm = False
        else:
            if not GROQ_API_KEY:
                print("⚠️ GROQ_API_KEY not found for explainer")
            self.use_llm = False
    
    def explain(self, solution, verification):
        """
        Create explanation for the solution
        """
        if self.use_llm and self.llm_client:
            return self._explain_with_llm(solution, verification)
        else:
            return self._explain_with_template(solution)
    
    def _explain_with_llm(self, solution, verification):
        """
        Use LLM to create explanation
        """
        try:
            answer = solution.get('final_answer', 'No answer provided')
            confidence = solution.get('confidence', 0.5)
            
            prompt = f"""You are a friendly math tutor explaining solutions to JEE students.

SOLUTION: {answer}

CONFIDENCE LEVEL: {confidence:.2%}

Create an explanation that:
1. Breaks down the solution step-by-step
2. Uses simple Hindi/English mix
3. Explains why each step works
4. Connects to key math concepts
5. Gives tips for similar problems

Make it engaging and easy to understand!"""

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800
            )
            
            explanation = response.choices[0].message.content
            
            return {
                'explanation': explanation,
                'method': f'Groq LLM ({self.model})',
                'model_used': self.model
            }
            
        except Exception as e:
            print(f"❌ Explanation error: {e}")
            return self._explain_with_template(solution)
    
    def _explain_with_template(self, solution):
        """
        Template-based explanation fallback
        """
        answer = solution.get('final_answer', 'No solution')
        
        template = f"""
## 📚 Solution Explanation

**What we did:**
1. Analyzed the problem
2. Applied appropriate math concepts
3. Calculated step-by-step
4. Verified the result

**Key points:**
- Solution provided: {answer[:100]}...
- Method used: {solution.get('method', 'Standard approach')}
- Confidence: {solution.get('confidence', 0.5):.2%}

**Tip:** For better explanations, configure Groq API in .env file.
"""
        
        return {
            'explanation': template,
            'method': 'Template-based (LLM unavailable)'
        }


# For backward compatibility
class EnhancedExplainerAgent(ExplainerAgent):
    """Alias for backward compatibility"""
    pass
