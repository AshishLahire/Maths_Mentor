"""
Bhai, ye Explainer Agent hai.
Iska kaam hai student-friendly step-by-step explanation dena.
"""

import re
import textwrap

class ExplainerAgent:
    def __init__(self):
        self.explanation_templates = {
            'derivative': {
                'steps': [
                    "Identify the function to differentiate",
                    "Apply differentiation rules (power rule, product rule, etc.)",
                    "Simplify the resulting expression",
                    "State the final derivative"
                ],
                'example': "For example, d/dx(x²) = 2x"
            },
            'integral': {
                'steps': [
                    "Identify the function to integrate",
                    "Check for standard integral forms",
                    "Apply integration rules",
                    "Add the constant of integration",
                    "Simplify if possible"
                ],
                'example': "For example, ∫2x dx = x² + C"
            },
            'equation': {
                'steps': [
                    "Isolate the variable on one side",
                    "Perform inverse operations",
                    "Simplify both sides",
                    "Check the solution by substitution"
                ],
                'example': "For example, 2x + 3 = 7 → 2x = 4 → x = 2"
            },
            'probability': {
                'steps': [
                    "Define the sample space",
                    "Identify the favorable outcomes",
                    "Count the number of favorable outcomes",
                    "Count the total number of outcomes",
                    "Calculate probability = favorable/total"
                ],
                'example': "For a fair coin, P(Heads) = 1/2"
            },
            'limit': {
                'steps': [
                    "Try direct substitution",
                    "If indeterminate, try factorization",
                    "Alternatively, try rationalization",
                    "Apply limit rules",
                    "State the limiting value"
                ],
                'example': "lim(x→0) sin(x)/x = 1"
            }
        }
        
        self.student_friendly_phrases = [
            "Let's break this down step by step",
            "First, we need to understand what's being asked",
            "The key concept here is",
            "Remember this important formula",
            "A common mistake to avoid is",
            "Let me explain why this works",
            "Think of it this way",
            "To make it easier to understand",
            "Here's a simpler way to look at it",
            "Let's verify our answer to be sure"
        ]
    
    def identify_problem_type(self, problem_text, solution):
        """Problem type identify karo for appropriate explanation"""
        text_lower = problem_text.lower()
        
        if any(word in text_lower for word in ['derivative', 'differentiate', 'rate of change']):
            return 'derivative'
        elif any(word in text_lower for word in ['integral', 'integrate', 'antiderivative']):
            return 'integral'
        elif any(word in text_lower for word in ['solve', 'equation', 'find x', '=']):
            return 'equation'
        elif any(word in text_lower for word in ['probability', 'chance', 'odds']):
            return 'probability'
        elif any(word in text_lower for word in ['limit', 'approaches', 'tends to']):
            return 'limit'
        elif 'matrix' in text_lower or 'determinant' in text_lower:
            return 'matrix'
        else:
            # Try to infer from solution
            if 'derivative' in str(solution).lower():
                return 'derivative'
            elif 'integral' in str(solution).lower():
                return 'integral'
            else:
                return 'general'
    
    def create_step_by_step_explanation(self, solution, problem_type):
        """Step-by-step explanation generate karo"""
        explanation_parts = []
        
        # Introduction
        intro = self.get_random_phrase()
        explanation_parts.append(f"🧠 **{intro}**\n")
        
        # Use template if available
        if problem_type in self.explanation_templates:
            template = self.explanation_templates[problem_type]
            
            explanation_parts.append(f"**For {problem_type} problems, we typically follow these steps:**")
            for i, step in enumerate(template['steps'], 1):
                explanation_parts.append(f"{i}. {step}")
            
            explanation_parts.append(f"\n**Example:** {template['example']}")
        
        # Add actual solution steps
        solution_steps = solution.get('solution_steps', [])
        if solution_steps:
            explanation_parts.append(f"\n**For this specific problem:**")
            
            for i, step in enumerate(solution_steps, 1):
                step_text = str(step)
                
                # Format step nicely
                if isinstance(step, dict):
                    if 'expression' in step:
                        step_text = f"Expression: `{step['expression']}`"
                    if 'tool' in step:
                        step_text += f" (using {step['tool']})"
                    if 'result' in step:
                        step_text += f" → Result: `{step['result']}`"
                
                explanation_parts.append(f"{i}. {step_text}")
        
        # Add final answer explanation
        final_answer = solution.get('final_answer')
        if final_answer:
            if isinstance(final_answer, dict) and 'method' in final_answer:
                # For stepwise approach
                explanation_parts.append(f"\n**Step-by-step approach:**")
                for step in final_answer.get('steps', []):
                    explanation_parts.append(f"• {step}")
            else:
                explanation_parts.append(f"\n**Therefore, the answer is:** `{final_answer}`")
        
        # Add learning tips
        explanation_parts.append(f"\n**💡 Learning Tips:**")
        
        tips = self.get_learning_tips(problem_type)
        for tip in tips:
            explanation_parts.append(f"• {tip}")
        
        # Add common mistakes warning
        common_mistakes = self.get_common_mistakes(problem_type)
        if common_mistakes:
            explanation_parts.append(f"\n**⚠️ Common Mistakes to Avoid:**")
            for mistake in common_mistakes:
                explanation_parts.append(f"• {mistake}")
        
        # Add verification note
        explanation_parts.append(f"\n**✓ Verification:** Always check your answer by substituting back or using logical reasoning.")
        
        return "\n".join(explanation_parts)
    
    def get_random_phrase(self):
        """Random student-friendly phrase select karo"""
        import random
        return random.choice(self.student_friendly_phrases)
    
    def get_learning_tips(self, problem_type):
        """Learning tips provide karo based on problem type"""
        tips = {
            'derivative': [
                "Practice the chain rule - it's the most important!",
                "Remember derivative of e^x is e^x",
                "d/dx(sin x) = cos x, d/dx(cos x) = -sin x",
                "Use product rule: (uv)' = u'v + uv'",
                "Use quotient rule: (u/v)' = (u'v - uv')/v²"
            ],
            'integral': [
                "Don't forget the constant of integration +C",
                "Remember ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1",
                "∫e^x dx = e^x + C",
                "∫sin x dx = -cos x + C",
                "Practice substitution method"
            ],
            'equation': [
                "Whatever you do to one side, do to the other",
                "Isolate the variable step by step",
                "Check your answer by substitution",
                "For quadratic equations, use the formula",
                "Watch out for sign changes"
            ],
            'probability': [
                "Probability is always between 0 and 1",
                "P(A') = 1 - P(A)",
                "For independent events, P(A∩B) = P(A)P(B)",
                "Remember conditional probability: P(A|B) = P(A∩B)/P(B)",
                "Draw Venn diagrams for complex problems"
            ],
            'limit': [
                "Try direct substitution first",
                "For 0/0 forms, try factorization",
                "Remember lim(x→0) sin(x)/x = 1",
                "Use L'Hopital's rule for indeterminate forms",
                "Check left and right limits for piecewise functions"
            ]
        }
        
        if problem_type in tips:
            import random
            return random.sample(tips[problem_type], 3)
        else:
            return [
                "Read the problem carefully",
                "Write down what's given and what's asked",
                "Check your units and constraints"
            ]
    
    def get_common_mistakes(self, problem_type):
        """Common mistakes bataye based on problem type"""
        mistakes = {
            'derivative': [
                "Forgetting chain rule",
                "Sign errors in differentiation",
                "Misapplying product/quotient rule",
                "Derivative of constant is 0 (not the constant itself)"
            ],
            'integral': [
                "Forgetting +C",
                "Wrong power rule application",
                "Integration by parts errors",
                "Forgetting to change limits in definite integral"
            ],
            'equation': [
                "Sign errors when moving terms",
                "Dividing by variable that could be zero",
                "Forgetting ± in square roots",
                "Not checking all solutions"
            ],
            'probability': [
                "Confusing P(A|B) and P(B|A)",
                "Forgetting to reduce fractions",
                "Adding probabilities for non-mutually exclusive events",
                "Not checking if probability ≤ 1"
            ]
        }
        
        if problem_type in mistakes:
            import random
            return random.sample(mistakes[problem_type], 2)
        else:
            return ["Calculation errors", "Misreading the problem"]
    
    def simplify_explanation(self, explanation, grade_level='high_school'):
        """Explanation ko simplify karo based on student level"""
        # Simple replacements for easier understanding
        simplifications = {
            'derivative': 'rate of change',
            'integral': 'area under curve',
            'probability': 'chance',
            'variable': 'unknown',
            'coefficient': 'number in front',
            'expression': 'math phrase',
            'equation': 'math sentence with =',
            'solve': 'find the value',
            'evaluate': 'find the value',
            'determine': 'find out',
            'hence': 'so',
            'thus': 'so',
            'therefore': 'so',
            'compute': 'calculate',
            'obtain': 'get'
        }
        
        simplified = explanation
        for complex_word, simple_word in simplifications.items():
            simplified = simplified.replace(f' {complex_word} ', f' {simple_word} ')
            simplified = simplified.replace(f' {complex_word},', f' {simple_word},')
            simplified = simplified.replace(f' {complex_word}.', f' {simple_word}.')
        
        # Add more line breaks for readability
        lines = simplified.split('\n')
        readable_lines = []
        
        for line in lines:
            if len(line) > 80:
                # Wrap long lines
                wrapped = textwrap.fill(line, width=80)
                readable_lines.append(wrapped)
            else:
                readable_lines.append(line)
        
        return '\n'.join(readable_lines)
    
    def explain(self, solution, verification):
        """
        Main explanation function
        """
        problem_text = solution.get('problem', 'No problem text')
        
        # Identify problem type
        problem_type = self.identify_problem_type(problem_text, solution)
        
        # Create explanation
        explanation = self.create_step_by_step_explanation(solution, problem_type)
        
        # Simplify for student
        simplified_explanation = self.simplify_explanation(explanation)
        
        # Add verification status
        verification_status = ""
        if verification.get('is_correct', False):
            verification_status = "✅ **Verified:** This solution has been checked and appears correct."
        else:
            verification_status = "⚠️ **Note:** Some issues were found. Please review carefully."
        
        if verification.get('requires_human_review', False):
            verification_status += "\n👨‍🏫 **Human Review Recommended:** A teacher should check this solution."
        
        # Final explanation
        final_explanation = f"""
# 📚 Solution Explanation

{verification_status}

---

## 🧮 Problem Type: {problem_type.title()}

{simplified_explanation}

---

## 🔍 Verification Details:
- Confidence Level: {verification.get('confidence', 0):.2%}
- Issues Found: {len(verification.get('issues', []))}
- Warnings: {len(verification.get('warnings', []))}

---

**Remember:** Practice makes perfect! Try similar problems to reinforce these concepts.
"""
        
        return {
            'explanation': final_explanation,
            'problem_type': problem_type,
            'simplified': simplified_explanation,
            'verification_summary': verification_status
        }

# Test function
if __name__ == "__main__":
    explainer = ExplainerAgent()
    
    # Test solution
    test_solution = {
        'problem': 'Find the derivative of x^2 + 3x',
        'final_answer': '2x + 3',
        'solution_steps': [
            {'expression': 'x^2', 'tool': 'differentiate', 'result': '2x'},
            {'expression': '3x', 'tool': 'differentiate', 'result': '3'},
            {'step': 'Combine results', 'result': '2x + 3'}
        ]
    }
    
    test_verification = {
        'is_correct': True,
        'confidence': 0.95,
        'issues': [],
        'warnings': []
    }
    
    explanation = explainer.explain(test_solution, test_verification)
    
    print("Generated Explanation:")
    print("="*80)
    print(explanation['explanation'])
    print("="*80)
    
    # Test another type
    print("\n\nTesting Probability Explanation:")
    print("="*80)
    
    prob_solution = {
        'problem': 'Probability of getting heads in coin toss',
        'final_answer': 0.5,
        'solution_steps': [
            {'step': 'Sample space = {Heads, Tails}', 'result': '2 outcomes'},
            {'step': 'Favorable outcome = {Heads}', 'result': '1 outcome'},
            {'step': 'Probability = favorable/total', 'result': '1/2 = 0.5'}
        ]
    }
    
    prob_explanation = explainer.explain(prob_solution, test_verification)
    print(prob_explanation['simplified'][:500] + "...")
