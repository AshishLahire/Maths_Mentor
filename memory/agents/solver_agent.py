"""
Bhai, ye enhanced solver agent hai.
LLM + RAG + Tools use karega.
"""

import re
import sympy
from sympy import symbols, solve, diff, integrate, limit
from llm.groq_handler import get_llm_response

class EnhancedSolverAgent:
    def __init__(self, use_llm=True):
        """
        Enhanced solver initialize karo
        """
        self.use_llm = use_llm
        self.tools = {
            'calculator': self.calculate_expression,
            'solve_equation': self.solve_equation,
            'differentiate': self.differentiate,
            'integrate': self.integrate,
            'probability_calc': self.calculate_probability,
            'limit_calc': self.calculate_limit
        }
        
        if use_llm:
            print("✅ Enhanced Solver with LLM enabled")
        else:
            print("✅ Enhanced Solver (rule-based only)")
    
    def extract_math_context(self, context):
        """
        Context se math formulas extract karo
        """
        math_context = ""
        for item in context:
            content = item.get('content', '')
            if any(math_keyword in content.lower() for math_keyword in 
                  ['formula', '=', 'derivative', 'integral', 'probability', 'limit']):
                math_context += f"- {content[:200]}\n"
        
        return math_context
    
    def solve_with_llm(self, problem_text, context):
        """
        LLM se solve karo
        """
        try:
            # Extract math context
            math_context = self.extract_math_context(context)
            
            # Get LLM response
            llm_response = get_llm_response(problem_text, math_context)
            
            # Parse LLM response
            solution_steps = self.parse_llm_response(llm_response)
            
            # Extract final answer
            final_answer = self.extract_final_answer(llm_response)
            
            return {
                'problem': problem_text,
                'solution_steps': solution_steps,
                'final_answer': final_answer,
                'llm_response': llm_response[:500] + "...",
                'method': 'LLM + RAG',
                'formulas_used': [item.get('content', '')[:100] for item in context[:3]]
            }
            
        except Exception as e:
            print(f"❌ LLM solving error: {e}")
            return self.solve_with_tools(problem_text, context)
    
    def solve_with_tools(self, problem_text, context):
        """
        Tools se solve karo (fallback)
        """
        # Extract expressions
        expressions = self.extract_math_expression(problem_text)
        
        solution_steps = []
        final_answer = None
        
        for expr in expressions:
            for tool_name, tool_func in self.tools.items():
                result = tool_func(expr)
                if result is not None:
                    solution_steps.append({
                        'expression': expr,
                        'tool': tool_name,
                        'result': result
                    })
                    
                    if final_answer is None:
                        final_answer = result
                    break
        
        return {
            'problem': problem_text,
            'solution_steps': solution_steps,
            'final_answer': final_answer or "Could not solve with tools",
            'method': 'Rule-based Tools',
            'formulas_used': []
        }
    
    def parse_llm_response(self, llm_response):
        """
        LLM response ko steps mein parse karo
        """
        steps = []
        
        # Split by numbered steps
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.\)]', line) or line.startswith('-') or line.startswith('*'):
                steps.append(line)
        
        return steps if steps else ["1. " + llm_response[:100] + "..."]
    
    def extract_final_answer(self, llm_response):
        """
        LLM response se final answer extract karo
        """
        # Look for patterns like "Answer:", "Final answer:", "= value"
        patterns = [
            r'[Aa]nswer[:\s]+([^\n]+)',
            r'[Ff]inal [Aa]nswer[:\s]+([^\n]+)',
            r'=\s*([^\n]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, llm_response)
            if match:
                return match.group(1).strip()
        
        # Return last line if no pattern found
        lines = llm_response.strip().split('\n')
        return lines[-1] if lines else "Answer not found"
    
    # Rest of tool functions (same as before)
    def extract_math_expression(self, text):
        """Math expression extract karo"""
        patterns = [
            r'([\d\.]+\s*[\+\-\*/]\s*[\d\.]+)',
            r'([a-zA-Z]+\s*=\s*[^\.]+)',
            r'(derivative of [^\.]+)',
            r'(integral of [^\.]+)',
            r'(probability of [^\.]+)',
            r'(limit [^\.]+)'
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend(matches)
        
        return expressions
    
    def calculate_expression(self, expression):
        """Calculate expression"""
        try:
            expression = expression.replace('^', '**')
            result = sympy.sympify(expression)
            return float(result.evalf())
        except:
            return None
    
    def solve_equation(self, equation):
        """Solve equation"""
        try:
            if '=' in equation:
                lhs, rhs = equation.split('=', 1)
                equation = f"{lhs} - ({rhs})"
            
            x = symbols('x')
            expr = sympy.sympify(equation)
            solutions = solve(expr, x)
            
            return [float(sol.evalf()) for sol in solutions]
        except:
            return None
    
    def differentiate(self, expression, var='x'):
        """Differentiate"""
        try:
            x = symbols(var)
            expr = sympy.sympify(expression)
            derivative = diff(expr, x)
            return str(derivative)
        except:
            return None
    
    def integrate(self, expression, var='x'):
        """Integrate"""
        try:
            x = symbols(var)
            expr = sympy.sympify(expression)
            integral = integrate(expr, x)
            return str(integral)
        except:
            return None
    
    def calculate_probability(self, description):
        """Calculate probability"""
        try:
            numbers = re.findall(r'\d+', description)
            
            if 'coin' in description.lower():
                return 0.5
            elif 'dice' in description.lower():
                if 'even' in description.lower():
                    return 3/6
                elif 'prime' in description.lower():
                    return 3/6
            
            if len(numbers) >= 2:
                favorable = int(numbers[0])
                total = int(numbers[1])
                if total > 0:
                    return favorable / total
            
            return None
        except:
            return None
    
    def calculate_limit(self, expression, var='x', point=0):
        """Calculate limit"""
        try:
            x = symbols(var)
            expr = sympy.sympify(expression)
            lim = limit(expr, x, point)
            return str(lim)
        except:
            return None
    
    def solve(self, parsed_problem, context):
        """
        Main solve function
        """
        problem_text = parsed_problem.get('problem_text', '')
        
        if self.use_llm:
            return self.solve_with_llm(problem_text, context)
        else:
            return self.solve_with_tools(problem_text, context)
