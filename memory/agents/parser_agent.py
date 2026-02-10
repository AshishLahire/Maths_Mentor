"""
Bhai, ye Parser Agent hai.
Iska kaam hai raw text ko structured format mein convert karna.
"""

import re
import json

class ParserAgent:
    def __init__(self):
        self.math_keywords = {
            'algebra': ['solve', 'equation', 'expression', 'polynomial', 'quadratic', 
                       '=', 'find x', 'value of', 'calculate', 'what is', 'how many'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate', 
                        'maxima', 'minima', 'rate of change', 'approaches', 'tends to'],
            'probability': ['probability', 'chance', 'odds', 'random', 
                          'event', 'heads', 'tails', 'likely', 'unlikely', 'coin', 'dice'],
            'linear_algebra': ['matrix', 'vector', 'determinant', 'eigenvalue'],
            'trigonometry': ['sin', 'cos', 'tan', 'angle', 'triangle', 
                           'trigonometry', 'sine', 'cosine', 'tangent']
        }
        
    def clean_text(self, text):
        """OCR/Audio se aaye text ko clean karo"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Common OCR errors fix
        corrections = {
            'l': '1',  # Small L ko 1 banaye
            'O': '0',  # Capital O ko 0 banaye
            '|': '1',  # Pipe ko 1 banaye
            '[': '(',  # Square bracket ko parenthesis
            ']': ')',
        }
        
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
            
        return text
    
    def identify_topic(self, text):
        """Question ka topic identify karo - IMPROVED VERSION"""
        text_lower = text.lower()
        
        # Pehle specific keywords check karo
        for topic, keywords in self.math_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return topic
        
        # Additional smart checks for common patterns
        # Probability related
        if any(word in text_lower for word in ['probability', 'chance', 'odds', 'heads', 'tails', 'coin', 'dice']):
            return 'probability'
        
        # Calculus related  
        if any(word in text_lower for word in ['derivative', 'differentiate', 'integral', 'limit', 'approaches']):
            return 'calculus'
        
        # Algebra related
        if any(word in text_lower for word in ['solve', 'equation', '=', 'find x', 'value of', '2x', '3x', '+']):
            return 'algebra'
        
        # Trigonometry related
        if any(word in text_lower for word in ['sin', 'cos', 'tan', 'angle', 'triangle']):
            return 'trigonometry'
        
        # Linear algebra
        if any(word in text_lower for word in ['matrix', 'determinant', 'vector']):
            return 'linear_algebra'
        
        # Check for math symbols
        if any(symbol in text_lower for symbol in ['+', '-', '*', '/', '=', '^', '√']):
            return 'algebra'
        
        # If contains numbers and variables, likely algebra
        if re.search(r'\d+', text_lower) and re.search(r'[a-zA-Z]', text_lower):
            return 'algebra'
        
        # Default to algebra for general math questions
        if any(word in text_lower for word in ['find', 'calculate', 'what is', 'how many', 'determine']):
            return 'algebra'
        
        return 'algebra'  # Default fallback
    
    def extract_variables(self, text):
        """Variables extract karo (like x, y, n, etc.)"""
        # Pattern for variables (single letters, sometimes with subscripts)
        var_pattern = r'\b([a-zA-Z])\b(?![^(]*\))'
        variables = re.findall(var_pattern, text)
        
        # Remove duplicates and common words
        common_words = ['a', 'an', 'the', 'of', 'to', 'in', 'is', 'and', 
                       'find', 'solve', 'what', 'how', 'calculate', 'value']
        unique_vars = []
        for var in set(variables):
            if var.lower() not in common_words:
                unique_vars.append(var)
        
        return unique_vars
    
    def extract_constraints(self, text):
        """Constraints extract karo (like x > 0, n is integer)"""
        constraints = []
        
        # Look for inequality constraints
        inequality_pattern = r'([a-zA-Z])\s*([<>]=?|≠)\s*(\d+|[a-zA-Z])'
        inequalities = re.findall(inequality_pattern, text)
        for var, op, val in inequalities:
            constraints.append(f"{var} {op} {val}")
        
        # Look for domain constraints
        if 'integer' in text.lower():
            constraints.append('integer domain')
        if 'real' in text.lower() or 'ℝ' in text:
            constraints.append('real numbers')
        if 'positive' in text.lower():
            constraints.append('positive numbers')
        
        return constraints
    
    def check_ambiguity(self, text):
        """Check if question is ambiguous"""
        ambiguous_phrases = [
            'find', 'calculate', 'what is', 'how many',
            'determine', 'evaluate'
        ]
        
        text_lower = text.lower()
        
        # If it's just a command without clear question
        has_ambiguous = any(phrase in text_lower for phrase in ambiguous_phrases)
        
        # Check if it has numbers or variables
        has_numbers = bool(re.search(r'\d+', text))
        has_variables = bool(self.extract_variables(text))
        
        # If ambiguous phrase but no numbers/variables, needs clarification
        needs_clarification = has_ambiguous and not (has_numbers or has_variables)
        
        return needs_clarification
    
    def parse(self, input_text):
        """
        Main parsing function
        Returns structured dictionary
        """
        # Clean the text
        cleaned_text = self.clean_text(input_text)
        
        # Identify components
        topic = self.identify_topic(cleaned_text)
        variables = self.extract_variables(cleaned_text)
        constraints = self.extract_constraints(cleaned_text)
        needs_clarification = self.check_ambiguity(cleaned_text)
        
        # Prepare clarification question if needed
        clarification_question = ""
        if needs_clarification:
            clarification_question = f"The question '{cleaned_text}' seems incomplete. What exactly needs to be found?"
        
        # Structure the output
        structured_output = {
            "problem_text": cleaned_text,
            "topic": topic,
            "variables": variables,
            "constraints": constraints,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "parsed_at": "timestamp_placeholder"  # Add actual timestamp in production
        }
        
        return structured_output

# Test function
if __name__ == "__main__":
    parser = ParserAgent()
    
    test_cases = [
        "Find the derivative of x^2 + 3x - 5",
        "What is the probability of getting heads when flipping a coin?",
        "Solve for x: 2x + 5 = 13",
        "Calculate the limit as x approaches 0 of sin(x)/x"
    ]
    
    for test in test_cases:
        print(f"\nInput: {test}")
        result = parser.parse(test)
        print(f"Topic: {result['topic']}")
        print(f"Variables: {result['variables']}")
