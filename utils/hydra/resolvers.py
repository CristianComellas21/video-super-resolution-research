import ast

## Custom resolver to evaluate expressions
def eval_expression(expression: str):   
    try: 
        return eval(expression)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {e}")
    
    
# Custom resolver to parse a string to a Python literal
def parse_to_type(value: str):
    try:
        # Attempt to evaluate the string to a Python literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If evaluation fails, return the original string
        return value