def create_lambda(expression: str, args: list):
    """
    Create a lambda function from a mathematical expression string.
    
    Args:
        expression (str): A string representing the mathematical expression, e.g. "a * x + b".
        args (list): A list of arguments to be used in the lambda function, e.g. ['x', 'a', 'b'].
        
    Returns:
        A lambda function based on the input expression.
    """
    # Join the argument names into a string to form the lambda signature
    arg_string = ", ".join(args)
    
    # Use eval to create a lambda function from the expression
    return eval(f"lambda {arg_string}: {expression}")
