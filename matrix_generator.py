# Gemini-based code from Google Gemini's LLM. Made solely for generating matrices

import numpy as np
import os

def generate_gaussian_test_data(filename, sizes, min_val=-100, max_val=100):
    """
    Generates random square matrices and saves them to a text file.
    
    Args:
        filename (str): Name of the output file.
        sizes (list): A list of integers representing dimensions (e.g., [10, 50, 100]).
        min_val (int): Minimum random value.
        max_val (int): Maximum random value.
    """
    with open(filename, 'w') as f:
        for n in sizes:
            f.write(f"SIZE: {n}\n")
            
            # Generating an (n) x (n+1) matrix to include the constants vector (b)
            # for the augmented matrix [A|b]
            matrix = np.random.uniform(min_val, max_val, size=(n, n + 1))
            
            # Save the matrix with formatting
            np.savetxt(f, matrix, fmt='%.4f')
            f.write("\n" + "="*20 + "\n\n")
            
    print(f"Successfully generated matrices of sizes {sizes} in '{filename}'.")

# Example Usage:
# This creates 3 matrices of increasing size to test complexity scaling
test_sizes = [5, 10, 50, 100]
generate_gaussian_test_data("matrices.txt", test_sizes)
