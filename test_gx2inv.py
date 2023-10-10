import numpy as np
import subprocess
from scipy.optimize import root
from scipy.stats import ncx2
from functions import gx2inv

import os

import warnings
warnings.filterwarnings('ignore')

# Define the tests
tests = [
    {
        'p': 0.9,
        'w': np.array([1, -5, 2]),
        'k': np.array([1, 2, 3]),
        'lambda_': np.array([2, 3, 7]),
        'm': 5,
        's': 0
    },
    {
        'p': 0.5,
        'w': np.array([1, 1, 1]),
        'k': np.array([2, 2, 2]),
        'lambda_': np.array([3, 3, 3]),
        'm': 0,
        's': 1
    },
    {
        'p': np.array([0.1, 0.5, 0.9]),
        'w': np.array([1, -2, 3]),
        'k': np.array([1, 2, 3]),
        'lambda_': np.array([2, 3, 4]),
        'm': 1,
        's': 0
    }
]

# Run the tests
for i, test in enumerate(tests, start=1):
    # Run the Python version
    x_python = gx2inv(test['p'], test['w'], test['k'], test['lambda_'], test['m'], test['s'])
    print(f'Python Test {i} result: {x_python}')

    # Get the absolute path of the MATLAB file
    matlab_file_path = os.path.abspath('./gx2-matlab')

    # Run the MATLAB version
    matlab_command = f"addpath('{matlab_file_path}'); x = gx2inv({test['p']}, {test['w']}, {test['k']}, {test['lambda_']}, {test['m']}, {test['s']}); disp(['[', num2str(x), ']'])"
    result = subprocess.run(['matlab', '-batch', matlab_command], capture_output=True, text=True)
    
    print(f'MATLAB Test {i} result: {result.stdout}')