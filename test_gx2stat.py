import numpy as np
import subprocess
import os
from functions import gx2stat

# Define the tests
tests = [
    {
        'w': np.array([1, -5, 2]),
        'k': np.array([1, 2, 3]),
        'lambda_': np.array([2, 3, 7]),
        'm': 5,
        's': 0
    },
    {
        'w': np.array([1, 1, 1]),
        'k': np.array([2, 2, 2]),
        'lambda_': np.array([3, 3, 3]),
        'm': 0,
        's': 1
    },
    {
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
    mu, v = gx2stat(test['w'], test['k'], test['lambda_'], test['m'], test['s'])
    print(f'Python Test {i} result: mu={mu}, v={v}')

    # Get the absolute path of the MATLAB file
    matlab_file_path = os.path.abspath('./gx2-matlab')

    # Run the MATLAB version
    # matlab_command = f"addpath('{matlab_file_path}'); disp(gx2stat({test['w']}, {test['k']}, {test['lambda_']}, {test['m']}, {test['s']}))"
    matlab_command = f"addpath('{matlab_file_path}'); [mu, v] = gx2stat({test['w']}, {test['k']}, {test['lambda_']}, {test['m']}, {test['s']}); disp(['mu=', num2str(mu), ', v=', num2str(v)])"
    result = subprocess.run(['matlab', '-batch', matlab_command], capture_output=True, text=True)
    
    print(f'MATLAB Test {i} result: {result.stdout}')