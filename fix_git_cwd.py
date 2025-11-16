"""
Fix subprocess.run calls to add cwd=self.repo_path

This script carefully adds cwd to each subprocess.run call.
"""

import re

# Read file
with open('xterminator/git_integrator.py', 'r') as f:
    lines = f.readlines()

# Process each line
in_subprocess_call = False
command_found = False
new_lines = []

for i, line in enumerate(lines):
    # Detect start of subprocess.run call
    if 'subprocess.run(' in line and 'cwd=' not in line:
        in_subprocess_call = True
        command_found = False

    # If we're in a subprocess call, look for the command list end
    if in_subprocess_call:
        # Check if this line has the closing ] of the command list
        if '],\n' in line or '],' in line:
            # Add cwd after the ],
            if 'cwd=' not in line:
                line = line.replace('],', '], cwd=self.repo_path,', 1)
            in_subprocess_call = False
        elif '],  # ' in line:  # Handle comments
            line = line.replace('],', '], cwd=self.repo_path,', 1)
            in_subprocess_call = False

    new_lines.append(line)

# Write back
with open('xterminator/git_integrator.py', 'w') as f:
    f.writelines(new_lines)

print("Updated subprocess.run calls")
