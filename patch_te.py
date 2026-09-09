import os
import sys
import site

# Find Transformer Engine source
for path in site.getsitepackages():
    te_path = os.path.join(path, 'transformer_engine')
    if os.path.exists(te_path):
        print(f"Found TE at: {te_path}")
        
        # Check build_tools/utils.py
        utils_path = os.path.join(te_path, 'build_tools/utils.py')
        if os.path.exists(utils_path):
            print(f"Found utils.py at: {utils_path}")
            with open(utils_path, 'r') as f:
                content = f.read()
            
            # Check if NCCL paths are included
            if 'nccl' in content.lower():
                print("NCCL references found in utils.py")
            else:
                print("No NCCL references found")
            break
