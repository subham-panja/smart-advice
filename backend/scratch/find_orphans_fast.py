import os
import re

def get_config_keys():
    with open('config.py', 'r') as f:
        content = f.read()
    keys = set()
    dict_keys = re.findall(r"'(\w+)':", content)
    keys.update(dict_keys)
    top_level = re.findall(r"^(\w+)\s*=", content, re.MULTILINE)
    keys.update(top_level)
    return sorted(list(keys))

if __name__ == "__main__":
    keys = get_config_keys()
    print(f"Total keys: {len(keys)}")
    
    # 1. Get ALL symbols from all .py files except config.py
    # We'll use a single pass over the project
    all_content = ""
    for root, dirs, files in os.walk('.'):
        if any(x in root for x in ['venv', '.venv', '__pycache__', 'scratch', '.pytest_cache']):
            continue
        for file in files:
            if file.endswith('.py') and file != 'config.py':
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        all_content += f.read() + "\n"
                except:
                    continue
    
    # 2. Check each key in the massive content blob
    orphans = []
    for key in keys:
        if len(key) < 4: continue # Skip short ones to avoid false positives
        if key not in all_content:
            orphans.append(key)
    
    print("\nPotential Orphans:")
    for o in orphans:
        print(f"- {o}")
