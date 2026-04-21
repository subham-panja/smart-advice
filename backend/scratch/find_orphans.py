import os
import re
import subprocess

def get_config_keys():
    with open('config.py', 'r') as f:
        content = f.read()
    
    # Simple regex to find keys in dicts and top-level variables
    # This finds 'KEY': or KEY =
    keys = set()
    
    # Find dictionary keys: 'key':
    dict_keys = re.findall(r"'(\w+)':", content)
    keys.update(dict_keys)
    
    # Find top-level variables: KEY =
    top_level = re.findall(r"^(\w+)\s*=", content, re.MULTILINE)
    keys.update(top_level)
    
    return sorted(list(keys))

def check_usage(key):
    # Search for key in all .py files except config.py
    # Return count of occurrences
    try:
        # Use grep to find occurrences
        result = subprocess.run(
            ['grep', '-rl', key, '.'],
            capture_output=True,
            text=True
        )
        files = result.stdout.splitlines()
        # Filter out config.py and common non-source files
        usable_files = [f for f in files if f.endswith('.py') and 'config.py' not in f and 'scratch' not in f and 'venv' not in f and '__pycache__' not in f]
        return usable_files
    except Exception:
        return []

if __name__ == "__main__":
    keys = get_config_keys()
    print(f"Checking {len(keys)} configuration keys...")
    
    orphans = []
    for key in keys:
        # Skip common python patterns or very short keys that might give false positives
        if len(key) < 3 or key in ['for', 'int', 'str', 'list', 'dict']:
            continue
            
        usage = check_usage(key)
        if not usage:
            orphans.append(key)
    
    print("\nPotential Orphaned Keys (Not found in any .py file except config.py):")
    for orphan in orphans:
        print(f"- {orphan}")
