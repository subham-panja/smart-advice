import os
import re

def check_config_usage():
    config_path = "config.py"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config_content = f.read()

    # Find all all-uppercase variable assignments
    variables = re.findall(r'^([A-Z0-9_]+)\s*=', config_content, re.MULTILINE)
    print(f"Found {len(variables)} config variables to check.")

    py_files = []
    for root, _, files in os.walk("."):
        if "venv" in root or "__pycache__" in root or ".venv" in root or "node_modules" in root: 
            continue
        for f in files:
            if f.endswith(".py") and f != "config.py":
                py_files.append(os.path.join(root, f))

    unused = []
    for var in variables:
        used = False
        for pf in py_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    if var in f.read():
                        used = True
                        break
            except Exception:
                pass
        
        if not used:
            unused.append(var)

    if unused:
        print("\n--- UNUSED CONFIG VARIABLES ---")
        for u in unused:
            print(f"- {u}")
        print("\nTotal unused variables:", len(unused))
    else:
        print("\nAll config variables are used in the codebase!")

if __name__ == "__main__":
    check_config_usage()
