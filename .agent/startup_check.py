#!/usr/bin/env python3
"""
Startup Check Script
====================

Reads all critical project documents and saves a read receipt.
AI agents MUST run this before starting any task.

Usage:
    python .agent/startup_check.py

Output:
    .agent/last_read.json - Timestamped receipt of all docs read
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AGENT_DIR = PROJECT_ROOT / ".agent"

DOCS_TO_READ = {
    "agent_md": PROJECT_ROOT / "AGENT.md",
    "skills": {
        "backtest_validation": PROJECT_ROOT / "skills" / "backtest_validation.md",
        "data_validation": PROJECT_ROOT / "skills" / "data_validation.md",
        "entry_pattern_optimization": PROJECT_ROOT
        / "skills"
        / "entry_pattern_optimization.md",
        "performance_debugging": PROJECT_ROOT / "skills" / "performance_debugging.md",
        "risk_management": PROJECT_ROOT / "skills" / "risk_management.md",
        "strategy_analysis": PROJECT_ROOT / "skills" / "strategy_analysis.md",
    },
    "workflows": {
        "analyze_stocks": PROJECT_ROOT / ".agent" / "workflows" / "analyze_stocks.md",
        "add_strategy": PROJECT_ROOT / ".agent" / "workflows" / "add_strategy.md",
        "frontend_development": PROJECT_ROOT
        / ".agent"
        / "workflows"
        / "frontend_development.md",
    },
}


def check_file_exists(path: Path) -> dict:
    """Check if file exists and get metadata."""
    if not path.exists():
        return {"exists": False, "error": f"File not found: {path}"}

    stat = path.stat()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "lines": content.count("\n") + 1,
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "read_at": datetime.now().isoformat(),
    }


def main():
    print("=" * 60)
    print("Smart Advice — Startup Document Check")
    print("=" * 60)
    print()

    read_receipt = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "documents": {},
        "summary": {"total": 0, "found": 0, "missing": 0},
    }

    # Check AGENT.md
    print("✓ Reading AGENT.md...")
    read_receipt["documents"]["agent_md"] = check_file_exists(DOCS_TO_READ["agent_md"])
    read_receipt["summary"]["total"] += 1
    if read_receipt["documents"]["agent_md"]["exists"]:
        read_receipt["summary"]["found"] += 1
        print(f"  ✓ AGENT.md ({read_receipt['documents']['agent_md']['lines']} lines)")
    else:
        read_receipt["summary"]["missing"] += 1
        print("  ✗ AGENT.md MISSING")

    # Check skills
    print("\n✓ Reading skills...")
    read_receipt["documents"]["skills"] = {}
    for skill_name, skill_path in DOCS_TO_READ["skills"].items():
        result = check_file_exists(skill_path)
        read_receipt["documents"]["skills"][skill_name] = result
        read_receipt["summary"]["total"] += 1
        if result["exists"]:
            read_receipt["summary"]["found"] += 1
            print(f"  ✓ {skill_name} ({result['lines']} lines)")
        else:
            read_receipt["summary"]["missing"] += 1
            print(f"  ✗ {skill_name} MISSING")

    # Check workflows
    print("\n✓ Reading workflows...")
    read_receipt["documents"]["workflows"] = {}
    for wf_name, wf_path in DOCS_TO_READ["workflows"].items():
        result = check_file_exists(wf_path)
        read_receipt["documents"]["workflows"][wf_name] = result
        read_receipt["summary"]["total"] += 1
        if result["exists"]:
            read_receipt["summary"]["found"] += 1
            print(f"  ✓ {wf_name} ({result['lines']} lines)")
        else:
            read_receipt["summary"]["missing"] += 1
            print(f"  ✗ {wf_name} MISSING")

    # Save read receipt
    receipt_path = AGENT_DIR / "last_read.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(read_receipt, f, indent=2)

    print()
    print("=" * 60)
    print(f"Read Receipt Saved: {receipt_path}")
    print(
        f"Total: {read_receipt['summary']['total']} | Found: {read_receipt['summary']['found']} | Missing: {read_receipt['summary']['missing']}"
    )
    print("=" * 60)

    if read_receipt["summary"]["missing"] > 0:
        print("\n⚠️  WARNING: Some documents are missing. Check paths above.")
        return 1
    else:
        print("\n✅ All documents loaded successfully.")
        print("\nAI Agent: You may now proceed with tasks.")
        print("Remember: Read relevant skills/*.md files completely before working.")
        return 0


if __name__ == "__main__":
    exit(main())
