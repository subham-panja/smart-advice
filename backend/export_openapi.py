"""
Export OpenAPI / Swagger JSON Specification
============================================
Generates the complete OpenAPI 2.0 / 3.0 specification from the Flask-RESTX
API and writes it to openapi.json.
"""

import json
import os
import sys

# Ensure backend root is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import create_app  # noqa: E402
from swagger import api  # noqa: E402


def export_openapi(output_path: str = None) -> dict:
    if output_path is None:
        output_path = os.path.join(CURRENT_DIR, "openapi.json")

    app = create_app()
    with app.test_request_context("/api"):
        spec = api.__schema__

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Successfully exported OpenAPI specification to: {output_path}")
    print(f"Title: {spec.get('info', {}).get('title')} v{spec.get('info', {}).get('version')}")
    print(f"Total paths documented: {len(spec.get('paths', {}))}")
    print(f"Total definitions: {len(spec.get('definitions', {}))}")
    return spec


if __name__ == "__main__":
    export_openapi()
