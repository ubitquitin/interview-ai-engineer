import json
import os
import sys
import time
from typing import Any, Dict

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please run 'uv sync' or 'pip install requests'.")
    sys.exit(1)

# Configuration - uses environment variable with a local fallback
API_URL = os.getenv("API_URL", "http://localhost:8000/pipeline")

DEFAULT_INPUT = (
    "Our manufacturing process uses a batch record system that tracks all weighing operations. "
    "We maintain audit trails for all critical process parameters and perform periodic "
    "validation of our sterilization equipment."
)

def print_header(title: str):
    print(f"\n{'='*20} {title} {'='*20}")

def run_pipeline_test(input_text: str):
    print(f"Testing FDA Compliance Pipeline...")
    print(f"Target URL: {API_URL}")
    print(f"\nInput Text: {input_text[:100]}...")

    start_time = time.time()
    
    try:
        response = requests.post(
            API_URL, 
            json={"input_text": input_text},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Request Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Body: {e.response.text}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # 1. Structured Output Section
    print_header("STRUCTURED OUTPUT")
    structured = data.get("structured", {})
    print(f"Summary: {structured.get('summary', 'N/A')}\n")

    print("Entities Identified:")
    for entity in structured.get("entities", []):
        print(f"  • {entity.get('name')} ({entity.get('type')}): {entity.get('description')}")

    print("\nSystems Detected:")
    for system in structured.get("systems", []):
        print(f"  • {system}")

    # 2. Compliance Report Section
    print_header("COMPLIANCE REPORT")
    compliance = data.get("compliance", {})
    print(f"OVERALL RISK LEVEL: {compliance.get('overall_risk', 'UNKNOWN')}")
    print(f"\nExecutive Summary:\n{compliance.get('summary', 'N/A')}")

    print("\nDetailed Risks & Deficiencies:")
    risks = compliance.get("risks", [])
    if not risks:
        print("  (No specific risks identified)")
    
    for i, risk in enumerate(risks, 1):
        deficiency = risk.get("deficiency", {})
        print(f"\n{i}. {deficiency.get('title', 'Untitled Deficiency')}")
        print(f"   Reference: {deficiency.get('cfr_reference', 'N/A')}")
        print(f"   Severity:  {risk.get('risk_level', 'N/A')}")
        print(f"\n   Description:\n   {deficiency.get('description')}")
        print(f"\n   Evidence:\n   {deficiency.get('evidence')}")
        print(f"\n   Required Action:\n   {deficiency.get('required_action')}")
        print(f"\n   Reasoning:\n   {risk.get('reasoning')}")
        print("-" * 50)

    print(f"\nPipeline test complete! (Time taken: {elapsed:.2f}s)")

if __name__ == "__main__":
    # Allow passing custom text via command line: uv run tests/test_pipeline.py "my custom text"
    input_to_test = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    run_pipeline_test(input_to_test)