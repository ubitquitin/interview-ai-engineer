#!/usr/bin/env python3
"""
Python client for testing the FDA Compliance Pipeline
"""

import httpx
import json
import sys
from typing import Optional

PIPELINE_URL = "http://localhost:8000/pipeline"

DEFAULT_TEST_INPUT = """
Our manufacturing facility uses a batch record system for tracking operations.
We maintain electronic audit trails for all critical process parameters.
The sterilization equipment is validated quarterly, and we have SOPs for deviation handling.
All weighing operations are performed on calibrated equipment with documented verification.
"""


def test_pipeline(input_text: str) -> dict:
    """
    Send text to the FDA compliance pipeline and return the result.

    Args:
        input_text: Raw process description text

    Returns:
        dict containing the compliance analysis result
    """
    try:
        print("=" * 80)
        print("FDA COMPLIANCE PIPELINE TEST")
        print("=" * 80)
        print(f"\nInput Text:\n{input_text}\n")
        print("Sending to pipeline...")

        response = httpx.post(
            PIPELINE_URL,
            json={"input_text": input_text},
            timeout=120.0  # Allow 2 minutes for processing
        )
        response.raise_for_status()

        result = response.json()

        print("\n" + "=" * 80)
        print("PIPELINE RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2))

        # Extract and display key findings
        if "compliance" in result:
            compliance = result["compliance"]
            print("\n" + "=" * 80)
            print("COMPLIANCE SUMMARY")
            print("=" * 80)

            if "overall_risk" in compliance:
                print(f"\nOverall Risk Level: {compliance['overall_risk']}")

            if "summary" in compliance:
                print(f"\nSummary:\n{compliance['summary']}")

            if "risks" in compliance and compliance["risks"]:
                print(f"\nIdentified Risks: {len(compliance['risks'])}")
                for i, risk in enumerate(compliance["risks"], 1):
                    print(f"\n  Risk {i}:")
                    print(f"    Level: {risk.get('risk_level', 'N/A')}")
                    if "deficiency" in risk:
                        deficiency = risk["deficiency"]
                        print(f"    CFR: {deficiency.get('cfr_reference', 'N/A')}")
                        print(f"    Title: {deficiency.get('title', 'N/A')}")

        return result

    except httpx.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def check_services():
    """Check if all services are healthy"""
    services = [
        ("Main Pipeline", "http://localhost:8000/health"),
        ("Structuring Agent", "http://localhost:8001/docs"),
        ("Compliance Agent", "http://localhost:8002/docs"),
    ]

    print("Checking service health...")
    all_healthy = True

    for name, url in services:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                print(f"  ✓ {name}: OK")
            else:
                print(f"  ✗ {name}: Status {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_healthy = False

    print()
    return all_healthy


if __name__ == "__main__":
    # Check services first
    if not check_services():
        print("⚠️  Some services are not responding. Make sure docker-compose is running:")
        print("    docker-compose up -d")
        print()

    # Get input from command line or use default
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = DEFAULT_TEST_INPUT
        print("Using default test input. To provide custom input:")
        print('    python test_pipeline.py "Your process description here"\n')

    # Run the test
    test_pipeline(input_text)
