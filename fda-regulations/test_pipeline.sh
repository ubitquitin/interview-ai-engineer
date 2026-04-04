#!/bin/bash

# Test script for FDA Compliance Pipeline
# Usage: ./test_pipeline.sh "Your process description here"

set -e

# Default test input if none provided
DEFAULT_INPUT="Our manufacturing process uses a batch record system that tracks all weighing operations. We maintain audit trails for all critical process parameters and perform periodic validation of our sterilization equipment."

INPUT="${1:-$DEFAULT_INPUT}"

echo "Testing FDA Compliance Pipeline..."
echo "Input: $INPUT"
echo ""

# Make the request
RESPONSE=$(curl -s -X POST "http://localhost:8000/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"input_text\":\"$INPUT\"}")

echo "Response:"
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

echo ""
echo "Pipeline test complete!"
