#!/bin/bash

set -e

DEFAULT_INPUT="Our manufacturing process uses a batch record system that tracks all weighing operations. We maintain audit trails for all critical process parameters and perform periodic validation of our sterilization equipment."
INPUT="${1:-$DEFAULT_INPUT}"

echo "Testing FDA Compliance Pipeline..."
echo "Input: $INPUT"
echo ""

START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST "http://localhost:8000/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"input_text\":\"$INPUT\"}")

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Response:"
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

echo ""
echo "⏱️ Time taken: ${ELAPSED}s"
echo "Pipeline test complete!"