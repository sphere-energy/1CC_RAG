#!/bin/bash

# Base URL
API_URL="https://rag.api.bap.sphere-energy.eu"

echo "========================================================"
echo "Testing 1CC RAG API (Authentication Disabled)"
echo "========================================================"

# 1. Non-streaming Chat Request (New Conversation)
echo -e "\n1. Testing Non-streaming Chat Request (New Conversation)..."
echo "Request:"
echo 'curl -X POST "'$API_URL'/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '"'"'{
    "messages": [
      {"role": "user", "content": "Based on given sources for EPR legislation in Denmark, could please tell me which is the newest enforced legislation regarding packaging and packaging waste legislation? Please summarize the most important aspects of this legislation. Please indicate the legal source and the relevant articles/paragraphs?h"}
    ],
    "stream": false
  }'"'"''

echo -e "\nResponse:"
curl -X POST "$API_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Based on given sources for EPR legislation in Denmark, could please tell me which is the newest enforced legislation regarding packaging and packaging waste legislation? Please summarize the most important aspects of this legislation. Please indicate the legal source and the relevant articles/paragraphs?"}
    ],
    "stream": false
  }'

# echo -e "\n\n--------------------------------------------------------"

# # 2. Streaming Chat Request (New Conversation)
# echo -e "\n2. Testing Streaming Chat Request..."
# echo "Request:"
# echo 'curl -N -X POST "'$API_URL'/api/v1/chat" \
#   -H "Content-Type: application/json" \
#   -d '"'"'{
#     "messages": [
#       {"role": "user", "content": "Explain the concept of Extended Producer Responsibility."}
#     ],
#     "stream": true
#   }'"'"''

# echo -e "\nResponse (Streamed):"
# curl -N -X POST "$API_URL/api/v1/chat" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "messages": [
#       {"role": "user", "content": "Explain the concept of Extended Producer Responsibility."}
#     ],
#     "stream": true
#   }'

# echo -e "\n\n--------------------------------------------------------"

# # 3. Chat Request with Conversation History (Simulated)
# # Note: In a real scenario, you would use the conversation_id returned from the previous request.
# # Here we are just showing the syntax. If you have a valid UUID from a previous run, replace it below.

# # Generate a random UUID for testing if uuidgen is available, else use a placeholder
# if command -v uuidgen &> /dev/null; then
#     TEST_CONV_ID=$(uuidgen)
# else
#     TEST_CONV_ID="123e4567-e89b-12d3-a456-426614174000" # Example UUID
# fi

# echo -e "\n3. Testing Chat Request with Conversation History (Simulated ID: $TEST_CONV_ID)..."
# echo "Request:"
# echo 'curl -X POST "'$API_URL'/api/v1/chat" \
#   -H "Content-Type: application/json" \
#   -d '"'"'{
#     "conversation_id": "'$TEST_CONV_ID'",
#     "messages": [
#       {"role": "user", "content": "What are the battery regulations?"},
#       {"role": "assistant", "content": "The EU Battery Regulation 2023/1542..."},
#       {"role": "user", "content": "Does this apply to portable batteries?"}
#     ],
#     "stream": false
#   }'"'"''

# echo -e "\nResponse:"
# curl -X POST "$API_URL/api/v1/chat" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "conversation_id": "'$TEST_CONV_ID'",
#     "messages": [
#       {"role": "user", "content": "What are the battery regulations?"},
#       {"role": "assistant", "content": "The EU Battery Regulation 2023/1542..."},
#       {"role": "user", "content": "Does this apply to portable batteries?"}
#     ],
#     "stream": false
#   }'

# echo -e "\n\n========================================================"
# echo "Done."
