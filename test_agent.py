from agent import chat

print("=== Test 1: Math question (should use calculate tool) ===\n")
result = chat("What is 847 * 213 + 1200?")
print("Response:", result["response"])
print("Tools used:", result["tools_used"])
print("\n" + "=" * 50 + "\n")

print("=== Test 2: Web search still works ===\n")
result = chat("What's the current weather in Kathmandu?")
print("Response:", result["response"])
print("Tools used:", result["tools_used"])