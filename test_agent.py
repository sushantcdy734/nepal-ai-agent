from agent import chat


def run_test(label, message):
    print(f"\n=== {label} ===\n")
    full = ""
    tools = []
    for event in chat(message):
        if event["type"] == "text":
            full += event["data"]
        elif event["type"] == "status":
            print(f"[status] {event['data']}")
        elif event["type"] == "tool_used":
            tools.append(event["data"])
    print(f"\nResponse: {full}")
    print(f"Tools used: {tools}")


run_test("Test 1: Math", "What is 847 * 213 + 1200?")
run_test("Test 2: Web search", "What's the current weather in Kathmandu?")