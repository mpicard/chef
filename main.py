from llm import llm
from orchestrator import Orchestrator
from tools_registry import TOOLS


def main():
    orch = Orchestrator(provider=llm, tools=TOOLS)

    GOAL = "Create sandbox/hello.py that prints Hello, chef"

    result = orch.run(GOAL)

    print(f"Status:  {result.status}")
    print()
    print("--- PLAN ---")
    print(result.plan)
    print()
    print("--- SUMMARY ---")
    print(result.report)


if __name__ == "__main__":
    main()
