import os
os.environ["EVAL_MODE"] = "1"

from langchain_core.messages import HumanMessage
from agent import app
from eval_set import EVAL

def run():
    rows, correct = [], 0
    for i, item in enumerate(EVAL):
        cfg = {"configurable": {"thread_id": f"eval-{i}"}}   # fresh memory per question
        out = app.invoke({"messages": [HumanMessage(content=item["q"])], "question": item["q"],
                          "query": "", "documents": [], "decision": None, "sufficient": None,
                          "grounded": None, "critique_feedback": "", "r_attempts": 0,
                          "c_attempts": 0}, cfg)
        d = out["decision"]
        got = d["decision"] if isinstance(d, dict) else d.decision   # works for dict or pydantic
        ok = got == item["expected"]
        correct += ok
        rows.append((item["q"][:55], item["expected"], got, "OK" if ok else "X"))

    print(f"\n{'QUESTION':<57}{'EXP':<11}{'GOT':<11}RESULT")
    print("-" * 90)
    for q, e, g, r in rows:
        print(f"{q:<57}{e:<11}{g:<11}{r}")
    print("-" * 90)
    print(f"Decision accuracy: {correct}/{len(EVAL)} = {correct/len(EVAL):.0%}")

if __name__ == "__main__":
    run()