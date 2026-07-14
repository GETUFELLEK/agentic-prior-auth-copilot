import os
os.environ["EVAL_MODE"] = "1"

out = app.invoke(make_initial_state(q), cfg)
from eval_set import EVAL

def run():
    rows, correct = [], 0
    for i, item in enumerate(EVAL):
        cfg = {"configurable": {"thread_id": f"eval-{i}"}}   # fresh memory per question
        out = app.invoke(make_initial_state(item["q"]), cfg)
        d = out["decision"]
        got = d["decision"] if isinstance(d, dict) else d.decision   # works for dict or pydantic
        ok = got == item["expected"]
        correct += ok
        rows.append((item["q"][:55], item["expected"], got, "OK" if ok else "X"))
        from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    for item, (_, _, got, _) in zip(EVAL, rows):
        cat = item.get("category", "core")
        by_cat[cat][1] += 1
        by_cat[cat][0] += (got == item["expected"])
    print("\nBy category:")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"  {cat:<22} {c}/{n} = {c/n:.0%}")

    print(f"\n{'QUESTION':<57}{'EXP':<11}{'GOT':<11}RESULT")
    print("-" * 90)
    for q, e, g, r in rows:
        print(f"{q:<57}{e:<11}{g:<11}{r}")
    print("-" * 90)
    print(f"Decision accuracy: {correct}/{len(EVAL)} = {correct/len(EVAL):.0%}")

if __name__ == "__main__":
    run()