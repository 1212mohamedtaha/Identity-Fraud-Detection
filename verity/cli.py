"""Command line: run the server, play in the terminal, evaluate and train policies.

    verity packs                               list packs and their policies
    verity serve                               web app on http://127.0.0.1:8000
    verity play cv --input cv=@my_cv.txt       answer questions in the terminal
    verity evaluate identity                   compare policies on simulated people
    verity train identity --episodes 2000      train an RL policy (saved to models/)
    verity data generate cv --size 3000        make a synthetic dataset (datasets/cv/)
"""
import argparse
import logging
from pathlib import Path


def parse_inputs(pairs):
    """``key=value`` pairs; ``key=@file.txt`` reads the value from a file."""
    inputs = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if value.startswith("@"):
            value = Path(value[1:]).read_text(encoding="utf-8")
        inputs[key] = value
    return inputs


def cmd_packs(args):
    from .packs import PACKS, get_pack

    for name in PACKS:
        pack = get_pack(name)
        print(f"{name:10} {pack.title} - policies: {', '.join(pack.policy_names())}")


def cmd_serve(args):
    import uvicorn

    uvicorn.run("verity.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


def cmd_play(args):
    from .core.engine import Session
    from .llm import get_llm
    from .packs import get_pack

    pack = get_pack(args.pack, llm=get_llm())
    session = Session(pack, parse_inputs(args.input), policy=pack.policy(args.policy))
    print(f"\n{pack.title}. Claims to check:")
    for claim in session.state.claims:
        print("  -", claim.text)
    while not session.finished:
        probe = session.current
        print(f"\nQ{len(session.history) + 1}: {probe.question}")
        for choice in probe.choices:
            print(f"   {choice.id}) {choice.text}")
        answer = input("> ").strip()
        try:
            observation = session.answer(answer)
        except ValueError as exc:
            print(exc)
            continue
        if pack.show_feedback:
            print(f"   score {observation.score:.0%}: {observation.feedback}")

    verdict = session.verdict
    print(f"\nVerdict: {verdict.status} ({verdict.probability:.0%} that every claim is true)")
    for result in verdict.claims:
        print(f"  [{result.status:9}] {result.claim.text} - {result.explanation}")
    for note in verdict.notes:
        print("  note:", note)


def simulation_pack(args):
    """The pack used for simulations: never a real LLM; optionally the pack's mock LLM."""
    from .packs import get_pack

    pack = get_pack(args.pack)
    if getattr(args, "mock_llm", False):
        pack.llm = pack.mock_llm(seed=args.seed)
        if pack.llm is None:
            raise SystemExit(f"Pack {pack.name!r} has no mock LLM.")
    return pack


def dataset_cases(args, split):
    from .core.dataset import load_cases

    if not args.data:
        return None
    cases = load_cases(args.data, split)
    if not cases:
        raise SystemExit(f"No {split!r} cases in {args.data}.")
    return cases


def cmd_evaluate(args):
    from .core.simulation import evaluate

    pack = simulation_pack(args)
    cases = dataset_cases(args, args.split)
    if cases:
        print(f"{len(cases)} {args.split} cases from {args.data}")
    policies = args.policies or pack.policy_names()
    print(f"{'policy':10} {'accuracy':>9} {'uncertain':>10} {'wrong':>7} {'questions':>10} {'reward':>7}")
    for name in policies:
        r = evaluate(pack, name, episodes=args.episodes, seed=args.seed, cases=cases)
        print(f"{name:10} {r['accuracy']:9.1%} {r['uncertain']:10.1%} {r['wrong']:7.1%} "
              f"{r['avg_questions']:10.1f} {r['reward']:+7.3f}")


def cmd_train(args):
    from .rl.train import train

    pack = simulation_pack(args)
    out = args.out or pack.learned_policy_path()
    train(pack, episodes=args.episodes, lr=args.lr, question_cost=args.question_cost, seed=args.seed, out=out,
          cases=dataset_cases(args, "train"))
    print(f"Saved the trained policy to {out}. Compare it with: verity evaluate {pack.name}")


def cmd_data_generate(args):
    from .core.dataset import generate_dataset
    from .packs import get_pack

    pack = get_pack(args.pack)
    out = Path(args.out or Path("datasets") / pack.name)
    cases = generate_dataset(pack, args.size, args.seed, out)
    print(f"Wrote {len(cases)} cases to {out}/ (train 70%, val 15%, test 15%).")


def cmd_data_fit(args):
    from .packs import get_pack

    pack = get_pack(args.pack)
    try:
        fitted = pack.fit_from_dataset(args.data, args.out)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    questions = fitted.get("questions", fitted)
    print(f"Fitted {len(questions)} questions from {args.data} (train split).")
    for question, f in list(questions.items())[:5]:
        print(f"  difficulty {f['difficulty']:5.2f}  discrimination {f['discrimination']:.1f}  {question[:60]}")
    if len(questions) > 5:
        print(f"  ... and {len(questions) - 5} more")
    for name, value in fitted.get("score_noise", {}).items():
        print(f"  score noise ({name} grading): {value}")


def cmd_data_grader_eval(args):
    from .packs import get_pack

    pack = get_pack(args.pack)
    try:
        report = pack.grader_report(args.data, args.split)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{'grader':10} {'answers':>8} {'error':>7} {'bias':>7} {'pass agreement':>15}")
    for r in report:
        print(f"{r['grader']:10} {r['answers']:8d} {r['mae']:7.3f} {r['bias']:+7.3f} {r['pass_agreement']:15.1%}")


def add_simulation_options(p):
    p.add_argument("--data", help="dataset folder (from `verity data generate`); default: fresh simulated people")
    p.add_argument("--mock-llm", action="store_true", help="run the pack's LLM code path with its mock LLM")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="verity", description="Verify claims with adaptive questions.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("packs", help="list packs").set_defaults(func=cmd_packs)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("play", help="answer questions in the terminal")
    p.add_argument("pack")
    p.add_argument("--input", action="append", metavar="KEY=VALUE", help="pack input; KEY=@file reads a file")
    p.add_argument("--policy", default="greedy")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("evaluate", help="compare policies on simulated people")
    p.add_argument("pack")
    p.add_argument("--policies", nargs="*", help="default: every policy the pack offers")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="dataset split (with --data)")
    add_simulation_options(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="train an RL policy on simulated people")
    p.add_argument("pack")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--question-cost", type=float, default=0.02, help="reward penalty per question asked")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="default: models/<pack>/policy.pt")
    add_simulation_options(p)
    p.set_defaults(func=cmd_train)

    data = sub.add_parser("data", help="synthetic datasets").add_subparsers(dest="data_command", required=True)
    p = data.add_parser("generate", help="generate a synthetic dataset")
    p.add_argument("pack")
    p.add_argument("--size", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="default: datasets/<pack>")
    p.set_defaults(func=cmd_data_generate)

    p = data.add_parser("fit", help="fit question parameters from a dataset")
    p.add_argument("pack")
    p.add_argument("--data", required=True)
    p.add_argument("--out", help="default: the pack's fitted-parameters file")
    p.set_defaults(func=cmd_data_fit)

    p = data.add_parser("grader-eval", help="measure graders against the true answer quality")
    p.add_argument("pack")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.set_defaults(func=cmd_data_grader_eval)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
