# Making reinforcement learning worth it

Status: proposal. Background and current results: [modeling.md](modeling.md) §9 and §13.

## Where we are

On the CV pack, a policy trained with imitation + actor-critic does **not** beat the greedy
rule (3,000 held-out test candidates, keyword grading):

| Policy | Right | Wrong | Questions | Reward |
| --- | --- | --- | --- | --- |
| Greedy (default) | 65.0% | 6.0% | 6.8 | **+0.261** |
| Learned, 3 seeds | 62–63% | 6–7% | 6.8 | +0.20 to +0.23 |

## Greedy vs RL in one paragraph each

**Greedy** is a fixed rule that looks **one step ahead**: take the claim most likely to be false,
ask the question expected to reduce doubt about it the most, stop when the verdict cannot change.
No training, fully explainable.

**RL** is a small neural network that learns a questioning strategy from thousands of simulated
interviews, rewarded for right verdicts and penalised for wrong ones and for every question.
It can, in principle, plan **several steps ahead** and learn when stopping early is worth the risk.

## Why RL does not win today

The belief model already captures everything we know (fitted difficulties, person factor,
fatigue, base rates of over-claiming). In that world the one-step rule is close to optimal,
because:

- every question costs the same,
- answers are independent once the person factor is accounted for,
- the question set is small and fixed,
- claims do not inform each other except through the person factor.

So there is no multi-step advantage to learn, and RL, which learns from noisy rewards, ends up
slightly below the rule it started from.

**Rule of thumb:** RL beats a good one-step rule only when *planning ahead* pays off. The
improvements below make the problem one where it does.

## Improvements

Ordered roughly by expected value. Each one is a change to the *world* (simulator, rewards,
actions), except the last group, which improves the *training*. #8 (graph-guided questioning)
is the long-term direction that ties most of the others together.

### 1. Questions with different costs ⭐
Real interviews mix 30-second questions with 10-minute tasks.
- **Change:** give each probe a `cost` (minutes); the reward charges the real cost instead of a
  flat 0.02 per question.
- **Why RL helps:** greedy maximises information per question; the right target is information per
  minute, and whether an expensive probe is worth it depends on what is still undecided.
- **Measure:** reward and average interview minutes vs greedy (and a "greedy per minute" baseline).

### 2. A fixed time budget with more claims than time ⭐
For example a 10-minute interview for a CV with five skills.
- **Change:** session budget in minutes; the verdict covers the claims that matter most for the job
  (job-weighted claims).
- **Why RL helps:** deciding which claims to skip and how deep to go on each is multi-step planning.
- **Measure:** weighted accuracy of job-relevant claims within the budget.

### 3. Warm-up and nerves
Candidates often under-perform on the first one or two questions.
- **Change:** add a start-of-session penalty to the persona simulator (and later fit it from real
  data, like fatigue).
- **Why RL helps:** the best plan becomes "easy question first, hard ones later", a sequence
  greedy cannot find because the first question looks uninformative on its own.

### 4. Related skills
Knowing Django predicts knowing Python; React implies JavaScript.
- **Change:** a skill graph with correlations in the simulator; optionally in the belief model.
- **Why RL helps:** one answer informs several claims, so the order of questions matters.
  (If we add the correlations to the belief model, greedy gets better too; measure both.)

### 5. LLM-generated questions
Questions written on the fly vary in quality and there are thousands of possible ones.
- **Change:** actions become "ask about topic T at difficulty D"; an LLM (mock in simulation)
  writes the question; question quality is noisy.
- **Why RL helps:** a learned policy can discover which kinds of questions are reliably informative,
  where a fixed formula needs trustworthy parameters for every single question.

### 6. Candidates who try to game the interview
Some candidates look answers up or use AI help.
- **Change:** an adversarial persona that answers well on searchable or predictable questions.
- **Why RL helps:** a predictable rule can be exploited; a policy trained against cheaters can learn
  to prefer questions that are hard to fake (and stay unpredictable).

### 7. Train on real sessions ⭐ (long term)
The simulator is the real limit: RL is only as good as the world it trains in.
- **Change:** log real practice sessions (with consent, no personal data kept), fit the simulator
  to them, and later use offline RL on the logs.
- **Why it matters:** real data contains effects we did not think to model; that is where a learned
  policy has the most to gain.

### 8. Graph-guided questioning: drill down and grow the graph ⭐
This was the original idea of the project, and it is the most natural place for RL.

**What exists today**
- The original identity model (`packs/identity/legacy/`) already picks *graph nodes*: a
  manager chooses a personal attribute node, a worker chooses a neighbouring place node, with a
  GNN embedding the graph. But it only goes **one hop** and never creates nodes.
- In Verity the knowledge graph is built (CV pack: skill → topic nodes such as "joins",
  "indexes", "transactions") but **not used for questioning**: probes come from a fixed bank,
  topics are not linked to questions, and every policy (greedy or RL) only picks from a fixed
  candidate list or stops. No policy walks the graph, goes deeper on a weak spot, or adds nodes.

**The idea**
Questioning becomes navigation of the knowledge graph, like a good interviewer:
"SQL → indexes → why are writes slower with an index? → covering indexes".
- **Actions** (hierarchical, like the original manager/worker design):
  1. choose a claim / top node (manager),
  2. choose a node to probe: stay, go **deeper** (child topic), go **sideways** (related or
     prerequisite topic), or **back up**,
  3. **expand**: ask an LLM to create a new sub-topic node and its question when the graph has
     no deeper node (the mock LLM does this in simulation, from a topic list),
  4. stop.
- **State**: the belief per node, plus graph structure. A GNN (as in the original model, now
  with relation types) summarises "what we know where".

**What else it needs**
- **Topic-level belief**: a level per topic node, linked to its skill and to related topics (a
  child topic's knowledge informs its parent; prerequisites constrain each other). Without this,
  drilling down tells the model nothing new.
- **Topic-level simulator**: personas know some topics deeply and others not at all (real
  "patchy" knowledge), so drilling down has something to discover.
- **Graph-aware greedy baseline**: one-step information gain over neighbouring nodes, so RL is
  compared with a fair rule and not with one that cannot walk the graph.
- Question bank or LLM questions tagged with their topic node.

**Why it should make the solution better**
- Finds the real edge of someone's knowledge with fewer, more targeted questions.
- Much better coaching feedback: "solid on joins, weak on transactions", not only "mid-level SQL".
- Harder to game: follow-up questions depend on earlier answers.

**Why it makes RL worthwhile**
Going deeper pays off only several questions later, and expanding the graph creates actions
that did not exist before. That is multi-step planning over a large, structured, changing action
space: exactly where a one-step rule is weak and a learned (hierarchical) policy can win.

**Measure:** reward, questions, and topic-level accuracy (did we find the right weak and strong
topics?) against the graph-aware greedy baseline, on held-out personas, at least three seeds.

### 9. Better training (helps, but not enough on its own)
- **Train against greedy, not against zero.** Play greedy and the learned policy on the *same*
  candidate and reward the difference. This removes most of the noise (paired comparison).
- **Fine-tune slowly from the greedy copy** and stop when validation gets worse.
- **PPO** (a more stable policy-gradient method) instead of plain actor-critic.
- **More episodes and a larger validation set** for checkpoint selection, to avoid keeping
  checkpoints that were only lucky (we measured this: +0.27 on validation, +0.20 to +0.23 on test).

## Recommended plan

1. **Implement #1 and #2 together** (question costs + a time budget with job-weighted claims).
   They are realistic for interview practice, cheap to simulate, and exactly the setting where
   planning ahead should beat a one-step rule.
2. Add a **"greedy per minute"** baseline, so RL is compared with the strongest simple rule.
3. Train with the **paired reward** from #8.
4. Accept RL only if it beats both greedy baselines on a few thousand held-out test candidates
   (reward ± 0.024), across at least three seeds.
5. In parallel, prepare #7 (consented session logging), which unlocks everything else.
6. Then move to #8, graph-guided questioning: topic-level belief and simulator first, a
   graph-aware greedy baseline second, then a hierarchical RL policy that navigates and grows
   the graph.

## How we will know it worked

| Check | Bar |
| --- | --- |
| Test reward vs best greedy baseline | higher, beyond ±0.024, for every seed |
| Wrong verdicts | not higher than greedy |
| Interview length | same or shorter |
| Explanation | per decision, show the top features behind the choice |
