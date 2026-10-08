# How Verity models claim verification

This document explains, with the mathematics, how Verity decides whether a person's
claims are true: what is modelled, how the model is fitted, how questions are chosen, how
the reinforcement-learning policy is trained, and what we learned while building it.
Code references are given as `file:function`. Plain-language summaries open each section.

> All numbers below come from **synthetic** data (see [specs/dataset.md](specs/dataset.md)).
> They show whether one method beats another under stated assumptions, not real-world accuracy.

---

## 1. The problem

*In plain words: someone claims things; we may ask a limited number of questions; we must
end with "supported", "refuted" or "not sure yet", while asking as few questions as possible
and almost never being confidently wrong.*

A session has a set of claims $\mathcal{C}$ (e.g. "knows SQL at mid level", "works at X").
At each step the system either asks one question (a **probe**) from a candidate set, or stops.
Each answer is graded into a score $s \in [0, 1]$. At the end every claim gets a status, and the
session a verdict:

$$
\text{verdict} = \begin{cases}
\text{refuted} & \text{if any claim is refuted} \\
\text{supported} & \text{if every claim is supported} \\
\text{uncertain} & \text{otherwise.}
\end{cases}
$$

This is **sequential hypothesis testing with question selection**: a partially observable
decision problem in which the hidden state is "what the person really knows", observations
are graded answers, and actions are "ask probe $j$" or "stop".

## 2. Claims live on an ordered scale

*In plain words: a yes/no claim and a "senior-level SQL" claim are the same kind of object,
a point on a ladder.*

Every claim $c$ has ordered levels $0, \dots, L_c - 1$ and a claimed level $k_c$
(`core/types.py:Claim`). The person's real level is $\ell_c$. The claim **holds** iff

$$\ell_c \ge k_c .$$

A yes/no claim is the case $L_c = 2$, $k_c = 1$ (levels "false", "true"). A CV skill uses
$L = 5$: none, beginner, junior, mid, senior.

## 3. How a person answers a question: item response theory

*In plain words: each question has a difficulty; the better someone is, the more likely they
answer it well; the curve between "hopeless" and "sure" has a steepness.*

A probe $j$ testing claim $c$ has a difficulty $b_j$, a discrimination $a_j$, a guessing floor
$g$ and a slip rate $u$. A person whose effective ability on $c$ is $\theta$ passes it (or, for
free text, scores on average) with

$$
r_j(\theta) = g + (1 - g - u)\,\sigma\big(a_j(\theta - b_j)\big), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

(`core/belief.py:irt_rate`, with $g = u = 0.05$). This is the **three-parameter logistic
model** of item response theory, with a slip term. Probes store $r_j$ at each integer level as
`pass_rates`; between levels we interpolate linearly (`rate_at`). A yes/no probe can simply
give $r_j = (p_\text{false}, p_\text{true})$.

## 4. The person: real level, sharpness and fatigue

*In plain words: two people at the same level are not identical, and everyone gets a bit
worse as an interview goes on.*

The effective ability of a person on claim $c$, when $t$ questions have already been asked, is

$$
\theta_c(t) = \ell_c + \delta - f\,t, \qquad \delta \sim \mathcal{N}(0, \sigma_\delta^2),
$$

- $\delta$ is the **person factor**: one offset shared by all claims of the session
  (a sharp person is sharp on every skill);
- $f$ is **fatigue**, in levels lost per question already asked.

Both are optional (`BeliefModel(person_spread=σ_δ, fatigue=f)`); with $\sigma_\delta = f = 0$
the model is a plain per-claim model.

## 5. How an answer counts as evidence

*In plain words: a multiple-choice answer is right or wrong; a free-text answer gets a grade,
and a grade of 0.3 is perfectly normal for a mid-level person on a hard question, so it must
not be treated as "70% wrong".*

The likelihood of a score $s$ given ability $\theta$ (`BeliefModel.likelihood`):

- **Multiple choice** ($s \in \{0, 1\}$): Bernoulli,
  $\;p(s \mid \theta) = r^{s}(1 - r)^{1-s}$ with $r = r_j(\theta)$.
- **Free text**: the grade is the expected score plus noise,
  $\;s \sim \mathcal{N}\big(r_j(\theta), \sigma_s^2\big)$, so
  $\;p(s \mid \theta) \propto \exp\!\big(-(s - r_j(\theta))^2 / 2\sigma_s^2\big)$.

**Why not read $s$ as "the chance it passed"?** The first version used the mixture
$s\,r + (1 - s)(1 - r)$. For a probe with rates $(0.12, 0.32, 0.68)$ at junior / mid / senior,
a mid-level person typically scores $s = 0.32$. The mixture gives junior
$0.32 \cdot 0.12 + 0.68 \cdot 0.88 = 0.637$ but mid only $0.32 \cdot 0.32 + 0.68 \cdot 0.68 = 0.565$:
the typical mid-level answer counts as evidence for *junior*. Averaged over many answers this
systematically underrates true claims. We found it by measuring calibration (§10), and the
Gaussian model fixes it: the likelihood is highest at the level whose expected score is
closest to the observed one (decision record 0006).

## 6. The prior: base rates of over-claiming

*In plain words: before any question, how likely is it that someone's real level is what
they claimed, one below, two below, …? That is a fact about the population, and it can be
measured.*

The starting belief over the real level given the claimed level $k$ uses the **gap
distribution** $q(d) = \Pr(\ell - k = d)$, estimated from data (`BeliefModel(gap_prior=q)`):

$$
\pi(\ell \mid k) \propto q(\ell - k) + \epsilon, \qquad \epsilon = 0.01 .
$$

Without data, a neutral prior puts probability $p_0 = 0.5$ evenly on the levels where the
claim holds and $1 - p_0$ evenly below. In the synthetic data $q(0) \approx 0.60$ (claims are
exact), $q(-1) \approx 0.17$, $q(-2) \approx 0.09$, and a few percent are under-stated
($q(+1) \approx q(+2) \approx 0.04$, skills listed without a level). Using $q$ instead of the
neutral prior was the single largest improvement (§11).

## 7. The belief: exact Bayesian updating on a grid

*In plain words: we keep, for every possible person offset and every level of every claim,
how plausible it is, and update all of it after each answer.*

Given answers $D$, the posterior factorises given $\delta$:

$$
p(\delta, \ell_{1..C} \mid D) \;\propto\; p(\delta) \prod_{c} \Big[ \pi(\ell_c \mid k_c) \prod_{(j,t,s) \in D_c} p\big(s \mid \ell_c + \delta - f t\big) \Big].
$$

We represent $\delta$ on a grid of $K = 7$ points $\delta_i \in [-2\sigma_\delta, 2\sigma_\delta]$
with weights $w_i \propto \mathcal{N}(\delta_i; 0, \sigma_\delta^2)$ (`person_grid`). For each grid
point we keep each claim's level distribution $P_{i,c}(\ell)$. After an answer $s$ to probe $j$ on
claim $c$ at step $t$ (`BeliefModel.update`):

$$
\tilde P_{i,c}(\ell) = P_{i,c}(\ell)\, p(s \mid \ell + \delta_i - f t), \qquad
E_i = \sum_\ell \tilde P_{i,c}(\ell), \qquad
P_{i,c} \leftarrow \frac{\tilde P_{i,c}}{E_i}, \qquad w_i \leftarrow \frac{w_i E_i}{\sum_k w_k E_k}.
$$

$E_i$ is how well offset $\delta_i$ explains the answer, so an answer on one claim shifts the
offset weights and, through them, every other claim. Finally each $P_{i,c}$ gets a floor of
$10^{-3}$ and is renormalised, so a single answer never rules a level out. Marginals:

$$
P_c(\ell) = \sum_i w_i P_{i,c}(\ell), \qquad
\Pr(\text{claim } c \text{ holds}) = \sum_{\ell \ge k_c} P_c(\ell).
$$

The claim is **supported** when this is $\ge 0.9$, **refuted** when $\le 0.1$, else uncertain.

## 8. Choosing the next question

*In plain words: ask the question whose answer is expected to settle the most doubt, about the
claim most likely to sink the verdict.*

**Expected information gain** of probe $j$ on claim $c$ (`BeliefModel.expected_gain`), with
$H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ and $h_c = \Pr(\text{claim } c \text{ holds})$:

$$
G_j = H(h_c) - \sum_{s \in S} \Pr(s)\, H\big(h_c \mid s\big),
$$

where $S = \{0, 1\}$ for multiple choice and $S = \{0, 0.1, \dots, 1\}$ for free text, and
$\Pr(s) \propto \sum_{i,\ell} w_i P_{i,c}(\ell)\, p(s \mid \ell + \delta_i - f t)$.

**Greedy policy** (`core/policies.py:GreedyPolicy`): stop if a claim is refuted (the verdict
can no longer change) or no undecided claim has probes left; otherwise take the undecided
claim with the **lowest** $h_c$ and ask its probe with the largest $G_j$. Focusing on the
weakest claim is a deliberate choice: one refuted claim decides the whole verdict, so digging
into the most doubtful claim settles the session fastest. (Spreading questions evenly was worse
than random in our first measurements.)

## 9. Learning when to ask and when to stop: reinforcement learning

*In plain words: a small neural network looks at the current beliefs and decides what to ask or
whether to stop; it learns by playing many simulated interviews and being rewarded for correct
verdicts and penalised for wrong ones and for every question.*

### 9.1 The decision problem
- **State**: the belief (§7) plus counters; summarised by features (§9.3).
- **Actions**: any unasked probe of an undecided claim, or **stop**.
- **Reward** (`core/simulation.py:episode_reward`): at the end $+1$ for a correct verdict,
  $-3$ for a wrong one, $-0.25$ for "uncertain"; and $-0.02$ per question asked.

The weights encode a product decision: telling a genuine senior "you are not senior" is three
times as bad as a correct verdict is good. With these weights, deciding "supported" now
beats stopping undecided only when $p - 3(1-p) > -0.25$, i.e. $p > 0.69$, but asking another
question may be worth more still. That trade-off is what the policy learns.

### 9.2 Actor-critic training (`rl/train.py`)
The policy $\pi_\phi(a \mid s)$ is a softmax over one score per candidate probe and one for
"stop"; a value head $V_\psi(s)$ predicts the return. For decision $t$ with return-to-go
$G_t$ (final outcome minus the cost of the questions from $t$ on):

$$
A_t = G_t - V_\psi(s_t), \qquad
\mathcal{L} = -\frac{1}{N}\sum_t \hat A_t \log \pi_\phi(a_t \mid s_t)
+ \tfrac{1}{2} \frac{1}{N}\sum_t \big(V_\psi(s_t) - G_t\big)^2
- \beta \frac{1}{N}\sum_t \mathcal{H}\big(\pi_\phi(\cdot \mid s_t)\big),
$$

with advantages $\hat A_t$ normalised per batch of 16 episodes and $\beta = 0.01$ (Adam,
learning rate $3 \cdot 10^{-3}$).

Before RL, an **imitation warm start** trains the network to copy the greedy policy
(cross-entropy on its choices, 300 sessions), so RL starts from a sensible policy. Every 500
episodes the network's best-action version is scored on validation people; the best checkpoint
is kept.

### 9.3 Features (`rl/features.py`)
Domain-agnostic, so the same network works for every pack. Per candidate probe: $h_c$,
$1 - |2h_c - 1|$ (how undecided), $G_j$, the probe's separation $r_j(k_c) - r_j(k_c - 1)$,
$r_j(k_c)$, $k_c / (L_c - 1)$, questions already asked on $c$, the mean score on $c$ so far,
session progress, undecided flag, free-text flag, expected person offset. For the session
(stop score and value): share of decided claims, progress, min and mean $h_c$, mean
undecidedness, largest available gain, expected person offset.

## 10. Fitting the model to data

*In plain words: difficulty, steepness, fatigue, how much people differ, how noisy grades are,
and how often people over-claim are all measured from answers whose truth we know.*

All fits maximise the (quasi-)log-likelihood
$\;\sum s \log r + (1 - s)\log(1 - r)\;$ by grid search (`core/fitting.py`), on the
**train split only** (`packs/cv/tuning.py:fit_questions`):

1. **Items.** For each question, $(\hat b_j, \hat a_j)$ on a grid
   ($b \in [-1, L]$ step 0.1, $a \in \{0.8, 1.2, 1.7, 2.4, 3.2\}$), using answers at the
   respondents' real levels.
2. **Fatigue and person offsets.** For each $f$ on a grid (0–0.08), every person gets their
   best offset $\hat\delta_p$; $\hat f$ maximises the total likelihood.
3. **Items again**, with each answer placed at $\ell + \hat\delta_p - \hat f t$
   (one round of alternating maximisation).
4. **Person spread, free of estimation noise.** Estimating $\hat\delta_p$ from about 20 answers is
   noisy, so $\operatorname{sd}(\hat\delta_p)$ overstates $\sigma_\delta$. Instead we estimate the
   offset twice per person, from the odd-numbered and the even-numbered answers. Their errors are
   independent, so
   $$\operatorname{Cov}\big(\hat\delta^{A}_p, \hat\delta^{B}_p\big) = \operatorname{Var}(\delta) \quad\Rightarrow\quad \hat\sigma_\delta = \sqrt{\max(\widehat{\operatorname{Cov}}, 0)}.$$
5. **Score noise** $\hat\sigma_s$: root-mean-square distance between observed scores and the
   fitted expected score, separately for LLM-style grading and keyword grading.
6. **Gap prior** $\hat q(d)$: the share of claims with real − claimed level $= d$.

On the synthetic data the fit recovers the hidden truth well: question difficulty error 0.12
levels (vs 0.33 for the easy/medium/hard labels), fatigue $\hat f = 0.03$ (true mean 0.03).
The measured person spread is 0.47 levels. That is larger than the generator's "sharpness"
(0.35) because impostors who look answers up also behave like consistently sharper people.

**One tuned setting.** The CV pack uses $\sigma_\delta = 0.1$, not the measured 0.47: on the
validation split, larger values make the model cautious (fewer wrong verdicts, many more
"uncertain") and lower the reward. $\sigma_\delta$ therefore works as a **caution dial**.

## 11. Evaluation methodology

*In plain words: never measure on the data you tuned on, use enough people that differences
are real, and check that "90% likely" really means 90%.*

- **Splits.** Datasets are split train / val / test (70 / 15 / 15). Fitting and training use
  train; settings and checkpoints are chosen on val; results are reported on test.
- **Enough people.** The reward of one session has a standard deviation of about 1.3, so the
  mean over $n$ sessions is accurate to about $1.3 / \sqrt{n}$: ±0.06 with 450 people (too noisy
  to rank policies) and ±0.024 with 3,000. Results below use the 3,000-person test split of a
  20,000-person dataset (`datasets/cv-large`).
- **Calibration.** For each claim at the end of a session we record the model's
  $h_c$ and whether the claim really holds. The Brier score
  $\frac{1}{N}\sum (h_c - y_c)^2$ and a reliability table (stated vs. observed frequency per
  probability bin) show whether probabilities can be trusted. This is how the evidence bug
  (§5) and the prior problem (§6) were found.
- **Ablations.** Each new model part is switched off one at a time to see what it contributes.

## 12. What we found, step by step

Greedy policy, keyword grading, test split of the 3,000-person dataset of the time (450 people,
±0.06 on reward); each row builds on the one above. Rows 1–2 used the first synthetic CVs, rows
3–5 the same personas with answer positions (for fatigue); all before the realistic CVs.

| Step | Accuracy | Uncertain | Wrong | Questions | What changed |
| --- | --- | --- | --- | --- | --- |
| Mixture evidence, label difficulties | 44% | 45% | 11% | 9.2 | starting point |
| + fitted difficulties, Gaussian evidence for free text | 52% | 33% | 15% | 6.6 | evidence bug fixed; more decisions |
| + fatigue | 60% | 30% | 10% | 6.4 | late answers no longer count as fails |
| + empirical gap prior | 68% | 25% | 7% | 5.8 | honest candidates no longer underrated |
| + person factor (σδ = 0.1) | 68% | 26% | 6% | 6.0 | final belief model (decision record 0007) |

**Final model on the realistic CVs** (3,000-person test split of `datasets/cv-large`, ±0.024 on reward):

| Grading | Accuracy | Uncertain | Wrong | Questions | Reward |
| --- | --- | --- | --- | --- | --- |
| Keyword (offline) | 65.0% | 29.0% | 6.0% | 6.8 | +0.261 |
| Mock LLM | 78.8% | 14.5% | 6.7% | 4.9 | +0.451 |

The identity pack (multiple choice, yes/no claims) is unaffected by all of the above: every
extension reduces to the original model for it, and its results are identical.

**Reinforcement learning.** See §13 for the trained policy compared with greedy.

## 13. Reinforcement learning results

Setup: `verity train cv --data datasets/cv-large --episodes 6000 --eval-every 500` with seeds 0, 1, 2
(14,000 training people, checkpoints chosen on 1,500 validation people), keyword grading; then all
policies on the same 3,000 test people (±0.024 on reward).

| Policy | Accuracy | Uncertain | Wrong | Questions | Reward (test) | Best validation reward |
| --- | --- | --- | --- | --- | --- | --- |
| **Greedy** (§8) | 65.0% | 29.0% | 6.0% | 6.8 | **+0.261** | +0.266 |
| Learned, seed 0 | 63.0% | 29.7% | 7.3% | 6.8 | +0.199 | +0.263 |
| Learned, seed 1 | 62.3% | 32.0% | 5.8% | 6.9 | +0.232 | +0.273 |
| Learned, seed 2 | 63.4% | 30.1% | 6.5% | 6.8 | +0.228 | +0.268 |
| Random | 56.7% | 38.1% | 5.3% | 11.3 | +0.087 | – |

**The learned policy does not beat greedy.** Its best validation scores only *equal* greedy's,
and on the untouched test split all three seeds are below it. Two lessons:

1. **Selection noise is real.** Picking the best of twelve noisy validation scores inflates the
   winner (+0.27 on validation, +0.20–0.23 on test). Earlier, with only 450 validation people,
   this produced convincing-looking "wins" that were noise; hence the large dataset (§11).
2. **With a well-specified belief, greedy information gain is already near-optimal.** RL can
   only improve on it by exploiting structure that the belief model does not capture. Every time
   we found such structure (biased evidence, fatigue, base rates, person differences) the right
   move was to put it **into the belief model**, which improved every policy at once and is
   explainable, instead of hoping a network learns it implicitly.

**Recommendation.** Keep `greedy` as the default. Keep RL as an optional, measured component;
it becomes worthwhile when the action space or the world has structure a one-step rule cannot
use, for example:
- questions generated on the fly by an LLM (a large action space with uneven quality),
- costs that differ per question (time, money, candidate stress),
- multi-step strategies such as "ask an easy warm-up first, because nervous candidates
  under-perform at the start" (a measured effect the belief does not model),
- offline RL from logged real sessions, where the simulator is no longer the bottleneck.

The training code, the imitation warm start, checkpoints and the evaluation harness are in place
for those cases: any new policy must beat greedy on a few thousand held-out test people.

## 14. Limitations and next steps

- **Synthetic world.** Answer texts are templated; personas are independent across skills
  except through $\delta$; all rates are assumptions. Real sessions (with consent) should
  replace them; every fitting step in §10 runs unchanged on real data.
- **Honest candidates sit on the boundary.** A genuine "mid" claim is true at exactly the claimed
  level, where evidence is most ambiguous; this caps achievable accuracy for any method.
- **Fatigue in the belief is a population average**; individual fatigue varies.
- **Question bank** is small (8 skills × 7 questions); more questions per skill is the most
  direct way to reduce "uncertain".
- **RL** currently does not beat greedy (§13); it is kept as an optional, measured component.
  How to make it worthwhile: [rl-improvements.md](rl-improvements.md).
