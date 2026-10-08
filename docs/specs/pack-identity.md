# Spec: identity pack

Code: `verity/packs/identity/`. Name: `identity`.

Checks that someone really works, studied, lives and was born where they say, by asking
about places near each of those locations. It is built on the original project's data
(`data/`) and keeps the original model (`legacy/`) runnable as a policy.

## Inputs
None from the user. `inputs["graph"]` (optional, default `0`) selects the applicant graph
from `data/preprocessed_graphs/test.json`. That file currently holds one graph.

## Blocks
| Block | Class | Behaviour |
| --- | --- | --- |
| Claims | `IdentityClaims` | One claim per `personal_information` entry: `company`, `university`, `live_in`, `born_in` ("Works at …", "Studied at …", "Lives at …", "Was born in …"). |
| Knowledge | `PlacesGraph` | Node per place; edge `personal node -[relation id]-> nearby place`, provenance `google-places`. `graph.data["processed"]` holds the legacy preprocessed graph. |
| Probes | `PlaceQuestions` | One multiple-choice probe per (attribute, nearby place): template question (`data/languageTemplates.json`), correct place + 2 look-alike distractors (`data/answersLibrary.json`) + "Not Sure". Probe id `"<attribute node>:<place node>"`. `p_true = 0.85`, `p_false = 0.30`. |
| Assessor | core `ChoiceAssessor` | Exact match. |
| Belief | core default | prior 0.5, accept 0.9, reject 0.1. |
| Policies | `greedy`, `random`, `learned`, `legacy` | `legacy` = `LegacyRLPolicy`. |

`max_questions = 20`, `show_feedback = False` (never reveal correct answers).

## Legacy policy
`LegacyRLPolicy` wraps the original GNN + manager/worker model (`legacy/policy.py`,
checkpoint in `data/checkpoints/`). It keeps the model's own dialogue state:
- When the model's worker asks about a place, the matching probe is returned.
- After the answer, the model is told whether it was correct.
- When the model's manager decides fraud / not fraud, the session stops and the note
  "The original RL model decided: fraud." (or "not fraud.") is added to the verdict.

The verdict itself still comes from the belief model, so all policies are judged the same way.

## Simulation
`sample_case`: a graph, then the number of false attributes `k` with weights
`{0: 4, 1: 1, 2: 1, 3: 1, 4: 1}` (from the original simulator: 50% genuine, the rest
fraud types 1–4); `k` random attributes are false.

## Known limits
- Answers come from public map data, so a determined impostor can look them up. Measuring
  that ("searchability") per probe and lowering `p_true - p_false` accordingly is future work.
- Only one applicant graph ships with the repo; `notebooks/ConstructGraph.ipynb` builds more
  (needs a Google Places API key).
