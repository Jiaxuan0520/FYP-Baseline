# Technical Specification

## Common Input Schema

Each evaluation record contains:

- `session_id`: string
- `profile_prior`: float in `[0, 1]`
- `expert_ready`: boolean session label
- `turns`: ordered list of turn observations

Each turn observation contains:

- `competency`: one of `communication`, `teamwork`, `problem_solving`, `ownership`, `learning_agility`
- `star`: numeric STAR evidence levels for `situation`, `task`, `action`, `result` in `[0, 1]`
- `evidence_quality`: float in `[0, 1]`
- `evidence_spans`: optional span identifiers
- `duplicate_evidence`: boolean
- `response_irrelevant`: boolean

## Fixed-Script Question Order

The fixed script uses a locked 8-turn order:

1. Communication overview
2. Teamwork example
3. Problem-solving example
4. Ownership example
5. Learning-agility example
6. Communication follow-up
7. Teamwork follow-up
8. Closing prompt

## Score-Only Formula

Per turn evidence score:

- `evidence_score = mean([situation, task, action, result])`
- If `action == 0`, the turn is treated as insufficient evidence for strong support.
- Duplicate evidence is ignored when it repeats the same turn signature.

Competency score:

- `competency_score[c] = mean(distinct_accepted_scores[c])`

Session readiness:

- `session_ready = mean(competency_score) >= locked_threshold`

## Rule-Policy Decision Tree

1. Close if the turn budget is exhausted or readiness is already satisfied.
2. If the response is irrelevant, probe once or switch.
3. If action is missing or materially weak, probe action.
4. Else if result is missing or materially weak, probe result.
5. Else if context is unclear, probe context.
6. Else deepen the current competency or switch to the least covered competency.

Reason codes are recorded for every decision.

## BCT Assumptions

- The Bayesian controller uses the same shared STAR observation schema as all baselines.
- Profile priors are bounded and cannot satisfy a competency on their own.
- Uncertainty is represented with per-competency posteriors and entropy-based routing.
- Duplicate evidence is discounted.
- The full BCT policy can be replaced by rule-based sequencing for the sequencing ablation.

## Stopping Rules

- Stop at 8 turns.
- Stop when at least 4 competencies are judged ready.
- For the demo benchmark, the same stopping limits are used across all controllers.

## Planned Split

- Development: threshold tuning, probe limits, and prior settings.
- Test: final locked evaluation only.
- Live study: fixed-script versus full BCT, counterbalanced question sets.