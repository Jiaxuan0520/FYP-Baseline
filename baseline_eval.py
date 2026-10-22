from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Set, Tuple


COMPETENCIES = [
    "communication",
    "teamwork",
    "problem_solving",
    "ownership",
    "learning_agility",
]

STAR_COMPONENTS = ["situation", "task", "action", "result"]
STAR_WEIGHTS = {
    "situation": 0.15,
    "task": 0.10,
    "action": 0.50,
    "result": 0.25,
}

MAX_TURNS = 8
REQUIRED_READY_COUNT = 4
DEFAULT_LOCKED_THRESHOLD = 0.65


@dataclass
class StateObservation:
    competency: str
    star: Dict[str, float]
    evidence_quality: float = 1.0
    evidence_spans: Tuple[str, ...] = ()
    duplicate_evidence: bool = False
    response_irrelevant: bool = False
    profile_prior: float = 0.5
    expert_ready: Optional[bool] = None
    transcript_id: Optional[str] = None


@dataclass
class NextAction:
    action_type: str
    target_competency: Optional[str] = None
    template_id: Optional[str] = None
    reason_code: str = ""


@dataclass
class CompetencyState:
    posterior: float = 0.5
    score: float = 0.0
    seen_signatures: Set[str] = field(default_factory=set)
    probes: Dict[str, int] = field(default_factory=lambda: {"action": 0, "result": 0, "context": 0})
    evidence_count: int = 0


@dataclass
class SessionBenchmark:
    session_id: str
    profile_prior: float
    expert_ready: bool
    turns: List[StateObservation]


class BaseController:
    name = "base"

    def __init__(self, locked_threshold: float = DEFAULT_LOCKED_THRESHOLD, max_turns: int = MAX_TURNS):
        self.locked_threshold = locked_threshold
        self.max_turns = max_turns
        self.reset()

    def reset(self, profile_prior: float = 0.5) -> None:
        self.current_competency = COMPETENCIES[0]
        self.turn_index = 0
        self.profile_prior = profile_prior
        self.closed = False
        self.state: Dict[str, CompetencyState] = {
            competency: CompetencyState(posterior=profile_prior)
            for competency in COMPETENCIES
        }
        self.history: List[Tuple[StateObservation, NextAction]] = []

    def decide(self, observation: StateObservation) -> NextAction:
        raise NotImplementedError

    def observe(self, observation: StateObservation) -> None:
        signature = self._signature(observation)
        state = self.state[self.current_competency]
        if observation.duplicate_evidence or signature in state.seen_signatures:
            state.seen_signatures.add(signature)
            return

        state.seen_signatures.add(signature)
        state.evidence_count += 1
        evidence_score = self._evidence_score(observation.star)
        state.posterior = self._update_posterior(state.posterior, evidence_score, observation.evidence_quality)
        state.score = mean([state.score, evidence_score]) if state.evidence_count > 1 else evidence_score

    def apply_action(self, action: NextAction) -> None:
        if action.action_type == "CLOSE":
            self.closed = True
            return
        if action.target_competency in COMPETENCIES:
            self.current_competency = action.target_competency

    def session_probability(self) -> float:
        raise NotImplementedError

    def ready_count(self) -> int:
        raise NotImplementedError

    def _signature(self, observation: StateObservation) -> str:
        rounded = tuple(round(observation.star.get(component, 0.0), 2) for component in STAR_COMPONENTS)
        spans = ",".join(observation.evidence_spans)
        return f"{observation.competency}:{rounded}:{spans}"

    def _evidence_score(self, star: Dict[str, float]) -> float:
        return sum(STAR_WEIGHTS[key] * float(star.get(key, 0.0)) for key in STAR_COMPONENTS)

    def _update_posterior(self, prior: float, evidence_score: float, evidence_quality: float) -> float:
        bounded_prior = min(max(prior, 0.02), 0.98)
        effective_evidence = max(0.05, min(0.95, evidence_score * evidence_quality))
        odds = bounded_prior / (1.0 - bounded_prior)
        likelihood_ratio = math.exp(4.0 * (effective_evidence - 0.5))
        posterior = (odds * likelihood_ratio) / (1.0 + odds * likelihood_ratio)
        return min(max(posterior, 0.0), 1.0)

    def _target_by_uncertainty(self) -> str:
        def entropy(value: float) -> float:
            value = min(max(value, 1e-6), 1.0 - 1e-6)
            return -(value * math.log(value) + (1.0 - value) * math.log(1.0 - value))

        return max(COMPETENCIES, key=lambda competency: entropy(self.state[competency].posterior))

    def _least_covered_competency(self) -> str:
        return min(COMPETENCIES, key=lambda competency: self.state[competency].posterior)

    def _ready(self) -> bool:
        return self.ready_count() >= REQUIRED_READY_COUNT or self.turn_index >= self.max_turns


class FixedScriptController(BaseController):
    name = "fixed_script"

    def reset(self, profile_prior: float = 0.5) -> None:
        super().reset(profile_prior=profile_prior)
        self.script_index = 0

    def decide(self, observation: StateObservation) -> NextAction:
        if self.turn_index >= self.max_turns or self._ready():
            return NextAction(action_type="CLOSE", reason_code="TURN_LIMIT")
        template_id = f"fixed_q{self.script_index + 1}"
        return NextAction(action_type="SWITCH", target_competency=observation.competency, template_id=template_id, reason_code="LOCKED_SCRIPT")

    def apply_action(self, action: NextAction) -> None:
        super().apply_action(action)
        self.script_index += 1

    def session_probability(self) -> float:
        return mean(state.score for state in self.state.values())

    def ready_count(self) -> int:
        return sum(1 for state in self.state.values() if state.score >= self.locked_threshold)


class ScoreOnlyController(BaseController):
    name = "score_only"

    def reset(self, profile_prior: float = 0.5) -> None:
        super().reset(profile_prior=profile_prior)
        self._accepted_scores: Dict[str, List[float]] = {competency: [] for competency in COMPETENCIES}

    def decide(self, observation: StateObservation) -> NextAction:
        if self.turn_index >= self.max_turns or self._ready():
            return NextAction(action_type="CLOSE", reason_code="TURN_LIMIT")
        return NextAction(action_type="SWITCH", target_competency=observation.competency, template_id=f"score_q{self.turn_index + 1}", reason_code="FIXED_SEQUENCE")

    def observe(self, observation: StateObservation) -> None:
        signature = self._signature(observation)
        evidence_score = self._evidence_score(observation.star)
        state = self.state[self.current_competency]
        if observation.duplicate_evidence or signature in state.seen_signatures:
            state.seen_signatures.add(signature)
            return
        state.seen_signatures.add(signature)
        if observation.star.get("action", 0.0) <= 0.0:
            return
        self._accepted_scores[self.current_competency].append(evidence_score)
        state.score = mean(self._accepted_scores[self.current_competency])
        state.posterior = state.score

    def session_probability(self) -> float:
        return mean(state.score for state in self.state.values())

    def ready_count(self) -> int:
        return sum(1 for state in self.state.values() if state.score >= self.locked_threshold)


class RuleBasedController(BaseController):
    name = "rule_based"

    def decide(self, observation: StateObservation) -> NextAction:
        if self.turn_index >= self.max_turns or self._ready():
            return NextAction(action_type="CLOSE", reason_code="TURN_LIMIT")

        state = self.state[self.current_competency]
        action_missing = observation.star.get("action", 0.0) < 0.5
        result_missing = observation.star.get("result", 0.0) < 0.5
        context_missing = observation.star.get("situation", 0.0) < 0.5 or observation.star.get("task", 0.0) < 0.5

        if observation.response_irrelevant:
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="clarify", reason_code="IRRELEVANT")
        if action_missing and state.probes["action"] < 2:
            state.probes["action"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_action", reason_code="MISSING_ACTION")
        if result_missing and state.probes["result"] < 2:
            state.probes["result"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_result", reason_code="MISSING_RESULT")
        if context_missing and state.probes["context"] < 1:
            state.probes["context"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_context", reason_code="MISSING_CONTEXT")

        if state.posterior < self.locked_threshold:
            return NextAction(action_type="DEEPEN", target_competency=self.current_competency, template_id="deepen_current", reason_code="LOW_SUPPORT")
        return NextAction(action_type="SWITCH", target_competency=self._least_covered_competency(), template_id="switch_least_covered", reason_code="LOWEST_COVERAGE")

    def session_probability(self) -> float:
        return mean(state.posterior for state in self.state.values())

    def ready_count(self) -> int:
        return sum(1 for state in self.state.values() if state.posterior >= self.locked_threshold)


class BCTController(BaseController):
    name = "bct"

    def decide(self, observation: StateObservation) -> NextAction:
        if self.turn_index >= self.max_turns or self._ready():
            return NextAction(action_type="CLOSE", reason_code="TURN_LIMIT")

        state = self.state[self.current_competency]
        action_missing = observation.star.get("action", 0.0) < 0.5
        result_missing = observation.star.get("result", 0.0) < 0.5
        context_missing = observation.star.get("situation", 0.0) < 0.5 or observation.star.get("task", 0.0) < 0.5

        if action_missing and state.probes["action"] < 1:
            state.probes["action"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_action", reason_code="MISSING_ACTION")
        if result_missing and state.probes["result"] < 1:
            state.probes["result"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_result", reason_code="MISSING_RESULT")
        if context_missing and state.probes["context"] < 1:
            state.probes["context"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_context", reason_code="MISSING_CONTEXT")

        target = self._target_by_uncertainty()
        if target != self.current_competency:
            return NextAction(action_type="SWITCH", target_competency=target, template_id="entropy_switch", reason_code="MAX_ENTROPY")
        return NextAction(action_type="SWITCH", target_competency=target, template_id="entropy_retain", reason_code="MAX_ENTROPY_RETAIN")

    def session_probability(self) -> float:
        return mean(state.posterior for state in self.state.values())

    def ready_count(self) -> int:
        return sum(1 for state in self.state.values() if state.posterior >= self.locked_threshold)


class BCTUniformPriorController(BCTController):
    name = "bct_uniform_prior"

    def reset(self, profile_prior: float = 0.5) -> None:
        super().reset(profile_prior=0.5)


class BCTRuleSequencingController(BCTController):
    name = "bct_rule_sequencing"

    def decide(self, observation: StateObservation) -> NextAction:
        if self.turn_index >= self.max_turns or self._ready():
            return NextAction(action_type="CLOSE", reason_code="TURN_LIMIT")

        state = self.state[self.current_competency]
        action_missing = observation.star.get("action", 0.0) < 0.5
        result_missing = observation.star.get("result", 0.0) < 0.5
        context_missing = observation.star.get("situation", 0.0) < 0.5 or observation.star.get("task", 0.0) < 0.5

        if action_missing and state.probes["action"] < 1:
            state.probes["action"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_action", reason_code="MISSING_ACTION")
        if result_missing and state.probes["result"] < 1:
            state.probes["result"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_result", reason_code="MISSING_RESULT")
        if context_missing and state.probes["context"] < 1:
            state.probes["context"] += 1
            return NextAction(action_type="PROBE", target_competency=self.current_competency, template_id="probe_context", reason_code="MISSING_CONTEXT")

        if state.posterior < self.locked_threshold:
            return NextAction(action_type="DEEPEN", target_competency=self.current_competency, template_id="rule_deepen", reason_code="LOW_SUPPORT")
        return NextAction(action_type="SWITCH", target_competency=self._least_covered_competency(), template_id="rule_switch", reason_code="LOWEST_COVERAGE")


def build_demo_sessions() -> List[SessionBenchmark]:
    base_turns = [
        ("communication", {"situation": 1.0, "task": 0.8, "action": 0.0, "result": 0.0}),
        ("communication", {"situation": 1.0, "task": 0.8, "action": 0.5, "result": 0.2}),
        ("teamwork", {"situation": 1.0, "task": 1.0, "action": 1.0, "result": 0.4}),
        ("problem_solving", {"situation": 0.8, "task": 0.8, "action": 1.0, "result": 0.7}),
        ("ownership", {"situation": 1.0, "task": 1.0, "action": 1.0, "result": 0.8}),
        ("learning_agility", {"situation": 0.6, "task": 0.7, "action": 0.6, "result": 0.5}),
        ("teamwork", {"situation": 1.0, "task": 1.0, "action": 1.0, "result": 1.0}),
        ("ownership", {"situation": 1.0, "task": 1.0, "action": 1.0, "result": 1.0}),
    ]
    sessions: List[SessionBenchmark] = []
    difficulty_levels = [1.0, 0.95, 0.9, 0.75, 0.6, 0.5]
    label_threshold = 0.72
    for index, difficulty in enumerate(difficulty_levels):
        turns: List[StateObservation] = []
        for turn_index, (competency, star) in enumerate(base_turns):
            offset = 0.05 * ((index + turn_index) % 3)
            adjusted = {
                component: min(
                    1.0,
                    max(
                        0.0,
                        (value * difficulty)
                        + offset
                        - (0.08 if component == "action" and index % 2 == 1 else 0.0)
                        - (0.06 if component == "result" and index >= 4 else 0.0),
                    ),
                )
                for component, value in star.items()
            }
            turns.append(
                StateObservation(
                    competency=competency,
                    star=adjusted,
                    evidence_quality=0.75 + 0.05 * (turn_index % 3),
                    evidence_spans=(f"{competency}:{turn_index}",),
                    duplicate_evidence=turn_index in {6} and index % 2 == 0,
                    response_irrelevant=turn_index == 0 and index == 4,
                    profile_prior=min(0.8, 0.45 + 0.05 * index),
                    transcript_id=f"session-{index + 1}",
                )
            )
        expert_ready = mean(sum(turn.star.values()) / len(STAR_COMPONENTS) for turn in turns) >= label_threshold
        sessions.append(SessionBenchmark(session_id=f"session-{index + 1}", profile_prior=turns[0].profile_prior, expert_ready=expert_ready, turns=turns))
    return sessions


def load_sessions(path: Optional[Path]) -> List[SessionBenchmark]:
    if path is None:
        return build_demo_sessions()
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions: List[SessionBenchmark] = []
    for item in payload:
        turns = [
            StateObservation(
                competency=turn["competency"],
                star=turn["star"],
                evidence_quality=turn.get("evidence_quality", 1.0),
                evidence_spans=tuple(turn.get("evidence_spans", [])),
                duplicate_evidence=turn.get("duplicate_evidence", False),
                response_irrelevant=turn.get("response_irrelevant", False),
                profile_prior=item.get("profile_prior", 0.5),
                expert_ready=item.get("expert_ready"),
                transcript_id=item.get("session_id"),
            )
            for turn in item["turns"]
        ]
        sessions.append(
            SessionBenchmark(
                session_id=item["session_id"],
                profile_prior=item.get("profile_prior", 0.5),
                expert_ready=bool(item["expert_ready"]),
                turns=turns,
            )
        )
    return sessions


def acceptable_action_set(observation: StateObservation, controller: BaseController) -> Set[str]:
    if observation.response_irrelevant:
        return {"PROBE", "SWITCH"}
    action = observation.star.get("action", 0.0)
    result = observation.star.get("result", 0.0)
    context = min(observation.star.get("situation", 0.0), observation.star.get("task", 0.0))
    if action < 0.5:
        return {"PROBE"}
    if result < 0.5:
        return {"PROBE", "DEEPEN"}
    if context < 0.5:
        return {"PROBE", "DEEPEN"}
    if controller.ready_count() >= REQUIRED_READY_COUNT:
        return {"CLOSE"}
    return {"SWITCH", "DEEPEN"}


def probability_metrics(probabilities: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    brier = mean((prob - float(label)) ** 2 for prob, label in zip(probabilities, labels)) if probabilities else 0.0
    ece = expected_calibration_error(probabilities, labels, bins=10)
    auroc = roc_auc_score(probabilities, labels)
    balanced_accuracy = balanced_accuracy_score(probabilities, labels)
    return {"brier": brier, "ece": ece, "auroc": auroc, "balanced_accuracy": balanced_accuracy}


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    bin_totals = [0] * bins
    bin_correct = [0.0] * bins
    bin_confidence = [0.0] * bins
    for probability, label in zip(probabilities, labels):
        index = min(bins - 1, int(probability * bins))
        bin_totals[index] += 1
        bin_correct[index] += float(label)
        bin_confidence[index] += probability
    total = len(probabilities)
    ece = 0.0
    for count, correct, confidence in zip(bin_totals, bin_correct, bin_confidence):
        if not count:
            continue
        accuracy = correct / count
        avg_confidence = confidence / count
        ece += (count / total) * abs(accuracy - avg_confidence)
    return ece


def roc_auc_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [prob for prob, label in zip(probabilities, labels) if label]
    negatives = [prob for prob, label in zip(probabilities, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def balanced_accuracy_score(probabilities: Sequence[float], labels: Sequence[bool], threshold: float = 0.5) -> float:
    predictions = [prob >= threshold for prob in probabilities]
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    fp = sum(pred and (not label) for pred, label in zip(predictions, labels))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return (tpr + tnr) / 2.0


def evaluate_controller(controller: BaseController, sessions: Sequence[SessionBenchmark]) -> Dict[str, float]:
    labels: List[bool] = []
    probabilities: List[float] = []
    turns_used: List[int] = []
    action_correct = 0
    action_total = 0
    probe_gains: List[float] = []
    repetition_hits = 0
    repeated_actions = 0
    irrelevant_probes = 0

    for session in sessions:
        controller.reset(profile_prior=session.profile_prior)
        previous_action: Optional[NextAction] = None
        probe_history: List[Tuple[str, float, int]] = []

        for turn_index, observation in enumerate(session.turns):
            controller.turn_index = turn_index
            controller.current_competency = observation.competency
            action = controller.decide(observation)
            controller.history.append((observation, action))

            acceptable = acceptable_action_set(observation, controller)
            action_correct += int(action.action_type in acceptable)
            action_total += 1

            if previous_action and previous_action.action_type == action.action_type and previous_action.target_competency == action.target_competency and previous_action.template_id == action.template_id:
                repetition_hits += 1
            if action.action_type in {"PROBE", "DEEPEN"}:
                repeated_actions += 1
                if not observation.response_irrelevant and action.action_type == "PROBE":
                    irrelevant_probes += 1
                probe_history.append((observation.competency, controller._evidence_score(observation.star), turn_index))

            controller.observe(observation)
            controller.apply_action(action)
            previous_action = action

            if action.action_type == "CLOSE":
                turns_used.append(turn_index + 1)
                break
        else:
            turns_used.append(len(session.turns))

        probability = controller.session_probability()
        labels.append(session.expert_ready)
        probabilities.append(probability)

        if probe_history:
            gains = []
            for competency, score, turn_index in probe_history:
                if turn_index + 1 < len(session.turns) and session.turns[turn_index + 1].competency == competency:
                    next_score = controller._evidence_score(session.turns[turn_index + 1].star)
                    gains.append(max(0.0, next_score - score))
            if gains:
                probe_gains.append(mean(gains))

    prob_metrics = probability_metrics(probabilities, labels)
    return {
        **prob_metrics,
        "action_accuracy": action_correct / action_total if action_total else 0.0,
        "probe_gain": mean(probe_gains) if probe_gains else 0.0,
        "repetition_rate": repetition_hits / repeated_actions if repeated_actions else 0.0,
        "irrelevant_probe_rate": irrelevant_probes / repeated_actions if repeated_actions else 0.0,
        "turns_used": mean(turns_used) if turns_used else 0.0,
        "predicted_ready_rate": mean(probabilities) if probabilities else 0.0,
        "label_rate": mean(float(label) for label in labels) if labels else 0.0,
    }


def build_controllers() -> List[BaseController]:
    return [
        FixedScriptController(),
        ScoreOnlyController(),
        RuleBasedController(),
        BCTController(),
        BCTUniformPriorController(),
        BCTRuleSequencingController(),
    ]


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    headers = [
        "method",
        "brier",
        "ece",
        "auroc",
        "balanced_accuracy",
        "action_accuracy",
        "probe_gain",
        "repetition_rate",
        "irrelevant_probe_rate",
        "turns_used",
    ]
    rows = [headers]
    for method, metrics in results.items():
        rows.append([
            method,
            f"{metrics['brier']:.3f}",
            f"{metrics['ece']:.3f}",
            f"{metrics['auroc']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['action_accuracy']:.3f}",
            f"{metrics['probe_gain']:.3f}",
            f"{metrics['repetition_rate']:.3f}",
            f"{metrics['irrelevant_probe_rate']:.3f}",
            f"{metrics['turns_used']:.2f}",
        ])

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    lines = []
    for row_index, row in enumerate(rows):
        parts = [row[column].ljust(widths[column]) for column in range(len(headers))]
        lines.append(" | ".join(parts))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation for Bayesian Competency Tracing")
    parser.add_argument("--input", type=Path, default=None, help="Optional JSON benchmark file")
    parser.add_argument("--output", type=Path, default=Path("baseline_metrics.json"), help="Where to write metrics JSON")
    args = parser.parse_args()

    sessions = load_sessions(args.input)
    results: Dict[str, Dict[str, float]] = {}
    for controller in build_controllers():
        results[controller.name] = evaluate_controller(controller, sessions)

    print(format_table(results))
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()