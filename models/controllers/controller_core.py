from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple


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