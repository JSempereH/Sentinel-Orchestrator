"""Deterministic execution plans for analysis requests."""

from __future__ import annotations

from dataclasses import dataclass

from .request import AnalysisRequest


@dataclass(frozen=True)
class PlanStep:
    sensor: str
    action: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowPlan:
    request: AnalysisRequest
    steps: tuple[PlanStep, ...]


def build_plan(request: AnalysisRequest) -> WorkflowPlan:
    sensor_steps = tuple(
        PlanStep(sensor=sensor, action="discover -> acquire -> preprocess -> harmonize", variables=request.variables)
        for sensor in request.sensors
    )
    auxiliary_steps = tuple(
        PlanStep(sensor=f"auxiliary:{spec.provider}", action="acquire -> normalize -> harmonize", variables=spec.variables)
        for spec in request.auxiliary
    )
    steps = sensor_steps + auxiliary_steps
    return WorkflowPlan(request=request, steps=steps)
