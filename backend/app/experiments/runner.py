from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.app.experiments.configuration import ExperimentConfiguration


class ExperimentRunnerError(RuntimeError):
    """Raised when experiment execution fails."""


ExperimentHandler = Callable[[Any], Any]


@dataclass(frozen=True)
class ExperimentRunResult:
    """Result of one experiment execution."""

    name: str
    status: str
    result: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Summary of a complete experiment run."""

    experiment_count: int
    successful_count: int
    failed_count: int
    results: tuple[ExperimentRunResult, ...]

    @property
    def all_successful(self) -> bool:
        return self.failed_count == 0


class ExperimentRunner:
    """
    Execute configured experiments using registered handlers.

    The runner is intentionally execution-oriented:
    configuration defines what should run, while handlers define
    how an individual experiment is executed.
    """

    def __init__(
        self,
        config: ExperimentConfiguration,
        handlers: dict[str, ExperimentHandler],
    ) -> None:
        self.config = config
        self.handlers = handlers

    def run(self) -> ExperimentRunSummary:
        results: list[ExperimentRunResult] = []

        for experiment in self.config.experiments:
            name = experiment.name

            handler = self.handlers.get(name)

            if handler is None:
                results.append(
                    ExperimentRunResult(
                        name=name,
                        status="failed",
                        error=f"No handler registered for experiment '{name}'.",
                    )
                )
                continue

            try:
                handler_result = handler(experiment)

                results.append(
                    ExperimentRunResult(
                        name=name,
                        status="success",
                        result=handler_result,
                    )
                )

            except Exception as exc:
                results.append(
                    ExperimentRunResult(
                        name=name,
                        status="failed",
                        error=str(exc),
                    )
                )

        successful_count = sum(
            result.status == "success"
            for result in results
        )

        failed_count = len(results) - successful_count

        return ExperimentRunSummary(
            experiment_count=len(results),
            successful_count=successful_count,
            failed_count=failed_count,
            results=tuple(results),
        )


def run_experiments(
    config: ExperimentConfiguration,
    *,
    handlers: dict[str, ExperimentHandler],
) -> ExperimentRunSummary:
    """
    Execute all configured experiments.

    Parameters
    ----------
    config:
        Parsed AVF-TRPDE experiment configuration.

    handlers:
        Mapping of experiment name to execution function.
    """

    runner = ExperimentRunner(
        config=config,
        handlers=handlers,
    )

    return runner.run()