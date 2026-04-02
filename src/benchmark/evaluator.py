from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..llm.llm import BaseLLM
from .schemas import (
    BenchmarkSample,
    ChoiceJudgeResult,
    ComparisonJudgeResult,
    FactualJudgeResult,
    JUDGE_SCHEMA_BY_TYPE,
    MultiFactJudgeResult,
    NegativeJudgeResult,
    ProcedureJudgeResult,
)


JUDGE_PROMPT_TEMPLATE = """
Ты оцениваешь ответ RAG-системы по технической документации.
Оценивай строго по критериям и только на основе приведенных данных.
Не додумывай факты, которых нет в эталоне или критериях.
Верни только структуру, соответствующую pydantic схеме.

Тип вопроса: {question_type}
Вопрос: {question}
Эталонный ответ: {gold_answer}
Критерии: {criteria_json}
Фрагмент источника: {source_excerpt}
Ответ агента: {agent_answer}
""".strip()


JUDGE_RULES_BLOCK = """
Правила:
- Если утверждение агента не подтверждается критериями или фрагментом, считай его ложным или неподдержанным.
- Для negative вопросов правильный ответ должен отказаться от выдумывания и указать, что данных недостаточно.
- Для procedure вопросов порядок важен только там, где он явно указан в criteria.critical_order_pairs.
- Для multi_fact вопросов сравнивай ответ по atomic facts.
- Для comparison вопросов проверяй различия по заданным осям.
""".strip()


BENCHMARK_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BENCHMARK_PACKAGE_ROOT / "results"


class BenchmarkEvaluator:
    def __init__(
        self,
        judge_llm: BaseLLM,
        logger: Optional[logging.Logger] = None,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        judge_model_name: str = "gpt-5.2",
    ) -> None:
        self.judge_llm = judge_llm
        self.logger = logger or logging.getLogger(__name__)
        self.results_dir = Path(results_dir).resolve()
        self.judge_model_name = judge_model_name

    def load(self, benchmark_path: str) -> List[BenchmarkSample]:
        path = Path(benchmark_path)
        return [
            BenchmarkSample.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def evaluate(
        self,
        benchmark_path: str,
        answer_fn: Callable[[str], str],
        experiment_name: str,
    ) -> Dict[str, object]:
        samples = self.load(benchmark_path)
        answer_rows: List[Dict[str, Any]] = []

        for sample in samples:
            agent_answer = self._call_answer_fn(answer_fn=answer_fn, question=sample.question)
            judge_result = self._judge_sample(sample=sample, agent_answer=agent_answer)
            metrics = self._extract_metrics(judge_result=judge_result)
            answer_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "question_type": sample.question_type.value,
                    "question": sample.question,
                    "gold_answer": sample.gold_answer,
                    "agent_answer": agent_answer,
                    "source_path": sample.source_path,
                    "page_number": sample.page_number,
                    "judge_schema_name": sample.judge_schema_name,
                    "judge_result": json.dumps(judge_result.model_dump(mode="json"), ensure_ascii=False),
                    "criteria": json.dumps(sample.criteria, ensure_ascii=False),
                    **metrics,
                }
            )

        answers_df = pd.DataFrame(answer_rows)
        metrics_df = self._build_metrics_report(answers_df)
        output_dir = self._make_output_dir(experiment_name=experiment_name, prefix="evaluation")

        answers_path = output_dir / "answers.csv"
        metrics_path = output_dir / "metrics.csv"
        summary_path = output_dir / "summary.json"

        answers_df.to_csv(answers_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        summary = {
            "experiment_name": experiment_name,
            "created_at": self._timestamp(),
            "judge_model_name": self.judge_model_name,
            "benchmark_path": benchmark_path,
            "samples": len(samples),
            "question_types": sorted({sample.question_type.value for sample in samples}),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "output_dir": str(output_dir),
            "answers_path": str(answers_path),
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "answers_df": answers_df,
            "metrics_df": metrics_df,
        }

    def _call_answer_fn(self, answer_fn: Callable[[str], str], question: str) -> str:
        answer = answer_fn(question)
        return str(answer).strip()

    def _judge_sample(self, sample: BenchmarkSample, agent_answer: str):
        schema = JUDGE_SCHEMA_BY_TYPE[sample.question_type]
        prompt = self._build_judge_prompt(sample=sample, agent_answer=agent_answer)
        return self.judge_llm.parse(prompt, schema, temperature=0)

    def _build_judge_prompt(self, sample: BenchmarkSample, agent_answer: str) -> str:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question_type=sample.question_type.value,
            question=sample.question,
            gold_answer=sample.gold_answer,
            criteria_json=json.dumps(sample.criteria, ensure_ascii=False),
            source_excerpt=sample.source_excerpt,
            agent_answer=agent_answer,
        )
        return f"{prompt}\n\n{JUDGE_RULES_BLOCK}"

    def _extract_metrics(self, judge_result: object) -> Dict[str, Any]:
        if isinstance(judge_result, (FactualJudgeResult, ChoiceJudgeResult)):
            return {
                "score_accuracy": float(judge_result.is_correct),
                "score_precision": math.nan,
                "score_recall": math.nan,
                "score_f1": math.nan,
                "score_step_recall": math.nan,
                "score_order_accuracy": math.nan,
                "score_hallucination_rate": math.nan,
                "score_attribute_accuracy": math.nan,
                "score_abstention_accuracy": math.nan,
            }

        if isinstance(judge_result, MultiFactJudgeResult):
            supported = len(judge_result.supported_facts)
            missing = len(judge_result.missing_facts)
            false_claims = len(judge_result.false_claims)
            precision = supported / (supported + false_claims) if (supported + false_claims) else 0.0
            recall = supported / (supported + missing) if (supported + missing) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            return {
                "score_accuracy": math.nan,
                "score_precision": precision,
                "score_recall": recall,
                "score_f1": f1,
                "score_step_recall": math.nan,
                "score_order_accuracy": math.nan,
                "score_hallucination_rate": false_claims / (supported + false_claims) if (supported + false_claims) else 0.0,
                "score_attribute_accuracy": math.nan,
                "score_abstention_accuracy": math.nan,
            }

        if isinstance(judge_result, ProcedureJudgeResult):
            covered_required = len(judge_result.covered_required_steps)
            missing_required = len(judge_result.missing_required_steps)
            hallucinated = len(judge_result.hallucinated_steps)
            covered_optional = len(judge_result.covered_optional_steps)
            return {
                "score_accuracy": math.nan,
                "score_precision": math.nan,
                "score_recall": math.nan,
                "score_f1": math.nan,
                "score_step_recall": covered_required / (covered_required + missing_required) if (covered_required + missing_required) else 0.0,
                "score_order_accuracy": float(judge_result.critical_order_correct),
                "score_hallucination_rate": hallucinated / (covered_required + covered_optional + hallucinated) if (covered_required + covered_optional + hallucinated) else 0.0,
                "score_attribute_accuracy": math.nan,
                "score_abstention_accuracy": math.nan,
            }

        if isinstance(judge_result, ComparisonJudgeResult):
            correct_axes = len(judge_result.correct_axes)
            incorrect_axes = len(judge_result.incorrect_axes)
            swapped_axes = len(judge_result.swapped_axes)
            total_axes = correct_axes + incorrect_axes + swapped_axes
            return {
                "score_accuracy": math.nan,
                "score_precision": math.nan,
                "score_recall": math.nan,
                "score_f1": math.nan,
                "score_step_recall": math.nan,
                "score_order_accuracy": math.nan,
                "score_hallucination_rate": len(judge_result.hallucinated_axes) / (total_axes + len(judge_result.hallucinated_axes)) if (total_axes + len(judge_result.hallucinated_axes)) else 0.0,
                "score_attribute_accuracy": correct_axes / total_axes if total_axes else 0.0,
                "score_abstention_accuracy": math.nan,
            }

        if isinstance(judge_result, NegativeJudgeResult):
            return {
                "score_accuracy": math.nan,
                "score_precision": math.nan,
                "score_recall": math.nan,
                "score_f1": math.nan,
                "score_step_recall": math.nan,
                "score_order_accuracy": math.nan,
                "score_hallucination_rate": float(bool(judge_result.unsupported_claims)),
                "score_attribute_accuracy": math.nan,
                "score_abstention_accuracy": float(
                    judge_result.correct_abstention and judge_result.cited_missing_information and not judge_result.unsupported_claims
                ),
            }

        raise TypeError(f"Unsupported judge result type: {type(judge_result)!r}")

    def _build_metrics_report(self, answers_df: pd.DataFrame) -> pd.DataFrame:
        metric_columns = [
            "score_accuracy",
            "score_precision",
            "score_recall",
            "score_f1",
            "score_step_recall",
            "score_order_accuracy",
            "score_hallucination_rate",
            "score_attribute_accuracy",
            "score_abstention_accuracy",
        ]

        rows: List[Dict[str, Any]] = []
        grouped = answers_df.groupby("question_type", dropna=False)
        for question_type, frame in grouped:
            for metric in metric_columns:
                series = frame[metric].dropna()
                if series.empty:
                    continue
                rows.append(
                    {
                        "question_type": question_type,
                        "metric": metric,
                        "value": float(series.mean()),
                        "count": int(series.shape[0]),
                    }
                )

        for metric in metric_columns:
            series = answers_df[metric].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "question_type": "overall",
                    "metric": metric,
                    "value": float(series.mean()),
                    "count": int(series.shape[0]),
                }
            )

        return pd.DataFrame(rows).sort_values(["question_type", "metric"]).reset_index(drop=True)

    def _make_output_dir(self, experiment_name: str, prefix: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_name.strip()) or "experiment"
        path = self.results_dir / f"{prefix}_{safe_name}_{self._timestamp_for_path()}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    @staticmethod
    def _timestamp_for_path() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")
