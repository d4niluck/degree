from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

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
    QuestionType,
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


JUDGE_COMMON_RULES_BLOCK = """
Общие правила:
- Если утверждение агента не подтверждается критериями или фрагментом, не считай его автоматически ложным. Считай его ошибкой только в тех случаях, когда оно противоречит эталону, искажает требуемый факт, подменяет ответ на вопрос или для данного типа вопроса неподтвержденные добавления должны штрафоваться явно.
- Оценивай только ответ агента, а не то, что он мог иметь в виду.
- Допускай эквивалентные формулировки, если они передают тот же смысл и не добавляют новых неподтвержденных фактов.
""".strip()


JUDGE_RULES_BY_TYPE = {
    QuestionType.FACTUAL: """
Правила для factual:
- Определи, совпадает ли ответ агента с эталонным фактом по смыслу.
- Сравнивай ответ прежде всего с criteria.atomic_facts.
- Если агент дал эквивалентную формулировку без искажения смысла, считай ответ правильным.
- Если агент добавил детали, которые значительно меняют смысл или противоречат основному утверждению, считай ответ неправильным.
- Заполни reasoning и итоговое поле is_correct.
""".strip(),
    QuestionType.CHOICE: """
Правила для choice:
- Определи, выбрал ли агент правильный вариант ответа.
- Сравнивай выбор агента с criteria.correct_option_key и criteria.correct_option_text.
- Если агент явно указывает правильный вариант или однозначно воспроизводит его текст, считай ответ правильным.
- Если агент уклоняется от выбора, выбирает несколько вариантов или отрицает наличие данных при наличии правильного варианта, считай ответ неправильным.
- Заполни reasoning и итоговое поле is_correct.
""".strip(),
    QuestionType.MULTI_FACT: """
Правила для multi_fact:
- Разбери ответ агента на отдельные утверждения.
- Сравнивай каждое утверждение с criteria.atomic_facts.
- Факты из criteria, которые присутствуют корректно, внеси в supported_facts.
- Факты из criteria, которые отсутствуют или переданы некорректно, внеси в missing_facts.
- Утверждения агента, не подтвержденные criteria или фрагментом, внеси в false_claims.
- Заполни reasoning, supported_facts, missing_facts и false_claims.
""".strip(),
    QuestionType.PROCEDURE: """
Правила для procedure:
- Сравнивай ответ агента с criteria.required_steps и criteria.optional_steps как с шагами процедуры.
- Шаги из criteria.required_steps, которые отражены корректно, внеси в covered_required_steps.
- В covered_required_steps используй точные формулировки шагов из criteria.required_steps, даже если агент их перефразировал.
- В covered_required_steps сохраняй порядок появления шагов в ответе агента.
- Не сортируй covered_required_steps по эталонному порядку.
- Обязательные шаги, которые отсутствуют или искажены, внеси в missing_required_steps.
- Корректно отраженные шаги из criteria.optional_steps внеси в covered_optional_steps.
- Дополнительные шаги агента, не подтвержденные criteria или фрагментом, внеси в hallucinated_steps.
- Порядок оценивай только по criteria.critical_order_pairs.
- Если все критические пары соблюдены, critical_order_correct = true, иначе false.
""".strip(),
    QuestionType.COMPARISON: """
Правила для comparison:
- Сравнивай ответ только по criteria.comparison_axes.
- Если ось раскрыта корректно, внеси ее в correct_axes.
- Если ось раскрыта с фактической ошибкой или неполным/неверным сравнением, внеси ее в incorrect_axes.
- Если различия между сравниваемыми сущностями перепутаны местами, внеси ось в swapped_axes.
- Любые дополнительные оси или неподтвержденные направления сравнения внеси в hallucinated_axes.
- Заполни reasoning, correct_axes, incorrect_axes, swapped_axes и hallucinated_axes.
""".strip(),
    QuestionType.NEGATIVE: """
Правила для negative:
- Правильный ответ должен явно отказаться от выдумывания и сообщить, что по приведенным данным ответить нельзя.
- Проверь, ссылается ли агент на отсутствие нужной информации или недостаточность данных; это поле cited_missing_information.
- Если отказ корректный и агент не добавляет неподтвержденных фактов, correct_abstention = true.
- Любые неподтвержденные конкретные утверждения, числа, параметры или ссылки на отсутствующие в фрагменте сведения внеси в unsupported_claims.
- Если агент дает содержательный ответ на вопрос вместо корректного отказа, correct_abstention = false.
""".strip(),
}


BENCHMARK_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BENCHMARK_PACKAGE_ROOT / "results"


def _pairwise_order_accuracy(gold_steps: Sequence[str], agent_steps: Sequence[str]) -> float:
    if len(agent_steps) < 2:
        return math.nan

    gold_positions = {step: idx for idx, step in enumerate(gold_steps)}
    comparable_pairs = 0
    correct_pairs = 0

    for i in range(len(agent_steps)):
        left = agent_steps[i]
        if left not in gold_positions:
            continue
        for j in range(i + 1, len(agent_steps)):
            right = agent_steps[j]
            if right not in gold_positions:
                continue
            comparable_pairs += 1
            if gold_positions[left] < gold_positions[right]:
                correct_pairs += 1

    if comparable_pairs == 0:
        return math.nan
    return correct_pairs / comparable_pairs


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
        batch: bool = True,
        batch_size: int = 5,
        resume: bool = True,
    ) -> Dict[str, object]:
        samples = self.load(benchmark_path)
        output_dir = self._resolve_output_dir(
            experiment_name=experiment_name,
            prefix="evaluation",
            resume=resume,
        )
        answers_path = output_dir / "answers.csv"
        metrics_path = output_dir / "metrics.csv"
        summary_path = output_dir / "summary.json"
        existing_rows = self._load_existing_rows(answers_path)
        processed_sample_ids = {str(row["sample_id"]) for row in existing_rows}
        pending_samples = [
            sample for sample in samples if sample.sample_id not in processed_sample_ids
        ]

        answer_rows = list(existing_rows)
        error_message: Optional[str] = None

        self._persist_evaluation_state(
            answers_path=answers_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            answer_rows=answer_rows,
            experiment_name=experiment_name,
            benchmark_path=benchmark_path,
            total_samples=len(samples),
            status="running",
            error_message=None,
        )

        try:
            if pending_samples:
                new_rows = self._evaluate_samples(
                    samples=pending_samples,
                    answer_fn=answer_fn,
                    batch=batch,
                    batch_size=batch_size,
                    answer_rows=answer_rows,
                    answers_path=answers_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    experiment_name=experiment_name,
                    benchmark_path=benchmark_path,
                    total_samples=len(samples),
                )
                answer_rows = list(existing_rows) + new_rows
        except Exception as exc:
            error_message = str(exc)
            self._persist_evaluation_state(
                answers_path=answers_path,
                metrics_path=metrics_path,
                summary_path=summary_path,
                answer_rows=answer_rows,
                experiment_name=experiment_name,
                benchmark_path=benchmark_path,
                total_samples=len(samples),
                status="failed",
                error_message=error_message,
            )
            raise

        self._persist_evaluation_state(
            answers_path=answers_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            answer_rows=answer_rows,
            experiment_name=experiment_name,
            benchmark_path=benchmark_path,
            total_samples=len(samples),
            status="completed",
            error_message=error_message,
        )
        answers_df = pd.DataFrame(answer_rows)
        metrics_df = self._build_metrics_report(answers_df)

        return {
            "output_dir": str(output_dir),
            "answers_path": str(answers_path),
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "answers_df": answers_df,
            "metrics_df": metrics_df,
        }

    def _evaluate_samples(
        self,
        samples: List[BenchmarkSample],
        answer_fn: Callable[[str], str],
        batch: bool,
        batch_size: int,
        answer_rows: List[Dict[str, Any]],
        answers_path: Path,
        metrics_path: Path,
        summary_path: Path,
        experiment_name: str,
        benchmark_path: str,
        total_samples: int,
    ) -> List[Dict[str, Any]]:
        if not batch:
            rows: List[Dict[str, Any]] = []
            for sample in tqdm(samples, desc="Evaluating benchmark"):
                row = self._evaluate_single_sample(sample=sample, answer_fn=answer_fn)
                rows.append(row)
                answer_rows.append(row)
                self._persist_evaluation_state(
                    answers_path=answers_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    answer_rows=answer_rows,
                    experiment_name=experiment_name,
                    benchmark_path=benchmark_path,
                    total_samples=total_samples,
                    status="running",
                    error_message=None,
                )
            return rows

        max_workers = max(1, batch_size)
        completed_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample_id = {
                executor.submit(self._evaluate_single_sample, sample, answer_fn): sample.sample_id
                for sample in samples
            }
            for future in tqdm(as_completed(future_to_sample_id), total=len(samples), desc="Evaluating benchmark"):
                row = future.result()
                completed_rows.append(row)
                answer_rows.append(row)
                self._persist_evaluation_state(
                    answers_path=answers_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    answer_rows=answer_rows,
                    experiment_name=experiment_name,
                    benchmark_path=benchmark_path,
                    total_samples=total_samples,
                    status="running",
                    error_message=None,
                )
        return completed_rows

    def _evaluate_single_sample(
        self,
        sample: BenchmarkSample,
        answer_fn: Callable[[str], str],
    ) -> Dict[str, Any]:
        agent_answer = self._call_answer_fn(answer_fn=answer_fn, question=sample.question)
        judge_result = self._judge_sample(sample=sample, agent_answer=agent_answer)
        metrics = self._extract_metrics(sample=sample, judge_result=judge_result)
        return {
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
        type_rules = JUDGE_RULES_BY_TYPE[sample.question_type]
        return f"{prompt}\n\n{JUDGE_COMMON_RULES_BLOCK}\n\n{type_rules}"

    def _extract_metrics(self, sample: BenchmarkSample, judge_result: object) -> Dict[str, Any]:
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
            order_accuracy = _pairwise_order_accuracy(
                gold_steps=sample.criteria.get("required_steps", []),
                agent_steps=judge_result.covered_required_steps,
            )
            return {
                "score_accuracy": math.nan,
                "score_precision": math.nan,
                "score_recall": math.nan,
                "score_f1": math.nan,
                "score_step_recall": covered_required / (covered_required + missing_required) if (covered_required + missing_required) else 0.0,
                "score_order_accuracy": order_accuracy,
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
        if answers_df.empty:
            return pd.DataFrame(columns=["question_type", "metric", "value", "count"])
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

    def _persist_evaluation_state(
        self,
        answers_path: Path,
        metrics_path: Path,
        summary_path: Path,
        answer_rows: Sequence[Dict[str, Any]],
        experiment_name: str,
        benchmark_path: str,
        total_samples: int,
        status: str,
        error_message: Optional[str],
    ) -> None:
        answers_df = pd.DataFrame(answer_rows)
        answers_df.to_csv(answers_path, index=False)
        metrics_df = self._build_metrics_report(answers_df)
        metrics_df.to_csv(metrics_path, index=False)
        summary = {
            "experiment_name": experiment_name,
            "created_at": self._timestamp(),
            "judge_model_name": self.judge_model_name,
            "benchmark_path": benchmark_path,
            "samples": total_samples,
            "samples_completed": len(answer_rows),
            "status": status,
            "error_message": error_message,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _make_output_dir(self, experiment_name: str, prefix: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_name.strip()) or "experiment"
        path = self.results_dir / f"{prefix}_{safe_name}_{self._timestamp_for_path()}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _resolve_output_dir(self, experiment_name: str, prefix: str, resume: bool) -> Path:
        if not resume:
            return self._make_output_dir(experiment_name=experiment_name, prefix=prefix)
        existing_dir = self._find_latest_output_dir(experiment_name=experiment_name, prefix=prefix)
        if existing_dir is not None:
            self.logger.info("Resuming benchmark evaluation in %s", existing_dir)
            return existing_dir
        return self._make_output_dir(experiment_name=experiment_name, prefix=prefix)

    def _find_latest_output_dir(self, experiment_name: str, prefix: str) -> Optional[Path]:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_name.strip()) or "experiment"
        pattern = f"{prefix}_{safe_name}_*"
        matches = sorted(
            [path for path in self.results_dir.glob(pattern) if path.is_dir()],
            key=lambda path: path.name,
        )
        if not matches:
            return None
        return matches[-1]

    def _load_existing_rows(self, answers_path: Path) -> List[Dict[str, Any]]:
        if not answers_path.exists():
            return []
        dataframe = pd.read_csv(answers_path)
        if dataframe.empty:
            return []
        return dataframe.to_dict(orient="records")

    @staticmethod
    def _timestamp_for_path() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")
