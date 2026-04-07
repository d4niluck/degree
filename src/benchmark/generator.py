from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4
from tqdm import tqdm

import pandas as pd

from ..llm.llm import BaseLLM
from ..preprocessing.reader import Reader
from .schemas import (
    BenchmarkSample,
    ChoiceQuestionCandidate,
    ComparisonQuestionCandidate,
    FactualQuestionCandidate,
    GENERATION_SCHEMA_BY_TYPE,
    JUDGE_SCHEMA_BY_TYPE,
    MultiFactQuestionCandidate,
    NegativeQuestionCandidate,
    ProcedureQuestionCandidate,
    QuestionType,
)


GENERATION_COMMON_PROMPT_TEMPLATE = """
Ты создаешь benchmark для RAG по технической документации.
Источник: {source_path}
Страница: {page_number}

Правила:
- Используй только факты из приведенного текста.
- Не выдумывай информацию, отсутствующую в тексте.
- Формулируй вопросы на русском языке.
- Делай вопросы полезными для оценки качества RAG.
- Если по этому тексту нельзя корректно построить вопрос нужного типа, верни `can_generate=false`, кратко объясни причину в `reason` и не заполняй `payload`.
- Если `can_generate=true`, то `payload` обязан быть заполнен полностью и строго соответствовать нужному типу вопроса.
- Возвращай только структуру, соответствующую pydantic схеме.

Текст страницы:
{page_text}
""".strip()


GENERATION_TYPE_INSTRUCTIONS = {
    QuestionType.FACTUAL: """
Сгенерируй один короткий фактический вопрос с единственным правильным ответом.
Ответ должен быть проверяемым по одному конкретному факту.
Если на странице нет достаточно четкого одиночного факта, верни `can_generate=false`.
""".strip(),
    QuestionType.CHOICE: """
Сгенерируй один вопрос с выбором ответа.
Должно быть ровно 3 варианта ответа, только один правильный.
Неправильные варианты должны быть правдоподобными, но опровергаться текстом.
Вопрос должен включать варианты ответа прямо в тексте.
Если нельзя составить честный вопрос с 3 вариантами и 1 правильным ответом, верни `can_generate=false`.
""".strip(),
    QuestionType.MULTI_FACT: """
Сгенерируй вопрос, требующий перечисления нескольких конкретных фактов из текста.
Нужно минимум 2 atomic facts, лучше 3-5.
Если в тексте недостаточно независимых фактов, верни `can_generate=false`.
""".strip(),
    QuestionType.PROCEDURE: """
Сгенерируй вопрос на пошаговую инструкцию.
Выдели обязательные шаги, опциональные шаги и критичные пары порядка только если порядок действительно следует из текста.
Если явной процедуры нет, верни `can_generate=false`.
""".strip(),
    QuestionType.COMPARISON: """
Сгенерируй вопрос на сравнение двух сущностей, режимов, вариантов или процедур, реально присутствующих в тексте.
Сравнение должно опираться на 2 и более оси различий.
Если в тексте нет корректного сравнения, верни `can_generate=false`.
""".strip(),
    QuestionType.NEGATIVE: """
Сгенерируй вопрос-ловушку, на который этот текст не дает ответа.
Вопрос должен выглядеть правдоподобно для пользователя, но корректный ответ обязан отказаться от выдумывания и сослаться на отсутствие данных в тексте.
Укажи, каких именно данных в тексте не хватает.
Если не получается составить правдоподобный неподдерживаемый вопрос именно по этому тексту, верни `can_generate=false`.
""".strip(),
}


BENCHMARK_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BENCHMARK_PACKAGE_ROOT / "results"


class BenchmarkGenerator:
    def __init__(
        self,
        reader: Reader,
        llm: BaseLLM,
        logger: Optional[logging.Logger] = None,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        generation_model_name: str = "gpt-5.2",
    ) -> None:
        self.reader = reader
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.results_dir = Path(results_dir).resolve()
        self.generation_model_name = generation_model_name
        self._slug_pattern = re.compile(r"[^a-zA-Z0-9_-]+")

    def generate(
        self,
        pdf_paths: Sequence[str],
        experiment_name: str,
        questions_per_type: int = 50,
        seed: int = 42,
        min_page_length: int = 400,
        max_attempts_per_type: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, object]:
        pages = self._collect_pages(pdf_paths=pdf_paths, min_page_length=min_page_length)
        if not pages:
            raise ValueError("No suitable pages found for benchmark generation")

        output_dir = self._resolve_output_dir(
            experiment_name=experiment_name,
            prefix="generation",
            resume=resume,
        )
        dataset_path = output_dir / "benchmark.jsonl"
        metadata_path = output_dir / "metadata.json"
        dataframe_path = output_dir / "benchmark.csv"

        rng = random.Random(seed)
        samples = self._load_existing_samples(dataset_path)
        per_type_counts = self._count_samples_by_type(samples)
        skipped_per_type = self._load_skipped_counts(metadata_path)
        max_attempts_per_type = max_attempts_per_type or max(len(pages) * 10, questions_per_type * 3)
        error_message: Optional[str] = None

        try:
            for question_type in tqdm(QuestionType):
                attempts = 0
                while per_type_counts[question_type] < questions_per_type:
                    attempts += 1
                    if attempts > max_attempts_per_type:
                        raise RuntimeError(
                            f"Failed to generate enough samples for type '{question_type.value}'. "
                            f"Generated {per_type_counts[question_type]}/{questions_per_type} after "
                            f"{max_attempts_per_type} attempts."
                        )
                    page_payload = rng.choice(pages)
                    sample = self._generate_sample(
                        page_payload=page_payload,
                        question_type=question_type,
                        ordinal=per_type_counts[question_type] + 1,
                    )
                    if sample is None:
                        skipped_per_type[question_type] += 1
                        self.logger.info(
                            "Skipped %s page %s for %s generation",
                            page_payload["source_path"],
                            page_payload["page_number"],
                            question_type.value,
                        )
                        self._persist_generation_state(
                            dataset_path=dataset_path,
                            metadata_path=metadata_path,
                            dataframe_path=dataframe_path,
                            samples=samples,
                            experiment_name=experiment_name,
                            questions_per_type=questions_per_type,
                            seed=seed,
                            pdf_paths=pdf_paths,
                            per_type_counts=per_type_counts,
                            skipped_per_type=skipped_per_type,
                            max_attempts_per_type=max_attempts_per_type,
                            status="running",
                            error_message=None,
                        )
                        continue
                    samples.append(sample)
                    per_type_counts[question_type] += 1
                    self.logger.info(
                        "Generated %s sample %s/%s from %s page %s",
                        question_type.value,
                        per_type_counts[question_type],
                        questions_per_type,
                        page_payload["source_path"],
                        page_payload["page_number"],
                    )
                    self._persist_generation_state(
                        dataset_path=dataset_path,
                        metadata_path=metadata_path,
                        dataframe_path=dataframe_path,
                        samples=samples,
                        experiment_name=experiment_name,
                        questions_per_type=questions_per_type,
                        seed=seed,
                        pdf_paths=pdf_paths,
                        per_type_counts=per_type_counts,
                        skipped_per_type=skipped_per_type,
                        max_attempts_per_type=max_attempts_per_type,
                        status="running",
                        error_message=None,
                    )
        except Exception as exc:
            error_message = str(exc)
            self._persist_generation_state(
                dataset_path=dataset_path,
                metadata_path=metadata_path,
                dataframe_path=dataframe_path,
                samples=samples,
                experiment_name=experiment_name,
                questions_per_type=questions_per_type,
                seed=seed,
                pdf_paths=pdf_paths,
                per_type_counts=per_type_counts,
                skipped_per_type=skipped_per_type,
                max_attempts_per_type=max_attempts_per_type,
                status="failed",
                error_message=error_message,
            )
            raise

        self._persist_generation_state(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            dataframe_path=dataframe_path,
            samples=samples,
            experiment_name=experiment_name,
            questions_per_type=questions_per_type,
            seed=seed,
            pdf_paths=pdf_paths,
            per_type_counts=per_type_counts,
            skipped_per_type=skipped_per_type,
            max_attempts_per_type=max_attempts_per_type,
            status="completed",
            error_message=error_message,
        )

        dataframe = self.samples_to_dataframe(samples)
        return {
            "output_dir": str(output_dir),
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path),
            "dataframe_path": str(dataframe_path),
            "dataframe": dataframe,
            "samples": samples,
        }

    def load(self, benchmark_path: str) -> List[BenchmarkSample]:
        path = Path(benchmark_path)
        return [
            BenchmarkSample.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def samples_to_dataframe(samples: Iterable[BenchmarkSample]) -> pd.DataFrame:
        rows = []
        for sample in samples:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "question_type": sample.question_type.value,
                    "question": sample.question,
                    "gold_answer": sample.gold_answer,
                    "source_path": sample.source_path,
                    "page_number": sample.page_number,
                    "source_excerpt": sample.source_excerpt,
                    "judge_schema_name": sample.judge_schema_name,
                    "criteria": json.dumps(sample.criteria, ensure_ascii=False),
                    "metadata": json.dumps(sample.metadata, ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)

    def _collect_pages(
        self,
        pdf_paths: Sequence[str],
        min_page_length: int,
    ) -> List[Dict[str, object]]:
        pages: List[Dict[str, object]] = []
        for pdf_path in tqdm(pdf_paths):
            document = self.reader.read(pdf_path)
            if not document:
                self.logger.warning("Skipping unreadable document: %s", pdf_path)
                continue
            for page in document.pages:
                text = (page.text or "").strip()
                if len(text) < min_page_length:
                    continue
                pages.append(
                    {
                        "source_path": document.source_path,
                        "page_number": page.number,
                        "text": text,
                    }
                )
        return pages

    def _generate_sample(
        self,
        page_payload: Dict[str, object],
        question_type: QuestionType,
        ordinal: int,
    ) -> Optional[BenchmarkSample]:
        source_path = str(page_payload["source_path"])
        page_number = int(page_payload["page_number"])
        page_text = str(page_payload["text"])
        schema = GENERATION_SCHEMA_BY_TYPE[question_type]
        prompt = self._build_generation_prompt(
            question_type=question_type,
            source_path=source_path,
            page_number=page_number,
            page_text=page_text,
        )
        generation_result = self.llm.parse(prompt, schema, temperature=0.2)
        if not generation_result.can_generate:
            self.logger.debug(
                "Generation skipped for type=%s source=%s page=%s: %s",
                question_type.value,
                source_path,
                page_number,
                generation_result.reason,
            )
            return None
        candidate = generation_result.payload
        if candidate is None:
            self.logger.debug(
                "Generation returned can_generate=True but empty payload for type=%s source=%s page=%s",
                question_type.value,
                source_path,
                page_number,
            )
            return None
        return self._candidate_to_sample(
            candidate=candidate,
            question_type=question_type,
            source_path=source_path,
            page_number=page_number,
            page_text=page_text,
            ordinal=ordinal,
        )

    def _candidate_to_sample(
        self,
        candidate: object,
        question_type: QuestionType,
        source_path: str,
        page_number: int,
        page_text: str,
        ordinal: int,
    ) -> BenchmarkSample:
        sample_id = f"{question_type.value}-{ordinal:03d}-{uuid4().hex[:8]}"
        if isinstance(candidate, FactualQuestionCandidate):
            criteria = {"atomic_facts": [candidate.fact]}
            gold_answer = candidate.gold_answer
            question = candidate.question
        elif isinstance(candidate, ChoiceQuestionCandidate):
            criteria = {
                "options": [option.model_dump() for option in candidate.options],
                "correct_option_key": candidate.correct_option_key,
                "correct_option_text": next(
                    option.text
                    for option in candidate.options
                    if option.key == candidate.correct_option_key
                ),
                "atomic_facts": [candidate.explanation_fact],
            }
            gold_answer = candidate.gold_answer
            question = candidate.question
        elif isinstance(candidate, MultiFactQuestionCandidate):
            criteria = {"atomic_facts": candidate.atomic_facts}
            gold_answer = candidate.gold_answer
            question = candidate.question
        elif isinstance(candidate, ProcedureQuestionCandidate):
            criteria = {
                "required_steps": candidate.required_steps,
                "optional_steps": candidate.optional_steps,
                "critical_order_pairs": candidate.critical_order_pairs,
            }
            gold_answer = candidate.gold_answer
            question = candidate.question
        elif isinstance(candidate, ComparisonQuestionCandidate):
            criteria = {
                "compared_entities": candidate.compared_entities,
                "comparison_axes": candidate.comparison_axes,
                "expected_differences": candidate.expected_differences,
            }
            gold_answer = candidate.gold_answer
            question = candidate.question
        elif isinstance(candidate, NegativeQuestionCandidate):
            criteria = {
                "unsupported_reason": candidate.unsupported_reason,
                "forbidden_claims": candidate.forbidden_claims,
            }
            gold_answer = candidate.gold_answer
            question = candidate.question
        else:
            raise TypeError(f"Unsupported generation candidate type: {type(candidate)!r}")

        return BenchmarkSample(
            sample_id=sample_id,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            source_path=source_path,
            page_number=page_number,
            source_excerpt=self._trim_text(page_text, limit=2000),
            criteria=criteria,
            judge_schema_name=JUDGE_SCHEMA_BY_TYPE[question_type].__name__,
            metadata={
                "generation_model_name": self.generation_model_name,
                "source_characters": len(page_text),
            },
        )

    def _build_generation_prompt(
        self,
        question_type: QuestionType,
        source_path: str,
        page_number: int,
        page_text: str,
    ) -> str:
        common = GENERATION_COMMON_PROMPT_TEMPLATE.format(
            source_path=source_path,
            page_number=page_number,
            page_text=page_text,
        )
        instructions = GENERATION_TYPE_INSTRUCTIONS[question_type]
        return f"{common}\n\n{instructions}"

    def _persist_generation_state(
        self,
        dataset_path: Path,
        metadata_path: Path,
        dataframe_path: Path,
        samples: Sequence[BenchmarkSample],
        experiment_name: str,
        questions_per_type: int,
        seed: int,
        pdf_paths: Sequence[str],
        per_type_counts: Dict[QuestionType, int],
        skipped_per_type: Dict[QuestionType, int],
        max_attempts_per_type: int,
        status: str,
        error_message: Optional[str],
    ) -> None:
        self._write_jsonl(dataset_path, samples)
        dataframe = self.samples_to_dataframe(samples)
        dataframe.to_csv(dataframe_path, index=False)
        metadata = {
            "experiment_name": experiment_name,
            "created_at": self._timestamp(),
            "generation_model_name": self.generation_model_name,
            "questions_per_type": questions_per_type,
            "seed": seed,
            "pdf_paths": list(pdf_paths),
            "counts": {question_type.value: count for question_type, count in per_type_counts.items()},
            "skipped": {question_type.value: skipped_per_type[question_type] for question_type in QuestionType},
            "max_attempts_per_type": max_attempts_per_type,
            "samples_generated": len(samples),
            "status": status,
            "error_message": error_message,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _make_output_dir(self, experiment_name: str, prefix: str) -> Path:
        safe_name = self._slugify(experiment_name)
        path = self.results_dir / f"{prefix}_{safe_name}_{self._timestamp_for_path()}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _resolve_output_dir(self, experiment_name: str, prefix: str, resume: bool) -> Path:
        if not resume:
            return self._make_output_dir(experiment_name=experiment_name, prefix=prefix)
        existing_dir = self._find_latest_output_dir(experiment_name=experiment_name, prefix=prefix)
        if existing_dir is not None:
            self.logger.info("Resuming benchmark generation in %s", existing_dir)
            return existing_dir
        return self._make_output_dir(experiment_name=experiment_name, prefix=prefix)

    def _find_latest_output_dir(self, experiment_name: str, prefix: str) -> Optional[Path]:
        safe_name = self._slugify(experiment_name)
        pattern = f"{prefix}_{safe_name}_*"
        matches = sorted(
            [path for path in self.results_dir.glob(pattern) if path.is_dir()],
            key=lambda path: path.name,
        )
        if not matches:
            return None
        return matches[-1]

    def _load_existing_samples(self, dataset_path: Path) -> List[BenchmarkSample]:
        if not dataset_path.exists():
            return []
        return self.load(str(dataset_path))

    def _count_samples_by_type(self, samples: Sequence[BenchmarkSample]) -> Dict[QuestionType, int]:
        counts: Dict[QuestionType, int] = defaultdict(int)
        for sample in samples:
            counts[sample.question_type] += 1
        return counts

    def _load_skipped_counts(self, metadata_path: Path) -> Dict[QuestionType, int]:
        skipped: Dict[QuestionType, int] = defaultdict(int)
        if not metadata_path.exists():
            return skipped
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        for question_type in QuestionType:
            skipped[question_type] = int(payload.get("skipped", {}).get(question_type.value, 0))
        return skipped

    def _write_jsonl(self, path: Path, samples: Sequence[BenchmarkSample]) -> None:
        lines = [json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) for sample in samples]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _slugify(self, value: str) -> str:
        stripped = value.strip().replace(" ", "_")
        return self._slug_pattern.sub("", stripped) or "experiment"

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _timestamp_for_path() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")
