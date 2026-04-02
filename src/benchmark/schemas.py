from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    FACTUAL = "factual"
    CHOICE = "choice"
    MULTI_FACT = "multi_fact"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    NEGATIVE = "negative"


class BenchmarkSample(BaseModel):
    sample_id: str = Field(..., description="Stable unique question id")
    question_type: QuestionType
    question: str
    gold_answer: str
    source_path: str
    page_number: int
    source_excerpt: str
    criteria: Dict[str, object] = Field(
        default_factory=dict,
        description="Structured evaluation criteria used by the judge model",
    )
    judge_schema_name: str
    metadata: Dict[str, object] = Field(default_factory=dict)


class GenerationBatch(BaseModel):
    items: List[BenchmarkSample]


class FactualQuestionCandidate(BaseModel):
    question: str = Field(..., description="Short factual question with one correct answer")
    gold_answer: str = Field(..., description="Canonical correct answer")
    fact: str = Field(..., description="Normalized atomic fact behind the answer")


class ChoiceOption(BaseModel):
    key: Literal["A", "B", "C"]
    text: str


class ChoiceQuestionCandidate(BaseModel):
    question: str
    options: List[ChoiceOption] = Field(..., min_length=3, max_length=3)
    correct_option_key: Literal["A", "B", "C"]
    gold_answer: str = Field(..., description="Correct option text with key")
    explanation_fact: str = Field(..., description="Fact that makes the option correct")


class MultiFactQuestionCandidate(BaseModel):
    question: str
    gold_answer: str
    atomic_facts: List[str] = Field(..., min_length=2)


class ProcedureQuestionCandidate(BaseModel):
    question: str
    gold_answer: str
    required_steps: List[str] = Field(..., min_length=2)
    optional_steps: List[str] = Field(default_factory=list)
    critical_order_pairs: List[str] = Field(
        default_factory=list,
        description="Pairs in format 'step A -> step B'",
    )


class ComparisonQuestionCandidate(BaseModel):
    question: str
    gold_answer: str
    compared_entities: List[str] = Field(..., min_length=2, max_length=2)
    comparison_axes: List[str] = Field(..., min_length=2)
    expected_differences: List[str] = Field(..., min_length=2)


class NegativeQuestionCandidate(BaseModel):
    question: str
    gold_answer: str = Field(
        ...,
        description="Reference abstaining answer grounded in the missing information",
    )
    unsupported_reason: str = Field(
        ...,
        description="Why the documentation cannot support the answer",
    )
    forbidden_claims: List[str] = Field(
        default_factory=list,
        description="Claims that would be hallucinations in the answer",
    )


class FactualGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[FactualQuestionCandidate] = None


class ChoiceGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[ChoiceQuestionCandidate] = None


class MultiFactGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[MultiFactQuestionCandidate] = None


class ProcedureGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[ProcedureQuestionCandidate] = None


class ComparisonGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[ComparisonQuestionCandidate] = None


class NegativeGenerationResult(BaseModel):
    can_generate: bool
    reason: str = ""
    payload: Optional[NegativeQuestionCandidate] = None


class FactualJudgeResult(BaseModel):
    reasoning: List[str] = Field(..., description="Short justification grounded in source facts")
    is_correct: bool


class ChoiceJudgeResult(BaseModel):
    reasoning: List[str]
    is_correct: bool


class MultiFactJudgeResult(BaseModel):
    reasoning: List[str]
    supported_facts: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    false_claims: List[str] = Field(default_factory=list)


class ProcedureJudgeResult(BaseModel):
    reasoning: List[str]
    covered_required_steps: List[str] = Field(default_factory=list)
    missing_required_steps: List[str] = Field(default_factory=list)
    covered_optional_steps: List[str] = Field(default_factory=list)
    hallucinated_steps: List[str] = Field(default_factory=list)
    critical_order_correct: bool


class ComparisonJudgeResult(BaseModel):
    reasoning: List[str]
    correct_axes: List[str] = Field(default_factory=list)
    incorrect_axes: List[str] = Field(default_factory=list)
    swapped_axes: List[str] = Field(default_factory=list)
    hallucinated_axes: List[str] = Field(default_factory=list)


class NegativeJudgeResult(BaseModel):
    reasoning: List[str]
    correct_abstention: bool
    cited_missing_information: bool
    unsupported_claims: List[str] = Field(default_factory=list)


GENERATION_SCHEMA_BY_TYPE: Dict[QuestionType, Type[BaseModel]] = {
    QuestionType.FACTUAL: FactualGenerationResult,
    QuestionType.CHOICE: ChoiceGenerationResult,
    QuestionType.MULTI_FACT: MultiFactGenerationResult,
    QuestionType.PROCEDURE: ProcedureGenerationResult,
    QuestionType.COMPARISON: ComparisonGenerationResult,
    QuestionType.NEGATIVE: NegativeGenerationResult,
}


JUDGE_SCHEMA_BY_TYPE: Dict[QuestionType, Type[BaseModel]] = {
    QuestionType.FACTUAL: FactualJudgeResult,
    QuestionType.CHOICE: ChoiceJudgeResult,
    QuestionType.MULTI_FACT: MultiFactJudgeResult,
    QuestionType.PROCEDURE: ProcedureJudgeResult,
    QuestionType.COMPARISON: ComparisonJudgeResult,
    QuestionType.NEGATIVE: NegativeJudgeResult,
}
