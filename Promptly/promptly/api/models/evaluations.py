"""
Evaluation API Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TestCase(BaseModel):
    """Test case model"""
    id: Optional[str] = Field(default=None, description="Test case ID")
    inputs: Dict[str, Any] = Field(..., description="Input variables")
    expected: Optional[str] = Field(default=None, description="Expected output")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class EvaluationRequest(BaseModel):
    """Run evaluation request"""
    prompt_name: str = Field(..., description="Prompt to evaluate")
    test_cases: List[TestCase] = Field(..., min_items=1, description="Test cases")
    evaluator: str = Field(default="keyword", description="Evaluator to use")
    model_endpoint: Optional[str] = Field(default=None, description="Optional model endpoint")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt_name": "summarizer",
                "test_cases": [
                    {
                        "inputs": {"text": "Long text here..."},
                        "expected": "Summary here..."
                    }
                ],
                "evaluator": "semantic"
            }
        }


class EvaluationResult(BaseModel):
    """Single evaluation result"""
    test_case_id: Optional[str]
    formatted_prompt: str
    actual: Optional[str]
    expected: Optional[str]
    score: Optional[float]
    metrics: Dict[str, Any] = {}


class EvaluationResponse(BaseModel):
    """Evaluation response"""
    evaluation_id: str
    prompt_name: str
    commit_hash: str
    results: List[EvaluationResult]
    average_score: Optional[float]
    total_tests: int
    passed_tests: int
    failed_tests: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationListItem(BaseModel):
    """Evaluation list item"""
    evaluation_id: str
    prompt_name: str
    commit_hash: str
    average_score: Optional[float]
    total_tests: int
    created_at: str


class EvaluationCompareRequest(BaseModel):
    """Compare evaluations request"""
    evaluation_ids: List[str] = Field(..., min_items=2, max_items=10)


class EvaluationCompareResponse(BaseModel):
    """Compare evaluations response"""
    evaluations: List[EvaluationListItem]
    comparison: Dict[str, Any]
    winner: Optional[str] = None
