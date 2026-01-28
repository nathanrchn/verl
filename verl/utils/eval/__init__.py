"""
Training-time evaluation module for VERL.
"""
from verl.utils.eval.config import EvalConfig, RolloutConfig
from verl.utils.eval.dataset_factory import DatasetFactory
from verl.utils.eval.generation_logger import GenerationLogger
from verl.utils.eval.manager import EvalManager
from verl.utils.eval.scorer import AutoScoringJudge, get_scorer
from verl.utils.eval.tasks import (
    RolloutParams,
    TASK_REGISTRY,
    get_task,
    math_task,
    code_task,
    code_thinking_task,
    fact_verification_task,
    ifeval_task,
    ifbench_task,
    mcq_task,
    extract_answer_from_tool_call,
)
from verl.utils.eval.utils import (
    compute_token_ttr,
    compute_text_ttr,
    is_degenerating,
)

__all__ = [
    "EvalConfig",
    "RolloutConfig",
    "DatasetFactory",
    "GenerationLogger",
    "EvalManager",
    "AutoScoringJudge",
    "get_scorer",
    "RolloutParams",
    "TASK_REGISTRY",
    "get_task",
    "compute_token_ttr",
    "compute_text_ttr",
    "is_degenerating",
    "extract_answer_from_tool_call",
    "math_task",
    "code_task",
    "code_thinking_task",
    "fact_verification_task",
    "ifeval_task",
    "ifbench_task",
    "mcq_task",
]
