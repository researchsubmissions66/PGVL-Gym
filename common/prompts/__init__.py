"""Dataset-level prompt specifications and method-native asset compilation."""

from .cod_mil import (
    MIN_BACKGROUND_PROMPTS,
    PROMPT_FEATURE_SCHEMA,
    expected_row_roles,
    file_sha256,
    load_prompt_bank_csv,
    prompt_bank_sha256,
    prompt_feature_metadata,
    validate_prompt_feature_metadata,
)
from .compiler import compile_task_prompt_assets, load_prompt_profile
from .convlm import (
    ATTRIBUTE_EMBEDDING_SCHEMA,
    ConVLMAttributeBank,
    ConVLMPromptBank,
    convlm_prompt_bank_sha256,
    load_convlm_attribute_embeddings,
    load_convlm_prompt_bank,
)
from .focus import (
    FOCUS_PROMPT_FORMAT,
    FOCUSPromptBank,
    focus_prompt_bank_sha256,
    load_focus_prompt_bank,
)
from .maple import MAPLEPromptBank, load_maple_prompt_bank
from .muse import (
    MUSEPromptBank,
    MUSEPromptCSV,
    load_muse_prompt_bank,
    load_muse_prompt_csv,
)
from .slip import SLIPPromptBank, load_slip_prompt_bank
from .sldpc import (
    SLDPC_ZERO_SHOT_TEMPLATES,
    SLDPCZeroShotPromptBank,
    load_sldpc_zero_shot_prompt_bank,
    sldpc_prompt_classname_sha256,
    sldpc_prompt_classnames,
    sldpc_zero_shot_templates_sha256,
)
from .vila_mil import (
    VILA_PROMPT_FORMAT,
    ViLAPromptBank,
    load_vila_prompt_bank,
    vila_prompt_bank_sha256,
)
from .top import (
    TOP_PROMPT_FORMAT,
    TOPBagPromptBank,
    TOPInstancePromptBank,
    TOPPromptCondition,
    load_top_bag_prompt_bank,
    load_top_instance_prompt_bank,
    load_top_prompt_condition,
    resolve_top_prompt_condition,
    top_bag_prompt_bank_sha256,
    top_instance_prompt_bank_sha256,
)
from .wsi_five import (
    WSI_FIVE_PROMPT_FORMAT,
    WSIFiVEAnswerBank,
    WSIFiVEEvaluationBank,
    WSIFiVEQuestionBank,
    load_wsi_five_answer_bank,
    load_wsi_five_evaluation_bank,
    load_wsi_five_question_bank,
)

__all__ = [
    "MIN_BACKGROUND_PROMPTS", "PROMPT_FEATURE_SCHEMA",
    "compile_task_prompt_assets",
    "ATTRIBUTE_EMBEDDING_SCHEMA", "ConVLMAttributeBank", "ConVLMPromptBank",
    "convlm_prompt_bank_sha256", "load_convlm_attribute_embeddings",
    "load_convlm_prompt_bank",
    "FOCUS_PROMPT_FORMAT", "FOCUSPromptBank", "focus_prompt_bank_sha256",
    "load_focus_prompt_bank",
    "expected_row_roles", "file_sha256", "load_prompt_bank_csv",
    "load_prompt_profile", "prompt_bank_sha256", "prompt_feature_metadata",
    "MAPLEPromptBank", "load_maple_prompt_bank",
    "MUSEPromptBank", "MUSEPromptCSV", "load_muse_prompt_bank",
    "load_muse_prompt_csv",
    "SLIPPromptBank", "load_slip_prompt_bank",
    "SLDPC_ZERO_SHOT_TEMPLATES", "SLDPCZeroShotPromptBank",
    "load_sldpc_zero_shot_prompt_bank", "sldpc_prompt_classname_sha256",
    "sldpc_prompt_classnames", "sldpc_zero_shot_templates_sha256",
    "VILA_PROMPT_FORMAT", "ViLAPromptBank", "load_vila_prompt_bank",
    "vila_prompt_bank_sha256",
    "TOP_PROMPT_FORMAT", "TOPBagPromptBank", "TOPInstancePromptBank",
    "TOPPromptCondition", "load_top_bag_prompt_bank",
    "load_top_instance_prompt_bank", "load_top_prompt_condition",
    "resolve_top_prompt_condition", "top_bag_prompt_bank_sha256",
    "top_instance_prompt_bank_sha256",
    "WSI_FIVE_PROMPT_FORMAT", "WSIFiVEAnswerBank",
    "WSIFiVEEvaluationBank", "WSIFiVEQuestionBank",
    "load_wsi_five_answer_bank", "load_wsi_five_evaluation_bank",
    "load_wsi_five_question_bank",
    "validate_prompt_feature_metadata",
]
