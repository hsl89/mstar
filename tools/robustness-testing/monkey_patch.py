from lm_eval.api.task import PromptSourceTask
from lm_eval.models import MODEL_API_REGISTRY
from patched_functions import doc_to_perturbed_text, fewshot_context_w_perturber_logged
from mstar_autolm import MStarAutoCausalLM, MStarAutoSeq2SeqLM, AutoUL2

PromptSourceTask.doc_to_text = doc_to_perturbed_text
PromptSourceTask.fewshot_context = fewshot_context_w_perturber_logged

MODEL_API_REGISTRY["mstar-causal"] = MStarAutoCausalLM
MODEL_API_REGISTRY["mstar-seq2seq"] = MStarAutoSeq2SeqLM
MODEL_API_REGISTRY['ul2'] = AutoUL2