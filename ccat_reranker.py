from cat import hook, log, RecallSettings
from sentence_transformers import CrossEncoder

from .rankers import litm, sbert_ranker

_MODELS_CACHE = {}


@hook(priority=1)
def after_cat_recalls_memories(config: RecallSettings, cat) -> None:
    if not cat.working_memory.context_memories:
        return

    settings = cat.mad_hatter.get_plugin().load_settings()

    final_docs = None
    if settings["sbert"]:
        model_name = settings["ranker"]
        
        if model_name not in _MODELS_CACHE:
            log.debug(f"Loading SBERT model: {model_name}...")
            _MODELS_CACHE[model_name] = CrossEncoder(model_name)
        
        model = _MODELS_CACHE[model_name]
        final_docs = sbert_ranker(
            cat.working_memory.context_memories,
            cat.working_memory.history[0].content.text,
            model,
        )

    if settings["litm"]:
        final_docs = litm(final_docs or cat.working_memory.context_memories)

    if final_docs:
        cat.working_memory.context_memories = final_docs
