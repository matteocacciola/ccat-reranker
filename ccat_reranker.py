from cat import hook, RecallSettings
from sentence_transformers import CrossEncoder

from .rankers import litm, sbert_ranker


@hook(priority=1)
def after_cat_recalls_memories(config: RecallSettings, cat) -> None:
    if not cat.working_memory.context_memories:
        return

    settings = cat.mad_hatter.get_plugin().load_settings()

    final_docs = None
    if settings["sbert"]:
        model = CrossEncoder(settings["ranker"])
        final_docs = sbert_ranker(
            cat.working_memory.context_memories, cat.working_memory.history[0].content.text, model
        )

    if settings["litm"]:
        final_docs = litm(final_docs or cat.working_memory.context_memories)

    if final_docs:
        cat.working_memory.context_memories = final_docs
