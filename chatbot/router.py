from langchain_core.callbacks import (
    UsageMetadataCallbackHandler
)

from chatbot.prompts import ROUTER_PROMPT

from chatbot.schemas import KnowledgeRoute


def identify_route(
    question,
    model
):

    router_model = (
        model.with_structured_output(
            KnowledgeRoute
        )
    )

    router_chain = (
        ROUTER_PROMPT
        | router_model
    )

    usage_callback = (
        UsageMetadataCallbackHandler()
    )

    result = router_chain.invoke(
        {
            "question": question
        },
        config={
            "callbacks": [
                usage_callback
            ]
        }
    )

    usage = (
        usage_callback.usage_metadata
    )

    return (
        result.route,
        usage
    )