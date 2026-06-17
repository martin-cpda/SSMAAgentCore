from typing import Any

from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client
from memory.session import get_memory_session_manager
from knowledge_base.retrieve import retrieve_ssma_docs

app = BedrockAgentCoreApp()
log = app.logger

# Define a Streamable HTTP MCP Client
mcp_clients = [get_streamable_http_mcp_client()]

DEFAULT_SYSTEM_PROMPT = """
Eres el asistente SSMA de Braskem Idesa. Respondes preguntas sobre seguridad,
salud y medio ambiente apoyándote en la documentación corporativa.

Usa la herramienta retrieve_ssma_docs para buscar en la base de conocimiento
antes de responder cualquier pregunta sobre procedimientos, normas o documentos.
Basa tus respuestas en la información recuperada y no inventes datos que no estén
en los documentos. Si no encuentras información relevante, indícalo con claridad.
"""


# Define a collection of tools used by the model
tools = []

# Recuperación RAG sobre la Knowledge Base (S3 Vectors) desplegada por el IaC.
tools.append(retrieve_ssma_docs)


# Add MCP client to tools if available
for mcp_client in mcp_clients:
    if mcp_client:
        tools.append(mcp_client)


def agent_factory():
    cache = {}
    def get_or_create_agent(session_id, user_id):
        key = f"{session_id}/{user_id}"
        if key not in cache:
            cache[key] = Agent(
                model=load_model(),
                session_manager=get_memory_session_manager(session_id, user_id),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                tools=tools
            )
        return cache[key]
    return get_or_create_agent
get_or_create_agent = agent_factory()


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")


    session_id = getattr(context, 'session_id', 'default-session')
    user_id = getattr(context, 'user_id', 'default-user')
    agent = get_or_create_agent(session_id, user_id)

    # Execute and format response
    stream = agent.stream_async(payload.get("prompt"))

    async for event in stream:
        # Handle Text parts of the response
        if "data" in event and isinstance(event["data"], str):
            yield event["data"]


if __name__ == "__main__":
    app.run()
