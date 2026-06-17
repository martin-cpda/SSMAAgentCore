import os
import logging

import boto3
from strands import tool

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
REGION = os.getenv("AWS_REGION")
# Número de pasajes a recuperar por consulta.
NUM_RESULTS = int(os.getenv("KB_NUM_RESULTS", "5"))

_client = None


def _get_client():
    """Cliente bedrock-agent-runtime perezoso (reutilizado entre invocaciones)."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agent-runtime", region_name=REGION)
    return _client


@tool
def retrieve_ssma_docs(query: str) -> str:
    """Busca en la documentación corporativa SSMA de Braskem Idesa.

    Úsala siempre que necesites información sobre procedimientos, normas o
    documentos de seguridad, salud y medio ambiente para responder al usuario.

    Args:
        query: La pregunta o términos de búsqueda en lenguaje natural.

    Returns:
        Los pasajes más relevantes de la base de conocimiento, o un mensaje
        indicando que no se encontró información.
    """
    if not KNOWLEDGE_BASE_ID:
        return "La base de conocimiento no está configurada (falta KNOWLEDGE_BASE_ID)."

    response = _get_client().retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}
        },
    )

    results = response.get("retrievalResults", [])
    if not results:
        return "No se encontró información relevante en la base de conocimiento."

    chunks = []
    for i, r in enumerate(results, 1):
        text = r.get("content", {}).get("text", "")
        source = r.get("location", {}).get("s3Location", {}).get("uri", "desconocida")
        score = r.get("score", 0.0)
        chunks.append(f"[{i}] (fuente: {source}, score: {score:.3f})\n{text}")

    return "\n\n".join(chunks)
