"""
Due esempi di Output Strutturati (Structured Outputs) con l'API Chat Completions.

Esempio 1 - Tutor di matematica: schema JSON passato a mano con strict=True.
Esempio 2 - Riassunto di articoli: schema generato da un modello Pydantic.

Il modello viene servito da Groq, compatibile con il client OpenAI.
La chiave API si legge dal file .env (mai scritta nel codice).
"""

import json
import os
from textwrap import dedent

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # legge il file .env

MODEL = "openai/gpt-oss-120b"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)


# ---------------------------------------------------------------------------
# Esempio 1: tutor di matematica, con schema JSON scritto a mano
# ---------------------------------------------------------------------------

math_tutor_prompt = """
    Sei un tutor di matematica.
    Ti verrà fornito un problema matematico e il tuo obiettivo sarà fornire
    una soluzione passo dopo passo, insieme a una risposta finale.
    Per ogni passaggio, metti l'equazione nel campo output e il ragionamento
    nel campo explanation. Rispondi in JSON.
"""

math_schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["explanation", "output"],
                "additionalProperties": False,
            },
        },
        "final_answer": {"type": "string"},
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": False,
}


def get_math_solution(question: str) -> dict:
    """Chiede la soluzione al modello e restituisce il JSON come dizionario."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_reasoning",
                "strict": True,
                "schema": math_schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content)


def print_math_solution(solution: dict) -> None:
    for i, step in enumerate(solution["steps"], start=1):
        print(f"Passo {i}: {step['explanation']}")
        print(f"         {step['output']}\n")
    print(f"Risposta finale: {solution['final_answer']}")


# ---------------------------------------------------------------------------
# Esempio 2: riassunto di articoli, con schema generato da Pydantic
# ---------------------------------------------------------------------------

summarization_prompt = """
    Ti verrà fornito il contenuto di un articolo su un'invenzione.
    Il tuo obiettivo sarà riassumere l'articolo seguendo lo schema JSON fornito.
    Ecco una descrizione dei parametri:
    - invented_year: anno in cui è stata inventata l'invenzione
    - summary: riassunto in una frase che descrive l'invenzione
    - inventors: elenco dei nomi completi degli inventori
    - concepts: concetti chiave, ognuno con un titolo e una descrizione
    - description: breve descrizione dell'invenzione
"""


class Concept(BaseModel):
    title: str
    description: str


class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str
    concepts: list[Concept]

    def __str__(self) -> str:
        inventors = "\n".join(f"- {i}" for i in self.inventors)
        concepts = "\n".join(f"- {c.title}: {c.description}" for c in self.concepts)
        return (
            f"Anno: {self.invented_year}\n\n"
            f"In una frase: {self.summary}\n\n"
            f"Inventori:\n{inventors}\n\n"
            f"Concetti chiave:\n{concepts}\n\n"
            f"Descrizione: {self.description}"
        )


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """Adatta lo schema di Pydantic al formato richiesto da strict:
    sostituisce i riferimenti ($defs) con il loro contenuto e aggiunge
    additionalProperties: false a ogni oggetto."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def risolvi(nodo):
        if isinstance(nodo, dict):
            if "$ref" in nodo:
                nome = nodo.pop("$ref").split("/")[-1]
                nodo.update(risolvi(dict(defs[nome])))
            if nodo.get("type") == "object":
                nodo["additionalProperties"] = False
            for valore in nodo.values():
                risolvi(valore)
        elif isinstance(nodo, list):
            for valore in nodo:
                risolvi(valore)
        return nodo

    return risolvi(schema)


def get_article_content(path: str) -> str:
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def get_article_summary(text: str) -> ArticleSummary:
    """Riassume l'articolo: lo schema viene ricavato dal modello Pydantic."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": dedent(summarization_prompt)},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "article_summary",
                "strict": True,
                "schema": pydantic_to_strict_schema(ArticleSummary),
            },
        },
    )
    # Pydantic controlla che la risposta rispetti davvero lo schema
    return ArticleSummary.model_validate_json(response.choices[0].message.content)


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("ESEMPIO 1 - Tutor di matematica (schema JSON scritto a mano)")
    print("=" * 70)
    question = "come posso risolvere 8x + 7 = -23"
    print(f"Domanda: {question}\n")
    print_math_solution(get_math_solution(question))

    print("\n" + "=" * 70)
    print("ESEMPIO 2 - Riassunto di articoli (schema da modello Pydantic)")
    print("=" * 70)
    articles = [
        "./structured_outputs_articles/cnns.md",
        "./structured_outputs_articles/llms.md",
        "./structured_outputs_articles/moe.md",
    ]
    for path in articles:
        print(f"\n--- {path} ---")
        print(get_article_summary(get_article_content(path)))


if __name__ == "__main__":
    main()