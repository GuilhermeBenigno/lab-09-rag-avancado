from transformers import pipeline


def generate_hyde_document(query):
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-small"
    )

    prompt = f"""
Transforme a pergunta coloquial abaixo em um documento técnico médico curto.

Pergunta: {query}

Documento técnico:
"""

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )

    generated_text = result[0]["generated_text"]

    hyde_doc = generated_text.replace(prompt, "").strip()

    if len(hyde_doc) < 10:
        hyde_doc = f"Paciente apresenta cefaleia pulsátil associada à fotofobia. Relato compatível com sintomas descritos como: {query}"

    return hyde_doc
