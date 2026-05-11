from transformers import pipeline


def generate_hyde_document(query):
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    prompt = f"""
Transforme a pergunta coloquial abaixo em um documento técnico médico curto.

Pergunta: {query}

Documento técnico:
"""

    result = generator(prompt, max_new_tokens=80, do_sample=False)

    hyde_doc = result[0]["generated_text"]

    if len(hyde_doc.strip()) < 10:
        hyde_doc = f"Paciente apresenta sintomas compatíveis com quadro clínico relacionado a: {query}"

    return hyde_doc
