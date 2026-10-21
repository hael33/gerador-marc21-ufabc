import os
import json
import gradio as gr
from pypdf import PdfReader
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")

def extrair_texto_pdf(pdf_path, max_paginas=10):
    reader = PdfReader(pdf_path)
    total_paginas = len(reader.pages)
    texto = ""
    for i in range(min(max_paginas, total_paginas)):
        extraido = reader.pages[i].extract_text()
        if extraido:
            texto += f"\n--- PÁGINA {i+1} ---\n" + extraido
    return texto, total_paginas

def processar_tese(arquivo_pdf):
    if not API_KEY:
        return "❌ Erro: Chave GEMINI_API_KEY não configurada no Render.", "", "", ""
    if arquivo_pdf is None:
        return "Aguardando envio do arquivo...", "", "", ""

    try:
        texto_pdf, total_paginas = extrair_texto_pdf(arquivo_pdf.name)
        
        client = genai.Client(api_key=API_KEY)
        
        prompt = f"""
        Analise o texto das páginas iniciais desta tese/dissertação da UFABC.
        Extraia as informações e responda ESTRITAMENTE em formato JSON com as chaves:
        - autor_sobrenome (string)
        - autor_nome (string)
        - titulo (string)
        - subtitulo (string)
        - ano (string)
        - cidade (string, padrão: São Bernardo do Campo)
        - instituicao (string, padrão: Universidade Federal do ABC)
        - programa (string)
        - grau (string, ex: Dissertação (Mestrado) ou Tese (Doutorado))
        - orientador (string)
        - resumo (string)
        - palavras_chave (lista de strings)

        Texto do PDF:
        {texto_pdf}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        dados = json.loads(response.text)

        sobrenome = dados.get("autor_sobrenome", "").upper()
        nome = dados.get("autor_nome", "")
        titulo = dados.get("titulo", "")
        subtitulo = f": {dados['subtitulo']}" if dados.get("subtitulo") else ""
        ano = dados.get("ano", "2024")
        grau = dados.get("grau", "Dissertação (Mestrado)")
        inst = dados.get("instituicao", "Universidade Federal do ABC")
        cidade = dados.get("cidade", "São Bernardo do Campo")

        abnt = f"**{sobrenome}, {nome}**. *{titulo}*{subtitulo}. {ano}. {grau} – {inst}, {cidade}, {ano}."

        sub_marc = f" : |b {dados['subtitulo']}" if dados.get("subtitulo") else ""
        linhas_marc = [
            f"100 1_ |a {dados.get('autor_sobrenome', '')}, {nome}",
            f"245 10 |a {titulo}{sub_marc} / |c {nome} {dados.get('autor_sobrenome', '')}.",
            f"264 _1 |a {cidade} : |b [s.n.], |c {ano}.",
            f"300 __ |a {total_paginas} f.",
        ]

        if dados.get("orientador"):
            linhas_marc.append(f"500 __ |a Orientador(a): {dados['orientador']}.")
        
        linhas_marc.append(f"502 __ |a {grau} - {inst}, {ano}.")

        if dados.get("resumo"):
            linhas_marc.append(f"520 3_ |a {dados['resumo']}")

        for kw in dados.get("palavras_chave", []):
            linhas_marc.append(f"650 _0 |a {kw.strip('.')}.")

        if dados.get("programa"):
            linhas_marc.append(f"710 2_ |a {inst}. |b {dados['programa']}.")

        return abnt, "\n".join(linhas_marc), dados.get("resumo", ""), "✅ Processado com sucesso!"

    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}", "", "", ""

with gr.Blocks(title="Gerador MARC21 - UFABC") as demo:
    gr.Markdown("# 📚 Gerador Automático de MARC21 e Referência (UFABC)")
    with gr.Row():
        file_input = gr.File(label="Arraste o PDF da Tese/Dissertação aqui", file_types=[".pdf"], type="filepath")
    btn_processar = gr.Button("⚙️ Processar Documento", variant="primary")
    status_output = gr.Markdown()
    with gr.Column():
        abnt_output = gr.Markdown(label="Referência ABNT")
        marc_output = gr.Code(label="Registro MARC21", language=None, lines=12)
        resumo_output = gr.Textbox(label="Resumo", lines=5)

    btn_processar.click(
        fn=processar_tese,
        inputs=[file_input],
        outputs=[abnt_output, marc_output, resumo_output, status_output]
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
