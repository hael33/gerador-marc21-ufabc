import os
import json
import time
import gradio as gr
from pypdf import PdfReader
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")

MODELOS_DISPONIVEIS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

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
        
        ATENÇÃO EXTREMA À ACENTUAÇÃO E CORREÇÃO DE TEXTO:
        O texto do PDF contém marcas de acentuação corrompidas vindas de compilação em LaTeX 
        (exemplo: 'Aplica¸ c˜ ao' deve virar 'Aplicação', 'M´ edias' deve virar 'Médias', 
        'Resolu¸ c˜ ao' deve virar 'Resolução', 'Pedag´ ogico' deve virar 'Pedagógico', 
        'M´ arcio' deve virar 'Márcio', 'Matem´ atica' deve virar 'Matemática').
        
        Você DEVE corrigir e normalizar toda a ortografia, acentos e cedilhas na língua portuguesa ao gerar o JSON.

        Extraia as informações e responda ESTRITAMENTE em formato JSON com as chaves:
        - autor_sobrenome (string, ex: COSTA)
        - autor_nome (string, ex: Marcelo Rodrigues da)
        - titulo (string, totalmente corrigido e acentuado)
        - subtitulo (string, totalmente corrigido e acentuado, ou vazio se não houver)
        - ano (string, ex: 2025)
        - cidade (string, padrão: Santo André ou São Bernardo do Campo conforme o documento)
        - instituicao (string, padrão: Universidade Federal do ABC)
        - programa (string, totalmente corrigido e acentuado, ex: Mestrado Profissional em Matemática em Rede Nacional)
        - grau (string, ex: Dissertação (Mestrado) ou Tese (Doutorado))
        - orientador (string, totalmente corrigido e acentuado, ex: Prof. Dr. Márcio Fabiano da Silva)
        - resumo (string, totalmente corrigido e acentuado)
        - palavras_chave (lista de strings, totalmente corrigidas e acentuadas)

        Texto extraído do PDF:
        {texto_pdf}
        """

        response = None
        ultimo_erro = None

        for modelo in MODELOS_DISPONIVEIS:
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                if response and response.text:
                    break
            except Exception as err:
                ultimo_erro = err
                time.sleep(1)
                continue

        if not response or not response.text:
            raise Exception(f"Serviço temporariamente indisponível. Detalhe: {str(ultimo_erro)}")

        dados = json.loads(response.text)

        sobrenome = dados.get("autor_sobrenome", "").upper()
        nome = dados.get("autor_nome", "")
        titulo = dados.get("titulo", "")
        subtitulo = f": {dados['subtitulo']}" if dados.get("subtitulo") else ""
        ano = dados.get("ano", "2025")
        grau = dados.get("grau", "Dissertação (Mestrado)")
        inst = dados.get("instituicao", "Universidade Federal do ABC")
        cidade = dados.get("cidade", "Santo André")

        abnt = f"**{sobrenome}, {nome}**. *{titulo}*{subtitulo}. {ano}. {grau} – {inst}, {cidade}, {ano}."

        sub_marc = f" : |b {dados['subtitulo']}" if dados.get("subtitulo") else ""
        linhas_marc = [
            f"100 1_ |a {sobrenome}, {nome}",
            f"245 10 |a {titulo}{sub_marc} / |c {nome} {sobrenome}.",
            f"264 _1 |a {cidade} : |b [s.n.], |c {ano}.",
            f"300 __ |a {total_paginas} f.",
        ]

        if dados.get("orientador"):
            linhas_marc.append(f"500 __ |a Orientador(a): {dados['orientador']}.")
        
        linhas_marc.append(f"502 __ |a {grau} - {inst}, {ano}.")

        if dados.get("resumo"):
            linhas_marc.append(f"520 3_ |a {dados['resumo']}")

        for kw in dados.get("palavras_chave", []):
            kw_limpa = kw.strip('.')
            linhas_marc.append(f"650 _0 |a {kw_limpa}.")

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
