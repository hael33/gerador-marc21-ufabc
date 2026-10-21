import os
import json
import time
import re
import unicodedata
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

def sanitizar_texto_utf8(texto):
    """
    Higieniza e normaliza qualquer texto extraído do PDF para UTF-8 padrão (NFC).
    Remove caracteres invisíveis de controle, nulos e repara combinadores soltos.
    """
    if not texto:
        return ""
    
    # 1. Normaliza para Form N-C (transforma 'a' + '˜' no caractere precomposto 'ã')
    texto = unicodedata.normalize("NFC", texto)
    
    # 2. Remove caracteres nulos e de controle não imprimíveis (exceto \n e \t)
    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', texto)
    
    # 3. Normaliza espaços múltiplos mantendo a estrutura
    texto = re.sub(r'[ \t]+', ' ', texto)
    
    return texto.strip()

def extrair_texto_pdf(pdf_path, max_paginas=12):
    """Lê as páginas iniciais do PDF tratando erros de extração e encodings."""
    reader = PdfReader(pdf_path)
    total_paginas = len(reader.pages)
    texto_acumulado = []
    
    for i in range(min(max_paginas, total_paginas)):
        try:
            extraido = reader.pages[i].extract_text()
            if extraido:
                limpo = sanitizar_texto_utf8(extraido)
                texto_acumulado.append(f"\n--- PÁGINA {i+1} ---\n{limpo}")
        except Exception:
            continue
            
    texto_completo = "\n".join(texto_acumulado)
    return sanitizar_texto_utf8(texto_completo), total_paginas

def processar_tese(arquivo_pdf):
    if not API_KEY:
        return "❌ Erro: Chave GEMINI_API_KEY não configurada no Render.", "", "", ""
    if arquivo_pdf is None:
        return "Aguardando envio do arquivo PDF...", "", "", ""

    try:
        texto_pdf, total_paginas = extrair_texto_pdf(arquivo_pdf.name)
        
        if not texto_pdf.strip():
            return "❌ Erro: Não foi possível extrair texto legível deste PDF. O arquivo pode ser uma imagem digitalizada sem OCR.", "", "", ""

        client = genai.Client(api_key=API_KEY)
        
        prompt = f"""
        Você é um especialista em catalogação bibliográfica (MARC21 e ABNT) da Biblioteca da UFABC.
        Sua tarefa é analisar o texto extraído das páginas iniciais de um trabalho acadêmico e gerar os metadados em JSON.

        REGRAS CRÍTICAS DE TRATAMENTO DE TEXTO E CODIFICAÇÃO (UTF-8):
        1. CORREÇÃO DE ACENTUAÇÃO E LATEX:
           - Repare acentos e cedilhas corrompidos por compilação LaTeX (ex: 'Aplica¸ c˜ ao' -> 'Aplicação', 'M´ edias' -> 'Médias', 'Resolu¸ c˜ ao' -> 'Resolução').
           - Repare falhas de mojibake/encoding (ex: 'Ã§Ã£o' -> 'ção', 'SÃ£o' -> 'São').
           - Repare ligaduras tipográficas se houver (ex: 'ﬁ' -> 'fi', 'ﬂ' -> 'fl').
        2. NORMALIZAÇÃO DE NOMES E TÍTULOS:
           - Remova hifenizações quebras no final de linha (ex: 'Desenvol- vimento' -> 'Desenvolvimento').
           - Mantenha títulos em 'Sentence case' ou capitalização padrão da língua portuguesa.
        3. IDIOMA: Responda TODO o conteúdo estritamente em português correto, com acentuação ortográfica perfeita.

        Extraia os campos e responda ESTRITAMENTE em formato JSON com a seguinte estrutura:
        {{
          "autor_sobrenome": "Sobrenome do autor em MAIÚSCULAS (ex: COSTA)",
          "autor_nome": "Nome e prenomes do autor (ex: Marcelo Rodrigues da)",
          "titulo": "Título principal sem o subtítulo",
          "subtitulo": "Subtítulo (se houver, sem a coluna inicial) ou vazio",
          "ano": "Ano da defesa/publicação (4 dígitos)",
          "cidade": "Cidade do campus/instituição (padrão: Santo André ou São Bernardo do Campo)",
          "instituicao": "Nome da instituição (padrão: Universidade Federal do ABC)",
          "programa": "Nome completo do Programa de Pós-Graduação ou Curso",
          "grau": "Tipo do trabalho (ex: Dissertação (Mestrado) ou Tese (Doutorado))",
          "orientador": "Nome completo do(a) orientador(a) com titulação se houver",
          "resumo": "Texto do resumo extraído do documento",
          "palavras_chave": ["Lista", "de", "palavras-chave"]
        }}

        Texto bruto extraído do PDF:
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
            raise Exception(f"Falha de comunicação com a API. Detalhe: {str(ultimo_erro)}")

        # Garante a normalização UTF-8 também na resposta JSON da IA
        resposta_json_limpa = sanitizar_texto_utf8(response.text)
        dados = json.loads(resposta_json_limpa)

        sobrenome = sanitizar_texto_utf8(dados.get("autor_sobrenome", "")).upper()
        nome = sanitizar_texto_utf8(dados.get("autor_nome", ""))
        titulo = sanitizar_texto_utf8(dados.get("titulo", ""))
        subtitulo_raw = sanitizar_texto_utf8(dados.get("subtitulo", ""))
        subtitulo_fmt = f" : {subtitulo_raw}" if subtitulo_raw else ""
        ano = sanitizar_texto_utf8(str(dados.get("ano", "2025")))
        grau = sanitizar_texto_utf8(dados.get("grau", "Dissertação (Mestrado)"))
        inst = sanitizar_texto_utf8(dados.get("instituicao", "Universidade Federal do ABC"))
        cidade = sanitizar_texto_utf8(dados.get("cidade", "Santo André"))
        orientador = sanitizar_texto_utf8(dados.get("orientador", ""))
        resumo = sanitizar_texto_utf8(dados.get("resumo", ""))
        programa = sanitizar_texto_utf8(dados.get("programa", ""))

        # Formatação Referência ABNT
        abnt = f"**{sobrenome}, {nome}**. *{titulo}*{subtitulo_fmt}. {ano}. {total_paginas} f. {grau} – {inst}, {cidade}, {ano}."

        # Formatação Campos MARC21
        sub_marc = f" : |b {subtitulo_raw}" if subtitulo_raw else ""
        linhas_marc = [
            f"100 1_ |a {sobrenome}, {nome}",
            f"245 10 |a {titulo}{sub_marc} / |c {nome} {sobrenome}.",
            f"264 _1 |a {cidade} : |b [s.n.], |c {ano}.",
            f"300 __ |a {total_paginas} f.",
        ]

        if orientador:
            linhas_marc.append(f"500 __ |a Orientador(a): {orientador}.")
        
        linhas_marc.append(f"502 __ |a {grau} - {inst}, {ano}.")

        if resumo:
            linhas_marc.append(f"520 3_ |a {resumo}")

        for kw in dados.get("palavras_chave", []):
            kw_limpa = sanitizar_texto_utf8(kw).strip('.')
            if kw_limpa:
                linhas_marc.append(f"650 _0 |a {kw_limpa}.")

        if programa:
            linhas_marc.append(f"710 2_ |a {inst}. |b {programa}.")

        marc_final = "\n".join(linhas_marc)

        return abnt, marc_final, resumo, "✅ Processado e codificado com sucesso em UTF-8!"

    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}", "", "", ""

# Interface Gradio
with gr.Blocks(title="Gerador MARC21 - UFABC") as demo:
    gr.Markdown("# 📚 Gerador Automático de MARC21 e Referência (UFABC)")
    gr.Markdown("Suporte universal a arquivos PDF (LaTeX, Word, LibreOffice) com normalização UTF-8.")
    
    with gr.Row():
        file_input = gr.File(label="Arraste o PDF da Tese/Dissertação aqui", file_types=[".pdf"], type="filepath")
    
    btn_processar = gr.Button("⚙️ Processar Documento", variant="primary")
    status_output = gr.Markdown()
    
    with gr.Column():
        abnt_output = gr.Markdown(label="Referência ABNT")
        marc_output = gr.Code(label="Registro MARC21", language=None, lines=14)
        resumo_output = gr.Textbox(label="Resumo Extraído", lines=5)

    btn_processar.click(
        fn=processar_tese,
        inputs=[file_input],
        outputs=[abnt_output, marc_output, resumo_output, status_output]
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
