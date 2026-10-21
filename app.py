import os
import json
import time
import re
import unicodedata
import requests
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

# --- UTILITÁRIOS E EXTRAÇÃO DE TESES ---

def sanitizar_texto_utf8(texto):
    if not texto:
        return ""
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

def extrair_texto_pdf(pdf_path, max_paginas=15):
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
        return "❌ Erro: Chave GEMINI_API_KEY não configurada no Render.", "", "", "", ""
    if arquivo_pdf is None:
        return "Aguardando envio do arquivo PDF...", "", "", "", ""

    try:
        texto_pdf, total_paginas = extrair_texto_pdf(arquivo_pdf.name)
        
        if not texto_pdf.strip():
            return "❌ Erro: Não foi possível extrair texto legível deste PDF.", "", "", "", ""

        client = genai.Client(api_key=API_KEY)
        
        prompt = f"""
        Você é um catalogador sênior de biblioteca universitária (UFABC) especialista em MARC21 e Objetivos de Desenvolvimento Sustentável (ODS) da ONU.
        Analise o texto das primeiras páginas do trabalho acadêmico (Folha de Rosto, Folha de Aprovação, Resumos) e extraia os metadados.

        REGRAS RÍGIDAS DE FORMATAÇÃO MARC21 E ODS:
        1. AUTOR (100): Formato 'Sobrenome, Nome' em capitalização normal (Title Case, ex: 'Costa, Marcelo Rodrigues da').
        2. TÍTULO E SUBTÍTULO (245):
           - O subtítulo DEVE começar com letra MINÚSCULA (ex: 'proposta de uso do GPT...').
        3. RESUMOS (520): Extraia o Resumo em Português E o Abstract em Inglês completos.
        4. PALAVRAS-CHAVE (650): 
           - Extraia palavras-chave em PT e EN em CAIXA ALTA (MAIÚSCULAS).
        5. TIPO DE DOCUMENTO (990): Determine estritamente se é 'Dissertação' ou 'Tese'.
        6. ANÁLISE ODS (ONU): Identifique de 1 a 3 Objetivos de Desenvolvimento Sustentável (ODS da ONU) aos quais este trabalho se alinha, justificando brevemente.

        Responda ESTRITAMENTE em JSON com a seguinte estrutura:
        {{
          "autor_invertido": "Costa, Marcelo Rodrigues da",
          "titulo": "Aplicação das desigualdades das médias na resolução de problemas de geometria plana e espacial no ensino médio",
          "subtitulo": "proposta de uso do GPT como ferramenta de apoio pedagógico",
          "ano": "2025",
          "cidade": "Santo André",
          "instituicao": "Universidade Federal do ABC",
          "programa_completo": "Programa de Pós-Graduação em Mestrado Profissional em Matemática em Rede Nacional - PROFMAT",
          "programa_sigla": "PROFMAT",
          "grau_tipo": "Dissertação",
          "grau_especificacao": "Dissertação (mestrado)",
          "orientador_nome_completo": "Prof. Dr. Márcio Fabiano da Silva",
          "orientador_invertido": "Silva, Márcio Fabiano da",
          "membros_banca_invertidos": ["Guéron, Eduardo", "Lymberopoulos, Alexandre"],
          "resumo_pt": "Texto completo do resumo em português...",
          "resumo_en": "Texto completo do abstract em inglês...",
          "palavras_chave_pt": ["DESIGUALDADES DAS MÉDIAS", "GEOMETRIA", "RESOLUÇÃO DE PROBLEMAS", "GEORGE PÓLYA", "IA GENERATIVA", "GPT", "EDUCAÇÃO MATEMÁTICA", "OTIMIZAÇÃO"],
          "palavras_chave_en": ["MEAN INEQUALITIES", "GEOMETRY", "PROBLEM-SOLVING", "GENERATIVE AI", "MATHEMATICS EDUCATION", "OPTIMIZATION"],
          "tem_ilustracoes": true,
          "ods_recomendados": [
            "ODS 4 - Educação de Qualidade: Foco no aprimoramento do ensino de matemática e metodologias pedagógicas no Ensino Médio.",
            "ODS 9 - Indústria, Inovação e Infraestrutura: Aplicação de Inteligência Artificial Generativa (GPT) como inovação tecnológica na educação."
          ]
        }}

        Texto do PDF:
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
            raise Exception(f"Falha na resposta da API. Detalhe: {str(ultimo_erro)}")

        dados = json.loads(sanitizar_texto_utf8(response.text))

        autor = sanitizar_texto_utf8(dados.get("autor_invertido", ""))
        titulo = sanitizar_texto_utf8(dados.get("titulo", ""))
        subtitulo = sanitizar_texto_utf8(dados.get("subtitulo", ""))
        ano = sanitizar_texto_utf8(str(dados.get("ano", "2025")))
        cidade = sanitizar_texto_utf8(dados.get("cidade", "Santo André"))
        inst = sanitizar_texto_utf8(dados.get("instituicao", "Universidade Federal do ABC"))
        prog_completo = sanitizar_texto_utf8(dados.get("programa_completo", ""))
        prog_sigla = sanitizar_texto_utf8(dados.get("programa_sigla", ""))
        grau_tipo = sanitizar_texto_utf8(dados.get("grau_tipo", "Dissertação"))
        grau_spec = sanitizar_texto_utf8(dados.get("grau_especificacao", "Dissertação (mestrado)"))
        orientador_completo = sanitizar_texto_utf8(dados.get("orientador_nome_completo", ""))
        orientador_inv = sanitizar_texto_utf8(dados.get("orientador_invertido", ""))
        
        resumo_pt = sanitizar_texto_utf8(dados.get("resumo_pt", ""))
        resumo_en = sanitizar_texto_utf8(dados.get("resumo_en", ""))

        sub_abnt = f": {subtitulo}" if subtitulo else ""
        abnt = f"**{autor.upper()}**. *{titulo}*{sub_abnt}. {ano}. {total_paginas} f. {grau_spec} – {inst}, {cidade}, {ano}."

        linhas_marc = []

        if autor:
            linhas_marc.append(f"100 __ |a {autor}")

        sub_marc = f": |b {subtitulo}" if subtitulo else ""
        linhas_marc.append(f"245 _0 |a {titulo}{sub_marc}")
        linhas_marc.append(f"260 __ |c {ano}")

        il_str = " |b il." if dados.get("tem_ilustracoes", True) else ""
        linhas_marc.append(f"300 __ |a {total_paginas} f.{il_str}")

        if orientador_completo:
            linhas_marc.append(f"500 __ |a Orientador: {orientador_completo}")

        linhas_marc.append(f"502 __ |a {grau_spec} - {inst}, {prog_completo}, {cidade}, {ano}.")

        if resumo_pt:
            linhas_marc.append(f"520 __ |a {resumo_pt}")
        if resumo_en:
            linhas_marc.append(f"520 __ |a {resumo_en}")

        kw_unicas = []
        for kw in dados.get("palavras_chave_pt", []) + dados.get("palavras_chave_en", []):
            kw_f = sanitizar_texto_utf8(kw).strip('.').upper()
            if kw_f and kw_f not in kw_unicas:
                kw_unicas.append(kw_f)
                linhas_marc.append(f"650 __ |a {kw_f}")

        if prog_completo:
            prog_upper = prog_completo.upper()
            if "PROGRAMA DE PÓS-GRADUAÇÃO" in prog_upper:
                prog_tag = f"{prog_upper} - UFABC"
            else:
                prog_tag = f"PROGRAMA DE PÓS-GRADUAÇÃO EM {prog_upper} - UFABC"
            linhas_marc.append(f"650 __ |a {prog_tag}")

        if orientador_inv:
            linhas_marc.append(f"700 __ |a {orientador_inv} |e Orientador")

        for membro in dados.get("membros_banca_invertidos", []):
            m_f = sanitizar_texto_utf8(membro)
            if m_f and m_f != orientador_inv:
                linhas_marc.append(f"700 __ |a {m_f} |e Membro de Banca")

        linhas_marc.append(f"990 __ |a {grau_tipo}")

        marc_final = "\n".join(linhas_marc)

        ods_lista = dados.get("ods_recomendados", [])
        ods_texto = "\n".join([f"• {item}" for item in ods_lista]) if ods_lista else "Nenhum ODS específico identificado."

        return abnt, marc_final, resumo_pt, ods_texto, "✅ Registro MARC21 e ODS gerados com sucesso!"

    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}", "", "", "", ""

# --- CONSULTA EXTERNA DE LIVROS E FUSÃO MARC21 ---

def consultar_library_of_congress(isbn):
    """Consulta a Library of Congress via protocolo SRU (MARCXML)."""
    url = f"https://lx2.loc.gov/LCDB?operation=searchRetrieve&version=1.1&query=bath.isbn={isbn}&recordSchema=marcxml"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and "<marc:record" in resp.text:
            return resp.text
    except Exception:
        pass
    return None

def consultar_open_library(isbn):
    """Consulta a Open Library via API JSON."""
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.json():
            return json.dumps(resp.json())
    except Exception:
        pass
    return None

def processar_livro_externo(isbn):
    if not API_KEY:
        return "❌ Erro: Chave GEMINI_API_KEY não configurada.", "", "", ""
    if not isbn or len(isbn.strip()) < 10:
        return "Por favor, insira um ISBN válido com 10 ou 13 dígitos.", "", "", ""

    isbn_limpo = re.sub(r'[^0-9X]', '', isbn.upper())
    
    dados_loc = consultar_library_of_congress(isbn_limpo)
    dados_openlib = consultar_open_library(isbn_limpo)

    if not dados_loc and not dados_openlib:
        return f"❌ Nenhum registro encontrado para o ISBN {isbn_limpo} nas fontes externas.", "", "", ""

    prompt_fusao = f"""
    Você é um bibliotecário catalogador sênior especialista em MARC21, ABNT e ODS da ONU.
    Recebi dados bibliográficos de diferentes fontes para o livro ISBN {isbn_limpo}:

    FONTE 1 (Library of Congress - MARCXML):
    {dados_loc if dados_loc else "Nenhum dado retornado."}

    FONTE 2 (Open Library - JSON):
    {dados_openlib if dados_openlib else "Nenhum dado retornado."}

    TAREFA:
    Combine os dados dessas fontes de forma a criar o registro MARC21 MAIS COMPLETO E RICO possível.
    - Resolva conflitos mantendo os dados mais precisos.
    - Faça a união (merge) de assuntos (650), autores secundários (700), notas (500/520).
    - Identifique os ODS pertinentes ao tema da obra.

    Responda ESTRITAMENTE em JSON com a seguinte estrutura:
    {{
      "autor_invertido": "Sobrenome, Nome",
      "titulo": "Título do livro",
      "subtitulo": "subtítulo do livro",
      "ano": "2023",
      "cidade": "Local de publicação",
      "editora": "Nome da Editora",
      "paginas": "350",
      "resumo_pt": "Descrição ou nota de resumo...",
      "palavras_chave_pt": ["ASSUNTO 1", "ASSUNTO 2"],
      "ods_recomendados": ["ODS 4 - Educação de Qualidade: Justificativa..."]
    }}
    """

    client = genai.Client(api_key=API_KEY)
    response = None
    ultimo_erro = None

    for modelo in MODELOS_DISPONIVEIS:
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=prompt_fusao,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            if response and response.text:
                break
        except Exception as err:
            ultimo_erro = err
            time.sleep(1)
            continue

    if not response or not response.text:
        return f"❌ Falha ao processar a fusão dos dados: {str(ultimo_erro)}", "", "", ""

    try:
        dados = json.loads(sanitizar_texto_utf8(response.text))
        
        autor = sanitizar_texto_utf8(dados.get("autor_invertido", ""))
        titulo = sanitizar_texto_utf8(dados.get("titulo", ""))
        subtitulo = sanitizar_texto_utf8(dados.get("subtitulo", ""))
        ano = sanitizar_texto_utf8(str(dados.get("ano", "")))
        cidade = sanitizar_texto_utf8(dados.get("cidade", ""))
        editora = sanitizar_texto_utf8(dados.get("editora", ""))
        paginas = sanitizar_texto_utf8(str(dados.get("paginas", "")))
        resumo_pt = sanitizar_texto_utf8(dados.get("resumo_pt", ""))

        sub_abnt = f": {subtitulo}" if subtitulo else ""
        ed_str = f" {editora}," if editora else ""
        cid_str = f"{cidade}: " if cidade else ""
        abnt = f"**{autor.upper()}**. *{titulo}*{sub_abnt}. {cid_str}{ed_str} {ano}. {paginas} p."

        linhas_marc = []
        if autor:
            linhas_marc.append(f"100 __ |a {autor}")
        sub_marc = f": |b {subtitulo}" if subtitulo else ""
        linhas_marc.append(f"245 _0 |a {titulo}{sub_marc}")
        loc_ed_str = f"{cidade} |b {editora}," if editora else ""
        linhas_marc.append(f"260 __ |a {loc_ed_str} |c {ano}")
        if paginas:
            linhas_marc.append(f"300 __ |a {paginas} p.")
        if resumo_pt:
            linhas_marc.append(f"520 __ |a {resumo_pt}")
        
        for kw in dados.get("palavras_chave_pt", []):
            kw_f = sanitizar_texto_utf8(kw).strip('.').upper()
            if kw_f:
                linhas_marc.append(f"650 __ |a {kw_f}")

        marc_final = "\n".join(linhas_marc)
        ods_lista = dados.get("ods_recomendados", [])
        ods_texto = "\n".join([f"• {item}" for item in ods_lista]) if ods_lista else "Nenhum ODS específico identificado."

        return abnt, marc_final, resumo_pt, ods_texto
    except Exception as e:
        return f"❌ Erro ao formatar dados do livro: {str(e)}", "", "", ""

# --- INTERFACE GRADIO UNIFICADA ---

with gr.Blocks(title="Catalogador Multimídia - UFABC") as demo:
    gr.Markdown("# 📚 Sistema de Catalogação Automática e ODS (UFABC)")
    
    with gr.Tabs():
        with gr.Tab("📄 Teses e Dissertações (PDF)"):
            with gr.Row():
                file_input = gr.File(label="Arraste o PDF da Tese/Dissertação aqui", file_types=[".pdf"], type="filepath")
            
            btn_processar_tese = gr.Button("⚙️ Processar Tese/Dissertação", variant="primary")
            status_tese = gr.Markdown()
            
            with gr.Column():
                abnt_tese = gr.Markdown(label="Referência ABNT")
                marc_tese = gr.Code(label="Registro MARC21 (Padrão Sophia/UFABC)", lines=16)
                ods_tese = gr.Textbox(label="🎯 Enquadramento nos ODS da ONU", lines=4)
                resumo_tese = gr.Textbox(label="Resumo Extraído", lines=5)

            btn_processar_tese.click(
                fn=processar_tese,
                inputs=[file_input],
                outputs=[abnt_tese, marc_tese, resumo_tese, ods_tese, status_tese]
            )

        with gr.Tab("📖 Livros (Busca Externa SRU / APIs)"):
            with gr.Row():
                isbn_input = gr.Textbox(label="Informe o ISBN do Livro", placeholder="Ex: 9788535902778")
            
            btn_processar_livro = gr.Button("🔍 Buscar e Unificar Registros MARC21", variant="primary")
            
            with gr.Column():
                abnt_livro = gr.Markdown(label="Referência ABNT")
                marc_livro = gr.Code(label="Registro MARC21 Mesclado (Mais Completo)", lines=16)
                ods_livro = gr.Textbox(label="🎯 Enquadramento nos ODS da ONU", lines=4)
                resumo_livro = gr.Textbox(label="Resumo / Notas", lines=5)

            btn_processar_livro.click(
                fn=processar_livro_externo,
                inputs=[isbn_input],
                outputs=[abnt_livro, marc_livro, resumo_livro, ods_livro]
            )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
