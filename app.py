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
        return "❌ Erro: Chave GEMINI_API_KEY não configurada no Render.", "", "", ""
    if arquivo_pdf is None:
        return "Aguardando envio do arquivo PDF...", "", "", ""

    try:
        texto_pdf, total_paginas = extrair_texto_pdf(arquivo_pdf.name)
        
        if not texto_pdf.strip():
            return "❌ Erro: Não foi possível extrair texto legível deste PDF.", "", "", ""

        client = genai.Client(api_key=API_KEY)
        
        prompt = f"""
        Você é um catalogador sênior de biblioteca universitária (UFABC) especialista em MARC21.
        Analise o texto das primeiras páginas do trabalho acadêmico (incluindo Folha de Rosto, Folha de Aprovação, Resumo e Abstract) e extraia os metadados.

        REGRAS RÍGIDAS DE FORMATAÇÃO MARC21 UFABC:
        1. AUTOR (100): Formato 'Sobrenome, Nome' em capitalização normal (Title Case, EX: 'Costa, Marcelo Rodrigues da'). NÃO use caixa alta no sobrenome.
        2. TÍTULO E SUBTÍTULO (245):
           - Subtítulo DEVE iniciar com letra MINÚSCULA (ex: 'proposta de uso do GPT...').
        3. RESUMOS (520): Extraia o Resumo em Português E o Abstract em Inglês completos, se presentes.
        4. PALAVRAS-CHAVE (650): 
           - Extraia TODAS as palavras-chave em Português E em Inglês.
           - TODAS as palavras-chave devem estar estritamente em CAIXA ALTA (MAIÚSCULAS).
        5. ORIENTADOR E BANCA (700):
           - Identifique o orientador e os membros da banca examinadora (na Folha de Aprovação) no formato 'Sobrenome, Nome'.
        6. ILUSTRAÇÕES (300): Verifique se há menção/presença de figuras, gráficos, tabelas ou ilustrações no trabalho.

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
          "tem_ilustracoes": true
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

        # Variáveis sanitizadas
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

        # 1. Referência ABNT
        sub_abnt = f": {subtitulo}" if subtitulo else ""
        abnt = f"**{autor.upper()}**. *{titulo}*{sub_abnt}. {ano}. {total_paginas} f. {grau_spec} – {inst}, {cidade}, {ano}."

        # 2. Montagem dos Campos MARC21 Exatos
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

        # Campo 502 Padronizado
        linhas_marc.append(f"502 __ |a {grau_spec} - {inst}, {prog_completo}, {cidade}, {ano}.")

        # Campos 520 (Resumos PT e EN)
        if resumo_pt:
            linhas_marc.append(f"520 __ |a {resumo_pt}")
        if resumo_en:
            linhas_marc.append(f"520 __ |a {resumo_en}")

        # Campos 650 (Palavras-chave em CAIXA ALTA - PT e EN)
        for kw in dados.get("palavras_chave_pt", []):
            kw_f = sanitizar_texto_utf8(kw).strip('.').upper()
            if kw_f:
                linhas_marc.append(f"650 __ |a {kw_f}")

        for kw in dados.get("palavras_chave_en", []):
            kw_f = sanitizar_texto_utf8(kw).strip('.').upper()
            if kw_f:
                linhas_marc.append(f"650 __ |a {kw_f}")

        # Campo 650 do Programa UFABC
        if prog_completo:
            prog_tag = f"PROGRAMA DE PÓS-GRADUAÇÃO EM {prog_completo.upper()}"
            if prog_sigla and prog_sigla.upper() not in prog_tag:
                prog_tag += f" - {prog_sigla.upper()}"
            prog_tag += " - UFABC"
            linhas_marc.append(f"650 __ |a {prog_tag}")

        # Campos 700 (Orientador e Banca)
        if orientador_inv:
            linhas_marc.append(f"700 __ |a {orientador_inv} |e Orientador")

        for membro in dados.get("membros_banca_invertidos", []):
            m_f = sanitizar_texto_utf8(membro)
            if m_f and m_f != orientador_inv:
                linhas_marc.append(f"700 __ |a {m_f} |e Membro de Banca")

        # Campo 990 Local
        linhas_marc.append(f"990 __ |a {grau_tipo}")

        marc_final = "\n".join(linhas_marc)

        return abnt, marc_final, resumo_pt, "✅ Registro MARC21 gerado no padrão exato da Biblioteca UFABC!"

    except Exception as e:
        return f"❌ Erro ao processar: {str(e)}", "", "", ""

# Interface Gradio
with gr.Blocks(title="Gerador MARC21 - UFABC") as demo:
    gr.Markdown("# 📚 Gerador Automático de MARC21 e Referência (UFABC)")
    
    with gr.Row():
        file_input = gr.File(label="Arraste o PDF da Tese/Dissertação aqui", file_types=[".pdf"], type="filepath")
    
    btn_processar = gr.Button("⚙️ Processar Documento", variant="primary")
    status_output = gr.Markdown()
    
    with gr.Column():
        abnt_output = gr.Markdown(label="Referência ABNT")
        marc_output = gr.Code(label="Registro MARC21 (Padrão UFABC)", language=None, lines=18)
        resumo_output = gr.Textbox(label="Resumo Extraído", lines=5)

    btn_processar.click(
        fn=processar_tese,
        inputs=[file_input],
        outputs=[abnt_output, marc_output, resumo_output, status_output]
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
