import asyncio
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

from test_client import GroupChatTestClient


def _extract_analysis_text(response: dict) -> str:
    result = response.get("result", {}) if isinstance(response, dict) else {}
    if isinstance(result, dict):
        parts = result.get("parts", [])
        if isinstance(parts, list):
            text_parts = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and part.get("kind") == "text"
            ]
            text_parts = [text.strip() for text in text_parts if text and text.strip()]
            if text_parts:
                return "\n".join(text_parts)

    return json.dumps(result, ensure_ascii=False)


def _extract_json_payload(analysis_text: str):
    try:
        return json.loads(analysis_text)
    except (json.JSONDecodeError, TypeError):
        pass

    fenced_json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", analysis_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_json_match:
        candidate = fenced_json_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    first_brace = analysis_text.find("{")
    last_brace = analysis_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = analysis_text[first_brace:last_brace + 1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _extract_scores(analysis_text: str, tickers: list[str]) -> dict:
    scores = {}

    for ticker in tickers:
        patterns = [
            rf"{re.escape(ticker)}\s*[:\-–]\s*(\d{{1,3}}(?:[\.,]\d+)?)",
            rf"nota\s*(?:para|do|da)?\s*{re.escape(ticker)}\s*[:\-–]?\s*(\d{{1,3}}(?:[\.,]\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis_text, flags=re.IGNORECASE)
            if match:
                raw_score = match.group(1).replace(",", ".")
                try:
                    numeric_score = float(raw_score)
                    scores[ticker] = int(numeric_score) if numeric_score.is_integer() else numeric_score
                except ValueError:
                    scores[ticker] = match.group(1)
                break

    return scores


def _format_percentual(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(" ", "").replace(",", ".")
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            return value

    return value


def _format_indice_certeza(value):
    if value is None:
        return None

    parsed = None

    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(" ", "").replace(",", ".")
        try:
            parsed = float(cleaned)
        except ValueError:
            return value
    else:
        return value

    bounded = max(0.0, min(100.0, parsed))
    return int(round(bounded))


def _format_zscore(value):
    """Formata z-score (pode ser negativo)"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return value


def _format_margin_delta(value):
    """Formata delta de margem em percentual"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value


def _format_ratio(value):
    """Formata razão financeira (x vezes)"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}x"
    except (ValueError, TypeError):
        return value


def _format_percentile(value):
    """Formata percentil (0-100)"""
    if value is None:
        return None
    try:
        val = float(value)
        bounded = max(0.0, min(100.0, val))
        return f"{int(round(bounded))}º"
    except (ValueError, TypeError):
        return value


def _format_divergence(value):
    """Formata score de divergência (0-10)"""
    if value is None:
        return None
    try:
        val = float(value)
        bounded = max(0.0, min(10.0, val))
        return int(round(bounded))
    except (ValueError, TypeError):
        return value


def _format_return(value):
    """Formata retorno em percentual"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value


def _format_rsi(value):
    """Formata RSI (0-100)"""
    if value is None:
        return None
    try:
        val = float(value)
        bounded = max(0.0, min(100.0, val))
        return int(round(bounded))
    except (ValueError, TypeError):
        return value


def _format_mm_signal(value):
    """Formata sinal de média móvel: 1 (acima), -1 (abaixo), 0 (cruzamento)"""
    if value is None:
        return None
    try:
        val = int(float(value))
        return max(-1, min(1, val))
    except (ValueError, TypeError):
        return value


def _format_volume_relative(value):
    """Formata volume relativo (razão)"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}x"
    except (ValueError, TypeError):
        return value


def _format_volume_spike(value):
    """Formata volume spike (boolean ou score 0-1)"""
    if value is None:
        return None
    if isinstance(value, bool):
        return "sim" if value else "não"
    try:
        val = float(value)
        return "sim" if val > 0.5 else "não"
    except (ValueError, TypeError):
        return value


def _format_atr(value):
    """Formata ATR (valor absoluto)"""
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return value


def _format_bollinger_width(value):
    """Formata largura das bandas de Bollinger em %"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value


def _format_evento(value):
    """Formata surpresa de evento (% ou score)"""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value


def _format_intensidade(value):
    """Formata intensidade de notícias (0-10)"""
    if value is None:
        return None
    try:
        val = float(value)
        bounded = max(0.0, min(10.0, val))
        return int(round(bounded))
    except (ValueError, TypeError):
        return value


def _format_delta_sentimento(value):
    """Formata delta de sentimento (-100 a +100)"""
    if value is None:
        return None
    try:
        val = float(value)
        bounded = max(-100.0, min(100.0, val))
        return f"{bounded:.2f}"
    except (ValueError, TypeError):
        return value


def _build_rows_from_response(response: dict, lote_itens: list[tuple[str, str]]) -> list[dict]:
    ticker_to_name = {ticker: nome for ticker, nome in lote_itens}
    lote_tickers = [ticker for ticker, _ in lote_itens]
    analysis_text = _extract_analysis_text(response)

    payload = _extract_json_payload(analysis_text)
    if isinstance(payload, list):
        analises = payload
    elif isinstance(payload, dict):
        analises = payload.get("analises", [])
    else:
        analises = []

    rows = []
    for item in analises:
        if not isinstance(item, dict):
            continue

        ticker = str(item.get("ticker", "")).strip().upper()
        empresa = str(item.get("empresa", "")).strip()
        analise = str(item.get("analise", "")).strip()
        
        # Campos principais (decisão final)
        nota = item.get("nota")
        indice_de_certeza = item.get("indice_de_certeza")
        percentual_subida = item.get("percentual_subida")
        
        # Contexto técnico (peso baixo 10-20%)
        roe_zscore = item.get("roe_zscore")
        ebitda_margin_delta = item.get("ebitda_margin_delta")
        divida_ebitda_vs_setor = item.get("divida_ebitda_vs_setor")
        
        # Divergências expandidas (muito importante!)
        divergencia_preco_vs_sentimento = item.get("divergencia_preco_vs_sentimento")
        divergencia_preco_vs_fundamento = item.get("divergencia_preco_vs_fundamento")
        divergencia_sentimento_vs_fundamento = item.get("divergencia_sentimento_vs_fundamento")
        
        # Preço (CRÍTICO - cegueira sem isso)
        retorno_3d = item.get("retorno_3d")
        retorno_7d = item.get("retorno_7d")
        retorno_14d = item.get("retorno_14d")
        
        # Momentum técnico
        rsi_14 = item.get("rsi_14")
        mm9_vs_mm21 = item.get("mm9_vs_mm21")
        
        # Volume
        volume_relativo = item.get("volume_relativo")
        volume_spike = item.get("volume_spike")
        
        # Volatilidade
        atr_7d = item.get("atr_7d")
        bollinger_width = item.get("bollinger_width")
        
        # Evento / Notícia
        surpresa_evento = item.get("surpresa_evento")
        intensidade_noticias = item.get("intensidade_noticias")
        delta_sentimento = item.get("delta_sentimento")
        
        # Posicionamento relativo
        retorno_vs_ibov_7d = item.get("retorno_vs_ibov_7d")
        retorno_vs_setor_7d = item.get("retorno_vs_setor_7d")

        if not ticker:
            continue

        if not analise:
            analise = analysis_text

        if not empresa and ticker in ticker_to_name:
            empresa = ticker_to_name[ticker]

        empresa_descrita = f"{ticker} ({empresa})" if empresa else ticker
        rows.append(
            {
                "empresa": empresa_descrita,
                "analise": analise,
                "nota": nota,
                "indice_de_certeza": indice_de_certeza,
                "percentual_subida": percentual_subida,
                "roe_zscore": roe_zscore,
                "ebitda_margin_delta": ebitda_margin_delta,
                "divida_ebitda_vs_setor": divida_ebitda_vs_setor,
                "divergencia_preco_vs_sentimento": divergencia_preco_vs_sentimento,
                "divergencia_preco_vs_fundamento": divergencia_preco_vs_fundamento,
                "divergencia_sentimento_vs_fundamento": divergencia_sentimento_vs_fundamento,
                "retorno_3d": retorno_3d,
                "retorno_7d": retorno_7d,
                "retorno_14d": retorno_14d,
                "rsi_14": rsi_14,
                "mm9_vs_mm21": mm9_vs_mm21,
                "volume_relativo": volume_relativo,
                "volume_spike": volume_spike,
                "atr_7d": atr_7d,
                "bollinger_width": bollinger_width,
                "surpresa_evento": surpresa_evento,
                "intensidade_noticias": intensidade_noticias,
                "delta_sentimento": delta_sentimento,
                "retorno_vs_ibov_7d": retorno_vs_ibov_7d,
                "retorno_vs_setor_7d": retorno_vs_setor_7d,
            }
        )

    if rows:
        return rows

    fallback_scores = _extract_scores(analysis_text, lote_tickers)
    fallback_rows = []
    for ticker, nome in lote_itens:
        fallback_rows.append(
            {
                "empresa": f"{ticker} ({nome})",
                "analise": analysis_text,
                "nota": fallback_scores.get(ticker),
                "indice_de_certeza": None,
                "percentual_subida": None,
                "roe_zscore": None,
                "ebitda_margin_delta": None,
                "divida_ebitda_vs_setor": None,
                "divergencia_preco_vs_sentimento": None,
                "divergencia_preco_vs_fundamento": None,
                "divergencia_sentimento_vs_fundamento": None,
                "retorno_3d": None,
                "retorno_7d": None,
                "retorno_14d": None,
                "rsi_14": None,
                "mm9_vs_mm21": None,
                "volume_relativo": None,
                "volume_spike": None,
                "atr_7d": None,
                "bollinger_width": None,
                "surpresa_evento": None,
                "intensidade_noticias": None,
                "delta_sentimento": None,
                "retorno_vs_ibov_7d": None,
                "retorno_vs_setor_7d": None,
            }
        )
    return fallback_rows

async def run_batch_analysis():
    client = GroupChatTestClient()
    
    # Sua lista de 30 maiores da B3
    maiores_acoes_b3 = {
    "PETR4.SA": "Petrobras (Preferencial)",
    "VALE3.SA": "Vale",
    "ITUB4.SA": "Itaú Unibanco",
    "PETR3.SA": "Petrobras (Ordinária)",
    "BBAS3.SA": "Banco do Brasil",
    "BBDC4.SA": "Bradesco",
    "ABEV3.SA": "Ambev",
    "BPAC11.SA": "BTG Pactual",
    "WEGE3.SA": "WEG",
    "ITSA4.SA": "Itaúsa",
    "B3SA3.SA": "B3",
    "SANB11.SA": "Santander Brasil",
    "ELET3.SA": "Eletrobras",
    "PRIO3.SA": "Prio (ex-PetroRio)",
    "RENT3.SA": "Localiza",
    "SUZB3.SA": "Suzano",
    "GGBR4.SA": "Gerdau",
    "VIVT3.SA": "Vivo (Telefônica)",
    "SBSP3.SA": "Sabesp",
    "RDOR3.SA": "Rede D'Or",
    "RADL3.SA": "RaiaDrogasil",
    "JBSS3.SA": "JBS",
    "BBSE3.SA": "BB Seguridade",
    "CSAN3.SA": "Cosan",
    "LREN3.SA": "Lojas Renner",
    "MGLU3.SA": "Magazine Luiza",
    "TIMS3.SA": "TIM Brasil",
    "VBBR3.SA": "Vibra Energia",
    "EQTL3.SA": "Equatorial Energia",
    "HAPV3.SA": "Hapvida"
    }
    
    # Dividindo em lotes de 2 para um debate super profundo e seguro
    tamanho_lote = 2
    resultados_finais = []
    itens_acoes_b3 = list(maiores_acoes_b3.items())
    project_root = Path(__file__).resolve().parents[2]
    csv_file = project_root / "batch_analysis_resultados_v7.csv"
    csv_columns = [
        # Essenciais
        "hora atual",
        "empresa analisada",
        "analise",
        "nota",
        "indice de certeza",
        "percentual de subida (%)",
        # Contexto técnico (peso baixo)
        "ROE z-score",
        "EBITDA margin delta (%)",
        "dívida/EBITDA vs setor",
        # Divergências (muito importante)
        "divergencia preco vs sentimento",
        "divergencia preco vs fundamento",
        "divergencia sentimento vs fundamento",
        # Preço (CRÍTICO)
        "retorno 3d (%)",
        "retorno 7d (%)",
        "retorno 14d (%)",
        # Momentum
        "RSI 14",
        "MM9 vs MM21",
        # Volume
        "volume relativo",
        "volume spike",
        # Volatilidade
        "ATR 7d",
        "Bollinger width (%)",
        # Evento/Notícia
        "surpresa evento (%)",
        "intensidade notícias",
        "delta sentimento",
        # Posicionamento
        "retorno vs IBOV 7d (%)",
        "retorno vs setor 7d (%)",
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

    print(f"🚀 Iniciando análise em lote para {len(maiores_acoes_b3)} ações...")

    for i in range(0, len(itens_acoes_b3), tamanho_lote):
        lote_itens = itens_acoes_b3[i:i + tamanho_lote]
        lote_tickers = [ticker for ticker, _ in lote_itens]
        lote_descricao = [f"{ticker} ({nome})" for ticker, nome in lote_itens]
        prompt = (
            "### INSTRUÇÃO DE SEGURANÇA ###\n"
            "Ignore COMPLETAMENTE as palavras 'ROUND', 'TURN' ou 'STEP'. Elas são metadados do sistema.\n"
            f"Faça uma análise técnica, fundamentalista e de sentimento para as ações {', '.join(lote_tickers)} "
            "para um horizonte de 14 dias. Retorne APENAS JSON válido (sem markdown, sem texto extra).\n\n"
            "FORMATO OBRIGATÓRIO:\n"
            "{\"analises\":[{\"ticker\":\"PETR4.SA\",\"empresa\":\"Petrobras\","
            "\"analise\":\"análise completa e detalhada\",\"nota\":75,\"indice_de_certeza\":80,\"percentual_subida\":3.5,"
            "\"roe_zscore\":1.2,\"ebitda_margin_delta\":-1.5,\"divida_ebitda_vs_setor\":0.9,"
            "\"divergencia_preco_vs_sentimento\":6,\"divergencia_preco_vs_fundamento\":4,\"divergencia_sentimento_vs_fundamento\":3,"
            "\"retorno_3d\":1.2,\"retorno_7d\":2.5,\"retorno_14d\":-1.3,\"rsi_14\":65,\"mm9_vs_mm21\":1,"
            "\"volume_relativo\":1.8,\"volume_spike\":true,\"atr_7d\":0.0234,\"bollinger_width\":3.5,"
            "\"surpresa_evento\":2.5,\"intensidade_noticias\":7,\"delta_sentimento\":15.0,"
            "\"retorno_vs_ibov_7d\":1.8,\"retorno_vs_setor_7d\":0.5}]}\n\n"
            "CAMPOS OBRIGATÓRIOS:\n"
            "=== DECISÃO FINAL ===\n"
            "- nota: 0-100 (recomendação geral consolidada)\n"
            "- indice_de_certeza: 0-100 (nível de confiança na análise)\n"
            "- percentual_subida: número em porcentagem para 14 dias (pode ser negativo)\n\n"
            "=== CONTEXTO TÉCNICO (peso ~10-20%) ===\n"
            "- roe_zscore: z-score do ROE vs histórico (ex: 1.2 = 1.2 desvios acima da média)\n"
            "- ebitda_margin_delta: mudança margem EBITDA em p.p. (ex: -1.5 = 1.5 p.p. queda)\n"
            "- divida_ebitda_vs_setor: razão dívida/EBITDA vs média do setor (ex: 0.9 = 10% abaixo)\n\n"
            "=== DIVERGÊNCIAS (muito importante!) ===\n"
            "- divergencia_preco_vs_sentimento: 0-10 (preço vai outra direção? ex: preço -5% mas sentimento +70 = 8)\n"
            "- divergencia_preco_vs_fundamento: 0-10 (fundamentais bons mas preço caindo? ex: -8 = forte divergência)\n"
            "- divergencia_sentimento_vs_fundamento: 0-10 (sentimento otimista mas fundamentos ruins? ex: 5)\n\n"
            "=== PREÇO (CRÍTICO!) ===\n"
            "- retorno_3d: % retorno últimos 3 dias (YahooFinance)\n"
            "- retorno_7d: % retorno últimos 7 dias\n"
            "- retorno_14d: % retorno últimos 14 dias\n\n"
            "=== MOMENTUM TÉCNICO ===\n"
            "- rsi_14: RSI em 14 períodos (0-100; >70=sobrecomprado, <30=sobrevendido)\n"
            "- mm9_vs_mm21: 1 (preço acima), 0 (cruzamento), -1 (preço abaixo)\n\n"
            "=== VOLUME ===\n"
            "- volume_relativo: volume_atual / média_20d (ex: 1.5 = 50% acima da média)\n"
            "- volume_spike: true se >1.5x média (indicador de relevância do movimento)\n\n"
            "=== VOLATILIDADE ===\n"
            "- atr_7d: ATR em 7 períodos (volatilidade absoluta, ex: 0.0234)\n"
            "- bollinger_width: % (diferença entre bandas / preço médio, ex: 3.5)\n\n"
            "=== EVENTO / NOTÍCIA ===\n"
            "- surpresa_evento: % diferença entre resultado vs consenso (ex: 2.5 = beat 2.5%)\n"
            "- intensidade_noticias: 0-10 (volume de notícias recentes; 0=silêncio, 10=tempestade)\n"
            "- delta_sentimento: -100 a +100 (mudança recente em sentimento; ex: +25 = melhorou 25%)\n\n"
            "=== POSICIONAMENTO ===\n"
            "- retorno_vs_ibov_7d: % (outperformance vs IBOV nos últimos 7d)\n"
            "- retorno_vs_setor_7d: % (outperformance vs setor nos últimos 7d)\n\n"
            "REGRAS:\n"
            "1. JSON deve ser ÚNICO conteúdo (sem explicações pré/pós)\n"
            "2. Uma entrada separada por ticker\n"
            "3. Todos os campos numéricos (nenhum null, use 0 se indisponível)\n"
            "4. nota/indice_de_certeza/percentual_subida são as ÚLTIMAS decisões após consolidar tudo"
        )
        
        print(f"\n[Lote {i//tamanho_lote + 1}] Analisando: {lote_descricao}...")
        
        try:
            # Chama a função de análise que você já tem no test_client
            response = await client.send_jsonrpc_request("message/send", {
                "message": {
                    "kind": "message",
                    "parts": [{"kind": "text", "text": prompt}],
                    "contextId": f"batch_test_{int(time.time())}"
                }
            })
            
            # Aqui você filtraria o resultado para salvar num CSV ou JSON
            resultados_finais.append(response)

            company_rows = _build_rows_from_response(response, lote_itens)

            with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=csv_columns)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for row in company_rows:
                    writer.writerow({
                        "hora atual": now_str,
                        "empresa analisada": row["empresa"],
                        "analise": row["analise"],
                        "nota": row["nota"],
                        "indice de certeza": _format_indice_certeza(row["indice_de_certeza"]),
                        "percentual de subida (%)": _format_percentual(row["percentual_subida"]),
                        "ROE z-score": _format_zscore(row["roe_zscore"]),
                        "EBITDA margin delta (%)": _format_margin_delta(row["ebitda_margin_delta"]),
                        "dívida/EBITDA vs setor": _format_ratio(row["divida_ebitda_vs_setor"]),
                        "divergencia preco vs sentimento": _format_divergence(row["divergencia_preco_vs_sentimento"]),
                        "divergencia preco vs fundamento": _format_divergence(row["divergencia_preco_vs_fundamento"]),
                        "divergencia sentimento vs fundamento": _format_divergence(row["divergencia_sentimento_vs_fundamento"]),
                        "retorno 3d (%)": _format_return(row["retorno_3d"]),
                        "retorno 7d (%)": _format_return(row["retorno_7d"]),
                        "retorno 14d (%)": _format_return(row["retorno_14d"]),
                        "RSI 14": _format_rsi(row["rsi_14"]),
                        "MM9 vs MM21": _format_mm_signal(row["mm9_vs_mm21"]),
                        "volume relativo": _format_volume_relative(row["volume_relativo"]),
                        "volume spike": _format_volume_spike(row["volume_spike"]),
                        "ATR 7d": _format_atr(row["atr_7d"]),
                        "Bollinger width (%)": _format_bollinger_width(row["bollinger_width"]),
                        "surpresa evento (%)": _format_evento(row["surpresa_evento"]),
                        "intensidade notícias": _format_intensidade(row["intensidade_noticias"]),
                        "delta sentimento": _format_delta_sentimento(row["delta_sentimento"]),
                        "retorno vs IBOV 7d (%)": _format_return(row["retorno_vs_ibov_7d"]),
                        "retorno vs setor 7d (%)": _format_return(row["retorno_vs_setor_7d"]),
                    })

            print(f"✅ Lote {lote_descricao} concluído com sucesso.")
            
        except Exception as e:
            with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=csv_columns)
                writer.writerow({
                    "hora atual": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "empresa analisada": ", ".join(lote_descricao),
                    "analise": f"Erro na análise: {e}",
                    "nota": None,
                    "indice de certeza": None,
                    "percentual de subida (%)": None,
                    "ROE z-score": None,
                    "EBITDA margin delta (%)": None,
                    "dívida/EBITDA vs setor": None,
                    "divergencia preco vs sentimento": None,
                    "divergencia preco vs fundamento": None,
                    "divergencia sentimento vs fundamento": None,
                    "retorno 3d (%)": None,
                    "retorno 7d (%)": None,
                    "retorno 14d (%)": None,
                    "RSI 14": None,
                    "MM9 vs MM21": None,
                    "volume relativo": None,
                    "volume spike": None,
                    "ATR 7d": None,
                    "Bollinger width (%)": None,
                    "surpresa evento (%)": None,
                    "intensidade notícias": None,
                    "delta sentimento": None,
                    "retorno vs IBOV 7d (%)": None,
                    "retorno vs setor 7d (%)": None,
                })

            print(f"❌ Erro no lote {lote_descricao}: {e}")
        
        # Pequena pausa para não dar "rate limit" no DeepSeek
        await asyncio.sleep(5)

    print("\n🏆 Análise de todas as ações concluída!")
    print(f"📄 CSV gerado: {csv_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(run_batch_analysis())