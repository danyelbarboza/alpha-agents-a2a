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
        nota = item.get("nota")
        indice_de_certeza = item.get("indice_de_certeza")
        percentual_subida = item.get("percentual_subida")

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
    csv_file = project_root / "batch_analysis_resultados_v6.csv"
    csv_columns = [
        "hora atual",
        "empresa analisada",
        "analise",
        "nota",
        "indice de certeza",
        "percentual de subida (%)",
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
            f"Faça uma análise fundamentalista e de sentimento para as ações {', '.join(lote_tickers)} "
            "para um horizonte de 14 dias. Retorne APENAS JSON válido (sem markdown, sem texto extra, mas com uma analise extremamente detalhada), "
            "no formato: {\"analises\":[{\"ticker\":\"NOVA3.SA\",\"empresa\":\"Empresa Fictícia Alpha\","
            "\"analise\":\"texto da análise da empresa\",\"nota\":85,\"indice_de_certeza\":78,\"percentual_subida\":4.2}]}. "
            "A nota deve ser de 0 a 100 e deve haver uma entrada separada para cada ticker solicitado. "
            "O campo indice_de_certeza deve ser um número de 0 a 100. "
            "O campo percentual_subida deve ser um número em porcentagem estimada para o fim do período (pode ser negativo). "
            "Certifique-se de que o JSON seja o único conteúdo retornado, sem explicações ou texto adicional. Se um agente falar, indique no campo da analise."
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
                })

            print(f"❌ Erro no lote {lote_descricao}: {e}")
        
        # Pequena pausa para não dar "rate limit" no DeepSeek
        await asyncio.sleep(5)

    print("\n🏆 Análise de todas as ações concluída!")
    print(f"📄 CSV gerado: {csv_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(run_batch_analysis())