from __future__ import annotations

from flask import Flask, jsonify, render_template, request
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "Ты — поддерживающий чат-бот платформы психологической поддержки жертв "
    "кибер-буллинга. Говори спокойным, нейтральным и сочувственным тоном. "
    "Не ставь диагнозы, не назначай лечение и не давай медицинских советов. "
    "Помогай пользователю осмыслить ситуацию, предлагай безопасные шаги "
    "самопомощи и способы получить поддержку. Если пользователь сообщает "
    "о самоповреждении, суицидальных мыслях, угрозах или насилии, мягко "
    "рекомендуй обратиться к специалисту, кризисным службам или доверенному "
    "человеку. Никогда не поощряй вред себе или другим."
)


def build_prompt(messages: list[dict[str, str]]) -> str:
    """Собирает диалог в единый prompt для Ollama."""
    parts = [f"Системные инструкции:\n{SYSTEM_PROMPT}\n"]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        role_label = "Пользователь" if role == "user" else "Ассистент"
        parts.append(f"{role_label}: {content}")
    parts.append("Ассистент:")
    return "\n".join(parts)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat() -> tuple[str, int] | tuple[dict[str, str], int]:
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Сообщения не получены."}), 400

    prompt = build_prompt(messages)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return (
            jsonify({
                "error": "Не удалось связаться с Ollama. Убедитесь, что сервис запущен.",
                "details": str(exc),
            }),
            502,
        )
    except ValueError:
        return jsonify({"error": "Некорректный ответ от Ollama."}), 502

    reply = result.get("response", "").strip()
    if not reply:
        return jsonify({"error": "Пустой ответ от модели."}), 502

    return jsonify({"reply": reply}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
