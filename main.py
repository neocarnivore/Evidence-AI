import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import discord
from discord.ext import commands
from openai import AsyncOpenAI


# --------------------------------------------------
# 環境変数
# --------------------------------------------------


def require_environment_variable(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"環境変数 {name} が設定されていません。"
            "RailwayのVariablesで設定してください。"
        )
    return value


DISCORD_TOKEN: Final[str] = require_environment_variable("DISCORD_TOKEN")
OPENAI_API_KEY: Final[str] = require_environment_variable("OPENAI_API_KEY")
OPENAI_MODEL: Final[str] = os.getenv("OPENAI_MODEL", "gpt-5").strip()

# 任意。設定していなければWeb検索だけを使う
OPENAI_VECTOR_STORE_ID: Final[str] = os.getenv(
    "OPENAI_VECTOR_STORE_ID", ""
).strip()

BODY_CHAR_LIMIT: Final[int] = 300
MAX_LINKS: Final[int] = 2
MAX_OUTPUT_TOKENS: Final[int] = int(
    os.getenv("MAIN_MAX_OUTPUT_TOKENS", "500")
)
STREAM_EDIT_INTERVAL_SECONDS: Final[float] = float(
    os.getenv("STREAM_EDIT_INTERVAL_SECONDS", "1.5")
)


# --------------------------------------------------
# ログ
# --------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evidence-ai")


# --------------------------------------------------
# Discord / OpenAI
# --------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None,
)

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=180.0,
    max_retries=2,
)


# --------------------------------------------------
# プロンプト
# --------------------------------------------------

SYSTEM_PROMPT: Final[str] = """
あなたは「Evidence AI」です。
カーニボア、ケトジェニック、低糖質、人類進化、代謝医学に特化した
リサーチAIとして回答してください。

【調査方針】
・必ず最新のWeb情報を検索する
・内部知識ベースが設定されている場合は、それも検索して照合する
・原著論文、系統的レビュー、メタ解析、RCTを優先する
・一般記事よりPubMed、DOI、学術誌原文、公的機関、大学を優先する
・カーニボアや低糖質を支持する研究、機序、臨床的主張も積極的に探索する
・結論を先に決めて証拠を歪めない
・カーニボアに不都合な研究や安全性の懸念も隠さない
・存在しない論文、著者、DOI、数値、URLを作らない

【カーニボアの視点】
・高糖質の混合食で得られた結果を、カーニボアへ無条件に当てはめない
・背景食、糖質摂取量、インスリン抵抗性、体重変化、エネルギー摂取量、
  ApoB、LDL-P、TG/HDL比、既往歴、家族歴などを必要に応じて考慮する
・カーニボアを直接検証していない研究は、その点を短く明示する
・カーニボア側の文脈を組み込むが、都合のよい証拠だけを選ばない

【回答の絶対ルール】
・本文はリンクを除いて必ず300字以内
・本文は自然な2〜3文だけ
・見出し、箇条書き、番号、参考文献一覧を作らない
・質問に最も直接関係するエビデンスの要点だけを書く
・出典は最も関連性が高いものを1〜2件だけ選ぶ
・本文中にURLを書かない
・同じURLを重複させない
・質問が曖昧でも、長い確認質問や選択肢一覧を出さず、
  最も自然な解釈で短く回答する
・断定できない場合も、何が分かっていて何が未確定かを短く示す
""".strip()


# --------------------------------------------------
# データ型
# --------------------------------------------------


@dataclass(frozen=True)
class WebCitation:
    title: str
    url: str


@dataclass(frozen=True)
class EvidenceAnswer:
    text: str
    citations: tuple[WebCitation, ...] = ()


# --------------------------------------------------
# 共通処理
# --------------------------------------------------


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    return re.sub(rf"<@!?{bot_user_id}>", "", content).strip()


def build_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "web_search",
            "search_context_size": "low",
        }
    ]

    if OPENAI_VECTOR_STORE_ID:
        tools.insert(
            0,
            {
                "type": "file_search",
                "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
                "max_num_results": 5,
            },
        )

    return tools


def clean_source_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        filtered_query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
        ]
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(filtered_query, doseq=True),
                parts.fragment,
            )
        )
    except Exception:
        return url


def add_citation(
    citations: list[WebCitation],
    seen_urls: set[str],
    title: str,
    url: str,
) -> None:
    if not url:
        return

    cleaned_url = clean_source_url(url)
    if not cleaned_url or cleaned_url in seen_urls:
        return

    cleaned_title = re.sub(r"\s+", " ", title or "").strip()
    if not cleaned_title:
        cleaned_title = urlsplit(cleaned_url).netloc or "出典"

    citations.append(
        WebCitation(
            title=cleaned_title,
            url=cleaned_url,
        )
    )
    seen_urls.add(cleaned_url)


def collect_citations_from_object(
    obj: Any,
    citations: list[WebCitation],
    seen_urls: set[str],
) -> None:
    """
    response.output_item.done 等のイベントからURL注釈を回収する。
    SDKの型差異に備えて、属性を安全にたどる。
    """
    if obj is None:
        return

    for content_item in getattr(obj, "content", []) or []:
        for annotation in getattr(content_item, "annotations", []) or []:
            if getattr(annotation, "type", "") != "url_citation":
                continue

            add_citation(
                citations,
                seen_urls,
                getattr(annotation, "title", ""),
                getattr(annotation, "url", ""),
            )


def collect_citations_from_event(
    event: Any,
    citations: list[WebCitation],
    seen_urls: set[str],
) -> None:
    for attr_name in ("item", "output_item", "part", "response"):
        collect_citations_from_object(
            getattr(event, attr_name, None),
            citations,
            seen_urls,
        )


def collect_urls_from_text(text: str) -> tuple[WebCitation, ...]:
    found: list[WebCitation] = []
    seen_urls: set[str] = set()

    for raw_url in re.findall(r"https?://[^\s)>\]}]+", text):
        url = raw_url.rstrip(".,、。")
        add_citation(
            found,
            seen_urls,
            urlsplit(url).netloc or "出典",
            url,
        )

    return tuple(found)


def strip_links_and_formatting(text: str) -> str:
    cleaned = text

    # Markdownリンクはリンク部分だけ削除し、表示文字も出典名なら除去
    cleaned = re.sub(
        r"\s*\[[^\]]*\]\(https?://[^)]+\)",
        "",
        cleaned,
    )

    # 生URLを削除
    cleaned = re.sub(r"https?://[^\s)>\]}]+", "", cleaned)

    # 見出しを削除
    cleaned = re.sub(r"【[^】\n]{1,40}】", "", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.fullmatch(
            r"(?:出典|参考|参考文献|主要出典|Sources?|References?)[:：]?",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        line = re.sub(r"^(?:[-•*]|\d+[.)、])\s*", "", line)
        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"（\s*）", "", cleaned)
    cleaned = re.sub(r"\s+([、。！？!?])", r"\1", cleaned)

    return cleaned.strip()


def keep_first_sentences(text: str, max_sentences: int = 3) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])", text)
        if part.strip()
    ]

    if not sentences:
        return text.strip()

    return "".join(sentences[:max_sentences]).strip()


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    candidate = text[:limit]
    sentence_end = max(
        candidate.rfind("。"),
        candidate.rfind("！"),
        candidate.rfind("？"),
        candidate.rfind("!"),
        candidate.rfind("?"),
    )

    if sentence_end >= max(80, limit // 2):
        return candidate[: sentence_end + 1].strip()

    return candidate[: limit - 1].rstrip("、, ") + "…"


def compact_body(text: str) -> str:
    body = strip_links_and_formatting(text)
    body = keep_first_sentences(body, max_sentences=3)
    body = truncate_text(body, BODY_CHAR_LIMIT)
    return body.strip()


def shorten_title(title: str, limit: int = 55) -> str:
    title = re.sub(r"\s+", " ", title).strip()

    if not title:
        return "出典"

    if len(title) <= limit:
        return title

    return title[: limit - 1].rstrip() + "…"


def format_final_answer(
    raw_text: str,
    citations: list[WebCitation],
) -> EvidenceAnswer:
    body = compact_body(raw_text)

    if not body:
        raise RuntimeError("OpenAIから回答本文を取得できませんでした。")

    selected = tuple(citations[:MAX_LINKS])

    if not selected:
        selected = collect_urls_from_text(raw_text)[:MAX_LINKS]

    if selected:
        links = "\n".join(
            f"[{shorten_title(citation.title)}]({citation.url})"
            for citation in selected
        )
        text = f"{body}\n\n{links}"
    else:
        text = body

    return EvidenceAnswer(
        text=text,
        citations=selected,
    )


def format_stream_preview(raw_text: str) -> str:
    body = compact_body(raw_text)
    return body or "🔎 エビデンスを検索しています…"


# --------------------------------------------------
# OpenAI
# --------------------------------------------------


async def stream_evidence_answer(
    question: str,
    preview_message: discord.Message,
) -> EvidenceAnswer:
    """
    ストリーム中に本文を表示する。
    response.completed内の完全レスポンスには依存せず、
    受信したdelta本文を最終回答として使う。
    """
    stream = await openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        tools=build_tools(),
        include=["file_search_call.results"] if OPENAI_VECTOR_STORE_ID else [],
        input=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    raw_text = ""
    displayed_preview = ""
    last_edit_time = 0.0

    citations: list[WebCitation] = []
    seen_urls: set[str] = set()

    async for event in stream:
        event_type = getattr(event, "type", "")

        collect_citations_from_event(
            event,
            citations,
            seen_urls,
        )

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if not delta:
                continue

            raw_text += delta
            now = time.monotonic()

            if now - last_edit_time >= STREAM_EDIT_INTERVAL_SECONDS:
                preview = format_stream_preview(raw_text)

                if preview != displayed_preview:
                    try:
                        await preview_message.edit(
                            content=preview,
                            suppress=True,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                        displayed_preview = preview
                        last_edit_time = now
                    except discord.HTTPException:
                        logger.warning(
                            "Discord streaming preview update failed"
                        )

        elif event_type == "response.failed":
            response_obj = getattr(event, "response", None)
            error = getattr(response_obj, "error", None)
            raise RuntimeError(
                f"OpenAI response failed: {error or response_obj}"
            )

        elif event_type == "error":
            raise RuntimeError(
                str(getattr(event, "message", "OpenAI stream error"))
            )

    if not raw_text.strip():
        raise RuntimeError("OpenAIから回答本文を取得できませんでした。")

    return format_final_answer(
        raw_text=raw_text,
        citations=citations,
    )


# --------------------------------------------------
# Discordイベント
# --------------------------------------------------


@bot.event
async def on_ready() -> None:
    logger.info(
        "Logged in as %s (%s) | model=%s | vector_store=%s",
        bot.user,
        bot.user.id if bot.user else "?",
        OPENAI_MODEL,
        OPENAI_VECTOR_STORE_ID or "disabled",
    )

    if bot.user:
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="@Evidence AI",
            )
        )


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or bot.user is None:
        return

    if bot.user not in message.mentions:
        await bot.process_commands(message)
        return

    question = remove_bot_mention(
        message.content,
        bot.user.id,
    )

    if not question:
        await message.reply(
            "質問を書いてください。例：`@Evidence AI 卵とコレステロールについて`",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    preview_message: discord.Message | None = None

    try:
        preview_message = await message.reply(
            "🔎 エビデンスを検索しています…",
            mention_author=False,
            suppress_embeds=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

        answer = await stream_evidence_answer(
            question=question,
            preview_message=preview_message,
        )

        await preview_message.edit(
            content=answer.text,
            suppress=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    except Exception:
        logger.exception("Evidence AI request failed")

        error_text = (
            "検索または回答生成中にエラーが発生しました。"
            "少し待ってから、もう一度試してください。"
        )

        if preview_message is not None:
            try:
                await preview_message.edit(
                    content=error_text,
                    suppress=True,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except discord.HTTPException:
                await message.reply(
                    error_text,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        else:
            await message.reply(
                error_text,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
