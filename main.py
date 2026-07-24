import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import discord
from discord.ext import commands
from openai import AsyncOpenAI


# ==================================================
# 環境変数
# ==================================================


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

# 任意。設定していない場合はWeb検索のみ
OPENAI_VECTOR_STORE_ID: Final[str] = os.getenv(
    "OPENAI_VECTOR_STORE_ID", ""
).strip()

BODY_CHAR_LIMIT: Final[int] = 300
MAX_LINKS: Final[int] = 2


# ==================================================
# ログ
# ==================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("carnivore-ai")


# ==================================================
# Discord / OpenAI
# ==================================================

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


# ==================================================
# プロンプト
# ==================================================

BASE_PROMPT: Final[str] = """
あなたは「カーニボアAI」です。
Discordコミュニティ内で、一般的な質問から健康・栄養・代謝の質問まで
自然な日本語で回答します。

【共通ルール】
・回答本文はリンクを除いて必ず300字以内
・自然な2〜3文で回答する
・見出し、箇条書き、番号、長い前置きは使わない
・質問へ直接答え、同じ説明を繰り返さない
・曖昧な質問でも長い確認項目を並べず、最も自然な意味で答える
・分からないことを作らない
""".strip()

RESEARCH_PROMPT: Final[str] = """
あなたは「カーニボアAI」です。
カーニボア、ケトジェニック、低糖質、人類進化、代謝医学に強い
リサーチAIとして回答してください。

【調査方針】
・質問に最も直接関係する最新のWeb情報を検索する
・内部知識ベースが設定されている場合は、そこも検索して照合する
・原著論文、系統的レビュー、メタ解析、RCTを優先する
・一般記事よりPubMed、DOI、学術誌原文、公的機関、大学を優先する
・カーニボアや低糖質を支持する研究、機序、臨床的主張も探索する
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
・自然な2〜3文だけで答える
・見出し、箇条書き、番号、参考文献一覧を作らない
・最も重要なエビデンスの要点だけを書く
・本文中にURLを書かない
・出典は最も関連性が高いものを1〜2件だけ選ぶ
・長い確認質問や選択肢一覧を出さない
""".strip()


# ==================================================
# データ型
# ==================================================


@dataclass(frozen=True)
class WebCitation:
    title: str
    url: str


@dataclass(frozen=True)
class CarnivoreAnswer:
    text: str
    citations: tuple[WebCitation, ...] = ()


# ==================================================
# 質問判定
# ==================================================


RESEARCH_PATTERNS: Final[tuple[str, ...]] = (
    r"エビデンス",
    r"論文",
    r"研究",
    r"出典",
    r"根拠",
    r"ソース",
    r"最新",
    r"調べ",
    r"検索",
    r"比較",
    r"安全性",
    r"リスク",
    r"PubMed",
    r"DOI",
    r"RCT",
    r"メタ解析",
    r"系統的レビュー",
    r"ガイドライン",
    r"コレステロール",
    r"LDL",
    r"ApoB",
    r"LDL-P",
    r"血糖",
    r"HbA1c",
    r"糖尿病",
    r"心血管",
    r"腎機能",
    r"肝機能",
    r"甲状腺",
    r"尿酸",
    r"栄養",
    r"医学",
    r"医療",
    r"カーニボア",
    r"ケト",
    r"低糖質",
)


def needs_research(question: str) -> bool:
    return any(
        re.search(pattern, question, flags=re.IGNORECASE)
        for pattern in RESEARCH_PATTERNS
    )


# ==================================================
# 共通処理
# ==================================================


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    return re.sub(rf"<@!?{bot_user_id}>", "", content).strip()


def build_research_tools() -> list[dict[str, Any]]:
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

    # Markdownリンクと生URLを除去
    cleaned = re.sub(
        r"\s*\[[^\]]*\]\(https?://[^)]+\)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"https?://[^\s)>\]}]+", "", cleaned)

    # 見出し・リスト記号を除去
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


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.findall(r".+?(?:[。！？!?]|$)", text)
        if sentence.strip()
    ]


def fit_complete_sentences(
    text: str,
    limit: int = BODY_CHAR_LIMIT,
    max_sentences: int = 3,
) -> str:
    """
    文章の途中では切らず、完結した文だけを300字以内に収める。
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    selected: list[str] = []
    total_length = 0

    for sentence in sentences:
        if len(selected) >= max_sentences:
            break

        if total_length + len(sentence) <= limit:
            selected.append(sentence)
            total_length += len(sentence)
        else:
            break

    if selected:
        return "".join(selected).strip()

    # 最初の1文自体が300字を超える異常ケースだけ、安全に短縮
    first = sentences[0]
    candidate = first[:limit]
    punctuation_position = max(
        candidate.rfind("。"),
        candidate.rfind("！"),
        candidate.rfind("？"),
        candidate.rfind("!"),
        candidate.rfind("?"),
    )

    if punctuation_position >= 80:
        return candidate[: punctuation_position + 1].strip()

    return candidate[: limit - 1].rstrip("、, ") + "…"


def compact_body(text: str) -> str:
    cleaned = strip_links_and_formatting(text)
    return fit_complete_sentences(cleaned)


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
    include_sources: bool,
) -> CarnivoreAnswer:
    body = compact_body(raw_text)

    if not body:
        raise RuntimeError("OpenAIから回答本文を取得できませんでした。")

    selected: tuple[WebCitation, ...] = ()

    if include_sources:
        selected = tuple(citations[:MAX_LINKS])
        if not selected:
            selected = collect_urls_from_text(raw_text)[:MAX_LINKS]

    if selected:
        links = "\n".join(
            f"[{shorten_title(citation.title)}]({citation.url})"
            for citation in selected
        )
        final_text = f"{body}\n\n{links}"
    else:
        final_text = body

    return CarnivoreAnswer(
        text=final_text,
        citations=selected,
    )


# ==================================================
# OpenAI回答
# ==================================================


async def ask_carnivore_ai(
    question: str,
    research_mode: bool,
) -> CarnivoreAnswer:
    citations: list[WebCitation] = []
    seen_urls: set[str] = set()

    request: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "developer",
                "content": RESEARCH_PROMPT if research_mode else BASE_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": True,
    }

    if research_mode:
        request["reasoning"] = {"effort": "low"}
        request["tools"] = build_research_tools()

        if OPENAI_VECTOR_STORE_ID:
            request["include"] = ["file_search_call.results"]

    stream = await openai_client.responses.create(**request)

    raw_text = ""

    async for event in stream:
        event_type = getattr(event, "type", "")

        if research_mode:
            collect_citations_from_event(
                event,
                citations,
                seen_urls,
            )

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                raw_text += delta

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
        include_sources=research_mode,
    )


# ==================================================
# Discordイベント
# ==================================================


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
                name="@カーニボアAI",
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
            f"質問を書いてください。例：`{bot.user.mention} 卵とコレステロールについて`",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    research_mode = needs_research(question)

    status_text = (
        "🔎 関連する研究を調べています…"
        if research_mode
        else "💭 回答を考えています…"
    )

    status_message: discord.Message | None = None

    try:
        # 生成途中の文章をDiscordへ何度も表示すると、
        # 文章の途中で止まったように見えるため、状態表示だけを出す。
        status_message = await message.reply(
            status_text,
            mention_author=False,
            suppress_embeds=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

        answer = await ask_carnivore_ai(
            question=question,
            research_mode=research_mode,
        )

        await status_message.edit(
            content=answer.text,
            suppress=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    except Exception:
        logger.exception("Carnivore AI request failed")

        error_text = (
            "回答生成中にエラーが発生しました。"
            "少し待ってから、もう一度試してください。"
        )

        if status_message is not None:
            try:
                await status_message.edit(
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
