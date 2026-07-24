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
OPENAI_MODEL: Final[str] = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()
OPENAI_VECTOR_STORE_ID: Final[str] = os.getenv(
    "OPENAI_VECTOR_STORE_ID", ""
).strip()

MAIN_BODY_CHAR_LIMIT: Final[int] = 300
MAIN_MAX_LINKS: Final[int] = 2
MAIN_MAX_OUTPUT_TOKENS: Final[int] = int(
    os.getenv("MAIN_MAX_OUTPUT_TOKENS", "700")
)
REFINEMENT_MAX_OUTPUT_TOKENS: Final[int] = int(
    os.getenv("REFINEMENT_MAX_OUTPUT_TOKENS", "3000")
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

BASE_RESEARCH_PROMPT: Final[str] = """
あなたは「Evidence AI」です。カーニボア、ケトジェニック、低糖質、
人類進化、代謝医学に特化したリサーチAIとして回答してください。

【調査方針】
- 最新のWeb情報と、設定されている場合は内部知識ベースの両方を調べる
- 原著論文、系統的レビュー、メタ解析、RCTを優先する
- 一般記事より、PubMed、DOI、学術誌原文、公的機関、大学を優先する
- カーニボアや低糖質を支持する研究・機序・臨床的主張も積極的に探索する
- 結論を先に決めて証拠を歪めない
- カーニボアに不都合な研究や安全性の懸念も隠さない
- 存在しない論文、著者、DOI、数値、URLを作らない

【カーニボアの視点】
- 一般的な高糖質の混合食で得られた結果を、カーニボアへ無条件に当てはめない
- 背景食、糖質摂取量、インスリン抵抗性、体重変化、エネルギー摂取量、
  TG/HDL比、ApoB、LDL-P、既往歴、家族歴などを質問に応じて考慮する
- カーニボアを直接検証していない研究は、何を実際に検証した研究か明示する
- カーニボア側の視点を組み込むが、有利な結論を作るための選択的引用はしない

【証拠の扱い】
- 比較的確立した事実、未確定の仮説、観察研究、症例報告、
  専門家の臨床経験、個人的見解を混同しない
- 研究デザイン、対象、期間、交絡、利益相反、代理指標と臨床アウトカムを区別する
- 個別の診断や治療を断定しない
""".strip()

MAIN_RESPONSE_RULES: Final[str] = """
【通常回答の絶対ルール】
- 回答本文は、出典リンクを除いて必ず300字以内
- 本文は自然な2〜3文だけにする
- 見出し、箇条書き、長い前置き、参考文献一覧を作らない
- 質問に最も直接関係するエビデンスの要点だけを書く
- カーニボア、低糖質、代謝状態の視点を本文に自然に組み込む
- 出典は最も関連性が高く、主張を直接支えるものを1〜2件だけ引用する
- 同じ出典を重複して引用しない
- 本文中にURLを羅列しない
- 単純な質問は100〜200字程度で答える
""".strip()

DETAIL_RESPONSE_RULES: Final[str] = """
【追加調査モード】
ユーザーが明示的に詳細表示を選んでいます。通常回答の300字制限は解除します。
ただしDiscordで読みやすく整理し、重要な主張には確認可能な出典を付けてください。
研究の対象人数、期間、主要結果、限界を、確認できる範囲で示してください。
""".strip()

MAIN_SYSTEM_PROMPT: Final[str] = (
    f"{BASE_RESEARCH_PROMPT}\n\n{MAIN_RESPONSE_RULES}"
)

DETAIL_SYSTEM_PROMPT: Final[str] = (
    f"{BASE_RESEARCH_PROMPT}\n\n{DETAIL_RESPONSE_RULES}"
)

REFINEMENT_INSTRUCTIONS: Final[dict[str, str]] = {
    "詳しく": (
        "元の質問をより詳しく再調査してください。研究ごとの対象人数、期間、"
        "効果量、限界を増やし、必要なら回答を複数メッセージに分けてください。"
    ),
    "論文だけ": (
        "元の質問に直接関係する学術論文だけを提示してください。各論文について"
        "タイトル、著者、年、研究デザイン、対象人数、主要結果、限界、DOIまたは"
        "原文URLを示してください。専門家発言や一般記事は除外してください。"
    ),
    "反対意見も": (
        "元の質問について、カーニボアまたは低糖質に不利な証拠と主要な反論を"
        "優先的に調べ、支持側の証拠と同じ厳しさで比較してください。"
    ),
    "初心者向け": (
        "元の回答を医学知識のない初心者向けに、用語を説明しながら簡潔に"
        "書き直してください。重要な注意点と出典は残してください。"
    ),
    "専門家向け": (
        "元の質問を医療・研究職向けに再調査してください。研究デザイン、統計、"
        "交絡、バイアス、代理指標と臨床アウトカムを重点的に評価してください。"
    ),
}


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
    knowledge_files: tuple[str, ...] = ()
    web_citations: tuple[WebCitation, ...] = ()


# --------------------------------------------------
# 共通処理
# --------------------------------------------------


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    return re.sub(rf"<@!?{bot_user_id}>", "", content).strip()


def split_discord_message(text: str, limit: int = 1900) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)

        if split_at < limit // 2:
            split_at = remaining.rfind("。", 0, limit + 1)
            if split_at >= limit // 2:
                split_at += 1

        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)

        if split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def build_tools(detailed: bool = False) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "web_search",
            "search_context_size": "medium" if detailed else "low",
        }
    ]

    if OPENAI_VECTOR_STORE_ID:
        tools.insert(
            0,
            {
                "type": "file_search",
                "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
                "max_num_results": 12 if detailed else 6,
            },
        )

    return tools


def collect_file_citations(response: Any) -> tuple[str, ...]:
    filenames: set[str] = set()

    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            for annotation in getattr(content_item, "annotations", []) or []:
                if getattr(annotation, "type", "") == "file_citation":
                    filename = getattr(annotation, "filename", "")
                    if filename:
                        filenames.add(filename)

    return tuple(sorted(filenames))


def clean_source_url(url: str) -> str:
    """OpenAI由来の追跡パラメータなどを除去する。"""
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


def collect_web_citations(response: Any) -> tuple[WebCitation, ...]:
    citations: list[WebCitation] = []
    seen_urls: set[str] = set()

    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            for annotation in getattr(content_item, "annotations", []) or []:
                if getattr(annotation, "type", "") != "url_citation":
                    continue

                raw_url = getattr(annotation, "url", "")
                if not raw_url:
                    continue

                url = clean_source_url(raw_url)
                if url in seen_urls:
                    continue

                title = getattr(annotation, "title", "") or urlsplit(url).netloc
                citations.append(WebCitation(title=title.strip(), url=url))
                seen_urls.add(url)

    return tuple(citations)


def collect_urls_from_text(text: str) -> tuple[WebCitation, ...]:
    """注釈が取れない場合だけ使う予備処理。"""
    urls = re.findall(r"https?://[^\s)>\]}]+", text)
    citations: list[WebCitation] = []
    seen_urls: set[str] = set()

    for raw_url in urls:
        url = clean_source_url(raw_url.rstrip(".,、。"))
        if url in seen_urls:
            continue

        citations.append(
            WebCitation(
                title=urlsplit(url).netloc or "出典",
                url=url,
            )
        )
        seen_urls.add(url)

    return tuple(citations)


def strip_links_and_formatting(text: str) -> str:
    """本文からリンク・見出し・箇条書き記号を除去する。"""
    cleaned = text

    # Markdownリンクを削除
    cleaned = re.sub(
        r"\s*\(?\[[^\]]*\]\(https?://[^)]+\)\)?",
        "",
        cleaned,
    )

    # 生URLを削除
    cleaned = re.sub(r"https?://[^\s)>\]}]+", "", cleaned)

    # 短い見出しを削除
    cleaned = re.sub(r"【[^】\n]{1,30}】", "", cleaned)

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
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])", text)
        if part.strip()
    ]

    if not parts:
        return text.strip()

    return "".join(parts[:max_sentences]).strip()


def truncate_japanese_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    candidate = text[:limit]
    last_sentence_end = max(
        candidate.rfind("。"),
        candidate.rfind("！"),
        candidate.rfind("？"),
        candidate.rfind("!"),
        candidate.rfind("?"),
    )

    if last_sentence_end >= max(80, limit // 2):
        return candidate[: last_sentence_end + 1].strip()

    return candidate[: limit - 1].rstrip("、, ") + "…"


def compact_body(text: str) -> str:
    body = strip_links_and_formatting(text)
    body = keep_first_sentences(body, max_sentences=3)
    body = truncate_japanese_text(body, MAIN_BODY_CHAR_LIMIT)
    return body.strip()


def shorten_source_title(title: str, limit: int = 70) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        return "出典"
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "…"


def format_compact_answer(response: Any) -> EvidenceAnswer:
    raw_text = getattr(response, "output_text", "").strip()
    if not raw_text:
        raise RuntimeError("OpenAIから回答本文が返されませんでした。")

    body = compact_body(raw_text)
    if not body:
        raise RuntimeError("回答本文を整形できませんでした。")

    citations = collect_web_citations(response)
    if not citations:
        citations = collect_urls_from_text(raw_text)

    selected_citations = citations[:MAIN_MAX_LINKS]

    if selected_citations:
        source_lines = "\n".join(
            f"[{shorten_source_title(citation.title)}]({citation.url})"
            for citation in selected_citations
        )
        final_text = f"{body}\n\n{source_lines}"
    else:
        final_text = body

    return EvidenceAnswer(
        text=final_text,
        knowledge_files=collect_file_citations(response),
        web_citations=selected_citations,
    )


def format_stream_preview(raw_text: str) -> str:
    body = compact_body(raw_text)
    if not body:
        return "🔎 エビデンスを検索しています…"
    return body


# --------------------------------------------------
# OpenAI呼び出し
# --------------------------------------------------


async def stream_main_answer(
    question: str,
    preview_message: discord.Message,
) -> EvidenceAnswer:
    """通常回答をストリーミングし、Discordの同一メッセージを更新する。"""
    stream = await openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        tools=build_tools(detailed=False),
        include=["file_search_call.results"] if OPENAI_VECTOR_STORE_ID else [],
        input=[
            {"role": "developer", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_output_tokens=MAIN_MAX_OUTPUT_TOKENS,
        stream=True,
    )

    raw_text = ""
    displayed_preview = ""
    last_edit_time = 0.0
    final_response: Any | None = None

    async for event in stream:
        event_type = getattr(event, "type", "")

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if not delta:
                continue

            raw_text += delta
            now = time.monotonic()

            if now - last_edit_time >= STREAM_EDIT_INTERVAL_SECONDS:
                preview = format_stream_preview(raw_text)

                if preview and preview != displayed_preview:
                    try:
                        await preview_message.edit(
                            content=preview,
                            suppress=True,
                        )
                        displayed_preview = preview
                        last_edit_time = now
                    except discord.HTTPException:
                        logger.warning("Discord streaming preview update failed")

        elif event_type == "response.completed":
            final_response = getattr(event, "response", None)

        elif event_type == "response.failed":
            failed_response = getattr(event, "response", None)
            error = getattr(failed_response, "error", None)
            raise RuntimeError(f"OpenAI response failed: {error or failed_response}")

        elif event_type == "error":
            message = getattr(event, "message", "OpenAI stream error")
            raise RuntimeError(str(message))

    if final_response is None:
        raise RuntimeError("OpenAIの完了レスポンスを取得できませんでした。")

    return format_compact_answer(final_response)


async def ask_refinement(
    question: str,
    refinement: str,
) -> EvidenceAnswer:
    user_content = (
        f"元の質問:\n{question}\n\n追加指示:\n"
        f"{REFINEMENT_INSTRUCTIONS[refinement]}"
    )

    response = await openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        tools=build_tools(detailed=True),
        include=["file_search_call.results"] if OPENAI_VECTOR_STORE_ID else [],
        input=[
            {"role": "developer", "content": DETAIL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=REFINEMENT_MAX_OUTPUT_TOKENS,
    )

    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("OpenAIから回答本文が返されませんでした。")

    knowledge_files = collect_file_citations(response)
    if knowledge_files:
        references = "\n".join(f"- `{name}`" for name in knowledge_files)
        answer += f"\n\n【内部知識ベース参照】\n{references}"

    return EvidenceAnswer(
        text=answer,
        knowledge_files=knowledge_files,
        web_citations=collect_web_citations(response),
    )


# --------------------------------------------------
# Discord UI
# --------------------------------------------------


class RefinementView(discord.ui.View):
    def __init__(self, question: str) -> None:
        super().__init__(timeout=3600)
        self.question = question

    async def run_refinement(
        self,
        interaction: discord.Interaction,
        mode: str,
    ) -> None:
        await interaction.response.defer(thinking=True)

        try:
            answer = await ask_refinement(self.question, refinement=mode)
            parts = split_discord_message(answer.text)

            for index, part in enumerate(parts):
                if index == 0:
                    await interaction.followup.send(
                        f"**{mode}**\n{part}",
                        suppress_embeds=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await interaction.followup.send(
                        part,
                        suppress_embeds=True,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )

        except Exception:
            logger.exception("Evidence AI refinement failed")
            await interaction.followup.send(
                "追加調査中にエラーが発生しました。少し待って再度お試しください。",
                ephemeral=True,
            )

    @discord.ui.button(label="詳しく", style=discord.ButtonStyle.primary)
    async def detail(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(interaction, "詳しく")

    @discord.ui.button(label="論文だけ", style=discord.ButtonStyle.secondary)
    async def papers(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(interaction, "論文だけ")

    @discord.ui.button(label="反対意見も", style=discord.ButtonStyle.secondary)
    async def counter(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(interaction, "反対意見も")

    @discord.ui.button(label="初心者向け", style=discord.ButtonStyle.secondary)
    async def beginner(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(interaction, "初心者向け")

    @discord.ui.button(label="専門家向け", style=discord.ButtonStyle.secondary)
    async def expert(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(interaction, "専門家向け")


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

    question = remove_bot_mention(message.content, bot.user.id)

    if not question:
        await message.reply(
            "質問を書いてください。\n"
            "例：`@Evidence AI LDL上昇について研究を比較して`",
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

        answer = await stream_main_answer(
            question=question,
            preview_message=preview_message,
        )

        await preview_message.edit(
            content=answer.text,
            suppress=True,
            view=RefinementView(question),
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
                    view=None,
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