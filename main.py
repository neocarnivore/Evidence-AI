import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Final

import discord
from discord.ext import commands
from openai import AsyncOpenAI


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
MAIN_DISPLAY_LIMIT: Final[int] = 300

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evidence-ai")

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

SYSTEM_PROMPT: Final[str] = """
あなたは「Evidence AI」です。カーニボア、ケトジェニック、低糖質、
人類進化、代謝医学に特化したリサーチAIとして回答してください。

【調査方針】
- カーニボアを支持する研究や臨床的主張を積極的に探索する
- 結論を先に決めて証拠を歪めない
- 質問ごとに、事前構築済みの内部知識ベースを最初に検索する
- 回答時にYouTubeへアクセスしたり字幕を取得したりしない
- 内部知識ベースと最新のWeb情報を照合する
- 原著論文、系統的レビュー、メタ解析、RCT、公的機関を優先する
- 一般的な栄養学的見解も、その根拠、研究デザイン、交絡、
  利益相反を検討する
- カーニボアに不都合な研究や安全性の懸念を隠さない

【証拠の区別】
以下を混同せず、該当する分類を明示してください。
- 比較的確立した事実
- 有力だが未確定の仮説
- 観察研究または症例報告
- 専門家の臨床経験
- 個人的見解
- 証拠が不足している主張

【研究の記載】
研究を提示するときは、確認できる範囲で次を記載してください。
- 研究タイトル、著者、発表年
- 研究デザイン、対象人数
- 主要結果、重要な限界
- DOIまたは原文URL

カーニボアを直接検証していない研究は、実際に何を検証した研究なのか
明示してください。存在しない論文、DOI、著者、数値を作らないでください。
出典を確認できない内容は「出典不明」または「確認できない」と明記します。

【YouTube字幕の扱い】
- YouTube字幕は査読済み研究や臨床試験ではなく、専門家による解説、
  臨床経験、症例・体験談、仮説・個人的見解のいずれかとして扱う
- speaker_roleがtarget_expertでない、またはunknown_speakerの場合、
  対象専門家本人の直接発言だと断定しない
- 動画内で研究に言及している場合はWeb検索で元論文を確認する
- 元研究を確認できない場合は
  「動画内では研究への言及がありますが、元論文は確認できていません」
  と明記する
- 発言者、公開日、字幕種別、該当時間、話者確度などのメタデータは
  回答を組み立てる際の判断材料として内部で確認する
- YouTube字幕の引用は原則として
  `発言者 — [短い動画タイトル 12:34](タイムスタンプ付きURL)`
  のように1行で簡潔に示す
- 公開日、情報源区分、手動・自動字幕、話者不確実性は、
  結論の信頼性や誤帰属防止に重要な場合だけ補足する
- 同じ動画を何度も引用する場合、出典情報を繰り返さない
- YouTube字幕だけを根拠に医学的事実が確立していると表現しない

【健康・医療上の扱い】
個別の診断や治療を断定しません。重大または緊急のリスクが考えられる場合は、
適切な医療相談を勧めてください。

検査値を扱う場合は、単一指標だけでなく、質問に応じてApoB、LDL-P、
TG/HDL比、既往歴、家族歴なども検討します。

【標準回答形式】
【結論】
【根拠】
【研究の強さ】
【注意点・反対意見】
【出典】

Discord向けに簡潔にまとめます。本文の理解に不要な書誌情報で文字数を
増やさず、重要な主張にだけ短いクリック可能な出典を付けてください。

同じ出典の重複、長いURLの単独表示、内部ファイル名の一覧は避けてください。

通常回答は、結論、最重要の根拠、必要な注意点、短い出典を残して
原則300字以内にまとめてください。詳しい説明は追加表示で行います。
""".strip()

REFINEMENT_INSTRUCTIONS: Final[dict[str, str]] = {
    "詳しく": (
        "元の質問をより詳しく再調査してください。研究ごとの対象人数、期間、"
        "効果量、限界を増やし、必要なら回答を複数メッセージに分けてください。"
    ),
    "論文だけ": (
        "元の質問に直接関係する学術論文だけを列挙してください。各論文について"
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


@dataclass(frozen=True)
class EvidenceAnswer:
    text: str
    knowledge_files: tuple[str, ...]


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    return re.sub(rf"<@!?{bot_user_id}>", "", content).strip()


def split_discord_message(
    text: str,
    limit: int = 1900,
) -> list[str]:
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


def compact_body(
    text: str,
    limit: int = MAIN_DISPLAY_LIMIT,
) -> str:
    """通常のDiscord回答を指定文字数以内へ収める。"""

    compact = re.sub(r"[ \t]+", " ", text).strip()
    compact = re.sub(r"\n{3,}", "\n\n", compact)

    if len(compact) <= limit:
        return compact

    cut = compact[: limit - 1]

    candidates = [
        cut.rfind("\n"),
        cut.rfind("。"),
        cut.rfind("！"),
        cut.rfind("？"),
    ]

    natural_end = max(candidates)

    if natural_end >= limit // 2:
        cut = cut[
            : natural_end
            + (0 if cut[natural_end] == "\n" else 1)
        ]

    return cut.rstrip() + "…"


def build_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "web_search",
        }
    ]

    if OPENAI_VECTOR_STORE_ID:
        tools.insert(
            0,
            {
                "type": "file_search",
                "vector_store_ids": [
                    OPENAI_VECTOR_STORE_ID
                ],
                "max_num_results": 12,
            },
        )

    return tools


def collect_file_citations(
    response: Any,
) -> tuple[str, ...]:
    filenames: set[str] = set()

    for output_item in getattr(response, "output", []):
        content_items = (
            getattr(output_item, "content", []) or []
        )

        for content_item in content_items:
            annotations = (
                getattr(content_item, "annotations", []) or []
            )

            for annotation in annotations:
                if (
                    getattr(annotation, "type", "")
                    == "file_citation"
                ):
                    filename = getattr(
                        annotation,
                        "filename",
                        "",
                    )

                    if filename:
                        filenames.add(filename)

    return tuple(sorted(filenames))


async def ask_evidence_ai(
    question: str,
    refinement: str | None = None,
) -> EvidenceAnswer:
    user_content = question

    if refinement:
        user_content = (
            f"元の質問:\n{question}\n\n"
            f"追加指示:\n"
            f"{REFINEMENT_INSTRUCTIONS[refinement]}"
        )

    stream = await openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={
            "effort": "low",
        },
        tools=build_tools(),
        include=(
            ["file_search_call.results"]
            if OPENAI_VECTOR_STORE_ID
            else []
        ),
        input=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        stream=True,
    )

    text_parts: list[str] = []
    completed_response: Any | None = None

    async for event in stream:
        event_type = getattr(event, "type", "")

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")

            if delta:
                text_parts.append(delta)

        elif event_type == "response.completed":
            completed_response = getattr(
                event,
                "response",
                None,
            )

    answer = "".join(text_parts).strip()

    if not answer:
        raise RuntimeError(
            "OpenAIから回答本文が返されませんでした。"
        )

    knowledge_files = (
        collect_file_citations(completed_response)
        if completed_response is not None
        else ()
    )

    return EvidenceAnswer(
        text=answer,
        knowledge_files=knowledge_files,
    )


async def send_answer_parts(
    destination: discord.abc.Messageable,
    answer: EvidenceAnswer,
) -> None:
    for part in split_discord_message(answer.text):
        await destination.send(
            part,
            suppress_embeds=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )


class RefinementView(discord.ui.View):
    def __init__(self, question: str) -> None:
        super().__init__(timeout=3600)
        self.question = question

    async def run_refinement(
        self,
        interaction: discord.Interaction,
        mode: str,
    ) -> None:
        await interaction.response.defer(
            thinking=True
        )

        try:
            answer = await ask_evidence_ai(
                self.question,
                refinement=mode,
            )

            parts = split_discord_message(
                answer.text
            )

            for index, part in enumerate(parts):
                if index == 0:
                    await interaction.followup.send(
                        f"**{mode}**\n{part}",
                        suppress_embeds=True,
                        allowed_mentions=(
                            discord.AllowedMentions.none()
                        ),
                    )
                else:
                    await interaction.followup.send(
                        part,
                        suppress_embeds=True,
                        allowed_mentions=(
                            discord.AllowedMentions.none()
                        ),
                    )

        except Exception:
            logger.exception(
                "Evidence AI refinement failed"
            )

            await interaction.followup.send(
                "追加調査中にエラーが発生しました。"
                "少し待って再度お試しください。",
                ephemeral=True,
            )

    @discord.ui.button(
        label="詳しく",
        style=discord.ButtonStyle.primary,
    )
    async def detail(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(
            interaction,
            "詳しく",
        )

    @discord.ui.button(
        label="論文だけ",
        style=discord.ButtonStyle.secondary,
    )
    async def papers(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(
            interaction,
            "論文だけ",
        )

    @discord.ui.button(
        label="反対意見も",
        style=discord.ButtonStyle.secondary,
    )
    async def counter(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(
            interaction,
            "反対意見も",
        )

    @discord.ui.button(
        label="初心者向け",
        style=discord.ButtonStyle.secondary,
    )
    async def beginner(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(
            interaction,
            "初心者向け",
        )

    @discord.ui.button(
        label="専門家向け",
        style=discord.ButtonStyle.secondary,
    )
    async def expert(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await self.run_refinement(
            interaction,
            "専門家向け",
        )


@bot.event
async def on_ready() -> None:
    logger.info(
        "Logged in as %s (%s) | vector_store=%s",
        bot.user,
        bot.user.id if bot.user else "?",
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
async def on_message(
    message: discord.Message,
) -> None:
    if message.author.bot:
        return

    if bot.user is None:
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
            "質問を書いてください。\n"
            "例：`@Evidence AI "
            "LDL上昇について研究を比較して`",
            mention_author=False,
            allowed_mentions=(
                discord.AllowedMentions.none()
            ),
        )
        return

    async with message.channel.typing():
        try:
            answer = await ask_evidence_ai(
                question
            )

            await message.reply(
                compact_body(answer.text),
                mention_author=False,
                suppress_embeds=True,
                allowed_mentions=(
                    discord.AllowedMentions.none()
                ),
            )

            await message.channel.send(
                "必要なら表示方法を選んでください。",
                view=RefinementView(question),
                allowed_mentions=(
                    discord.AllowedMentions.none()
                ),
            )

        except Exception:
            logger.exception(
                "Evidence AI request failed"
            )

            await message.reply(
                "検索または回答生成中に"
                "エラーが発生しました。"
                "少し待ってから、"
                "もう一度試してください。",
                mention_author=False,
                allowed_mentions=(
                    discord.AllowedMentions.none()
                ),
            )

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(
        DISCORD_TOKEN,
        log_handler=None,
    )