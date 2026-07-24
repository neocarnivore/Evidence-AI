import asyncio
import logging
import os
import re
from typing import Final

import discord
from discord.ext import commands
from openai import OpenAI

# --------------------------------------------------
# 環境変数
# --------------------------------------------------

DISCORD_TOKEN: Final[str] = os.environ["DISCORD_TOKEN"]
OPENAI_API_KEY: Final[str] = os.environ["OPENAI_API_KEY"]

OPENAI_MODEL: Final[str] = os.getenv("OPENAI_MODEL", "gpt-5.6")

# --------------------------------------------------
# ログ設定
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("evidence-ai")

# --------------------------------------------------
# Discord設定
# --------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None,
)

# --------------------------------------------------
# OpenAI設定
# --------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT: Final[str] = """
あなたは「Evidence AI」です。

健康、栄養、代謝、カーニボア、ケトジェニック、医学、食品科学に関する
質問を、最新のオンライン情報と学術資料を検索して回答します。

【基本姿勢】
・カーニボアやケトジェニックを頭ごなしに否定しない
・一般的な栄養学の見解を無批判に繰り返さない
・一方で、ユーザーが望む結論に合わせて証拠を選別しない
・支持する証拠、反対する証拠、不確実性を区別する
・確認できない内容は断定しない

【情報源の優先順位】
1. 原著論文
2. 系統的レビュー、メタ解析
3. ランダム化比較試験
4. 公的機関、大学、医学会、研究機関
5. 信頼できる二次資料

ブログ、まとめサイト、SNS投稿だけを医学的結論の根拠にしないでください。

【研究評価】
可能な範囲で、以下を区別してください。
・RCT
・メタ解析
・系統的レビュー
・観察研究
・症例報告
・動物研究
・細胞研究
・機序上の仮説

研究対象人数、期間、交絡、限界、代理指標と臨床アウトカムの違いも確認してください。

【回答形式】
原則として日本語で回答してください。

最初に結論を簡潔に示し、その後に根拠を説明してください。
重要な事実には出典を付けてください。
出典は、Discord上で開けるURLとして回答内に含めてください。

回答が長くなりすぎないようにしてください。
ただし、重要な注意点や研究上の限界は省略しないでください。
"""


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    """メッセージからBotへのメンション部分を削除する。"""
    patterns = [
        rf"<@{bot_user_id}>",
        rf"<@!{bot_user_id}>",
    ]

    result = content

    for pattern in patterns:
        result = re.sub(pattern, "", result)

    return result.strip()


def split_discord_message(text: str, limit: int = 1900) -> list[str]:
    """
    Discordの文字数制限を超えないように文章を分割する。
    コードブロックや複雑なMarkdownの完全保持までは行わない。
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, limit)

        if split_at < limit // 2:
            split_at = remaining.rfind("。", 0, limit)

        if split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at].strip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].strip()

    return chunks


def ask_evidence_ai(question: str) -> str:
    """
    OpenAI Responses APIを使用し、Web検索を含む回答を生成する。
    """
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        tools=[
            {
                "type": "web_search",
            }
        ],
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
    )

    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError("OpenAIから回答本文が返されませんでした。")

    return answer


@bot.event
async def on_ready() -> None:
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")

    if bot.user:
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="@Evidence AI",
            )
        )


@bot.event
async def on_message(message: discord.Message) -> None:
    # Bot自身や他Botのメッセージには反応しない
    if message.author.bot:
        return

    if bot.user is None:
        return

    # メンションされた場合のみ反応
    if bot.user not in message.mentions:
        await bot.process_commands(message)
        return

    question = remove_bot_mention(message.content, bot.user.id)

    if not question:
        await message.reply(
            "質問を書いてください。\n"
            "例：`@Evidence AI 飽和脂肪酸と心血管疾患の研究を調べて`",
            mention_author=False,
        )
        return

    async with message.channel.typing():
        try:
            # OpenAI SDKは同期処理なので別スレッドで実行
            answer = await asyncio.to_thread(
                ask_evidence_ai,
                question,
            )

            parts = split_discord_message(answer)

            first_message = True

            for part in parts:
                if first_message:
                    await message.reply(
                        part,
                        mention_author=False,
                        suppress_embeds=True,
                    )
                    first_message = False
                else:
                    await message.channel.send(
                        part,
                        suppress_embeds=True,
                    )

        except Exception as exc:
            logger.exception("Evidence AI request failed")

            await message.reply(
                "検索または回答生成中にエラーが発生しました。"
                "少し待ってから、もう一度試してください。",
                mention_author=False,
            )

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
