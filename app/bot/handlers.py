"""Telegram bot handlers using aiogram 3.x.

Supports two modes:
  - Webhook: for production (Render, etc.)
  - Polling: for local development

Commands:
  /start     — Welcome message
  /help      — Available commands
  /search    — Search documentation with RAG
  /status    — System status
  /clear     — Clear conversation history
  /docs      — List indexed documents
  /ingest    — Re-ingest documents (admin only)

Direct text messages are processed through RAG pipeline.
"""

from typing import Optional

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command

from app.core.config import settings
from app.core.security import sanitize_input
from app.services.rag import RAGPipeline

# Dispatcher (router-agnostic — can be used with webhook or polling)
dp = Dispatcher()

# RAG pipeline (initialized on first use)
_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    """Lazy initialization of RAG pipeline."""
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def is_admin(user_id: int) -> bool:
    """Check if user is admin."""
    if not settings.TELEGRAM_ADMIN_IDS:
        return True  # Open if not configured
    admin_ids = [int(x.strip()) for x in settings.TELEGRAM_ADMIN_IDS.split(",") if x.strip()]
    return user_id in admin_ids


# ========== Command Handlers ==========

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    welcome = (
        "👋 <b>Hello! I'm your AI Automation Agent.</b>\n\n"
        "I can answer questions about your business processes using the "
        "documentation you've provided.\n\n"
        "<b>Commands:</b>\n"
        "• /search &lt;query&gt; — Search docs\n"
        "• /status — System status\n"
        "• /clear — Reset conversation\n"
        "• /help — Show all commands\n\n"
        "Or just send me a question directly!"
    )
    await message.answer(welcome)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "📋 <b>Available Commands:</b>\n\n"
        "/start — Start the bot\n"
        "/help — This message\n"
        "/search &lt;query&gt; — Search documentation\n"
        "/status — Check system status\n"
        "/clear — Clear conversation history\n"
        "/docs — Show indexed document count\n"
        "/ingest — Re-ingest all documents (admin)\n\n"
        "You can also send any question directly — I'll search the docs and answer."
    )
    await message.answer(help_text)


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    rag = get_rag()
    stats = rag.stats()
    status = (
        "📊 <b>System Status</b>\n\n"
        f"🗂 Documents indexed: <b>{stats['documents_indexed']}</b>\n"
        f"🧠 Embedding model: {stats['embedding_model']}\n"
        f"📐 Vector dimension: {stats['embedding_dim']}\n"
        f"🤖 LLM model: {stats['llm_model']}\n"
        f"💬 Active conversations: {stats['active_conversations']}"
    )
    await message.answer(status)


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    rag = get_rag()
    user_id = str(message.from_user.id)
    rag.clear_history(user_id)
    await message.answer("✅ Conversation history cleared. Starting fresh!")


@dp.message(Command("docs"))
async def cmd_docs(message: types.Message):
    rag = get_rag()
    count = rag.vector.count()
    await message.answer(f"📄 Indexed documents: <b>{count}</b> chunks")


@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    query = message.text.replace("/search", "").strip()
    if not query:
        await message.answer("Usage: <code>/search your question here</code>")
        return
    await _process_query(message, query)


@dp.message(Command("ingest"))
async def cmd_ingest(message: types.Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔ Admin only command.")
        return

    await message.answer("📤 Starting document ingestion...")
    try:
        import asyncio
        from scripts.ingest_docs import ingest_all
        result = await ingest_all()
        await message.answer(
            f"✅ Ingestion complete!\n"
            f"Files: {result['files']}\n"
            f"Chunks: {result['chunks']}"
        )
    except Exception as e:
        await message.answer(f"❌ Ingestion failed: {e}")


# ========== Text Message Handler (RAG) ==========

@dp.message(F.text)
async def handle_text(message: types.Message):
    text = sanitize_input(message.text)
    if not text:
        return
    await _process_query(message, text)


async def _process_query(message: types.Message, query: str):
    """Process a query through RAG and send response."""
    try:
        rag = get_rag()
        user_id = str(message.from_user.id)
        result = await rag.query(query, user_id=user_id)

        response_text = result.answer
        if result.sources:
            sources_text = "\n".join([f"• {s['title']}" for s in result.sources])
            response_text += f"\n\n📚 <b>Sources:</b>\n{sources_text}"

        await message.answer(response_text)

    except Exception as e:
        await message.answer(
            f"❌ Error processing request.\n"
            f"<code>{str(e)[:200]}</code>"
        )


# ========== Webhook / Polling Setup ==========

def get_bot() -> Bot:
    """Create bot instance."""
    return Bot(token=settings.TELEGRAM_BOT_TOKEN, parse_mode=ParseMode.HTML)


async def setup_webhook(bot: Bot, base_url: str) -> None:
    """Set webhook for production."""
    webhook_url = f"{base_url.rstrip('/')}/bot/webhook"
    await bot.set_webhook(url=webhook_url, drop_pending_updates=True)


async def remove_webhook(bot: Bot) -> None:
    """Remove webhook."""
    await bot.delete_webhook(drop_pending_updates=True)


async def start_polling() -> None:
    """Start bot in polling mode (for local dev)."""
    bot = get_bot()
    await remove_webhook(bot)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
