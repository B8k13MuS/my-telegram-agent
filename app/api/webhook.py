"""Telegram webhook endpoint for production deployments.

Used instead of polling on Render, VPS, and other server environments.
Webhook is more efficient: Telegram pushes updates to your server
instead of the bot constantly asking "any news?"
"""

from typing import Optional

from aiogram import Bot
from fastapi import APIRouter, Request, HTTPException, Header

from app.bot.handlers import dp, get_bot
from app.core.config import settings

router = APIRouter(prefix="/bot", tags=["telegram"])

_bot: Optional[Bot] = None


def _get_bot() -> Bot:
    """Lazy bot initialization."""
    global _bot
    if _bot is None:
        _bot = get_bot()
    return _bot


@router.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    """Receive updates from Telegram."""
    try:
        update_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    bot = _get_bot()
    from aiogram.types import Update
    update = Update.model_validate(update_data)
    await dp.feed_update(bot=bot, update=update)

    return {"ok": True}


@router.get("/webhook/info")
async def webhook_info():
    """Get current webhook status from Telegram."""
    bot = _get_bot()
    info = await bot.get_webhook_info()
    return {
        "url": info.url,
        "has_custom_certificate": info.has_custom_certificate,
        "pending_update_count": info.pending_update_count,
    }
