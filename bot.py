import os
import logging
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not TELEGRAM_TOKEN or not DEEPSEEK_API_KEY:
    raise ValueError("❌ Установите TELEGRAM_TOKEN и DEEPSEEK_API_KEY")

client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
logging.basicConfig(level=logging.INFO)
user_histories = {}

async def call_deepseek(prompt: str, user_id: int) -> str:
    if user_id not in user_histories:
        user_histories[user_id] = []
    history = user_histories[user_id][-10:]
    messages = [{"role": "system", "content": "Ты — умный ИИ-агент. Отвечай кратко и по делу."}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.7, max_tokens=2000)
        reply = response.choices[0].message.content
        user_histories[user_id].append({"role": "user", "content": prompt})
        user_histories[user_id].append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"❌ Ошибка: {e}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Привет! Я ИИ-агент. Присылай текст!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(action="typing")
    response = await call_deepseek(update.message.text, update.effective_user.id)
    await update.message.reply_text(response)


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
