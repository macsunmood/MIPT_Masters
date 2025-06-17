#!/usr/bin/env python

import logging
import os
import requests
from dotenv import load_dotenv

from telegram import ForceReply, Update
from telegram.ext import (Application, CommandHandler, ContextTypes, 
                          MessageHandler, CallbackContext, filters)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
# Set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load variables from .env
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_OAUTH_TOKEN = os.getenv('YANDEX_OAUTH_TOKEN')
LLL_API_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'


def get_iam_token():
    # Get Yandex Cloud API IAM Token
    response = requests.post(
        'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        json={'yandexPassportOauthToken': YANDEX_OAUTH_TOKEN}
    )

    response.raise_for_status()
    return response.json()['iamToken']


# Define a few command handlers. These usually take the two arguments update and context
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def process_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text

    # Получаем IAM токен
    iam_token = get_iam_token()

    # Собираем запрос
    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
        "completionOptions": {
            "temperature": 0.3,
            "maxTokens": 1000
        },
        "messages": [{"role": "user", "text": user_text}]
    }

    # Отправляем запрос
    response = requests.post(
        LLL_API_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {iam_token}"
        },
        json=data,
        timeout=15
    ).json()

    # Распечатываем результат
    print(response)

    answer = response.get('result', {})\
                     .get('alternatives', [{}])[0]\
                     .get('message', {})\
                     .get('text', {})

    await update.message.reply_text(answer)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()
    logger.info('Bot successfully started!')

    # Message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
