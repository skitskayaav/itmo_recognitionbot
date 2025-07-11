import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import URLInputFile
import pandas as pd
import asyncio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = "ur token"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

EXCEL_FILE = "tasks.xlsx"


async def load_tasks():
    try:
        dtype_col = {
            "ID": str,
            "Условие": str
        }
        tasks_df = pd.read_excel(EXCEL_FILE, dtype=dtype_col)
        return tasks_df
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        return pd.DataFrame(columns=["ID", "Условие"])


def convert_yandex_link(original_url: str) -> str:
    """Преобразует ссылку Яндекс.Диска в прямую для скачивания"""
    if not isinstance(original_url, str):
        return ""

    try:
        if "yadi.sk" in original_url:
            file_id = original_url.split("/d/")[1].split("/")[0]
        elif "disk.yandex.ru" in original_url:
            if "/i/" in original_url:
                file_id = original_url.split("/i/")[1].split("/")[0]
            else:
                file_id = original_url.split("/d/")[1].split("/")[0]
        else:
            return original_url  # Если ссылка уже прямая

        return f"https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/{file_id}"
    except Exception as e:
        logger.error(f"Ошибка преобразования ссылки: {e}")
        return ""


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    tasks_df = await load_tasks()

    builder = InlineKeyboardBuilder()

    for _, row in tasks_df.iterrows():
        builder.add(types.InlineKeyboardButton(
            text=f"Задача {row['ID']}",
            callback_data=row['ID'])
        )

    builder.adjust(3)
    await message.answer("Выберите задачу:", reply_markup=builder.as_markup())


@dp.callback_query()
async def process_task_selection(callback: types.CallbackQuery):
    task_id = callback.data
    tasks_df = await load_tasks()

    try:
        task_row = tasks_df[tasks_df["ID"] == task_id].iloc[0]
        condition_url = task_row["Условие"]

        if not condition_url:
            await callback.message.answer("У задачи нет условия")
            return


        direct_url = convert_yandex_link(condition_url)
        if direct_url:
            try:
                image = URLInputFile(direct_url)
                await callback.message.answer_photo(
                    image,
                    caption=f"Условие задачи {task_id}"
                )
            except Exception as e:
                logger.error(f"Ошибка отправки изображения: {e}")
                await callback.message.answer(
                    f"Условие задачи {task_id} (не удалось загрузить изображение):\n{condition_url}"
                )
        else:
            await callback.message.answer(f"Условие задачи {task_id}:\n{condition_url}")

    except IndexError:
        await callback.message.answer("Задача не найдена!")
    except Exception as e:
        logger.error(f"Ошибка обработки задачи: {e}")
        await callback.message.answer("Произошла ошибка при обработке задачи")

    await callback.answer()


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())