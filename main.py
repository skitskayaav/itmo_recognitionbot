import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import URLInputFile, FSInputFile
from aiogram import F
import pandas as pd
import asyncio
from pix2text import Pix2Text
from ultralytics import YOLO
import cv2

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = "ur token"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

EXCEL_FILE = "tasks.xlsx"

model = YOLO("yolov8_h3.pt")


TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


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
            return original_url

        return f"https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/{file_id}"
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


def visualize_detections(image_path, results, output_path):
    """Визуализирует обнаруженные формулы на изображении"""
    image = cv2.imread(image_path)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Получаем координаты bounding box'ов
        confidences = result.boxes.conf.cpu().numpy()  

        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)


p2t = Pix2Text.from_config()


def recognize_formula(image_path):
    """Распознает формулу на изображении с помощью P2T"""
    try:
        result = p2t.recognize_formula(image_path, resized_shape=600)  # Увеличиваем размер для лучшего качества
        return result.text
    except Exception as e:
        logger.error(f"Ошибка распознавания формулы: {e}")
        return None


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    try:
        # 1. Скачиваем изображение
        photo = message.photo[-1]
        file_id = photo.file_id
        file = await bot.get_file(file_id)
        file_path = f"{TEMP_DIR}/{file_id}.jpg"
        await bot.download_file(file.file_path, file_path)

        # 2. Детекция формул
        results = model(file_path)
        conf_threshold = 0.75

        # 3. Сохраняем каждый box отдельно
        boxes_dir = f"{TEMP_DIR}/{file_id}_boxes"
        box_paths = save_individual_boxes(file_path, results, boxes_dir, conf_threshold)

        if not box_paths:
            await message.reply(f"Формулы с уверенностью ≥{conf_threshold} не найдены")
            os.remove(file_path)
            return

        # 4. Отправляем оригинал с разметкой
        output_path = f"{TEMP_DIR}/{file_id}_output.jpg"
        visualize_detections(file_path, results, output_path, conf_threshold)
        await message.reply_photo(
            FSInputFile(output_path),
            caption="Результаты обнаружения формул:"
        )

        # 5. Обрабатываем и отправляем каждый box
        for i, box_path in enumerate(box_paths):
            latex_code = recognize_formula(box_path)

            if latex_code:
                
                await message.answer_photo(
                    FSInputFile(box_path),
                    caption=f"Формула {i + 1}"
                )

                # Отправляем LaTeX код
                await message.answer(
                    f"LaTeX код формулы {i + 1}:\n"
                    f"```latex\n{latex_code}\n```\n",
                    parse_mode="Markdown"
                )
            else:
                await message.answer(
                    f"Не удалось распознать формулу {i + 1}",
                    reply_to_message_id=message.message_id
                )

            os.remove(box_path)

        os.remove(file_path)
        os.remove(output_path)
        os.rmdir(boxes_dir)

    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")
        await message.reply("Произошла ошибка при обработке изображения")


        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        if 'boxes_dir' in locals() and os.path.exists(boxes_dir):
            for f in os.listdir(boxes_dir):
                os.remove(f"{boxes_dir}/{f}")
            os.rmdir(boxes_dir)

@dp.message(F.document)
async def handle_documents(message: types.Message):
    if message.document.mime_type.startswith('image/'):
        await message.reply("Вы прислали изображение как файл")
    else:
        await message.reply("Пожалуйста, пришлите изображение")


@dp.message()
async def handle_non_photos(message: types.Message):
    await message.reply("Пожалуйста, пришлите фото")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())