import os
import logging
import base64
import io
import json
import random
import requests
import urllib3
import time
import re
from PIL import Image
import telebot
from docx import Document

from matrix_parser import extract_paragraph_full_text, format_matrix

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BOT_TOKEN = ""
OPENROUTER_API_KEY = ""

MODEL = "google/gemini-2.5-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_CONTEXT_CHARS = 6000
MAX_SEARCH_BLOCKS = 5

bot = telebot.TeleBot(BOT_TOKEN)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_FILE = "memory.json"
user_conversations = {}


def load_memory():
    global user_conversations
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            user_conversations = json.load(f)
    except:
        user_conversations = {}


def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(user_conversations, f, ensure_ascii=False, indent=2)



def load_docx_text():
    try:
        doc = Document("text.docx")
        lines = []
        for p in doc.paragraphs:
            text = extract_paragraph_full_text(p).strip()
            if text:
                lines.append(text)
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    lines.append(" | ".join(row_text))
        return "\n".join(lines)
    except FileNotFoundError:
        print("Файл text.docx не найден!")
        return ""
    except Exception as e:
        print(f"Ошибка загрузки docx: {e}")
        return ""


def parse_book_from_docx():
    try:
        doc = Document("text.docx")
    except Exception as e:
        print(f"Ошибка загрузки docx: {e}")
        return {"full_text": TEXT, "sections": [], "tasks": [], "examples": [], "answers": ""}

    paragraphs = doc.paragraphs
    sections = []
    tasks = []
    examples = []
    answers = ""

    current_section = None
    section_content_lines = []
    in_answers = False
    in_exercises = False

    def flush_section():
        nonlocal section_content_lines
        if current_section and section_content_lines:
            sections.append({
                "title": current_section,
                "content": "\n".join(section_content_lines)
            })
        section_content_lines = []

    def is_top_level_list(p):
        if p.style.name != "List Paragraph":
            return False
        pPr = p._element.pPr
        if pPr is None or pPr.numPr is None:
            return False
        ilvl = pPr.numPr.ilvl
        return ilvl is None or ilvl.val == 0

    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        text = extract_paragraph_full_text(p).strip()

        # ── Заголовки глав (Heading 1) ──────────────────────────
        if p.style.name == "Heading 1" and p.text.strip():
            flush_section()
            current_section = p.text.strip()
            in_exercises = False
            in_answers = "ответ" in current_section.lower()
            i += 1
            continue

        # ── Раздел ответов ──────────────────────────────────────
        if in_answers:
            if text:
                answers += text + "\n"
            i += 1
            continue

        # ── Маркер "Упражнения" ─────────────────────────────────
        if p.text.strip() == "Упражнения":
            in_exercises = True
            section_content_lines.append("\n[Упражнения]")
            i += 1
            continue

        # ── Параграфы внутри блока Упражнений ───────────────────
        if in_exercises and text:
            section_content_lines.append(text)

            if is_top_level_list(p):
                # Собираем всё условие: заголовок + уточнения из следующих строк
                task_lines = [text]
                j = i + 1
                while j < len(paragraphs) and j < i + 8:
                    np_ = paragraphs[j]
                    nt = extract_paragraph_full_text(np_).strip()
                    if not nt:
                        j += 1
                        continue
                    if is_top_level_list(np_) or np_.style.name == "Heading 1":
                        break
                    if np_.style.name == "List Paragraph" and nt:
                        task_lines.append(nt)
                    j += 1

                full_task = "\n".join(task_lines)
                if len(full_task.strip()) > 15:
                    label = current_section or "Упражнения"
                    tasks.append(f"[{label}]\n{full_task[:600]}")

            i += 1
            continue

        # ── Обычные параграфы (теория + раздел 9) ───────────────
        if text:
            section_content_lines.append(text)

            # Примеры с разбором
            if re.match(r'^Пример\s+\d+', text, re.IGNORECASE):
                ex_lines = [text]
                j = i + 1
                while j < len(paragraphs) and j < i + 20:
                    nt = extract_paragraph_full_text(paragraphs[j]).strip()
                    if nt and not re.match(r'^Пример\s+\d+', nt, re.IGNORECASE):
                        ex_lines.append(nt)
                        j += 1
                    else:
                        break
                examples.append("\n".join(ex_lines)[:800])

        i += 1

    flush_section()

    # ── Задачи из раздела 9 (Normal-параграфы, по Вариантам) ────
    for section in sections:
        if "самостоятельн" in section["title"].lower():
            lines = section["content"].split("\n")
            current_variant = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.match(r'^Вариант\s+\d+', line, re.IGNORECASE):
                    current_variant = line
                elif current_variant and len(line) > 20 and not any(
                    x in line.lower() for x in ["ответ", "решение", "список литературы"]
                ):
                    tasks.append(f"[{section['title']} — {current_variant}]\n{line[:500]}")

    tasks = list(dict.fromkeys(tasks))

    print(f"\nРезультаты парсинга:")
    print(f"  Разделов: {len(sections)}, Примеров: {len(examples)}, Задач: {len(tasks)}")
    if tasks:
        print("  Первые 3 задачи:")
        for t in tasks[:3]:
            print(f"    • {t[:80].replace(chr(10), ' ')}")

    return {
        "full_text": TEXT,
        "sections": sections,
        "tasks": tasks,
        "examples": examples,
        "answers": answers
    }


# Загружаем и парсим учебник
TEXT = load_docx_text()
BOOK = parse_book_from_docx()



def get_random_task():
    """Возвращает случайную задачу (без ответа)."""
    if not BOOK["tasks"]:
        return None
    return random.choice(BOOK["tasks"])


def format_task_for_telegram(raw_task):
    match = re.match(r'^\[([^\]]+)\]\n?(.*)', raw_task, re.DOTALL)
    if match:
        label = match.group(1)
        body = match.group(2).strip()
    else:
        label = ""
        body = raw_task.strip()

    # Разбиваем тело на фрагменты: обычный текст и матрицы
    # Матрица — это несколько строк подряд, начинающихся с ⎡ ⎢ ⎣ ( )
    matrix_line = r'[ \t]*[⎡⎢⎣()][^\n]*'
    matrix_block = r'(?:' + matrix_line + r'\n)+'  + matrix_line

    parts = re.split(r'(\n?' + matrix_block + r')', body)

    html_parts = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if re.search(r'[⎡⎢⎣]', stripped):
            html_parts.append(f"<pre>{stripped}</pre>")
        else:
            html_parts.append(stripped)

    body_html = "\n".join(html_parts) if html_parts else body

    header = f"<b>📘 {label}</b>\n\n" if label else ""
    footer = "\n\n<i>Реши и пришли своё решение — я проверю.\nИли напиши «покажи ответ».</i>"

    return header + body_html + footer


def find_relevant_context(query, max_chars=MAX_CONTEXT_CHARS):
    """Ищет наиболее релевантные разделы учебника по запросу."""
    if not BOOK["sections"]:
        return BOOK["full_text"][:max_chars] if BOOK["full_text"] else "Учебник не загружен"

    query_lower = query.lower()
    query_words = set(query_lower.split()) - {
        'это', 'как', 'что', 'для', 'на', 'в', 'с', 'по', 'из', 'за',
        'у', 'о', 'к', 'и', 'а', 'но', 'да', 'нет', 'или', 'же', 'мне',
        'мой', 'моя', 'дай', 'дайте', 'пожалуйста'
    }

    scored = []
    for section in BOOK["sections"]:
        combined = f"{section['title']}\n{section['content']}".lower()
        score = sum(1 for w in query_words if w in combined)
        if query_lower[:20] in combined:
            score += 5
        if score > 0:
            scored.append((score, section))

    scored.sort(key=lambda x: x[0], reverse=True)

    context_parts = []
    total = 0
    for _, section in scored[:MAX_SEARCH_BLOCKS]:
        block = f"{section['title']}\n\n{section['content'][:2000]}"
        if total + len(block) > max_chars:
            break
        context_parts.append(block)
        total += len(block)

    if context_parts:
        return "\n\n---\n\n".join(context_parts)

    if BOOK["sections"]:
        s = BOOK["sections"][0]
        return f"{s['title']}\n\n{s['content'][:max_chars]}"

    return BOOK["full_text"][:max_chars] if BOOK["full_text"] else "Учебник не загружен"


SYSTEM_PROMPT = """Ты преподаватель по тензорному исчислению. Работаешь по конкретному учебному пособию.

ВАЖНО — краткость:
- Максимум 3-4 абзаца или 5-6 пунктов.
- Для простых вопросов (определение, термин) — 2-3 предложения.
- Для задач — пошаговое решение без лишних предисловий.
- Не перечисляй всё из контекста подряд.

Содержание:
- Основывайся на переданном контексте из учебника.
- Можно дополнить общематематическими фактами там, где контекст неполный.
- Если темы нет совсем — скажи коротко.

Форматирование (Telegram HTML):
- Заголовки: <b>текст</b>
- Курсив: <i>текст</i>
- Матрицы/формулы: <pre>текст</pre>
- Списки: дефис или цифры. Без markdown (* # ` и т.п.)"""


def clean_response(text):
    """Конвертирует markdown в Telegram HTML."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^#{1,3}\s+(.+)', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'```[\w]*\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()




def ask(messages, max_tokens=500, retry_count=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/tensor_bot",
        "X-OpenRouter-Title": "Tensor Assistant"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.4
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=90, verify=False)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    return content if content else "Модель вернула пустой ответ"
                return "Неожиданный формат ответа"
            else:
                if attempt < retry_count - 1:
                    time.sleep(2)
                    continue
                return f"Ошибка API ({response.status_code}): {response.text[:200]}"
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return f"Ошибка запроса: {str(e)[:100]}"

    return "Не удалось получить ответ."



def is_request_task(text):
    t = text.lower()
    return any(x in t for x in [
        "дай задачу", "пришли задачу", "хочу задачу", "ещё задачу", "еще задачу",
        "задачу пожалуйста", "дайте задачу", "следующую задачу", "новую задачу",
        "дай другую задачу", "другую задачу", "задачу"
    ])


def is_request_answer(text):
    t = text.lower()
    return any(x in t for x in [
        "покажи ответ", "дай ответ", "какой ответ", "покажи решение",
        "проверь", "проверить", "правильно ли", "верно ли",
        "мой ответ", "моё решение", "я решил", "я получил",
        "у меня получилось", "у меня вышло", "я считаю", "ответ"
    ])


def process_text(user_id, text):
    conversation = user_conversations.get(str(user_id), [])
    if is_request_task(text):
        task = get_random_task()
        if not task:
            return "Задачи не найдены в учебнике. Проверьте файл text.docx"

        conversation.append({"role": "user", "content": text})
        conversation.append({"role": "assistant", "content": f"[ЗАДАЧА]{task}"})
        if len(conversation) > 20:
            conversation = conversation[-20:]
        user_conversations[str(user_id)] = conversation
        save_memory()
        return format_task_for_telegram(task)

    if is_request_answer(text):
        last_task = None
        for msg in reversed(conversation):
            if msg["role"] == "assistant" and msg["content"].startswith("[ЗАДАЧА]"):
                last_task = msg["content"][len("[ЗАДАЧА]"):].strip()
                break

        if last_task:
            context = find_relevant_context(last_task)
            wants_solution = any(x in text.lower() for x in [
                "покажи ответ", "дай ответ", "какой ответ", "покажи решение"
            ])
            if wants_solution:
                user_prompt = f"Задача:\n{last_task}\n\nДай пошаговое решение."
                max_tok = 800
            else:
                user_prompt = (
                    f"Задача:\n{last_task}\n\n"
                    f"Ответ студента:\n{text}\n\n"
                    f"Коротко: верно или нет? Если есть ошибки — укажи."
                )
                max_tok = 500
        else:
            context = find_relevant_context(text)
            user_prompt = text
            max_tok = 500

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation[-6:],
            {"role": "user", "content": f"Контекст из учебника:\n{context}\n\n---\n\n{user_prompt}"}
        ]
        final = clean_response(ask(messages, max_tokens=max_tok))
        conversation.append({"role": "user", "content": text})
        conversation.append({"role": "assistant", "content": final})
        if len(conversation) > 20:
            conversation = conversation[-20:]
        user_conversations[str(user_id)] = conversation
        save_memory()
        return final

    context = find_relevant_context(text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation[-6:],
        {
            "role": "user",
            "content": f"Контекст из учебника:\n{context}\n\n---\n\nВопрос: {text}\n\nОтветь кратко."
        }
    ]

    final = clean_response(ask(messages, max_tokens=500))

    conversation.append({"role": "user", "content": text})
    conversation.append({"role": "assistant", "content": final})
    if len(conversation) > 20:
        conversation = conversation[-20:]
    user_conversations[str(user_id)] = conversation
    save_memory()
    return final


def process_image(user_id, image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.width > 1024 or img.height > 1024:
            ratio = min(1024 / img.width, 1024 / img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        context = find_relevant_context("задача тензор", max_chars=2000)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Реши задачу с изображения. Контекст:\n{context}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
        return clean_response(ask(messages, max_tokens=800))
    except Exception as e:
        return f"Ошибка при обработке изображения: {str(e)[:100]}"


def send_reply(message, text):
    """Отправляет ответ с HTML, при ошибке — plain text."""
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            bot.reply_to(message, chunk, parse_mode='HTML')
        except Exception:
            plain = re.sub(r'<[^>]+>', '', chunk)
            bot.reply_to(message, plain)




@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message,
        f"<b>Привет! Я преподаватель по тензорам</b>\n\n"
        f"Модель: Google Gemini 2.5 Flash\n\n"
        f"<b>Статистика учебника:</b>\n"
        f"• Разделов: {len(BOOK['sections'])}\n"
        f"• Задач: {len(BOOK['tasks'])}\n\n"
        f"<b>Команды:</b>\n"
        f"/task — случайная задача\n"
        f"/sections — разделы учебника\n"
        f"/help — справка\n"
        f"/reset — сбросить контекст\n"
        f"/debug — отладочная информация",
        parse_mode='HTML'
    )


@bot.message_handler(commands=['help'])
def help_command(message):
    bot.reply_to(
        message,
        "<b>Как работать с ботом:</b>\n\n"
        "1. Задай вопрос по теории: <i>Что такое тензор?</i>\n"
        "2. Попроси решить задачу: <i>Найти тензорное произведение...</i>\n"
        "3. /task — получить задачу из учебника\n"
        "4. Пришли фото с задачей — разберу\n"
        "5. После получения задачи:\n"
        "   • пришли своё решение — проверю\n"
        "   • напиши «покажи ответ» — дам решение\n\n"
        "/reset — очистить историю диалога",
        parse_mode='HTML'
    )


@bot.message_handler(commands=['reset'])
def reset(message):
    uid = str(message.from_user.id)
    if uid in user_conversations:
        del user_conversations[uid]
        save_memory()
        bot.reply_to(message, "Контекст сброшен.")
    else:
        bot.reply_to(message, "Контекст и так пуст.")


@bot.message_handler(commands=['task'])
def send_task(message):
    bot.send_chat_action(message.chat.id, 'typing')
    task = get_random_task()
    if not task:
        bot.reply_to(message, "Задачи не найдены. Проверьте файл text.docx")
        return

    uid = str(message.from_user.id)
    conv = user_conversations.get(uid, [])
    conv.append({"role": "user", "content": "/task"})
    conv.append({"role": "assistant", "content": f"[ЗАДАЧА]{task}"})
    if len(conv) > 20:
        conv = conv[-20:]
    user_conversations[uid] = conv
    save_memory()

    send_reply(message, format_task_for_telegram(task))


@bot.message_handler(commands=['sections'])
def show_sections(message):
    if not BOOK["sections"]:
        bot.reply_to(message, "Разделы не найдены. Проверьте файл text.docx")
        return
    lines = ["<b>Разделы учебника:</b>\n"]
    for i, s in enumerate(BOOK["sections"][:20], 1):
        lines.append(f"{i}. {s['title'][:60]}")
    if len(BOOK["sections"]) > 20:
        lines.append(f"\n...и ещё {len(BOOK['sections']) - 20}")
    bot.reply_to(message, "\n".join(lines), parse_mode='HTML')


@bot.message_handler(commands=['debug'])
def debug(message):
    info = (
        f"<b>Отладка:</b>\n\n"
        f"Текст загружен: {'да' if TEXT else 'нет'}\n"
        f"Разделов: {len(BOOK['sections'])}\n"
        f"Задач: {len(BOOK['tasks'])}\n"
        f"Примеров: {len(BOOK['examples'])}\n"
    )
    if BOOK["sections"]:
        info += "\n<b>Первые 5 разделов:</b>\n"
        for s in BOOK["sections"][:5]:
            info += f"• {s['title'][:50]}\n"
    if BOOK["tasks"]:
        info += "\n<b>Первые 3 задачи:</b>\n"
        for t in BOOK["tasks"][:3]:
            info += f"• {t[:60].replace(chr(10), ' ')}...\n"
    bot.reply_to(message, info, parse_mode='HTML')


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        bot.reply_to(message, "Анализирую изображение...")
        file_info = bot.get_file(message.photo[-1].file_id)
        file_bytes = bot.download_file(file_info.file_path)
        answer = process_image(message.from_user.id, file_bytes)
        send_reply(message, answer)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)[:100]}")
        logger.error(f"Photo error: {e}")


@bot.message_handler(func=lambda m: True)
def handle_text(message):
    if message.text.startswith("/"):
        return
    bot.send_chat_action(message.chat.id, 'typing')
    answer = process_text(message.from_user.id, message.text)
    send_reply(message, answer)



if __name__ == "__main__":
    load_memory()

    print("БОТ ЗАПУЩЕН")

    print(f"Модель: {MODEL}")
    print(f"Учебник: {'загружен' if TEXT else 'НЕ НАЙДЕН'}")
    print(f"Разделов: {len(BOOK['sections'])}, Задач: {len(BOOK['tasks'])}")
    if not TEXT:
        print("Файл с пособием не найден!")
    try:
        bot.infinity_polling(timeout=60)
    except KeyboardInterrupt:
        print("\nБот остановлен")
    except Exception as e:
        print(f"Ошибка: {e}")