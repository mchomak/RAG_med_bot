#!/usr/bin/env python3
"""
RAG Medical Bot - MVP версия
Консольное приложение для работы с медицинской документацией через RAG

Использование:
    python main.py

Команды:
    /exit - выход из программы
    /reload - переиндексация документов
    /help - справка по командам
    /sources - показать источники последнего ответа
    /clear - очистить экран
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box

import config
from rag_engine import RAGEngine

console = Console()


def print_banner():
    """Вывод приветственного баннера"""
    banner_text = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🏥 RAG Medical Bot - MVP версия 🏥             ║
    ║                                                           ║
    ║      Интеллектуальный ассистент для работы с              ║
    ║           медицинской документацией                       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner_text, style="bold cyan")
    console.print("\n💡 Введите /help для списка доступных команд\n", style="yellow")


def print_help():
    """Вывод справки по командам"""
    table = Table(title="Доступные команды", box=box.ROUNDED)

    table.add_column("Команда", style="cyan", no_wrap=True)
    table.add_column("Описание", style="white")

    for cmd, description in config.COMMANDS.items():
        table.add_row(f"/{cmd}", description)

    console.print(table)
    console.print()


def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_sources(sources):
    """
    Вывод источников информации

    Args:
        sources: Список источников
    """
    if not sources:
        console.print("ℹ Источники не найдены", style="yellow")
        return

    console.print(f"\n📚 Источники информации ({len(sources)}):", style="bold magenta")

    for i, source in enumerate(sources, 1):
        panel_content = f"""
**Файл:** {source['file_name']}
**Релевантность:** {source['score']:.2%}

**Фрагмент текста:**
{source['text_preview']}
        """

        console.print(
            Panel(
                panel_content.strip(),
                title=f"Источник {i}",
                border_style="magenta",
                box=box.ROUNDED
            )
        )


def get_documents_folder() -> Optional[Path]:
    """
    Запросить у пользователя путь к папке с документами

    Returns:
        Path к папке или None
    """
    while True:
        console.print("\n📁 Укажите путь к папке с документами (txt, pdf, docx):", style="bold cyan")
        folder_path = Prompt.ask("Путь к папке", default="./documents")

        folder_path = Path(folder_path).expanduser().resolve()

        # Валидация пути
        if not folder_path.exists():
            console.print(f"✗ Папка не существует: {folder_path}", style="red")
            retry = Confirm.ask("Попробовать снова?", default=True)
            if not retry:
                return None
            continue

        if not folder_path.is_dir():
            console.print(f"✗ Путь не является папкой: {folder_path}", style="red")
            retry = Confirm.ask("Попробовать снова?", default=True)
            if not retry:
                return None
            continue

        # Проверяем наличие поддерживаемых файлов
        all_files = []
        for extension in config.SUPPORTED_FILE_EXTENSIONS:
            all_files.extend(folder_path.glob(f"*{extension}"))

        if not all_files:
            supported_formats = ", ".join(config.SUPPORTED_FILE_EXTENSIONS)
            console.print(f"✗ В папке нет поддерживаемых файлов ({supported_formats}): {folder_path}", style="red")
            retry = Confirm.ask("Попробовать снова?", default=True)
            if not retry:
                return None
            continue

        # Показываем статистику по типам файлов
        file_types_count = {}
        for file_path in all_files:
            ext = file_path.suffix.lower()
            file_types_count[ext] = file_types_count.get(ext, 0) + 1

        stats_str = ", ".join([f"{ext}: {count}" for ext, count in file_types_count.items()])
        console.print(f"✓ Найдено файлов: {len(all_files)} ({stats_str})", style="green")
        return folder_path


def handle_query(engine: RAGEngine, question: str):
    """
    Обработка вопроса пользователя

    Args:
        engine: RAG Engine
        question: Вопрос
    """
    # Засекаем время
    start_time = time.time()

    console.print(f"\n🤔 Ищу ответ...", style="cyan")

    # Выполняем запрос
    result = engine.query(question)

    # Время обработки
    elapsed_time = time.time() - start_time

    # Выводим ответ
    if result["error"]:
        console.print(
            Panel(
                result["answer"],
                title="❌ Ошибка",
                border_style="red",
                box=box.ROUNDED
            )
        )
    else:
        console.print(
            Panel(
                Markdown(result["answer"]),
                title="💬 Ответ",
                border_style="green",
                box=box.ROUNDED
            )
        )

        # Выводим источники, если они есть и настройка включена
        if config.SHOW_SOURCES and result["sources"]:
            print_sources(result["sources"])

    # Время обработки
    console.print(f"\n⏱ Время обработки: {elapsed_time:.2f} сек", style="dim")


def interactive_loop(engine: RAGEngine):
    """
    Основной интерактивный цикл

    Args:
        engine: RAG Engine
    """
    console.print("\n🚀 Готов к работе! Задавайте вопросы по документам.\n", style="bold green")

    while True:
        try:
            # Получаем вопрос от пользователя
            question = Prompt.ask("\n[bold cyan]Ваш вопрос[/bold cyan]")

            # Пустой ввод - пропускаем
            if not question.strip():
                continue

            # Обработка команд
            if question.startswith("/"):
                command = question[1:].lower().strip()

                if command == "exit" or command == "quit":
                    console.print("\n👋 До свидания!", style="bold yellow")
                    break

                elif command == "help":
                    print_help()
                    continue

                elif command == "clear":
                    clear_screen()
                    print_banner()
                    continue

                elif command == "sources":
                    sources = engine.get_sources()
                    if sources:
                        print_sources(sources)
                    else:
                        console.print("ℹ Источников нет. Сначала задайте вопрос.", style="yellow")
                    continue

                elif command == "reload":
                    console.print("\n🔄 Переиндексация документов...", style="cyan")
                    folder_path = get_documents_folder()
                    if folder_path:
                        success = engine.index_documents(str(folder_path))
                        if success:
                            console.print("✓ Переиндексация завершена!", style="green")
                        else:
                            console.print("✗ Ошибка при переиндексации", style="red")
                    continue

                else:
                    console.print(f"❓ Неизвестная команда: /{command}", style="yellow")
                    console.print("Введите /help для списка команд", style="dim")
                    continue

            # Обрабатываем вопрос
            handle_query(engine, question)

        except KeyboardInterrupt:
            console.print("\n\n👋 Прервано пользователем. До свидания!", style="bold yellow")
            break
        except EOFError:
            console.print("\n\n👋 До свидания!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"\n❌ Неожиданная ошибка: {str(e)}", style="red")
            continue


def main():
    """Главная функция приложения"""
    try:
        # Очищаем экран и показываем баннер
        clear_screen()
        print_banner()

        # Инициализация RAG Engine
        console.print("⚙ Инициализация системы...\n", style="bold cyan")

        try:
            engine = RAGEngine(verbose=True)
        except Exception as e:
            console.print(
                Panel(
                    f"Не удалось инициализировать RAG Engine.\n\n"
                    f"Возможные причины:\n"
                    f"1. Ollama не запущена (запустите: ollama serve)\n"
                    f"2. Модель {config.OLLAMA_MODEL} не загружена (загрузите: ollama pull {config.OLLAMA_MODEL})\n"
                    f"3. Не установлены зависимости (установите: pip install -r requirements.txt)\n\n"
                    f"Ошибка: {str(e)}",
                    title="❌ Ошибка инициализации",
                    border_style="red",
                    box=box.ROUNDED
                )
            )
            return 1

        # Проверяем наличие существующего индекса
        has_existing_index = engine.load_index()

        if has_existing_index:
            console.print(
                Panel(
                    f"✓ Найден существующий индекс\n"
                    f"Документов: {engine.document_count}",
                    title="📊 Загрузка индекса",
                    border_style="green",
                    box=box.ROUNDED
                )
            )

            # Спрашиваем, использовать ли существующий индекс
            use_existing = Confirm.ask(
                "\nИспользовать существующий индекс?",
                default=True
            )

            if not use_existing:
                has_existing_index = False

        # Если нет индекса или пользователь хочет переиндексировать
        if not has_existing_index:
            folder_path = get_documents_folder()

            if folder_path is None:
                console.print("❌ Работа программы прервана", style="red")
                return 1

            # Индексируем документы
            console.print("\n📑 Начинаю индексацию документов...\n", style="bold cyan")
            success = engine.index_documents(str(folder_path))

            if not success:
                console.print("❌ Не удалось проиндексировать документы", style="red")
                return 1

        # Показываем статистику
        stats = engine.get_stats()
        stats_table = Table(title="Статистика системы", box=box.ROUNDED)
        stats_table.add_column("Параметр", style="cyan")
        stats_table.add_column("Значение", style="white")

        stats_table.add_row("Документов в индексе", str(stats["documents_count"]))
        stats_table.add_row("LLM модель", stats["model"])
        stats_table.add_row("Эмбеддинг модель", stats["embedding_model"])
        stats_table.add_row("Хранилище", stats["storage_path"])

        console.print("\n")
        console.print(stats_table)

        # Запускаем интерактивный цикл
        interactive_loop(engine)

        return 0

    except Exception as e:
        console.print(f"\n❌ Критическая ошибка: {str(e)}", style="bold red")
        import traceback
        console.print(traceback.format_exc(), style="dim")
        return 1


if __name__ == "__main__":
    sys.exit(main())