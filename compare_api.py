import asyncio
import json
import logging
import os
import time
from typing import Any

from run_batch import SYSTEM_PROMPT, parse_and_validate

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("compare_api")

# Условные цены эксперимента, рубли
VM_T4_PRICE_PER_HOUR_RUB = 65.0
INPUT_PRICE_PER_MILLION_RUB = 20.0
OUTPUT_PRICE_PER_MILLION_RUB = 60.0

API_BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

MAX_ATTEMPTS = 3
BASE_DELAY = 2.0
MAX_DELAY = 120.0
REQUEST_TIMEOUT = 60.0
TEMPERATURE = 0.3
MAX_TOKENS = 350
PAUSE_BETWEEN_PRODUCTS = 0.5
PAUSE_BETWEEN_RETRIES = 1.0


def calculate_retry_delay(
    attempt: int,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
) -> float:
    """Экспоненциальная задержка с ограничением сверху."""
    return min(base_delay * 2 ** (attempt - 1), max_delay)


def get_retry_after(error: Exception) -> float | None:
    """Получить время ожидания из ответа."""
    try:
        value = error.response.headers.get("Retry-After")
        if value is not None:
            return max(0.0, float(value))
    except (AttributeError, TypeError, ValueError):
        pass

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", {})
        metadata = details.get("metadata", {})

        for source in (metadata, details):
            value = source.get("retry_after_seconds")
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass

        headers = metadata.get("headers", {})
        reset = headers.get("X-RateLimit-Reset")

        if reset is not None:
            try:
                reset_timestamp = float(reset)
                if reset_timestamp > 10_000_000_000:
                    reset_timestamp /= 1000.0

                return max(0.0, reset_timestamp - time.time())
            except (TypeError, ValueError):
                pass

    return None


def create_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )


def create_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.1,
        },
    )


async def run_api_comparison(
    sample_size: int = 20,
) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Переменная окружения DEEPSEEK_API_KEY не задана"
        )

    if sample_size < 1:
        raise ValueError("sample_size должен быть не меньше 1")

    client = create_client(api_key)

    items = []
    with open("dev.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))

    items = items[:sample_size]

    if not items:
        raise RuntimeError("В dev.jsonl нет товаров для обработки")

    logger.info(
        "Запуск эксперимента: модель=%s, товаров=%d, max_tokens=%d",
        MODEL,
        len(items),
        MAX_TOKENS,
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0
    valid_count = 0
    failed_count = 0
    api_error_count = 0
    validation_error_count = 0
    total_attempts = 0

    failure_details: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for item_index, item in enumerate(items):
        expected_id = item["product_id"]
        user_content = json.dumps(item, ensure_ascii=False)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Обращение пользователя:\n{user_content}",
            },
        ]

        succeeded = False
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            total_attempts += 1
            content: str | None = None

            logger.info(
                "Товар %s: попытка %d/%d, модель=%s",
                expected_id,
                attempt,
                MAX_ATTEMPTS,
                MODEL,
            )

            try:
                response = await asyncio.to_thread(
                    create_chat_completion,
                    client,
                    MODEL,
                    messages,
                )

                choice = response.choices[0]
                content = choice.message.content or ""

                if response.usage:
                    total_prompt_tokens += (
                        response.usage.prompt_tokens
                    )
                    total_completion_tokens += (
                        response.usage.completion_tokens
                    )

                logger.info(
                    "Товар=%s, finish_reason=%s",
                    expected_id,
                    choice.finish_reason,
                )

                if choice.finish_reason == "length":
                    raise ValueError(
                        f"Ответ обрезан из-за ограничения max_tokens={MAX_TOKENS}"
                    )

                if not content.strip():
                    raise ValueError("Модель вернула пустой ответ")

                parse_and_validate(content, expected_id)

                valid_count += 1
                succeeded = True

                logger.info(
                    "Товар %s прошёл валидацию на попытке %d",
                    expected_id,
                    attempt,
                )
                break

            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
                APIStatusError,
            ) as error:
                api_error_count += 1
                last_error = error

                status_code = getattr(error, "status_code", None)

                retryable = (
                    isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError))
                    or 
          			(status_code is not None and (status_code == 429 or status_code >= 500))
                )

                if not retryable or attempt == MAX_ATTEMPTS:
                    logger.warning(
                        "Запрос API завершился ошибкой: товар=%s, "
                        "код=%s, ошибка=%s",
                        expected_id,
                        status_code,
                        error,
                    )
                    break

                retry_after = get_retry_after(error)
                backoff = calculate_retry_delay(attempt)

                delay = max(backoff, retry_after if retry_after is not None else 0.0)

                logger.warning(
                    "Временная ошибка API: товар=%s, код=%s. "
                    "Повтор %d/%d через %.1f сек.",
                    expected_id,
                    status_code,
                    attempt + 1,
                    MAX_ATTEMPTS,
                    delay,
                )

                await asyncio.sleep(delay)
                continue

            except Exception as error:
                validation_error_count += 1
                last_error = error

                logger.warning(
                    "Ошибка ответа/валидации: товар=%s, попытка=%d: %s",
                    expected_id,
                    attempt,
                    error,
                )

                if attempt == MAX_ATTEMPTS:
                    break

                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                    })

                messages.append({
                    "role": "user",
                    "content": (
                        "Предыдущий ответ не прошёл проверку. "
						f"ОШИБКА: {error}."
						f"Поле product_id оставь строго '{json.dumps(expected_id, ensure_ascii=False)}'. "
						f"Исправь ТОЛЬКО эту ошибку и верни исправленный валидный JSON без лишнего текста."
                    ),
                })

                await asyncio.sleep(PAUSE_BETWEEN_RETRIES)

        if not succeeded:
            failed_count += 1

            failure_details.append({
                "product_id": expected_id,
                "error": (
                    str(last_error)
                    if last_error
                    else "Unknown error"
                ),
                "model": MODEL,
            })

            logger.error(
                "Товар %s не обработан за %d попытки",
                expected_id,
                MAX_ATTEMPTS,
            )

        if item_index < len(items) - 1:
            await asyncio.sleep(PAUSE_BETWEEN_PRODUCTS)

    elapsed_sec = time.perf_counter() - start_time
    n = len(items)
    valid_rate = round(valid_count / n, 3) if n else 0.0

    cost_per_1000 = (
        (total_prompt_tokens * INPUT_PRICE_PER_MILLION_RUB + total_completion_tokens * OUTPUT_PRICE_PER_MILLION_RUB)
        / 1_000_000
        / n
        * 1000
        if n
        else 0.0
    )

    api_metrics = {
        "model": MODEL,
        "n": n,
        "valid_rate": valid_rate,
        "input_tokens": total_prompt_tokens,
        "output_tokens": total_completion_tokens,
        "elapsed_sec": round(elapsed_sec, 2),
        "cost_per_1000": round(cost_per_1000, 2),
    }

    report_path = "outputs/report.json"
    os.makedirs("outputs", exist_ok=True)

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as file:
            report_data = json.load(file)
    else:
        report_data = {}

    report_data["api_comparison"] = api_metrics

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Эксперимент завершён: успешно=%d, ошибок=%d, "
        "ошибок API=%d, ошибок валидации=%d, попыток=%d",
        valid_count,
        failed_count,
        api_error_count,
        validation_error_count,
        total_attempts,
    )

    return api_metrics
