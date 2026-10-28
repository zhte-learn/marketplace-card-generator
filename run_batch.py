import json
import logging
import os
import re
import asyncio, time
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import Any
from openai import AsyncOpenAI
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("json_reliability")


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

MAX_ATTEMPTS = 3
MAX_INPUT_TOKENS = 1500
TEMPERATURE = 0.3
MAX_TOKENS = 350
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

MIN_DESCRIPTION_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 700
MIN_DESCRIPTION_SENTENCES = 2
MAX_DESCRIPTION_SENTENCES = 5
MIN_PROS_COUNT = 2
MAX_PROS_COUNT = 5
MIN_CONS_COUNT = 1
MAX_CONS_COUNT = 3
MIN_TAGS_COUNT = 3
MAX_TAGS_COUNT = 8

EXPECTED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags"
}

Message = dict[str, str]
RequestFn = Callable[[list[Message]], str]

SYSTEM_PROMPT = f"""
Ты — генератор карточек товаров для маркетплейса. Твоя задача — вернуть СТРОГО один валидный JSON-объект по шаблону ниже.

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
1. "product_id": скопируй ТОЧНО из запроса пользователя (без изменений и префиксов).
2. "description": пиши описание ровно в таком же объеме, как в примере (2-3 предложения средней длины, без длинных списков).
3. "pros": массив строк, содержащий от {MIN_PROS_COUNT} до {MAX_PROS_COUNT} достоинств.
4. "cons": массив строк, содержащий от {MIN_CONS_COUNT} до {MAX_CONS_COUNT} недостатков (ОБЯЗАТЕЛЬНО укажи хотя бы 1 минус!).
5. "tags": массив строк, содержащий от {MIN_TAGS_COUNT} до {MAX_TAGS_COUNT} тегов.
6. ВЕРНИ ТОЛЬКО JSON. Не добавляй никакого текста, пояснений или markdown-разметки вокруг него.

ПРИМЕР ОПИСАНИЯ И СТРУКТУРЫ:
{{
  "product_id": "P-1001",
  "description": "Беспроводная мышь с эргономичным корпусом и оптическим сенсором. Она подключается по радиоканалу и стабильно работает от одной батарейки.",
  "pros": ["Удобная форма", "Надежная связь"],
  "cons": ["Быстро садится батарейка"],
  "tags": ["мышь", "беспроводная", "периферия"]
}}
""".strip()


def count_tokens(text: str) -> int:
    """Подсчитать количество токенов в тексте."""
    return len(tokenizer.encode(text))


def count_sentences(text):
    """Число предложений в тексте. Точка внутри числа концом предложения не считается."""
    return len([s for s in re.split(r"[.!?]+(?!\d)", text) if s.strip()])


class ResponseValidationError(ValueError):
    """Ошибка парсинга или проверки ответа модели."""


async def summarize_reviews(reviews: list[str]) -> str:
    """Сжимает список отзывов в короткое сводное резюме через запрос к модели."""
    if len(reviews) > 25:
        reviews = reviews[:25]

    reviews_text = "\n".join(f"- {review}" for review in reviews)
    
    prompt = (
        "Ниже приведены отзывы покупателей о товаре. "
        "Сделай краткое сводное резюме (2-3 предложения), "
        "сохранив ключевые достоинства и недостатки:\n\n"
        + reviews_text
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=150
    )
    
    summary = response.choices[0].message.content
    return summary.strip()


def build_messages(user_request: str) -> list[Message]:
    """Собрать сообщения для генерации карточки товара.

    Аргументы:
        user_request: данные о товаре и отзывы.

    Возвращает:
        Список сообщений для Chat Completions API.
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Обращение пользователя:\n{user_request}",
        },
    ]
    

async def request_json(
    client: AsyncOpenAI,
    model_id: str,
    messages: list[Message],
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repetition_penalty: float = REPETITION_PENALTY,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Запросить у модели ответ в формате JSON.

    Аргументы:
        client: настроенный клиент OpenRouter.
        model_id: идентификатор выбранной модели.
        messages: история сообщений.

    Возвращает:
        Содержимое ответа модели.

    Исключения:
        ResponseValidationError: модель вернула пустой
            или оборванный ответ.
    """
    start_time = time.perf_counter()

    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"top_k": top_k, "repetition_penalty": repetition_penalty},
        response_format={
            "type": "json_object",
        }
    )

    elapsed_single = time.perf_counter() - start_time

    choice = response.choices[0]
    content = choice.message.content

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    if response.usage is not None:
        logger.info(
            (
                "API response: model=%s finish_reason=%s "
                "prompt_tokens=%d completion_tokens=%d time=%.2fs"
            ),
            model_id,
            choice.finish_reason,
            prompt_tokens,
            completion_tokens,
            elapsed_single,
        )
    else:
        logger.info(
            "API response: model=%s finish_reason=%s usage=unknown",
            model_id,
            choice.finish_reason,
        )
    
    if choice.finish_reason == "length":
        raise ResponseValidationError(
            "Ответ оборвался из-за ограничения max_tokens"
        )

    if not content:
        raise ResponseValidationError(
            "Модель вернула пустой ответ"
        )

    return content, prompt_tokens, completion_tokens, elapsed_single


def parse_and_validate(content: str, expected_product_id: str) -> dict[str, str]:
    """Распарсить JSON и проверить контракт ответа.

    Аргументы:
        content: строка, полученная от модели
        expected_product_id: ожидаемый идентификатор товара для проверки.

    Возвращает:
        Проверенный словарь с результатом.

    Исключения:
        ResponseValidationError: ответ не прошёл проверку.
    """
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"Ответ не является корректным JSON: {error.msg}"
        ) from error
    
    if not isinstance(data, dict):
        raise ResponseValidationError(
            "На верхнем уровне должен находиться JSON-объект"
        )
    
    received_fields = set(data)
    missing_fields = EXPECTED_FIELDS - received_fields
    unexpected_fields = received_fields - EXPECTED_FIELDS

    if missing_fields:
        raise ResponseValidationError(
            "Отсутствуют обязательные поля: "
            f"{sorted(missing_fields)}"
        )

    if unexpected_fields:
        raise ResponseValidationError(
            "Получены лишние поля: "
            f"{sorted(unexpected_fields)}"
        )

    product_id = data["product_id"]
    description = data["description"]
    pros = data["pros"]
    cons = data["cons"]
    tags = data["tags"]

    if not isinstance(product_id, str):
        raise ResponseValidationError("Поле product_id должно быть строкой")
    if product_id != expected_product_id:
        raise ResponseValidationError(f"product_id не совпадает: ожидался {expected_product_id}, получен {product_id}")
      
    if not isinstance(description, str):
        raise ResponseValidationError("Поле description должно быть строкой")

    description = description.strip()
    
    if not (MIN_DESCRIPTION_LENGTH <= len(description) <= MAX_DESCRIPTION_LENGTH):
        raise ResponseValidationError(
            f"Длина description ({len(description)}) вне диапазона "
            f"{MIN_DESCRIPTION_LENGTH}-{MAX_DESCRIPTION_LENGTH} символов"
        )
      
    sent_count = count_sentences(description)
    if not (MIN_DESCRIPTION_SENTENCES <= sent_count <= MAX_DESCRIPTION_SENTENCES):
        raise ResponseValidationError(
            f"Количество предложений в description ({sent_count}) вне диапазона "
            f"{MIN_DESCRIPTION_SENTENCES}-{MAX_DESCRIPTION_SENTENCES}"
        )

    if isinstance(pros, list):
        if len(pros) > MAX_PROS_COUNT:
            pros = pros[:MAX_PROS_COUNT]
    if not isinstance(pros, list) or not (MIN_PROS_COUNT <= len(pros) <= MAX_PROS_COUNT):
        raise ResponseValidationError(f"Количество элементов в pros вне диапазона")
    
    if isinstance(cons, list):
        if len(cons) > MAX_CONS_COUNT:
            cons = cons[:MAX_CONS_COUNT]
    if not isinstance(cons, list) or not (MIN_CONS_COUNT <= len(cons) <= MAX_CONS_COUNT):
        raise ResponseValidationError(f"Количество элементов в cons вне диапазона")

    if isinstance(tags, list):
        if len(tags) > MAX_TAGS_COUNT:
            tags = tags[:MAX_TAGS_COUNT]
    if not isinstance(tags, list) or not (MIN_TAGS_COUNT <= len(tags) <= MAX_TAGS_COUNT):
        raise ResponseValidationError(f"Количество элементов в tags вне диапазона")

    for item in pros + cons + tags:
        if not isinstance(item, str) or not item.strip():
            raise ResponseValidationError("Элементы списков должны быть непустыми строками")
    
    return {
        "product_id": product_id,
        "description": description,
        "pros": [p.strip() for p in pros],
        "cons": [c.strip() for c in cons],
        "tags": [t.strip() for t in tags],
    }


async def generate_one(
    user_request: str,
    request_fn: RequestFn,
    request_id: str,
    max_attempts: int = MAX_ATTEMPTS,
    params: dict[str, Any] | None = None
) -> tuple[dict[str, str], int]:
    """Получить корректный ответ с помощью корректирующих повторов.

    Аргументы:
        user_request: обращение пользователя.
        request_fn: функция отправки запроса.
        request_id: идентификатор обращения для логов.
        max_attempts: максимальное число попыток.

    Возвращает:
        Проверенный ответ и номер успешной попытки.

    Исключения:
        ResponseValidationError: все попытки завершились ошибкой.
    """

    item_data = json.loads(user_request)
    expected_product_id = item_data["product_id"]

    if max_attempts < 1:
        raise ValueError(
            "max_attempts должен быть не меньше 1"
        )

    reviews = item_data.get("reviews", [])
    if count_tokens(user_request) > MAX_INPUT_TOKENS or len(reviews) > 20:
        if reviews:
            summary = await summarize_reviews(reviews)
            item_data["reviews"] = [f"Резюме отзывов: {summary}"]
            user_request = json.dumps(item_data, ensure_ascii=False)

    messages = build_messages(user_request)

    gen_params = params or {}
    temperature = gen_params.get("temperature", TEMPERATURE)
    top_p = gen_params.get("top_p", TOP_P)
    top_k = gen_params.get("top_k", TOP_K)
    repetition_penalty = gen_params.get("repetition_penalty", REPETITION_PENALTY)
    max_tokens = gen_params.get("max_tokens", MAX_TOKENS)

    full_prompt_text = SYSTEM_PROMPT + user_request
    if count_tokens(full_prompt_text) > MAX_INPUT_TOKENS:
        raise ResponseValidationError(f"Превышен лимит входных токенов: {count_tokens(full_prompt_text)} > {MAX_INPUT_TOKENS}")

    last_error: ResponseValidationError | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    latencies = []

    for attempt in range(1, max_attempts + 1):
        content: str | None = None

        logger.info(
            "request_id=%s attempt=%d/%d",
            request_id,
            attempt,
            max_attempts,
        )

        try:
            content, p_toks, c_toks, lat = await request_fn(
                client=client,
                model_id=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            )

            total_prompt_tokens += p_toks
            total_completion_tokens += c_toks
            latencies.append(lat)

            data = parse_and_validate(content, expected_product_id)

            logger.info(
                (
                    "request_id=%s validation=success "
                    "attempt=%d product_id=%s desc_len=%d pros_count=%d cons_count=%d tags_count=%d"
                ),
                request_id,
                attempt,
                data["product_id"],
                len(data["description"]),
                len(data["pros"]),
                len(data["cons"]),
                len(data["tags"]),
            )

            return data, attempt, total_prompt_tokens, total_completion_tokens, latencies

        except ResponseValidationError as error:
            last_error = error

            logger.warning(
                (
                    "request_id=%s validation=failed "
                    "attempt=%d reason=%s"
                ),
                request_id,
                attempt,
                error,
            )

            if attempt == max_attempts:
                break

            if content is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
        
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Предыдущий ответ не прошёл проверку. "
                        f"ОШИБКА: {error}."
                        f"Поле product_id оставь строго '{expected_product_id}'. "
                        f"Исправь ТОЛЬКО эту ошибку и верни исправленный валидный JSON без лишнего текста."
                    ),
                }
            )

    if last_error is not None:
        raise last_error
    
    raise ResponseValidationError(
        f"Не удалось получить корректный ответ за {max_attempts} попытки"
    )


async def run_concurrent(items, concurrency, generate_one, request_fn):
    """Отправляет запросы параллельно, не больше `concurrency` одновременно."""
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(item):
        async with sem:
            user_request = json.dumps(item, ensure_ascii=False)
            request_id = item["product_id"]
            return await generate_one(
                user_request=user_request,
                request_fn=request_fn,
                request_id=request_id
            )

    started = time.perf_counter()
    results = await asyncio.gather(*[wrapped(it) for it in items], return_exceptions=True)
    return results, time.perf_counter() - started


async def evaluate_reliability(
    items: list[dict[str, Any]],
    request_fn: RequestFn,
    max_attempts: int = MAX_ATTEMPTS,
    concurrency: int = 16,
) -> dict[str, Any]:
    """Обработать запросы и рассчитать метрики надёжности и валидности карточек товаров.

    Аргументы:
        items: список словарей с данными товаров из dev.jsonl.
        request_fn: функция отправки запроса.
        max_attempts: максимальное число попыток.

    Возвращает:
        Словарь с метриками эксперимента (success rate, причины брака, токены и т.д.).
    """
    if not items:
        raise ValueError(
            "Список items не должен быть пустым"
        )

    success_by_attempt: Counter[int] = Counter()
    failed_requests: list[dict[str, Any]] = []
    error_reasons_counter: Counter[str] = Counter()

    logger.info("Запуск валидации на %d товарах из dev.jsonl...", len(items))
    
    results, elapsed = await run_concurrent(items, concurrency, generate_one, request_fn)

    for request_id, (item, res) in enumerate(zip(items, results), start=1):
        expected_product_id = item["product_id"]

        if isinstance(res, Exception):
            error_msg = str(res)
            error_reasons_counter[error_msg] += 1
            failed_requests.append({
                "request_id": request_id,
                "product_id": expected_product_id,
                "error": error_msg,
            })
            logger.error(f"product_id={expected_product_id} result=failed attempts={max_attempts} reason={error_msg}")
            continue

        data, attempt, _, _, _ = res
        success_by_attempt[attempt] += 1


        logger.info(
            (
                "product_id=%s result=success "
                "attempt=%d desc_len=%d pros_count=%d"
            ),
            expected_product_id,
            attempt,
            len(data["description"]),
            len(data["pros"]),
        )

    total_requests = len(items)
    cumulative_successes = 0
    success_rate_by_attempt: dict[int, float] = {}

    logger.info("=" * 70)
    logger.info("Накопительный success_rate (Доля валидных карточек)")

    for attempt in range(1, max_attempts + 1):
        cumulative_successes += success_by_attempt[attempt]
        success_rate = cumulative_successes / total_requests
        success_rate_by_attempt[attempt] = success_rate

        logger.info(
            "После попытки %d: %.2f (%d/%d)",
            attempt,
            success_rate,
            cumulative_successes,
            total_requests,
        )

    recovered_after_correction = sum(
        success_by_attempt[attempt]
        for attempt in range(2, max_attempts + 1)
    )

    failed_count = len(failed_requests)
    final_success_rate = cumulative_successes / total_requests

    total_generation_attempts = sum(
        attempt * success_by_attempt[attempt]
        for attempt in range(1, max_attempts + 1)
    )
    total_generation_attempts += failed_count * max_attempts

    logger.info("=" * 70)
    logger.info("Итоги эксперимента по карточкам товаров")
    logger.info(
        "Всего товаров: %d",
        total_requests,
    )
    logger.info(
        "Успешно с первой попытки: %d",
        success_by_attempt[1],
    )
    logger.info(
        "Исправлено после корректирующих повторов: %d",
        recovered_after_correction,
    )
    logger.info(
        "Брак (не прошло валидацию): %d",
        failed_count,
    )
    logger.info(
        "Всего попыток генерации: %d",
        total_generation_attempts,
    )
    logger.info(
        "Итоговый success_rate (доля валидных): %.2f (порог >= 0.80)",
        final_success_rate,
    )
    logger.info(
        "Разбивка брака по причинам: %s",
        dict(error_reasons_counter),
    )

    return {
        "total_requests": total_requests,
        "success_by_attempt": {
            attempt: success_by_attempt[attempt]
            for attempt in range(1, max_attempts + 1)
        },
        "success_rate_by_attempt": success_rate_by_attempt,
        "recovered_after_correction": recovered_after_correction,
        "failed_count": failed_count,
        "failed_requests": failed_requests,
        "error_reasons": dict(error_reasons_counter),
        "total_generation_attempts": total_generation_attempts,
        "final_success_rate": final_success_rate,
        "elapsed_seconds": elapsed,
    }


async def run_experiment_for_params(items: list, params: dict) -> dict:
    """Запуск сетки экспериментов на подмножестве товаров."""
    async def process_item(item):
        prod_id = item["product_id"]
        user_request = json.dumps(item, ensure_ascii=False)
        try:
            data, _, _, c_toks, _ = await generate_one(
                user_request=user_request,
                request_fn=request_json,
                request_id=prod_id,
                params=params
            )
            return 1, len(data["description"]), c_toks
        except Exception as e:
            logger.error(f"Ошибка на товаре {prod_id} при params {params}: {e}")
            return 0, 0, 0

    results = await asyncio.gather(*(process_item(item) for item in items))
    
    valid_results = [r for r in results if r[0] == 1]
    valid_rate = len(valid_results) / len(items) if items else 0.0
    avg_desc_len = sum(r[1] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_tokens = sum(r[2] for r in results) / len(items) if items else 0.0

    return {
        "params": params,
        "valid_rate": round(valid_rate, 2),
        "avg_description_length": round(avg_desc_len, 1),
        "avg_completion_tokens": round(avg_tokens, 1)
    }


async def main():
    items = []

    # with open("dev.jsonl", "r", encoding="utf-8") as f:
    with open("benchmark.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # # Этап 3. Подобрать параметры генерации
    # test_items = items[:20]
    # grid = [
    #     {"temperature": 0.0, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 20, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": -1, "repetition_penalty": 1.1, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2, "max_tokens": 350},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 250},
    #     {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 500},
    # ]

    # print(f"\n{'='*15} Запуск сетки экспериментов на {len(test_items)} товарах {'='*15}")
    # experiment_results = []
    
    # for params in grid:
    #     res = await run_experiment_for_params(test_items, params)
    #     experiment_results.append(res)
    #     param_desc = ", ".join(f"{k}={v}" for k, v in params.items())
    #     print(f"{param_desc:<45} | {res['valid_rate']:<10} | {res['avg_description_length']:<10} | {res['avg_completion_tokens']:<8}")

    # os.makedirs("outputs", exist_ok=True)
    # with open("outputs/grid_search_report.json", "w", encoding="utf-8") as f:
    #     json.dump(experiment_results, f, ensure_ascii=False, indent=2)
    # print(f"\nОтчёт по сетке экспериментов сохранен в outputs/grid_search_report.json\n" + "="*83 + "\n")

    best_params = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "max_tokens": 350
    }

    request_fn = partial(
        request_json,
        client=client,
        model_id=MODEL_NAME,
        temperature=best_params["temperature"],
        top_p=best_params["top_p"],
        top_k=best_params["top_k"],
        repetition_penalty=best_params["repetition_penalty"],
        max_tokens=best_params["max_tokens"]
    )

    # # Этап 5. Пакетная обработка и батчинг
    # # Запуск сетки Concurrency
    # # concurrency_levels = [1, 4, 16, 32]
    # # concurrency_table = []
    # # # замер на подмножестве для экономии времени
    # # grid_subset = items[:30]  
    # # logger.info("Запуск сетки Concurrency (throughput / latency)")
    # # for conc in concurrency_levels:
    # #     _, el_sec = await run_concurrent(grid_subset, conc, generate_one, request_fn)
    # #     thr = round(len(grid_subset) / el_sec, 2) if el_sec > 0 else 0.0
    # #     concurrency_table.append({"concurrency": conc, "elapsed_sec": round(el_sec, 2), "throughput": thr})
    # #     logger.info(f"Concurrency={conc} | Time={el_sec:.1f}s | Throughput={thr} it/s")

    # # print("\nРезультаты сетки Concurrency:", json.dumps(concurrency_table, ensure_ascii=False, indent=2))

    concurrency = 16
    results, elapsed_sec = await run_concurrent(items, concurrency, generate_one, request_fn)

    total_requests = len(items)
    valid_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    all_latencies = []
    total_attempts_sum = 0
    error_reasons = Counter()

    predictions = []

    for item, res in zip(items, results):
        product_id = item["product_id"]

        if isinstance(res, Exception):
            err_msg = str(res)
            error_reasons[err_msg] += 1

            predictions.append({
                "product_id": product_id,
                "status": "error",
                "error": err_msg,
            })
            continue

        data, attempt, p_toks, c_toks, lats = res
        valid_count += 1
        total_input_tokens += p_toks
        total_output_tokens += c_toks
        all_latencies.extend(lats)
        total_attempts_sum += attempt

        predictions.append(data)

    failed_count = total_requests - valid_count
    valid_rate = round(valid_count / total_requests, 3) if total_requests > 0 else 0.0
    throughput = round(total_requests / elapsed_sec, 2) if elapsed_sec > 0 else 0.0
    avg_latency = round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0.0
    avg_attempts = round(total_attempts_sum / total_requests, 2) if total_requests > 0 else 0.0

    cost_vllm_per_1000 = round(((elapsed_sec / 3600.0) * 65.0) / total_requests * 1000.0, 1)

    report_data = {
      "run": {
        "model": MODEL_NAME,
        "concurrency": concurrency,
        "params": best_params,
        "elapsed_sec": round(elapsed_sec, 1),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "n": total_requests,
        "avg_latency": avg_latency,
        "avg_attempts": avg_attempts
      },
      "metrics": {
        "valid": valid_count,
        "valid_rate": valid_rate,
        "throughput": throughput,
        "time_for_300_sec": round(elapsed_sec, 1),
        "cost_vllm_per_1000": cost_vllm_per_1000,
        "errors_by_reason": dict(error_reasons)
      }
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/predictions.jsonl", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    with open("outputs/report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\nМетрики эксперимента собраны и сохранены в outputs/report.json")
    print("Сгенерированные карточки сохранены в outputs/predictions.jsonl")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
