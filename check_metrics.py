
import json
import math
import sys
from collections import Counter
from pathlib import Path

from run_batch import parse_and_validate, ResponseValidationError


BENCHMARK_PATH = Path("benchmark.jsonl")
PREDICTIONS_PATH = Path("outputs/predictions_new.jsonl")
REPORT_PATH = Path("outputs/report_new.json")


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: некорректный JSON: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: ожидался JSON-объект"
                )

            records.append(record)

    return records


def check_metrics() -> bool:
    benchmark = read_jsonl(BENCHMARK_PATH)
    predictions = read_jsonl(PREDICTIONS_PATH)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    errors = []
    validation_errors = Counter()
    valid_count = 0
    failed_count = 0

	# Проверка структуры отчета
    if not isinstance(report.get("run"), dict):
        raise ValueError("В отчёте отсутствует объект run")

    if not isinstance(report.get("metrics"), dict):
        raise ValueError("В отчёте отсутствует объект metrics")

    run_report = report["run"]
    metrics_report = report["metrics"]

    required_run = {
        "model", "concurrency", "params", "elapsed_sec",
        "input_tokens", "output_tokens", "n",
        "avg_latency", "avg_attempts",
    }
    required_metrics = {
        "valid", "valid_rate", "throughput",
        "time_for_300_sec", "cost_vllm_per_1000",
        "errors_by_reason",
    }

    if set(run_report) != required_run:
        errors.append(
            "Структура run не соответствует ожидаемой схеме: "
            f"{sorted(run_report)}"
        )

    if set(metrics_report) != required_metrics:
        errors.append(
            "Структура metrics не соответствует ожидаемой схеме: "
            f"{sorted(metrics_report)}"
        )

    if "api_comparison" in report:
        api = report["api_comparison"]
        required_api = {
            "model", "n", "valid_rate", "input_tokens",
            "output_tokens", "elapsed_sec", "cost_per_1000",
        }

        if not isinstance(api, dict) or set(api) != required_api:
            errors.append(
                "Структура api_comparison не соответствует схеме"
            )

    # Проверка количества записей и соответствие id
    if len(predictions) != len(benchmark):
        errors.append(
            f"Количество строк: predictions={len(predictions)}, "
            f"benchmark={len(benchmark)}"
        )

    for index, item in enumerate(benchmark):
        if index >= len(predictions):
            break

        expected_id = item.get("product_id")
        actual_id = predictions[index].get("product_id")

        if actual_id != expected_id:
            errors.append(
                f"Строка {index + 1}: ожидался product_id={expected_id!r}, "
                f"получен {actual_id!r}"
            )

    # Валидация записей
    for index, prediction in enumerate(predictions, start=1):
        product_id = prediction.get("product_id")

        if prediction.get("status") == "error":
            failed_count += 1

            if not isinstance(prediction.get("error"), str) or not prediction["error"].strip():
                errors.append(
                    f"Строка {index}: отсутствует описание ошибки"
                )
            continue

        if "status" in prediction:
            failed_count += 1
            errors.append(
                f"Строка {index}: неизвестный статус "
                f"{prediction.get('status')!r}"
            )
            continue

        if not isinstance(product_id, str):
            failed_count += 1
            errors.append(
                f"Строка {index}: отсутствует строковый product_id"
            )
            continue

        try:
            parse_and_validate(
                json.dumps(prediction, ensure_ascii=False),
                product_id,
            )
            valid_count += 1

        except (ResponseValidationError, TypeError, ValueError) as error:
            failed_count += 1
            validation_errors[str(error)] += 1
            errors.append(
                f"Строка {index}, product_id={product_id!r}: {error}"
            )

    total = len(benchmark)
    valid_rate = round(valid_count / total, 3) if total else 0.0

    # Сверка с report.json
    if run_report.get("n") != total:
        errors.append(
            f"run.n={run_report.get('n')}, ожидалось {total}"
        )

    if len(predictions) != run_report.get("n"):
        errors.append(
            f"Количество predictions={len(predictions)} "
            f"не совпадает с run.n={run_report.get('n')}"
        )

    if metrics_report.get("valid") != valid_count:
        errors.append(
            f"metrics.valid={metrics_report.get('valid')}, "
            f"пересчитано {valid_count}"
        )

    reported_rate = metrics_report.get("valid_rate")

    if (
        not isinstance(reported_rate, (int, float))
        or not math.isclose(
            reported_rate,
            valid_rate,
            abs_tol=0.001,
        )
    ):
        errors.append(
            f"metrics.valid_rate={reported_rate}, "
            f"пересчитано {valid_rate}"
        )

    print("Вывод сверки check mertics:")
    print(f"benchmark:       {total}")
    print(f"predictions:     {len(predictions)}")
    print(f"run.n:           {run_report.get('n')}")
    print(f"valid:           {valid_count}")
    print(f"failed:          {failed_count}")
    print(f"valid_rate:      {valid_rate:.3f}")

    if validation_errors:
        print("\nОшибки валидации карточек:")
        for reason, count in validation_errors.most_common():
            print(f"  {count} x {reason}")

    if errors:
        print("\nПроверка не пройдена.")
        for error in errors:
            print(f"- {error}")
        return False

    print("\nПроверка пройдена.")
    return True


if __name__ == "__main__":
    try:
        success = check_metrics()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Ошибка проверки: {error}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if success else 1)
