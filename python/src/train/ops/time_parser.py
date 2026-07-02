from __future__ import annotations

"""时间解析算子：将多种时间格式解析为稳定整数特征。"""

from datetime import UTC, datetime
from typing import Any

from . import register_op

COMMON_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
)


@register_op("TimeParser")
class TimeParser:
    """Parse string or epoch time values into integer time features."""

    def __init__(
        self,
        input_format: str = "auto",
        output: str = "timestamp_s",
        formats: list[str] | None = None,
        default_val: int = 0,
    ) -> None:
        self.input_format = input_format
        self.output = output
        self.formats = formats or []
        self.default_val = default_val
        self._validate()

    @classmethod
    def from_config(cls, params: dict) -> TimeParser:
        return cls(
            input_format=str(params.get("input_format", "auto")),
            output=str(params.get("output", "timestamp_s")),
            formats=[str(fmt) for fmt in params.get("formats", [])],
            default_val=int(params.get("default_val", 0)),
        )

    def _validate(self) -> None:
        if self.input_format not in {"auto", "epoch_s", "epoch_ms", "rfc3339", "strftime"}:
            raise ValueError(f"TimeParser: unsupported input_format '{self.input_format}'")
        if self.output not in {
            "timestamp_s",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "weekday",
            "day_of_year",
            "yyyymmdd",
            "minute_of_day",
        }:
            raise ValueError(f"TimeParser: unsupported output '{self.output}'")

    def process(self, inputs: list[Any]) -> int:
        dt = self._parse_input(inputs[0] if inputs else None)
        if dt is None:
            return self.default_val
        return self._project(dt)

    def process_batch(self, inputs: list[Any]) -> list[int]:
        vals = inputs[0]
        return [self.process([val]) for val in vals]

    def _parse_input(self, value: Any) -> datetime | None:
        if self.input_format == "epoch_s":
            return _datetime_from_epoch(value, scale=1)
        if self.input_format == "epoch_ms":
            return _datetime_from_epoch(value, scale=1000)
        if self.input_format == "rfc3339":
            return _parse_rfc3339(value)
        if self.input_format == "strftime":
            return _parse_with_formats(value, self.formats)
        return self._parse_auto(value)

    def _parse_auto(self, value: Any) -> datetime | None:
        number = _to_int(value)
        if number is not None:
            scale = 1000 if abs(number) >= 100_000_000_000 else 1
            return _datetime_from_epoch(number, scale=scale)
        return (
            _parse_rfc3339(value)
            or _parse_with_formats(value, self.formats)
            or _parse_with_formats(value, COMMON_FORMATS)
        )

    def _project(self, dt: datetime) -> int:
        if self.output == "timestamp_s":
            value = int(dt.timestamp())
            if value < -(2**31) or value > 2**31 - 1:
                return self.default_val
            return value
        if self.output == "year":
            return dt.year
        if self.output == "month":
            return dt.month
        if self.output == "day":
            return dt.day
        if self.output == "hour":
            return dt.hour
        if self.output == "minute":
            return dt.minute
        if self.output == "weekday":
            return dt.weekday()
        if self.output == "day_of_year":
            return int(dt.strftime("%j"))
        if self.output == "yyyymmdd":
            return dt.year * 10_000 + dt.month * 100 + dt.day
        if self.output == "minute_of_day":
            return dt.hour * 60 + dt.minute
        return self.default_val


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _datetime_from_epoch(value: Any, scale: int) -> datetime | None:
    number = _to_int(value)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number / scale, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _parse_rfc3339(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_with_formats(value: Any, formats: list[str] | tuple[str, ...]) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)  # noqa: DTZ007
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None
