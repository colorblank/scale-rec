//! 时间解析算子：将多种时间格式解析为稳定整数特征。
use super::{CustomOp, Fv};
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Timelike, Utc};

/// 输入时间格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputFormat {
    Auto,
    EpochS,
    EpochMs,
    Rfc3339,
    Strftime,
}

impl InputFormat {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "auto" => Ok(Self::Auto),
            "epoch_s" => Ok(Self::EpochS),
            "epoch_ms" => Ok(Self::EpochMs),
            "rfc3339" => Ok(Self::Rfc3339),
            "strftime" => Ok(Self::Strftime),
            other => Err(format!("TimeParser: unsupported input_format '{}'", other)),
        }
    }
}

/// 输出字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputField {
    TimestampS,
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Weekday,
    DayOfYear,
    Yyyymmdd,
    MinuteOfDay,
}

impl OutputField {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "timestamp_s" => Ok(Self::TimestampS),
            "year" => Ok(Self::Year),
            "month" => Ok(Self::Month),
            "day" => Ok(Self::Day),
            "hour" => Ok(Self::Hour),
            "minute" => Ok(Self::Minute),
            "weekday" => Ok(Self::Weekday),
            "day_of_year" => Ok(Self::DayOfYear),
            "yyyymmdd" => Ok(Self::Yyyymmdd),
            "minute_of_day" => Ok(Self::MinuteOfDay),
            other => Err(format!("TimeParser: unsupported output '{}'", other)),
        }
    }
}

/// 将时间字符串、秒/毫秒时间戳解析为整数时间特征。
pub struct TimeParser {
    input_format: InputFormat,
    output: OutputField,
    formats: Vec<String>,
    default_val: i32,
}

impl TimeParser {
    /// 创建时间解析算子。
    pub fn new(
        input_format: &str,
        output: &str,
        formats: Vec<String>,
        default_val: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            input_format: InputFormat::parse(input_format)?,
            output: OutputField::parse(output)?,
            formats,
            default_val,
        })
    }

    fn parse_input(&self, input: &Fv) -> Option<DateTime<Utc>> {
        match self.input_format {
            InputFormat::EpochS => numeric_i64(input).and_then(datetime_from_epoch_s),
            InputFormat::EpochMs => numeric_i64(input).and_then(datetime_from_epoch_ms),
            InputFormat::Rfc3339 => string_value(input).and_then(parse_rfc3339),
            InputFormat::Strftime => {
                string_value(input).and_then(|s| parse_with_formats(s, &self.formats))
            }
            InputFormat::Auto => self.parse_auto(input),
        }
    }

    fn parse_auto(&self, input: &Fv) -> Option<DateTime<Utc>> {
        if let Some(value) = numeric_i64(input) {
            return parse_auto_epoch(value);
        }
        let value = string_value(input)?;
        if let Ok(number) = value.parse::<i64>() {
            return parse_auto_epoch(number);
        }
        parse_rfc3339(value)
            .or_else(|| parse_with_formats(value, &self.formats))
            .or_else(|| parse_common_formats(value))
    }

    fn project(&self, dt: DateTime<Utc>) -> Option<i32> {
        match self.output {
            OutputField::TimestampS => i32::try_from(dt.timestamp()).ok(),
            OutputField::Year => Some(dt.year()),
            OutputField::Month => Some(dt.month() as i32),
            OutputField::Day => Some(dt.day() as i32),
            OutputField::Hour => Some(dt.hour() as i32),
            OutputField::Minute => Some(dt.minute() as i32),
            OutputField::Weekday => Some(dt.weekday().num_days_from_monday() as i32),
            OutputField::DayOfYear => Some(dt.ordinal() as i32),
            OutputField::Yyyymmdd => {
                Some(dt.year() * 10_000 + dt.month() as i32 * 100 + dt.day() as i32)
            }
            OutputField::MinuteOfDay => Some((dt.hour() * 60 + dt.minute()) as i32),
        }
    }
}

impl CustomOp for TimeParser {
    fn name(&self) -> &str {
        "TimeParser"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let value = inputs
            .first()
            .and_then(|input| self.parse_input(input))
            .and_then(|dt| self.project(dt))
            .unwrap_or(self.default_val);
        Ok(Fv::Int(value))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for input in col.iter().take(n_rows) {
            let value = self
                .parse_input(input)
                .and_then(|dt| self.project(dt))
                .unwrap_or(self.default_val);
            results.push(Fv::Int(value));
        }
        Ok(results)
    }
}

/// 从 YAML params 创建 TimeParser 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let input_format = params
        .get("input_format")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");
    let output = params
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("timestamp_s");
    let formats = params
        .get("formats")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let default_val = params
        .get("default_val")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    Ok(Box::new(TimeParser::new(
        input_format,
        output,
        formats,
        default_val,
    )?))
}

fn numeric_i64(input: &Fv) -> Option<i64> {
    match input {
        Fv::Int(value) => Some(i64::from(*value)),
        Fv::Float(value) if value.is_finite() => Some(*value as i64),
        Fv::Str(value) => value.parse::<i64>().ok(),
        _ => None,
    }
}

fn string_value(input: &Fv) -> Option<&str> {
    match input {
        Fv::Str(value) => {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        }
        _ => None,
    }
}

fn parse_auto_epoch(value: i64) -> Option<DateTime<Utc>> {
    if value.abs() >= 100_000_000_000 {
        datetime_from_epoch_ms(value)
    } else {
        datetime_from_epoch_s(value)
    }
}

fn datetime_from_epoch_s(value: i64) -> Option<DateTime<Utc>> {
    DateTime::from_timestamp(value, 0)
}

fn datetime_from_epoch_ms(value: i64) -> Option<DateTime<Utc>> {
    DateTime::from_timestamp_millis(value)
}

fn parse_rfc3339(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

fn parse_with_formats(value: &str, formats: &[String]) -> Option<DateTime<Utc>> {
    for fmt in formats {
        if let Some(dt) = parse_with_format(value, fmt) {
            return Some(dt);
        }
    }
    None
}

fn parse_common_formats(value: &str) -> Option<DateTime<Utc>> {
    for fmt in COMMON_FORMATS {
        if let Some(dt) = parse_with_format(value, fmt) {
            return Some(dt);
        }
    }
    None
}

fn parse_with_format(value: &str, fmt: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_str(value, fmt) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, fmt) {
        return Some(dt.and_utc());
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, fmt) {
        return date.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }
    None
}

const COMMON_FORMATS: &[&str] = &[
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feats::ops::CustomOp;

    #[test]
    fn parses_rfc3339_to_hour_utc() {
        let op = TimeParser::new("rfc3339", "hour", vec![], -1).unwrap();
        let result = op.process(&[Fv::Str("2026-06-29T10:30:00+08:00".into())]);
        assert_eq!(result.unwrap(), Fv::Int(2));
    }

    #[test]
    fn parses_custom_format_to_weekday() {
        let op = TimeParser::new("strftime", "weekday", vec!["%Y.%m.%d %H:%M".into()], -1).unwrap();
        assert_eq!(
            op.process(&[Fv::Str("2026.06.29 12:00".into())]).unwrap(),
            Fv::Int(0)
        );
    }

    #[test]
    fn parses_epoch_ms_input() {
        let op = TimeParser::new("epoch_ms", "yyyymmdd", vec![], -1).unwrap();
        assert_eq!(
            op.process(&[Fv::Str("1782727200000".into())]).unwrap(),
            Fv::Int(20260629)
        );
    }

    #[test]
    fn falls_back_to_default_on_invalid_input() {
        let op = TimeParser::new("auto", "day_of_year", vec![], 366).unwrap();
        assert_eq!(op.process(&[Fv::Str("bad".into())]).unwrap(), Fv::Int(366));
    }
}
