"""AIND log formatters, filters, and handler setup for use with logging.yml.

Load with ``setup_logging`` at the capsule entry point so that every
downstream module inherits structured JSON output on the root logger.
"""

import datetime
import json
import logging
import logging.config
import os

import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logging.yml"
)


def setup_logging(log_stream: str, **fields) -> None:
    """Load logging.yml and configure the root logger.

    Also attempts to add a CloudWatch handler so logs are sent to the
    default AIND log group. Requires valid AWS credentials at runtime;
    falls back gracefully when watchtower is not installed or credentials
    are absent.

    Parameters
    ----------
    log_stream : str
        CloudWatch log stream name, e.g. the process or capsule name.
    **fields : str
        Arbitrary context fields to inject into every log record, e.g.
        ``acquisition_name="839909_2026-01-01_10-00-00"``,
        ``process_name="aind-pophys-converter"``.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        config["filters"]["aind_fields"].update(fields)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG)
        logging.warning(
            "logging.yml not found — falling back to basicConfig"
        )
    add_cloudwatch_handler(log_stream=log_stream)


def add_cloudwatch_handler(log_stream: str) -> None:
    """Add a watchtower CloudWatch handler to the root logger.

    No-op if watchtower is not installed or AWS credentials are
    unavailable. The handler reuses the ``AindJsonFormatter`` and
    ``AindContextFilter`` already attached to the console handler so
    that CloudWatch log events are structured identically.

    Parameters
    ----------
    log_stream : str
        CloudWatch log stream name, e.g. the process or capsule name.
    """
    try:
        import watchtower
    except ImportError:
        logging.warning(
            "watchtower not installed — skipping CloudWatch handler"
        )
        return

    try:
        handler = watchtower.CloudWatchLogHandler(
            log_stream_name=log_stream,
            log_group="aind/internal-logs",
        )
    except Exception as e:
        logging.warning(
            f"CloudWatch handler unavailable — skipping: {e}"
        )
        return

    root = logging.getLogger()
    formatter = AindJsonFormatter()
    for h in root.handlers:
        if isinstance(h.formatter, AindJsonFormatter):
            formatter = h.formatter
        for f in h.filters:
            if isinstance(f, AindContextFilter):
                handler.addFilter(f)
                break

    handler.setFormatter(formatter)
    root.addHandler(handler)


class AindJsonFormatter(logging.Formatter):
    """Format log records as JSON with AIND standard fields.

    The list of fields to include is configured via the ``fields``
    argument, which is passed directly from logging.yml via dictConfig.
    Standard ``LogRecord`` attributes (e.g. ``lineno``, ``process``)
    are read from the record automatically; custom fields injected by a
    filter (e.g. ``acquisition_name``, ``process_name``) fall back to
    ``"undefined"`` if absent.

    Special fields with non-trivial mappings:

    - ``timestamp`` — ISO8601 local time derived from ``record.created``
    - ``level``     — maps to ``record.levelname``
    - ``message``   — maps to ``record.getMessage()``
    """

    _DEFAULT_FIELDS = [
        "timestamp",
        "level",
        "message",
        "acquisition_name",
        "process_name",
        "event_type",
    ]

    def __init__(self, fields=None):
        """Initialize the formatter with an optional list of fields.

        Parameters
        ----------
        fields : list of str, optional
            Fields to include in each JSON log record. Defaults to
            ``_DEFAULT_FIELDS`` if not provided.
        """
        super().__init__()
        self.fields = fields or self._DEFAULT_FIELDS

    def format(self, record):
        """Format a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            JSON-encoded log entry.
        """
        entry = {}
        for field in self.fields:
            if field == "timestamp":
                entry["timestamp"] = (
                    datetime.datetime.fromtimestamp(
                        record.created, tz=datetime.timezone.utc
                    )
                    .astimezone()
                    .isoformat(timespec="milliseconds")
                )
            elif field == "level":
                entry["level"] = record.levelname
            elif field == "message":
                entry["message"] = record.getMessage()
            else:
                entry[field] = getattr(record, field, "undefined")
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry)


class AindContextFilter(logging.Filter):
    """Inject arbitrary context fields into every log record.

    Fields and their default values are declared in logging.yml and
    passed as keyword arguments by dictConfig. Any field listed here
    will be available to the formatter. Values can be overridden
    per-call via ``extra={}`` when logging.
    """

    def __init__(self, **fields):
        """Initialize the filter with context fields.

        Parameters
        ----------
        **fields : str
            Key-value pairs to inject into every log record, e.g.
            ``acquisition_name="undefined"``.
        """
        super().__init__()
        self.fields = fields

    def filter(self, record):
        """Inject context fields into the log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to augment.

        Returns
        -------
        bool
            Always ``True`` (record is never suppressed).
        """
        for key, value in self.fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True
