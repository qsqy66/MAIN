"""
OpenTelemetry 链路追踪 & structlog 配置模块

功能：
  - 初始化 TracerProvider + SpanProcessor
  - 支持 console / otlp 两种导出器
  - 配置 structlog，注入 trace_id / span_id 到日志
  - 提供便捷的 get_tracer() / trace_id 上下文变量
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

from config import settings

# ---------------------------------------------------------------------------
# 上下文变量 — 跨 async 任务传播 trace_id / span_id
# ---------------------------------------------------------------------------
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
span_id_var: ContextVar[str] = ContextVar("span_id", default="")


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------
_provider: Optional[TracerProvider] = None


def init_tracing() -> None:
    """
    初始化 OpenTelemetry TracerProvider + 导出器。

    在程序入口（main.py）中调用一次即可。
    测试或不需要追踪时，设置 TRACING_ENABLED=false 跳过。
    """
    global _provider

    if not settings.TRACING_ENABLED:
        return

    resource = Resource.create(
        {
            "service.name": settings.OTEL_SERVICE_NAME,
            "service.version": "0.1.0",
        }
    )

    _provider = TracerProvider(resource=resource)

    # ---- 导出器 ----
    exporter_type = settings.OTEL_EXPORTER_TYPE.lower()

    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_ENDPOINT,
                insecure=True,
            )
        except ImportError:
            # 降级为 console
            exporter = ConsoleSpanExporter()
    elif exporter_type == "none":
        exporter = None
    else:
        # 默认 console
        exporter = ConsoleSpanExporter()

    if exporter is not None:
        _provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(_provider)


def get_tracer(name: str = "office_agent") -> trace.Tracer:
    """获取一个 Tracer 实例，所有模块统一入口。"""
    return trace.get_tracer(name)


def _get_current_context() -> dict:
    """读取当前 span 上下文，方便 structlog 注入。"""
    span = trace.get_current_span()
    if span is not None and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {}


def configure_structlog() -> None:
    """
    配置 structlog：
    - 自动注入 trace_id / span_id（来自 OpenTelemetry current span）
    - 添加时间戳、日志级别
    - 标准库 logging → structlog 桥接
    """
    from structlog.contextvars import merge_contextvars

    structlog.configure(
        processors=[
            merge_contextvars,                                  # 读取 context vars
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 桥接标准库 logging → structlog
    logging.basicConfig(format="%(message)s", level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# 便捷装饰器 / 上下文管理器
# ---------------------------------------------------------------------------
def start_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict | None = None,
    tracer_name: str = "office_agent",
):
    """
    创建一个 OpenTelemetry span 并作为异步上下文管理器。

    用法:
        async with start_span("llm_chat", attributes={"model": "glm-4"}) as span:
            ... 业务代码 ...
            span.set_attribute("tokens", 1000)
    """
    tracer = get_tracer(tracer_name)
    return tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
    )


def set_span_ok(span, description: str = "") -> None:
    """标记 span 成功。"""
    span.set_status(Status(StatusCode.OK))
    if description:
        span.set_attribute("status_detail", description)


def set_span_error(span, error: Exception | str) -> None:
    """标记 span 失败。"""
    span.set_status(Status(StatusCode.ERROR, str(error)))
    if isinstance(error, Exception):
        span.record_exception(error)


def generate_trace_id() -> str:
    """生成一个手动 trace_id（用于自定义链路关联）。"""
    return uuid.uuid4().hex[:32]
