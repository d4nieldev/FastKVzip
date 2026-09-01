def parse_window_size(value: str | int | float) -> int | float:
    """Parse a fixed token count or a context-relative window ratio."""
    if isinstance(value, bool):
        raise ValueError("window size must be a non-negative integer or a ratio")
    size = float(value)
    if 0 < size < 1:
        return size
    if size >= 0 and size.is_integer():
        return int(size)
    raise ValueError("window size must be a non-negative integer or a ratio")


def resolve_window_size(
    value: int | float,
    context_length: int,
    prefill_chunk: int,
    short_context_ratio: float = 0.02,
) -> int:
    """Resolve a window setting to a context-token count."""
    size = parse_window_size(value)
    if isinstance(size, float):
        return int(size * context_length)
    if context_length < prefill_chunk:
        size = int(short_context_ratio * context_length)
    return min(size, context_length)
