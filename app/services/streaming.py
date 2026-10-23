"""Server-sent events, so a question does not look broken while it is working.

An answer here costs several seconds of provider round trips: contextualise,
route, retrieve, grade, maybe rewrite, maybe search, then generate. A single
JSON response at the end of that is a spinner for eight seconds, and the thing
that makes this product worth looking at -- the decisions it took on the way --
arrives only after the user has stopped watching.

So the pipeline emits as it goes: the stage it entered, the trace line it
recorded, the citations as soon as retrieval has them, the answer in pieces, and
a final event carrying the whole result. The console lights up its graph node by
node against the same events, which is why they are named after the nodes.

**SSE and not WebSockets**, because this is one-directional, it is plain HTTP
(so it survives every proxy that would need configuring for an upgrade), it
reconnects by itself in the browser, and the server side is a generator rather
than a connection to manage. Nothing here needs a socket.

**The work runs in a thread and the generator drains a queue.** It has to:
`Copilot.ask` is synchronous and spends its time blocking on HTTP, and running
it inside an async generator would block the event loop for every other request
in the process -- the same reason `/chat` and `/audit` are `def` rather than
`async def`. The route hands this iterator to a `StreamingResponse`, Starlette
iterates it off the event loop, and the pipeline is untouched.

**Back-pressure is resolved in favour of the worker, not the stream.** A browser
that walked away leaves nothing draining the queue; blocking the worker on a
full queue would pin a thread and its provider connections until the request
times out. So a full queue drops the oldest *progress* event and counts it,
while the terminal events are never dropped. A stream that lost three trace
lines is a cosmetic loss; a leaked thread is not.

**A heartbeat, because intermediaries kill idle connections.** A comment frame
every `heartbeat_seconds` keeps the connection alive through a proxy with a
read timeout, and it is a comment (`: ping`) rather than an event so no client
has to know about it.

**On a serverless platform this is bounded by the function's own lifetime.**
Vercel will stop the function at `maxDuration`, which cuts the stream wherever
it had got to. `vercel.json` sets the ceiling as high as the plan allows, the
budget in `app/agent/budgets.py` is set under it, and the client treats a
truncated stream as a failed request rather than as a short answer -- see
`STREAM_END` below, which is what tells it the difference.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_logger

log = get_logger("stream")

__all__ = [
    "EventStream",
    "STREAM_END",
    "StreamEvent",
    "format_sse",
    "stream_in_thread",
]

# The terminal event name. A client that reaches the end of the response body
# without having seen this knows the stream was *cut* -- the function timed out,
# the proxy gave up, the process died -- which is a different thing from an
# answer that finished. Without a sentinel the two are indistinguishable and the
# UI shows a truncated answer as if it were complete.
STREAM_END = "end"

# Event names, which double as the contract with `static/js/app.js`. Named after
# what the UI does with them rather than after the internals, so adding a node
# does not mean adding an event.
EVENT_STAGE = "stage"  # entered a node / a worker
EVENT_TRACE = "trace"  # one trace line
EVENT_CITATIONS = "citations"  # retrieval has sources
EVENT_TOKEN = "token"  # a piece of the answer
EVENT_ANSWER = "answer"  # the whole answer, once
EVENT_ERROR = "error"  # our own message, never a provider's
EVENT_RESULT = "result"  # the full AnswerResult payload

# Events that must never be dropped under back-pressure: without them a client
# cannot tell a finished stream from a cut one, or show anything at all.
_UNDROPPABLE = frozenset({STREAM_END, EVENT_RESULT, EVENT_ERROR, EVENT_ANSWER})


@dataclass(slots=True)
class StreamEvent:
    name: str
    data: dict[str, Any] = field(default_factory=dict)


def format_sse(name: str, data: Any) -> str:
    """One SSE frame.

    `json.dumps` with `default=str` so a stray datetime or Path in a payload
    degrades to its string form rather than failing the whole stream -- an
    answer the user can read beats a serialisation error they cannot.

    Newlines inside the payload have to be escaped, and JSON encoding does that
    for us. A raw multi-line string in a `data:` field would be read by the
    client as several fields and silently concatenated without the newlines,
    which is a bug that only shows up on an answer containing a list.
    """
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {name}\ndata: {payload}\n\n"


class EventStream:
    """A queue the worker writes to and the response iterates.

    Not thread-local and not per-event-loop: one instance belongs to one
    request, the worker thread holds a reference and calls `emit`, and the
    generator in the request's own thread yields frames. `queue.Queue` is the
    only synchronisation needed.
    """

    def __init__(
        self,
        *,
        heartbeat_seconds: float = 12.0,
        queue_size: int = 256,
        max_seconds: float = 120.0,
    ) -> None:
        self._queue: queue.Queue[StreamEvent | None] = queue.Queue(maxsize=queue_size)
        self.heartbeat_seconds = heartbeat_seconds
        self.max_seconds = max_seconds
        self._closed = threading.Event()
        self._dropped = 0
        self._emitted = 0

    # --- Producer side (the worker thread) -----------------------------------
    def emit(self, event: str, /, **data: Any) -> None:
        """Queue one event. Never raises, never blocks for long.

        A closed stream swallows silently: the worker finds out it is talking to
        nobody through `is_closed`, and making every `emit` in the pipeline
        handle an exception would put error handling for a cosmetic feature on
        the path of every answer.

        **`event` is positional-only, and that is not style.** The payload is
        `**data`, and a `stage` event's most natural field is the stage's
        `name` -- which collided with the parameter and raised "got multiple
        values for argument 'name'" from inside a feature whose whole job is to
        be invisible. The `/` makes every payload key available, including
        `name`, `kind` and `data`.
        """
        if self._closed.is_set():
            return
        queued = StreamEvent(name=event, data=data)
        try:
            self._queue.put_nowait(queued)
            self._emitted += 1
            return
        except queue.Full:
            pass

        if event in _UNDROPPABLE:
            # Make room by discarding the oldest droppable event. Worst case we
            # block briefly on a queue nothing is draining, which is why the
            # timeout is short and failure is silent.
            try:
                self._queue.get_nowait()
                self._dropped += 1
                self._queue.put(queued, timeout=0.5)
                self._emitted += 1
            except (queue.Empty, queue.Full):
                log.warning(
                    "Dropped a terminal stream event (%s); client is gone", event
                )
            return

        self._dropped += 1

    def emit_tokens(self, text: str, *, size: int = 24) -> None:
        """Emit a finished string in pieces, so a UI can type it out.

        Used when the answer is already complete but the model did not stream it
        -- a cached answer, an extractive answer, a composed supervisor answer.
        The alternative is a complete answer appearing instantly after eight
        seconds of nothing, which reads as a different kind of broken.

        Split on whitespace boundaries where possible: splitting mid-word makes
        a markdown renderer flicker on every partial `**`.
        """
        remaining = text
        while remaining:
            if len(remaining) <= size:
                piece, remaining = remaining, ""
            else:
                cut = remaining.rfind(" ", 0, size + 1)
                cut = cut if cut > size // 2 else size
                piece, remaining = remaining[:cut], remaining[cut:]
            self.emit(EVENT_TOKEN, text=piece)

    def close(self, **data: Any) -> None:
        """Send the terminal event and stop the generator."""
        if self._closed.is_set():
            return
        self.emit(STREAM_END, **data)
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - the sentinel is tried once more below
            pass

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    # --- Consumer side (the response) ----------------------------------------
    def __iter__(self) -> Iterator[str]:
        """Yield SSE frames until the stream closes, times out, or is cut."""
        deadline = time.monotonic() + self.max_seconds
        # Sent immediately so the browser's EventSource fires `open` and the UI
        # can swap its spinner for a live indicator before the first stage
        # arrives -- which on a cold start can be several seconds.
        yield ": stream open\n\n"
        while True:
            if time.monotonic() > deadline:
                log.warning("Stream exceeded %.0fs; closing", self.max_seconds)
                yield format_sse(
                    EVENT_ERROR,
                    {"message": "The answer took too long and was stopped."},
                )
                yield format_sse(STREAM_END, {"ok": False, "reason": "timeout"})
                return
            try:
                event = self._queue.get(timeout=self.heartbeat_seconds)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if event is None:
                return
            yield format_sse(event.name, event.data)
            if event.name == STREAM_END:
                return

    def stats(self) -> dict[str, Any]:
        return {
            "emitted": self._emitted,
            "dropped": self._dropped,
            "closed": self.is_closed,
        }


def stream_in_thread(
    work: Callable[[EventStream], None],
    *,
    heartbeat_seconds: float = 12.0,
    max_seconds: float = 120.0,
    on_error: str = "The copilot could not answer that right now.",
) -> Iterator[str]:
    """Run `work` on a thread and yield the SSE frames it produces.

    The thread is a daemon: a worker still blocked on a provider when the
    process is asked to shut down must not hold the shutdown open. It has
    already lost its reader by then.

    `work` is called with the stream and is expected to `emit` as it goes. It
    does **not** have to call `close()` -- this does, in a `finally`, so a
    `work` that raises still terminates the stream rather than leaving the
    client waiting for a sentinel that is never coming. A raised exception
    becomes one `error` event carrying `on_error`: our own sentence, never the
    provider's text, for the same reason no handler in `app/api/routes.py`
    returns `str(exc)`.
    """
    stream = EventStream(
        heartbeat_seconds=heartbeat_seconds, max_seconds=max_seconds
    )

    def run() -> None:
        try:
            work(stream)
        except Exception:  # noqa: BLE001 - never leak provider error text
            log.exception("The streamed pipeline raised")
            stream.emit(EVENT_ERROR, message=on_error)
            stream.close(ok=False, reason="error")
        else:
            if not stream.is_closed:
                stream.close(ok=True)

    thread = threading.Thread(target=run, name="hr-copilot-stream", daemon=True)
    thread.start()
    yield from stream
