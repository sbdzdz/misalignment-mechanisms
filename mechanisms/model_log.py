import hashlib
import json
from pathlib import Path

FORMAT = "model-calls-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Response:
    def __init__(self):
        self.model = None
        self.content, self.reasoning = ([], [])
        self.tool_calls = {}
        self.finish_reason = None
        self.usage = None
        self.errors = []

    def add(self, data):
        if not isinstance(data, dict):
            return
        self.model = data.get("model") or self.model
        if isinstance(data.get("usage"), dict):
            self.usage = data["usage"]
        if data.get("error"):
            self.errors.append(data["error"])
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            part = choice.get("delta") or choice.get("message") or {}
            if isinstance(part.get("content"), str):
                self.content.append(part["content"])
            reasoning = part.get("reasoning") or part.get("reasoning_content")
            if isinstance(reasoning, str):
                self.reasoning.append(reasoning)
            for position, call in enumerate(part.get("tool_calls") or []):
                slot = self.tool_calls.setdefault(
                    call.get("index", position),
                    {"id": None, "name": "", "arguments": ""},
                )
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    slot["arguments"] += arguments
                elif arguments is not None:
                    slot["arguments"] = json.dumps(arguments)
            self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def result(self):
        return {
            "model": self.model,
            "reasoning": "".join(self.reasoning) or None,
            "content": "".join(self.content) or None,
            "tool_calls": [self.tool_calls[k] for k in sorted(self.tool_calls)],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "errors": self.errors,
        }


class ModelLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.requests = []
        self.tools = set()

    def write(self, record):
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def record(self, payload, response, **metadata):
        messages = payload.get("messages")
        messages = messages if isinstance(messages, list) else []
        hashes = [digest(m) for m in messages]
        parent = next(
            (
                i
                for i in range(len(self.requests) - 1, -1, -1)
                if self.requests[i]
                and hashes[: len(self.requests[i])] == self.requests[i]
            ),
            None,
        )
        start = len(self.requests[parent]) if parent is not None else 0
        tools = payload.get("tools")
        tools_sha = digest(tools) if tools else None
        if tools_sha and tools_sha not in self.tools:
            self.tools.add(tools_sha)
            self.write({"type": "tools", "sha256": tools_sha, "tools": tools})
        self.write(
            {
                "type": "call",
                "index": len(self.requests),
                "parent": parent,
                "message_count": len(messages),
                "new_messages": messages[start:],
                "tools_sha256": tools_sha,
                "params": {
                    k: v for k, v in payload.items() if k not in ("messages", "tools")
                },
                "response": response.result(),
                **metadata,
            }
        )
        self.requests.append(hashes)


def read(path):
    transcript = {"format": FORMAT, "tools": {}, "calls": []}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            transcript.setdefault("unparsed_lines", []).append(line)
            continue
        if record.get("type") == "tools":
            transcript["tools"][record["sha256"]] = record["tools"]
        elif record.get("type") == "call":
            transcript["calls"].append(record)
    return transcript
