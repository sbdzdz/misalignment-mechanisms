import asyncio
import json

import httpx


async def conversation(config, env, credentials, client):
    messages = [{"role": "user", "content": env.instruction}]
    for _ in range(config["execution"]["max_turns"]):
        response = await client.post(
            credentials["url"] + "/chat/completions",
            headers={"Authorization": "Bearer " + credentials["token"]},
            json=dict(
                model=config["model"]["id"],
                messages=messages,
                tools=env.catalog,
                stream=False,
                **config["model"]["parameters"],
            ),
        )
        if response.status_code == 402:
            return "budget_limit"
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        reason = choice.get("finish_reason")
        if reason == "length":
            return "token_limit"
        if reason not in {"stop", "tool_calls"}:
            return "model_termination"
        message = choice["message"]
        messages.append(
            {
                k: v
                for k, v in message.items()
                if k
                in {
                    "role",
                    "content",
                    "tool_calls",
                    "reasoning_details",
                    "reasoning_content",
                }
            }
        )
        calls = message.get("tool_calls") or []
        if not calls:
            return "completed"
        for call in calls:
            function = call["function"]
            result = env.call(function["name"], function.get("arguments", ""))
            messages.append(
                dict(role="tool", tool_call_id=call["id"], content=json.dumps(result))
            )
            if env.state["accepted"]:
                return "accepted"
    return "turn_limit"


async def run(config, env, credentials):
    async with httpx.AsyncClient(
        timeout=config["execution"]["request_timeout_sec"], trust_env=False
    ) as client:
        try:
            async with asyncio.timeout(config["execution"]["timeout_sec"]):
                return await conversation(config, env, credentials, client)
        except TimeoutError:
            return "timeout"
