from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from collections.abc import AsyncGenerator
import asyncio

server = Server()


@server.agent()
async def echo(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Echoes the input message back."""
    for message in input:
        await asyncio.sleep(0.5)
        yield {"thought": "I should echo everything"}
        await asyncio.sleep(0.5)
        yield message

server.run()
