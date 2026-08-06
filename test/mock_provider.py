import asyncio


class MockProvider:

    def complete(self, system: str, user: str, role: str = "default") -> str:
        if role == "default":
            return f"Mock response: {system} {user}"
        if role == "plan":
            return f"Mock plan: {system} {user}"
        if role == "summary":
            return f"Mock summary: {system} {user}"
        if role == "judge":
            return f"Mock judge: {system} {user}"

    async def acomplete(self, system: str, user: str, role: str = "default") -> str:
        return await asyncio.to_thread(self.complete, system, user, role)
