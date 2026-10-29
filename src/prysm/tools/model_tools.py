"""Tool for querying and switching PRYSM's LLM model routing."""

from typing import Any

from prysm.tools.interfaces import BaseTool


class ModelRoutingTools(BaseTool):
    """Allows checking and switching between local models and secondary API harness."""

    def __init__(self, router: Any) -> None:
        self.router = router

    def get_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "switch_model_mode",
                "description": (
                    "Switch PRYSM's intelligence backend between 'local' (local Ollama coder + thinking models) "
                    "and 'api' (secondary cloud API harness)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["local", "api"],
                            "description": "Target intelligence mode: 'local' for offline Ollama models, 'api' for cloud harness.",
                        }
                    },
                    "required": ["mode"],
                },
            },
            {
                "name": "get_model_status",
                "description": "Get current LLM routing status, including whether local or API models are active.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "switch_model_mode":
            mode = arguments.get("mode", "local")
            if hasattr(self.router, "set_mode"):
                self.router.set_mode(mode)
            desc = (
                self.router.get_status_description()
                if hasattr(self.router, "get_status_description")
                else mode
            )
            return {"status": "success", "mode": mode, "description": desc}

        elif tool_name == "get_model_status":
            mode = self.router.get_mode() if hasattr(self.router, "get_mode") else "unknown"
            desc = (
                self.router.get_status_description()
                if hasattr(self.router, "get_status_description")
                else "unknown"
            )
            last_model = getattr(self.router, "last_used_model", "unknown")
            return {
                "active_mode": mode,
                "description": desc,
                "last_used_model": last_model,
            }

        return {"error": f"Unknown tool: {tool_name}"}

