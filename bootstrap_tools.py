"""Bootstrap — register all tools into the shared ToolRegistry at startup."""

from core.tools import ToolRegistry
from tools.filesystem import read_file

tool_registry = ToolRegistry()

tool_registry.register(
    name="read_file",
    description="Read the text content of a local file",
    input_schema={"path": "string"},
    function=read_file,
)
