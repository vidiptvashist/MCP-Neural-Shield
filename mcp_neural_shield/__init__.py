from mcp_neural_shield.middleware import MCPVectorShieldMiddleware, ShieldMiddleware
from mcp_neural_shield.verify import verify_tool_metadata
from mcp_neural_shield.mcp_registry import MCPSemanticRegistry
from mcp_neural_shield.mcp_classifier_engine import MCPNeuralShield

__version__ = "0.2.3"

__all__ = [
    "MCPVectorShieldMiddleware",
    "ShieldMiddleware",
    "verify_tool_metadata",
    "MCPSemanticRegistry",
    "MCPNeuralShield",
]
