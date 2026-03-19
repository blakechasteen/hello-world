"""
Proto Matrix Adapter
=====================

Matrix/Element ChatOps integration for Proto code agent.

Provides bot commands for code assistance in Matrix rooms:
    - !proto <query>          - Ask any code-related question
    - !proto review <file>    - Code review
    - !proto explain <code>   - Explain code
    - !proto test <file>      - Run tests
    - !proto security <file>  - Security scan
    - !proto git <cmd>        - Git operations
    - !proto run <code>       - Execute Python code
    - !proto help             - Show help

Features:
    - Room-based persistent sessions
    - Context file support
    - Conversation history
    - Async message handling
    - Audit trail for all operations
    - Integration with Tier 2 abilities

Architecture:
    - Built on HoloLoom ChatOps pattern
    - Routes through ProtoEngine.process()
    - Uses abilities (code_execution, git_operations, test_runner, security_scan)

Usage:
    from hololoom.apps.departments.proto.adapters.matrix import (
        ProtoMatrixHandlers,
        create_proto_handlers,
    )

    handlers = create_proto_handlers()
    await handlers.startup()
    handlers.register_all(matrix_client)

    # Handle messages
    response = await handlers.handle_message(message)

    await handlers.shutdown()

Status: Production ready (2025-12-04)
"""

from .proto_handlers import (
    MatrixMessage,
    ProtoMatrixHandlers,
    ProtoSession,
    create_proto_handlers,
)

__all__ = [
    "ProtoMatrixHandlers",
    "MatrixMessage",
    "ProtoSession",
    "create_proto_handlers",
]
