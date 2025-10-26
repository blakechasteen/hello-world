"""
MCP Server Entry Point (Standalone)
====================================
This is a standalone entry point that avoids HoloLoom package imports.
Used by Claude Desktop MCP integration.
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import logging
    from pathlib import Path
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Setup logging to stderr for Claude to see
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Import MCP components
        from mcp.server import Server
        from mcp.types import Resource, Tool, TextContent
        
        # Import protocol directly (avoid package __init__)
        import importlib.util
        
        protocol_path = Path(__file__).parent / "protocol.py"
        spec = importlib.util.spec_from_file_location("memory_protocol", protocol_path)
        protocol = importlib.util.module_from_spec(spec)
        sys.modules['memory_protocol'] = protocol
        spec.loader.exec_module(protocol)
        
        # Get what we need
        UnifiedMemoryInterface = protocol.UnifiedMemoryInterface
        Strategy = protocol.Strategy
        create_unified_memory = protocol.create_unified_memory
        
        logger.info("✓ Protocol loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to import dependencies: {e}", exc_info=True)
        sys.exit(1)
    
    # Global memory instance
    memory = None
    
    # Create server
    server = Server("holoLoom-memory")
    
    # ========================================================================
    # Resources
    # ========================================================================
    
    @server.list_resources()
    async def list_memories():
        """List recent memories as resources."""
        try:
            result = await memory.recall(
                query="",
                strategy=Strategy.TEMPORAL,
                limit=100
            )
            
            return [
                Resource(
                    uri=f"memory://{mem.id}",
                    name=mem.text[:60] + "..." if len(mem.text) > 60 else mem.text,
                    mimeType="text/plain",
                    description=f"Stored: {mem.timestamp.isoformat()}"
                )
                for mem in result.memories
            ]
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []
    
    @server.read_resource()
    async def read_memory(uri: str):
        """Read specific memory."""
        try:
            mem_id = uri.replace("memory://", "")
            result = await memory.recall(query=mem_id, strategy=Strategy.TEMPORAL, limit=100)
            
            for mem in result.memories:
                if mem.id == mem_id:
                    return f"Memory: {mem.text}\nContext: {mem.context}\nTags: {mem.tags}"
            return f"Memory {mem_id} not found"
        except Exception as e:
            return f"Error: {e}"
    
    # ========================================================================
    # Tools
    # ========================================================================
    
    @server.list_tools()
    async def list_tools():
        """List available tools."""
        return [
            Tool(
                name="recall_memories",
                description="Search memories using various strategies (temporal, semantic, graph, pattern, fused)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "strategy": {
                            "type": "string",
                            "enum": ["temporal", "semantic", "graph", "pattern", "fused"],
                            "default": "fused"
                        },
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="store_memory",
                description="Store a new memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Memory text"},
                        "context": {"type": "object"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="memory_health",
                description="Check memory system health",
                inputSchema={"type": "object", "properties": {}}
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Execute tool."""
        try:
            if name == "recall_memories":
                strategy_name = arguments.get("strategy", "fused").upper()
                strategy = Strategy[strategy_name]
                
                result = await memory.recall(
                    query=arguments["query"],
                    strategy=strategy,
                    limit=arguments.get("limit", 10)
                )
                
                if not result.memories:
                    return [TextContent(type="text", text=f"No memories found for: {arguments['query']}")]
                
                lines = [f"Found {len(result.memories)} memories:\n"]
                for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
                    lines.append(f"{i}. [{score:.3f}] {mem.text}")
                    if mem.tags:
                        lines.append(f"   Tags: {', '.join(mem.tags)}")
                
                return [TextContent(type="text", text="\n".join(lines))]
            
            elif name == "store_memory":
                # Extract arguments
                text = arguments["text"]
                context = arguments.get("context", {})
                tags = arguments.get("tags", [])
                
                # Add tags to context if provided
                if tags:
                    context["tags"] = tags
                
                mem_id = await memory.store(
                    text=text,
                    context=context,
                    user_id=arguments.get("user_id")
                )
                return [TextContent(type="text", text=f"✓ Memory stored: {mem_id}\n📝 {text[:100]}...")]
            
            elif name == "memory_health":
                # Get store stats directly
                store = memory._store
                memory_count = len(store.memories) if hasattr(store, 'memories') else 0
                backend_type = type(store).__name__
                
                return [TextContent(
                    type="text",
                    text=f"✅ Status: healthy\n🗄️ Backend: {backend_type}\n📊 Memories: {memory_count}"
                )]
            
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {e}")]
    
    # ========================================================================
    # Server Main
    # ========================================================================
    
    async def main():
        """Initialize and run server."""
        global memory
        
        logger.info("Initializing HoloLoom Memory MCP Server...")
        
        try:
            # Load InMemoryStore directly
            store_path = Path(__file__).parent / "stores" / "in_memory_store.py"
            spec = importlib.util.spec_from_file_location("in_memory_store", store_path)
            store_module = importlib.util.module_from_spec(spec)
            
            # Make protocol available for the store
            sys.modules['protocol'] = protocol
            spec.loader.exec_module(store_module)
            
            InMemoryStore = store_module.InMemoryStore
            logger.info("✓ InMemoryStore loaded")
            
            # Create memory interface directly
            memory = UnifiedMemoryInterface(_store=InMemoryStore())
            
            # Test basic functionality
            test_id = await memory.store("MCP server initialized")
            logger.info(f"Memory system ready (test store: {test_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}", exc_info=True)
            sys.exit(1)
        
        # Run server on stdin/stdout (for MCP protocol)
        logger.info("Starting MCP server on stdio...")
        
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
