import React, { useCallback, useRef } from 'react';
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    addEdge,
    useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import GlassNode from './GlassNode';

const nodeTypes = {
    glass: GlassNode,
};

const initialNodes = [
    {
        id: '1',
        type: 'glass',
        position: { x: 100, y: 100 },
        data: { label: 'HoloLoom Query', type: 'query', icon: 'Q', description: 'Input Query' }
    },
    {
        id: '2',
        type: 'glass',
        position: { x: 400, y: 100 },
        data: { label: 'Embedder', type: 'process', icon: 'E', description: 'Vectorize' }
    },
];

const initialEdges = [
    { id: 'e1-2', source: '1', target: '2', animated: true, style: { stroke: 'var(--color-accent-primary)' } },
];

let id = 3;
const getId = () => `${id++}`;

const WorkflowCanvas = ({ onNodeSelect }) => {
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const { project } = useReactFlow();

    const onConnect = useCallback((params) => setEdges((eds) => addEdge({
        ...params,
        animated: true,
        style: { stroke: 'var(--color-accent-primary)' }
    }, eds)), [setEdges]);

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow');
            const label = event.dataTransfer.getData('application/label');
            const icon = event.dataTransfer.getData('application/icon');
            const nodeType = event.dataTransfer.getData('application/nodeType');

            // check if the dropped element is valid
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
            const position = project({
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
            });

            const newNode = {
                id: getId(),
                type: 'glass', // Always use our custom node type
                position,
                data: {
                    label: label,
                    type: nodeType, // query, process, memory
                    icon: icon,
                    description: 'New Agent'
                },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [project, setNodes]
    );

    const onNodeClick = useCallback((event, node) => {
        onNodeSelect(node.id);
    }, [onNodeSelect]);

    const onPaneClick = useCallback(() => {
        onNodeSelect(null);
    }, [onNodeSelect]);

    return (
        <div className="workflow-wrapper" ref={reactFlowWrapper} style={{ width: '100%', height: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={onNodeClick}
                onPaneClick={onPaneClick}
                onDrop={onDrop}
                onDragOver={onDragOver}
                nodeTypes={nodeTypes}
                fitView
            >
                <Background color="var(--color-border)" gap={20} />
                <Controls style={{
                    background: 'var(--glass-bg)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    fill: 'var(--color-text-primary)'
                }} />
                <MiniMap
                    style={{
                        background: 'var(--glass-bg)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '8px'
                    }}
                    nodeColor={(n) => {
                        return 'var(--color-accent-primary)';
                    }}
                />
            </ReactFlow>
        </div>
    );
};

export default WorkflowCanvas;
