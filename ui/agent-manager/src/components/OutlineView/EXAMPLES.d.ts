/**
 * Example Usage of Control Components
 * Shows common patterns for using PriorityControls, ThreadControls, and InjectMenu
 */
import React from 'react';
interface ThreadRowExample1Props {
    threadId: string;
    name: string;
    status: 'running' | 'paused' | 'idle' | 'completed' | 'failed' | 'cancelled';
    priority: number;
}
export declare const ThreadRowExample1: React.FC<ThreadRowExample1Props>;
export declare const ThreadRowHorizontalExample: React.FC<ThreadRowExample1Props>;
interface Step {
    id: string;
    name: string;
    status: 'pending' | 'running' | 'completed';
}
interface ThreadViewExampleProps {
    threadId: string;
    name: string;
    status: 'running' | 'paused' | 'completed';
    priority: number;
    steps: Step[];
}
export declare const ThreadDetailViewExample: React.FC<ThreadViewExampleProps>;
export declare const CompactThreadRowExample: React.FC<ThreadRowExample1Props>;
interface ThreadWithCallbacksProps {
    threadId: string;
    name: string;
    status: 'running' | 'paused' | 'idle';
    priority: number;
    onPriorityChange?: (newPriority: number) => void;
    onStatusChange?: (newStatus: string) => void;
    onInject?: (type: 'mrf' | 'mcts', value: any) => void;
}
export declare const ThreadRowWithCallbacksExample: React.FC<ThreadWithCallbacksProps>;
interface ThreadGridExampleProps {
    threads: ThreadRowExample1Props[];
}
export declare const ThreadGridExample: React.FC<ThreadGridExampleProps>;
export declare const ResponsiveThreadRowExample: React.FC<ThreadRowExample1Props>;
export declare const StyledThreadRowExample: React.FC<ThreadRowExample1Props>;
declare const _default: {
    ThreadRowExample1: React.FC<ThreadRowExample1Props>;
    ThreadRowHorizontalExample: React.FC<ThreadRowExample1Props>;
    ThreadDetailViewExample: React.FC<ThreadViewExampleProps>;
    CompactThreadRowExample: React.FC<ThreadRowExample1Props>;
    ThreadRowWithCallbacksExample: React.FC<ThreadWithCallbacksProps>;
    ThreadGridExample: React.FC<ThreadGridExampleProps>;
    ResponsiveThreadRowExample: React.FC<ThreadRowExample1Props>;
    StyledThreadRowExample: React.FC<ThreadRowExample1Props>;
};
export default _default;
//# sourceMappingURL=EXAMPLES.d.ts.map