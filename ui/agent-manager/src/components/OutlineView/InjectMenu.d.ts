/**
 * InjectMenu Component
 * Dropdown menu for MRF/MCTS injection controls
 * Allows users to inject optimization strategies at specific steps
 */
import React from 'react';
interface MCTSConfig {
    budget: number;
    exploration: number;
}
interface InjectMenuProps {
    /** Thread ID for context */
    threadId: string;
    /** Step ID to inject at */
    stepId: string;
    /** Whether MRF injection is eligible for this step */
    mrfEligible: boolean;
    /** Whether MCTS injection is eligible for this step */
    mctsEligible: boolean;
    /** Callback when MRF strategy is selected */
    onInjectMRF?: (strategy: string) => void;
    /** Callback when MCTS config is selected */
    onInjectMCTS?: (config: MCTSConfig) => void;
    /** Already applied injection type */
    injected?: 'mrf' | 'mcts' | null;
    /** Already applied MRF strategy (if injected) */
    appliedStrategy?: string;
    /** Size of the button */
    size?: 'sm' | 'md';
    /** Custom CSS class */
    className?: string;
}
export declare const InjectMenu: React.FC<InjectMenuProps>;
export default InjectMenu;
//# sourceMappingURL=InjectMenu.d.ts.map