/**
 * TypeScript utilities module for LSP testing.
 *
 * Tests LSP support for TypeScript/JavaScript files, including:
 * - Interface definitions
 * - Function declarations with type annotations
 * - Class definitions
 * - Import statements
 */

import * as os from "os";
import * as fs from "fs";
import { EventEmitter } from "events";

/**
 * Represents options for a mathematical operation.
 */
interface OperationOptions {
    /** Enable verbose logging */
    verbose?: boolean;
    /** Maximum precision for calculations */
    precision?: number;
    /** Cache results for repeated operations */
    cache?: boolean;
}

/**
 * A simple math utility class for performing calculations.
 *
 * Supports basic arithmetic operations and maintains a history
 * of all operations performed.
 */
export class MathUtil extends EventEmitter {
    private history: Array<{ op: string; result: number }> = [];
    private options: OperationOptions;

    /**
     * Creates a new MathUtil instance.
     *
     * @param options Configuration options for the math utility
     */
    constructor(options: OperationOptions = {}) {
        super();
        this.options = {
            verbose: false,
            precision: 2,
            cache: true,
            ...options,
        };
    }

    /**
     * Add two numbers together.
     *
     * @param a First number
     * @param b Second number
     * @returns The sum of a and b
     */
    public add(a: number, b: number): number {
        const result = a + b;
        this.recordOperation("add", result);
        return result;
    }

    /**
     * Multiply two numbers.
     *
     * @param a First number
     * @param b Second number
     * @returns The product of a and b
     */
    public multiply(a: number, b: number): number {
        const result = a * b;
        this.recordOperation("multiply", result);
        return result;
    }

    /**
     * Divide two numbers with error handling.
     *
     * @param a Numerator
     * @param b Denominator
     * @returns The quotient, or null if division by zero
     */
    public divide(a: number, b: number): number | null {
        if (b === 0) {
            this.log("Warning: Division by zero");
            return null;
        }
        const result = a / b;
        this.recordOperation("divide", result);
        return result;
    }

    /**
     * Get the calculation history.
     *
     * @returns Array of operations and their results
     */
    public getHistory(): Array<{ op: string; result: number }> {
        return [...this.history];
    }

    /**
     * Clear the operation history.
     */
    public clearHistory(): void {
        this.history = [];
        this.log("History cleared");
    }

    /**
     * Record an operation in the history.
     *
     * @param operation The operation name
     * @param result The result of the operation
     */
    private recordOperation(operation: string, result: number): void {
        this.history.push({ op: operation, result });
        this.emit("operation", { operation, result });
        this.log(`${operation}: ${result}`);
    }

    /**
     * Log a message if verbose mode is enabled.
     *
     * @param message The message to log
     */
    private log(message: string): void {
        if (this.options.verbose) {
            console.log(`[MathUtil] ${message}`);
        }
    }
}

/**
 * Represents an arm in a multi-armed bandit problem.
 * Uses Thompson Sampling for exploration/exploitation.
 */
export interface BanditArmState {
    /** Number of successful trials */
    successes: number;
    /** Number of failed trials */
    failures: number;
    /** Alpha parameter for Beta distribution */
    alpha: number;
    /** Beta parameter for Beta distribution */
    beta: number;
}

/**
 * A multi-armed bandit arm using Thompson Sampling.
 *
 * Maintains a Beta distribution prior that is updated based on
 * reward signals.
 */
export class BanditArm {
    private state: BanditArmState;

    /**
     * Creates a new BanditArm with initial parameters.
     *
     * @param alpha Initial alpha parameter
     * @param beta Initial beta parameter
     */
    constructor(alpha: number = 1.0, beta: number = 1.0) {
        this.state = {
            successes: 0,
            failures: 0,
            alpha,
            beta,
        };
    }

    /**
     * Update the arm based on a reward signal.
     *
     * @param reward Reward signal (0 = failure, 1 = success)
     */
    public update(reward: number): void {
        if (reward > 0.5) {
            this.state.successes += 1;
            this.state.alpha += 1;
        } else {
            this.state.failures += 1;
            this.state.beta += 1;
        }
    }

    /**
     * Sample from the Thompson Sampling posterior.
     *
     * @returns A sample from Beta(alpha, beta)
     */
    public sampleThompson(): number {
        // Simplified: random between 0 and 1
        // Real implementation would use actual Beta sampling
        return Math.random();
    }

    /**
     * Get the current state of the arm.
     *
     * @returns The arm's state
     */
    public getState(): Readonly<BanditArmState> {
        return { ...this.state };
    }
}

/**
 * Helper function to create multiple bandit arms.
 *
 * @param count Number of arms to create
 * @returns Array of BanditArm instances
 */
export function createBanditArms(count: number): BanditArm[] {
    return Array.from({ length: count }, () => new BanditArm());
}

/**
 * Process numeric data and compute statistics.
 *
 * @param data Array of numeric values
 * @returns Object with min, max, and mean statistics
 */
export function computeStatistics(data: number[]): {
    min: number;
    max: number;
    mean: number;
} {
    if (data.length === 0) {
        return { min: 0, max: 0, mean: 0 };
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const mean = data.reduce((a, b) => a + b, 0) / data.length;

    return { min, max, mean };
}

// NOTE: Add error handling for file operations
/**
 * Read a JSON file and parse its contents.
 *
 * @param filePath Path to the JSON file
 * @returns Parsed JSON object or null if file doesn't exist
 */
export function readJsonFile(filePath: string): unknown | null {
    try {
        const content = fs.readFileSync(filePath, "utf-8");
        return JSON.parse(content);
    } catch (error) {
        console.error(`Failed to read file: ${filePath}`, error);
        return null;
    }
}

// TODO: Implement write JSON file function
// TODO: Add compression support for large files

// FIXME: Optimize for large arrays
/**
 * Find the index of maximum value in array.
 *
 * @param arr Array of numbers
 * @returns Index of maximum value
 */
export function findMaxIndex(arr: number[]): number {
    if (arr.length === 0) {
        return -1;
    }
    let maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > arr[maxIdx]) {
            maxIdx = i;
        }
    }
    return maxIdx;
}

/**
 * Main entry point for testing.
 */
if (require.main === module) {
    const math = new MathUtil({ verbose: true });
    console.log("Result:", math.add(5, 3));
    console.log("History:", math.getHistory());

    const arms = createBanditArms(3);
    console.log("Created", arms.length, "bandit arms");

    const stats = computeStatistics([1, 2, 3, 4, 5]);
    console.log("Statistics:", stats);
}
