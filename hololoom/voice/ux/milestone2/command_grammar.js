/**
 * Command Grammar Parser
 * Milestone 2: Structured command parsing with discrete syntax
 * Date: November 2025
 *
 * Features:
 * - Structured command grammar (VERB NOUN [MODIFIERS])
 * - Slot filling (extract parameters from commands)
 * - Command validation (check required/optional parameters)
 * - Command completion (suggest missing parameters)
 * - Multi-language command support
 * - Alias expansion (shortcuts for common commands)
 */

/**
 * Command Grammar
 * Defines the structure of discrete commands
 *
 * Command format: "loom <verb> <noun> [<modifier>=<value>]"
 * Example: "loom open thread machine-learning topic=AI"
 * Example: "loom filter threads by=topic value=AI"
 * Example: "loom activate background threads"
 */

class CommandGrammar {
    constructor() {
        // Command registry
        this.commands = new Map();

        // Aliases
        this.aliases = new Map();

        // Language mappings
        this.languageMappings = new Map();

        // Initialize built-in commands
        this._registerBuiltinCommands();
    }

    /**
     * Register a command
     */
    registerCommand(commandDef) {
        const { verb, noun, description, params, examples, handler } = commandDef;

        const commandKey = `${verb}_${noun}`;
        this.commands.set(commandKey, {
            verb,
            noun,
            description,
            params: params || [],
            examples: examples || [],
            handler: handler || null
        });

        console.log(`[CommandGrammar] Registered command: ${verb} ${noun}`);
    }

    /**
     * Register an alias
     */
    registerAlias(alias, fullCommand) {
        this.aliases.set(alias, fullCommand);
        console.log(`[CommandGrammar] Registered alias: ${alias} → ${fullCommand}`);
    }

    /**
     * Parse command from transcript
     */
    parse(transcript) {
        const startTime = Date.now();

        // Normalize transcript
        const normalized = transcript.toLowerCase().trim();

        // Check for wake word
        const hasWakeWord = normalized.startsWith('loom,') || normalized.startsWith('loom ');
        let commandText = normalized;

        if (hasWakeWord) {
            // Remove wake word
            commandText = normalized.replace(/^loom,?\s*/, '');
        }

        // Expand aliases
        commandText = this._expandAliases(commandText);

        // Tokenize
        const tokens = this._tokenize(commandText);

        if (tokens.length < 2) {
            return {
                success: false,
                error: 'Command too short (need at least verb + noun)',
                transcript: transcript
            };
        }

        // Extract verb and noun
        const verb = tokens[0];
        const noun = tokens[1];
        const commandKey = `${verb}_${noun}`;

        // Look up command
        const commandDef = this.commands.get(commandKey);

        if (!commandDef) {
            return {
                success: false,
                error: `Unknown command: ${verb} ${noun}`,
                transcript: transcript,
                suggestions: this._findSimilarCommands(verb, noun)
            };
        }

        // Extract parameters
        const params = this._extractParams(tokens.slice(2), commandDef.params);

        // Validate required parameters
        const validation = this._validateParams(params, commandDef.params);

        if (!validation.valid) {
            return {
                success: false,
                error: validation.error,
                transcript: transcript,
                missingParams: validation.missingParams,
                suggestions: this._suggestCompletion(commandDef, params)
            };
        }

        // Success!
        return {
            success: true,
            verb: verb,
            noun: noun,
            params: params,
            commandDef: commandDef,
            hasWakeWord: hasWakeWord,
            latencyMs: Date.now() - startTime,
            timestamp: new Date()
        };
    }

    /**
     * Expand aliases
     */
    _expandAliases(text) {
        for (const [alias, fullCommand] of this.aliases.entries()) {
            if (text.startsWith(alias)) {
                return text.replace(alias, fullCommand);
            }
        }
        return text;
    }

    /**
     * Tokenize command text
     */
    _tokenize(text) {
        // Split by spaces, but preserve "key=value" pairs
        const tokens = [];
        const parts = text.split(/\s+/);

        for (const part of parts) {
            if (part.includes('=')) {
                // Key-value pair
                tokens.push(part);
            } else {
                // Regular token
                tokens.push(part);
            }
        }

        return tokens;
    }

    /**
     * Extract parameters from tokens
     */
    _extractParams(tokens, paramDefs) {
        const params = {};

        let i = 0;
        while (i < tokens.length) {
            const token = tokens[i];

            if (token.includes('=')) {
                // Explicit key=value
                const [key, value] = token.split('=');
                params[key] = value;
                i++;
            } else {
                // Positional parameter
                // Match to first unfilled parameter
                for (const paramDef of paramDefs) {
                    if (!params[paramDef.name]) {
                        params[paramDef.name] = token;
                        break;
                    }
                }
                i++;
            }
        }

        return params;
    }

    /**
     * Validate parameters
     */
    _validateParams(params, paramDefs) {
        const missingParams = [];

        for (const paramDef of paramDefs) {
            if (paramDef.required && !params[paramDef.name]) {
                missingParams.push(paramDef.name);
            }
        }

        if (missingParams.length > 0) {
            return {
                valid: false,
                error: `Missing required parameters: ${missingParams.join(', ')}`,
                missingParams: missingParams
            };
        }

        return { valid: true };
    }

    /**
     * Find similar commands (for suggestions)
     */
    _findSimilarCommands(verb, noun) {
        const suggestions = [];

        for (const [commandKey, commandDef] of this.commands.entries()) {
            // Match on verb
            if (commandDef.verb === verb) {
                suggestions.push(`${commandDef.verb} ${commandDef.noun}`);
            }

            // Match on noun
            if (commandDef.noun === noun) {
                suggestions.push(`${commandDef.verb} ${commandDef.noun}`);
            }
        }

        return suggestions.slice(0, 3); // Top 3 suggestions
    }

    /**
     * Suggest command completion
     */
    _suggestCompletion(commandDef, currentParams) {
        const suggestions = [];

        for (const paramDef of commandDef.params) {
            if (paramDef.required && !currentParams[paramDef.name]) {
                suggestions.push({
                    param: paramDef.name,
                    type: paramDef.type,
                    description: paramDef.description,
                    examples: paramDef.examples || []
                });
            }
        }

        return suggestions;
    }

    /**
     * Get all commands (for help)
     */
    getAllCommands() {
        const commands = [];

        for (const [commandKey, commandDef] of this.commands.entries()) {
            commands.push({
                command: `${commandDef.verb} ${commandDef.noun}`,
                description: commandDef.description,
                params: commandDef.params,
                examples: commandDef.examples
            });
        }

        return commands;
    }

    /**
     * Register built-in commands
     */
    _registerBuiltinCommands() {
        // Thread commands
        this.registerCommand({
            verb: 'open',
            noun: 'thread',
            description: 'Open a thread by name',
            params: [
                { name: 'name', type: 'string', required: true, description: 'Thread name' }
            ],
            examples: [
                'loom open thread machine-learning',
                'loom open thread name=machine-learning'
            ]
        });

        this.registerCommand({
            verb: 'create',
            noun: 'thread',
            description: 'Create a new thread',
            params: [
                { name: 'name', type: 'string', required: true, description: 'Thread name' },
                { name: 'topic', type: 'string', required: false, description: 'Thread topic' }
            ],
            examples: [
                'loom create thread machine-learning',
                'loom create thread name=ML topic=AI'
            ]
        });

        this.registerCommand({
            verb: 'close',
            noun: 'thread',
            description: 'Close a thread',
            params: [
                { name: 'name', type: 'string', required: true, description: 'Thread name' }
            ],
            examples: [
                'loom close thread machine-learning'
            ]
        });

        this.registerCommand({
            verb: 'list',
            noun: 'threads',
            description: 'List all threads',
            params: [
                { name: 'filter', type: 'string', required: false, description: 'Filter by state (active, background, archived)' }
            ],
            examples: [
                'loom list threads',
                'loom list threads filter=active'
            ]
        });

        this.registerCommand({
            verb: 'filter',
            noun: 'threads',
            description: 'Filter threads by criteria',
            params: [
                { name: 'by', type: 'string', required: true, description: 'Filter field (topic, state, name)' },
                { name: 'value', type: 'string', required: true, description: 'Filter value' }
            ],
            examples: [
                'loom filter threads by=topic value=AI',
                'loom filter threads by=state value=active'
            ]
        });

        this.registerCommand({
            verb: 'activate',
            noun: 'background',
            description: 'Activate all background threads',
            params: [],
            examples: [
                'loom activate background threads'
            ]
        });

        // System commands
        this.registerCommand({
            verb: 'set',
            noun: 'language',
            description: 'Set voice recognition language',
            params: [
                { name: 'lang', type: 'string', required: true, description: 'Language code (en-US, es-ES, etc.)' }
            ],
            examples: [
                'loom set language en-US',
                'loom set language es-ES'
            ]
        });

        this.registerCommand({
            verb: 'set',
            noun: 'mode',
            description: 'Set voice mode',
            params: [
                { name: 'mode', type: 'string', required: true, description: 'Mode (conversational, command, streaming)' }
            ],
            examples: [
                'loom set mode conversational',
                'loom set mode command'
            ]
        });

        this.registerCommand({
            verb: 'show',
            noun: 'help',
            description: 'Show help for commands',
            params: [
                { name: 'command', type: 'string', required: false, description: 'Specific command to show help for' }
            ],
            examples: [
                'loom show help',
                'loom show help open thread'
            ]
        });

        // Aliases
        this.registerAlias('ls', 'list threads');
        this.registerAlias('new', 'create thread');
        this.registerAlias('switch', 'open thread');
        this.registerAlias('activate all', 'activate background threads');
    }
}


/**
 * Command Completion Engine
 * Suggests completions for partial commands
 */
class CommandCompletionEngine {
    constructor(grammar) {
        this.grammar = grammar;
    }

    /**
     * Get completions for partial command
     */
    getCompletions(partialCommand) {
        const tokens = partialCommand.toLowerCase().trim().split(/\s+/);

        if (tokens.length === 0) {
            return [];
        }

        if (tokens.length === 1) {
            // Complete verb
            return this._completeVerb(tokens[0]);
        }

        if (tokens.length === 2) {
            // Complete noun
            return this._completeNoun(tokens[0], tokens[1]);
        }

        // Complete parameters
        return this._completeParams(tokens);
    }

    _completeVerb(partial) {
        const verbs = new Set();

        for (const [commandKey, commandDef] of this.grammar.commands.entries()) {
            if (commandDef.verb.startsWith(partial)) {
                verbs.add(commandDef.verb);
            }
        }

        return Array.from(verbs);
    }

    _completeNoun(verb, partial) {
        const nouns = [];

        for (const [commandKey, commandDef] of this.grammar.commands.entries()) {
            if (commandDef.verb === verb && commandDef.noun.startsWith(partial)) {
                nouns.push(commandDef.noun);
            }
        }

        return nouns;
    }

    _completeParams(tokens) {
        const verb = tokens[0];
        const noun = tokens[1];
        const commandKey = `${verb}_${noun}`;

        const commandDef = this.grammar.commands.get(commandKey);
        if (!commandDef) {
            return [];
        }

        // Extract current params
        const currentParams = this.grammar._extractParams(tokens.slice(2), commandDef.params);

        // Find unfilled params
        const suggestions = [];

        for (const paramDef of commandDef.params) {
            if (!currentParams[paramDef.name]) {
                suggestions.push({
                    param: paramDef.name,
                    type: paramDef.type,
                    required: paramDef.required,
                    description: paramDef.description,
                    format: `${paramDef.name}=<value>`
                });
            }
        }

        return suggestions;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CommandGrammar,
        CommandCompletionEngine
    };
}
