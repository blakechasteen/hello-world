/**
 * Production LLM Client Integration
 * Supports Anthropic Claude, OpenAI GPT-4, and Ollama (local)
 * Date: November 2025
 *
 * Features:
 * - Multi-provider support (Anthropic, OpenAI, Ollama)
 * - Unified interface
 * - Cost tracking
 * - Automatic retries
 * - Fallback providers
 * - Streaming support
 */

/**
 * LLM Provider Types
 */
const LLMProvider = {
    ANTHROPIC: 'anthropic',
    OPENAI: 'openai',
    OLLAMA: 'ollama',
    MOCK: 'mock'
};

/**
 * Cost Tracker
 */
class CostTracker {
    constructor() {
        this.costs = {
            total: 0.0,
            byProvider: {},
            byModel: {},
            requestCount: 0
        };

        // Pricing (per 1M tokens)
        this.pricing = {
            // Anthropic Claude
            'claude-3-5-sonnet-20241022': { input: 3.00, output: 15.00 },
            'claude-3-opus-20240229': { input: 15.00, output: 75.00 },
            'claude-3-haiku-20240307': { input: 0.25, output: 1.25 },

            // OpenAI GPT
            'gpt-4-turbo-preview': { input: 10.00, output: 30.00 },
            'gpt-4': { input: 30.00, output: 60.00 },
            'gpt-3.5-turbo': { input: 0.50, output: 1.50 },

            // Ollama (local - free)
            'ollama': { input: 0.0, output: 0.0 }
        };
    }

    trackRequest(provider, model, inputTokens, outputTokens) {
        const pricing = this.pricing[model] || { input: 0, output: 0 };

        const cost = (inputTokens / 1_000_000) * pricing.input +
                    (outputTokens / 1_000_000) * pricing.output;

        this.costs.total += cost;
        this.costs.requestCount++;

        if (!this.costs.byProvider[provider]) {
            this.costs.byProvider[provider] = 0.0;
        }
        this.costs.byProvider[provider] += cost;

        if (!this.costs.byModel[model]) {
            this.costs.byModel[model] = 0.0;
        }
        this.costs.byModel[model] += cost;

        return cost;
    }

    getCosts() {
        return { ...this.costs };
    }

    reset() {
        this.costs = {
            total: 0.0,
            byProvider: {},
            byModel: {},
            requestCount: 0
        };
    }
}

/**
 * Base LLM Client (Abstract)
 */
class BaseLLMClient {
    constructor(config = {}) {
        this.config = config;
        this.costTracker = config.costTracker || new CostTracker();
        this.requestCount = 0;
        this.errorCount = 0;
    }

    async complete(prompt, systemPrompt = null, options = {}) {
        throw new Error('complete() must be implemented by subclass');
    }

    async chat(messages, options = {}) {
        throw new Error('chat() must be implemented by subclass');
    }

    async *stream(prompt, systemPrompt = null, options = {}) {
        throw new Error('stream() must be implemented by subclass');
    }

    estimateTokens(text) {
        // Rough estimate: 1 token ≈ 4 characters
        return Math.ceil(text.length / 4);
    }

    getStats() {
        return {
            requests: this.requestCount,
            errors: this.errorCount,
            costs: this.costTracker.getCosts()
        };
    }
}

/**
 * Anthropic Claude Client
 */
class AnthropicClient extends BaseLLMClient {
    constructor(config = {}) {
        super(config);
        this.apiKey = config.apiKey || process.env.ANTHROPIC_API_KEY;
        this.model = config.model || 'claude-3-5-sonnet-20241022';
        this.baseURL = 'https://api.anthropic.com/v1';
        this.maxRetries = config.maxRetries || 3;

        if (!this.apiKey) {
            console.warn('[AnthropicClient] No API key provided');
        }
    }

    async complete(prompt, systemPrompt = null, options = {}) {
        const messages = [{ role: 'user', content: prompt }];

        const requestBody = {
            model: this.model,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
            ...(systemPrompt && { system: systemPrompt })
        };

        let lastError;
        for (let attempt = 0; attempt < this.maxRetries; attempt++) {
            try {
                const response = await fetch(`${this.baseURL}/messages`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': this.apiKey,
                        'anthropic-version': '2023-06-01'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(`Anthropic API error: ${error.error?.message || response.statusText}`);
                }

                const data = await response.json();

                // Track costs
                const inputTokens = data.usage.input_tokens;
                const outputTokens = data.usage.output_tokens;
                this.costTracker.trackRequest(LLMProvider.ANTHROPIC, this.model, inputTokens, outputTokens);

                this.requestCount++;

                return data.content[0].text;

            } catch (error) {
                lastError = error;
                this.errorCount++;

                if (attempt < this.maxRetries - 1) {
                    // Exponential backoff
                    const delay = Math.pow(2, attempt) * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw new Error(`Anthropic request failed after ${this.maxRetries} attempts: ${lastError.message}`);
    }

    async chat(messages, options = {}) {
        const requestBody = {
            model: this.model,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
            ...(options.system && { system: options.system })
        };

        const response = await fetch(`${this.baseURL}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': this.apiKey,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Anthropic API error: ${response.statusText}`);
        }

        const data = await response.json();

        this.costTracker.trackRequest(
            LLMProvider.ANTHROPIC,
            this.model,
            data.usage.input_tokens,
            data.usage.output_tokens
        );

        this.requestCount++;

        return data.content[0].text;
    }

    async *stream(prompt, systemPrompt = null, options = {}) {
        const messages = [{ role: 'user', content: prompt }];

        const requestBody = {
            model: this.model,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
            stream: true,
            ...(systemPrompt && { system: systemPrompt })
        };

        const response = await fetch(`${this.baseURL}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': this.apiKey,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Anthropic API error: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'content_block_delta') {
                        yield data.delta.text;
                    }
                }
            }
        }

        this.requestCount++;
    }
}

/**
 * OpenAI GPT Client
 */
class OpenAIClient extends BaseLLMClient {
    constructor(config = {}) {
        super(config);
        this.apiKey = config.apiKey || process.env.OPENAI_API_KEY;
        this.model = config.model || 'gpt-4-turbo-preview';
        this.baseURL = 'https://api.openai.com/v1';
        this.maxRetries = config.maxRetries || 3;

        if (!this.apiKey) {
            console.warn('[OpenAIClient] No API key provided');
        }
    }

    async complete(prompt, systemPrompt = null, options = {}) {
        const messages = [];
        if (systemPrompt) {
            messages.push({ role: 'system', content: systemPrompt });
        }
        messages.push({ role: 'user', content: prompt });

        return await this.chat(messages, options);
    }

    async chat(messages, options = {}) {
        const requestBody = {
            model: this.model,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7
        };

        let lastError;
        for (let attempt = 0; attempt < this.maxRetries; attempt++) {
            try {
                const response = await fetch(`${this.baseURL}/chat/completions`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.apiKey}`
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
                }

                const data = await response.json();

                // Track costs
                const inputTokens = data.usage.prompt_tokens;
                const outputTokens = data.usage.completion_tokens;
                this.costTracker.trackRequest(LLMProvider.OPENAI, this.model, inputTokens, outputTokens);

                this.requestCount++;

                return data.choices[0].message.content;

            } catch (error) {
                lastError = error;
                this.errorCount++;

                if (attempt < this.maxRetries - 1) {
                    const delay = Math.pow(2, attempt) * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw new Error(`OpenAI request failed after ${this.maxRetries} attempts: ${lastError.message}`);
    }

    async *stream(prompt, systemPrompt = null, options = {}) {
        const messages = [];
        if (systemPrompt) {
            messages.push({ role: 'system', content: systemPrompt });
        }
        messages.push({ role: 'user', content: prompt });

        const requestBody = {
            model: this.model,
            messages: messages,
            max_tokens: options.maxTokens || 4096,
            temperature: options.temperature || 0.7,
            stream: true
        };

        const response = await fetch(`${this.baseURL}/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;

                    const parsed = JSON.parse(data);
                    const delta = parsed.choices[0]?.delta?.content;
                    if (delta) {
                        yield delta;
                    }
                }
            }
        }

        this.requestCount++;
    }
}

/**
 * Ollama Local LLM Client
 */
class OllamaClient extends BaseLLMClient {
    constructor(config = {}) {
        super(config);
        this.model = config.model || 'llama3.2:3b';
        this.baseURL = config.baseURL || 'http://localhost:11434';
        this.maxRetries = config.maxRetries || 3;
    }

    async complete(prompt, systemPrompt = null, options = {}) {
        const fullPrompt = systemPrompt ?
            `${systemPrompt}\n\nUser: ${prompt}\n\nAssistant:` :
            prompt;

        const requestBody = {
            model: this.model,
            prompt: fullPrompt,
            stream: false,
            options: {
                temperature: options.temperature || 0.7,
                num_predict: options.maxTokens || 4096
            }
        };

        let lastError;
        for (let attempt = 0; attempt < this.maxRetries; attempt++) {
            try {
                const response = await fetch(`${this.baseURL}/api/generate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    throw new Error(`Ollama API error: ${response.statusText}`);
                }

                const data = await response.json();

                // Track (no cost for local)
                this.costTracker.trackRequest(LLMProvider.OLLAMA, this.model, 0, 0);
                this.requestCount++;

                return data.response;

            } catch (error) {
                lastError = error;
                this.errorCount++;

                if (attempt < this.maxRetries - 1) {
                    const delay = Math.pow(2, attempt) * 500;  // Shorter delay for local
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw new Error(`Ollama request failed after ${this.maxRetries} attempts: ${lastError.message}`);
    }

    async chat(messages, options = {}) {
        const requestBody = {
            model: this.model,
            messages: messages,
            stream: false,
            options: {
                temperature: options.temperature || 0.7,
                num_predict: options.maxTokens || 4096
            }
        };

        const response = await fetch(`${this.baseURL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Ollama API error: ${response.statusText}`);
        }

        const data = await response.json();
        this.costTracker.trackRequest(LLMProvider.OLLAMA, this.model, 0, 0);
        this.requestCount++;

        return data.message.content;
    }

    async *stream(prompt, systemPrompt = null, options = {}) {
        const fullPrompt = systemPrompt ?
            `${systemPrompt}\n\nUser: ${prompt}\n\nAssistant:` :
            prompt;

        const requestBody = {
            model: this.model,
            prompt: fullPrompt,
            stream: true,
            options: {
                temperature: options.temperature || 0.7,
                num_predict: options.maxTokens || 4096
            }
        };

        const response = await fetch(`${this.baseURL}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Ollama API error: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                const data = JSON.parse(line);
                if (data.response) {
                    yield data.response;
                }
            }
        }

        this.requestCount++;
    }
}

/**
 * Mock LLM Client (for testing)
 */
class MockLLMClient extends BaseLLMClient {
    async complete(prompt, systemPrompt = null, options = {}) {
        this.requestCount++;

        // Simulate delay
        await new Promise(resolve => setTimeout(resolve, 100));

        return JSON.stringify({
            response: "Mock LLM response",
            confidence: 0.75
        });
    }

    async chat(messages, options = {}) {
        return await this.complete(messages[messages.length - 1].content);
    }

    async *stream(prompt, systemPrompt = null, options = {}) {
        const response = await this.complete(prompt, systemPrompt, options);
        for (const char of response) {
            yield char;
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }
}

/**
 * LLM Client Factory
 * Creates appropriate client based on provider
 */
class LLMClientFactory {
    static create(provider, config = {}) {
        const sharedCostTracker = new CostTracker();

        switch (provider) {
            case LLMProvider.ANTHROPIC:
                return new AnthropicClient({ ...config, costTracker: sharedCostTracker });

            case LLMProvider.OPENAI:
                return new OpenAIClient({ ...config, costTracker: sharedCostTracker });

            case LLMProvider.OLLAMA:
                return new OllamaClient({ ...config, costTracker: sharedCostTracker });

            case LLMProvider.MOCK:
                return new MockLLMClient({ ...config, costTracker: sharedCostTracker });

            default:
                throw new Error(`Unknown LLM provider: ${provider}`);
        }
    }

    /**
     * Create client with automatic fallback
     */
    static createWithFallback(preferredProvider, fallbackProviders = [], config = {}) {
        const providers = [preferredProvider, ...fallbackProviders];

        for (const provider of providers) {
            try {
                const client = LLMClientFactory.create(provider, config);

                // Wrap with fallback logic
                return new FallbackLLMClient(client, providers.slice(1), config);

            } catch (error) {
                console.warn(`[LLMFactory] Failed to create ${provider}: ${error.message}`);
            }
        }

        throw new Error('All LLM providers failed to initialize');
    }
}

/**
 * Fallback LLM Client
 * Automatically falls back to alternative providers on failure
 */
class FallbackLLMClient extends BaseLLMClient {
    constructor(primaryClient, fallbackProviders, config) {
        super(config);
        this.primary = primaryClient;
        this.fallbacks = fallbackProviders.map(p => LLMClientFactory.create(p, config));
        this.currentClient = this.primary;
    }

    async complete(prompt, systemPrompt = null, options = {}) {
        try {
            return await this.currentClient.complete(prompt, systemPrompt, options);
        } catch (error) {
            console.warn('[Fallback] Primary failed, trying fallback...');

            for (const fallback of this.fallbacks) {
                try {
                    this.currentClient = fallback;
                    return await fallback.complete(prompt, systemPrompt, options);
                } catch (fallbackError) {
                    console.warn(`[Fallback] ${fallback.constructor.name} failed`);
                }
            }

            throw new Error('All LLM providers failed');
        }
    }

    async chat(messages, options = {}) {
        try {
            return await this.currentClient.chat(messages, options);
        } catch (error) {
            for (const fallback of this.fallbacks) {
                try {
                    this.currentClient = fallback;
                    return await fallback.chat(messages, options);
                } catch (fallbackError) {
                    // Continue to next fallback
                }
            }
            throw new Error('All LLM providers failed');
        }
    }

    getStats() {
        return {
            primary: this.primary.getStats(),
            fallbacks: this.fallbacks.map(f => f.getStats()),
            currentProvider: this.currentClient.constructor.name
        };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LLMProvider,
        LLMClientFactory,
        AnthropicClient,
        OpenAIClient,
        OllamaClient,
        MockLLMClient,
        CostTracker
    };
}
