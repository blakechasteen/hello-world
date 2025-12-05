/**
 * Architecture Visualization Tour
 * "How a Player Action Flows Through BigPlay"
 *
 * 12-step guided tour through the entire system architecture
 */

const architectureTourSteps = [
    {
        target: '[data-component="unity"]',
        title: '1. Player Input from Game Client',
        content: `
            <p>The journey begins when a player interacts with your game. This could be in <strong>Unity</strong>, Godot, Unreal, or a web browser.</p>
            <p>Example: A player types "Tell me about the dragon" to an innkeeper NPC.</p>
        `,
        before: (el) => {
            // Scroll to client layer
            document.querySelector('.client-layer')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    },
    {
        target: '[data-component="fastapi"]',
        title: '2. Request Reaches FastAPI Server',
        content: `
            <p>The game client sends an HTTP POST request to <code>/elle/game/action</code> with:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>Current game state (scene, NPCs present)</li>
                <li>Player intent ("talk_to_npc")</li>
                <li>Raw input text</li>
            </ul>
            <p><strong>Latency so far: ~5ms</strong> (network + routing)</p>
        `
    },
    {
        target: '[data-component="auth"]',
        title: '3. Authentication & Rate Limiting',
        content: `
            <p>Before processing, the request passes through security:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>✅ <strong>API Key validation</strong> (is this a valid user?)</li>
                <li>✅ <strong>Rate limiting</strong> (100-1000 requests/min depending on tier)</li>
                <li>✅ <strong>IP whitelisting</strong> (production environments)</li>
            </ul>
            <p><strong>Latency: +2ms</strong></p>
        `
    },
    {
        target: '[data-component="safety"]',
        title: '4. Safety Guardrails Check',
        content: `
            <p>Player input is scanned for inappropriate content:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>🔍 Pattern matching (hate speech, PII, profanity)</li>
                <li>📊 Risk scoring (0.0 = safe, 1.0 = block)</li>
                <li>🎮 Context-aware (combat violence gets lower risk)</li>
                <li>📝 Complete audit trail</li>
            </ul>
            <p><strong>Latency: +3ms</strong> (total: 10ms)</p>
        `
    },
    {
        target: '[data-component="action-processor"]',
        title: '5. Intent Classification',
        content: `
            <p>The Action Processor determines <em>what the player wants to do</em>:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li><code>talk_to_npc</code> - Dialogue</li>
                <li><code>attack</code> - Combat</li>
                <li><code>investigate</code> - Examine object</li>
                <li><code>use_item</code> - Inventory action</li>
            </ul>
            <p>Our example: "Tell me about the dragon" → <strong>talk_to_npc</strong></p>
            <p><strong>Latency: +5ms</strong></p>
        `
    },
    {
        target: '[data-component="hololoom"]',
        title: '6. Memory Retrieval',
        content: `
            <p>HoloLoom's Knowledge Graph recalls what the NPC knows:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>📚 <strong>Past interactions</strong> with this player</li>
                <li>🧠 <strong>Facts about the dragon</strong></li>
                <li>👥 <strong>Relationships</strong> (who else knows about it?)</li>
                <li>⏰ <strong>Temporal context</strong> (what happened last time?)</li>
            </ul>
            <p><strong>Latency: +25ms</strong> (total: 40ms)</p>
        `
    },
    {
        target: '[data-component="emotion-engine"]',
        title: '7. Emotional State Calculation',
        content: `
            <p>The NPC's current emotions are computed using the <strong>PAD model</strong>:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li><strong>Valence</strong>: 0.3 (slightly positive - friendly innkeeper)</li>
                <li><strong>Arousal</strong>: 0.5 (alert but not excited)</li>
                <li><strong>Dominance</strong>: 0.6 (confident in his tavern)</li>
                <li><strong>Trust</strong>: 0.5 (neutral - first meeting)</li>
            </ul>
            <p>Mentioning the <em>dragon</em> will affect these values!</p>
            <p><strong>Latency: +5ms</strong></p>
        `
    },
    {
        target: '[data-component="llm-bridge"]',
        title: '8. LLM Generation',
        content: `
            <p>The enriched prompt is sent to the LLM (OpenAI, Anthropic, or local):</p>
            <pre style="background: var(--color-bg-tertiary); padding: var(--space-3); border-radius: var(--radius-md); font-size: var(--font-size-sm); overflow-x: auto; margin-top: var(--space-2);">
You are Bob, a friendly innkeeper.
Emotions: slightly happy, alert, confident.
Context: Player asks about the dragon.
Memories: [dragon threat, village fear]

Respond in character:
            </pre>
            <p><strong>Latency: +180ms</strong> (LLM API call)</p>
            <p><strong>Total so far: 225ms</strong></p>
        `
    },
    {
        target: '[data-component="emotion-engine"]',
        title: '9. Emotion Update',
        content: `
            <p>Based on the LLM's response, emotions are updated:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>Valence: 0.3 → <strong>0.1</strong> (concerned about dragon)</li>
                <li>Arousal: 0.5 → <strong>0.7</strong> (nervous tension)</li>
                <li>Dominance: 0.6 → <strong>0.5</strong> (feels less in control)</li>
                <li>Trust: 0.5 → <strong>0.4</strong> (wary of player's intentions)</li>
            </ul>
            <p>These changes persist and affect future interactions!</p>
            <p><strong>Latency: +5ms</strong></p>
        `
    },
    {
        target: '[data-component="hololoom"]',
        title: '10. Memory Storage',
        content: `
            <p>The conversation is stored in the Knowledge Graph:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>🔗 New relationships (Player ASKED_ABOUT Dragon)</li>
                <li>📝 Conversation summary</li>
                <li>😰 Emotional state snapshot</li>
                <li>⏱️ Timestamp for temporal queries</li>
            </ul>
            <p>This memory will be recalled in future interactions!</p>
            <p><strong>Latency: +15ms</strong></p>
        `
    },
    {
        target: '[data-component="analytics"]',
        title: '11. Metrics & Monitoring',
        content: `
            <p>Performance data is exported to Prometheus:</p>
            <ul style="margin-top: var(--space-2); margin-left: var(--space-4);">
                <li>📊 Total latency: <strong>245ms</strong></li>
                <li>🎯 Intent classification accuracy</li>
                <li>💰 LLM token usage (for cost tracking)</li>
                <li>❌ Error rate (if any failures occurred)</li>
            </ul>
            <p>Grafana dashboards visualize these metrics in real-time.</p>
            <p><strong>Latency: +5ms</strong></p>
        `
    },
    {
        target: '[data-component="fastapi"]',
        title: '12. Response Returns to Game',
        content: `
            <p><strong>Total journey: 250ms</strong></p>
            <p>The game client receives a JSON response:</p>
            <pre style="background: var(--color-bg-tertiary); padding: var(--space-3); border-radius: var(--radius-md); font-size: var(--font-size-sm); overflow-x: auto; margin-top: var(--space-2);">
{
  "npc_response": "Ah, the dragon...
    *nervous* That's dangerous talk,
    friend. But I suppose you deserve
    to know...",
  "emotional_state": {
    "valence": 0.1,
    "arousal": 0.7,
    ...
  },
  "quest_triggered": "dragon_investigation"
}
            </pre>
            <p>✨ <strong>The NPC feels alive!</strong></p>
        `
    }
];

// Create tour button
function initializeArchitectureTour() {
    // Check if already initialized
    if (document.getElementById('tour-button')) return;

    // Create "Take a Tour" button
    const tourButton = document.createElement('button');
    tourButton.id = 'tour-button';
    tourButton.className = 'btn btn-primary';
    tourButton.innerHTML = '🎓 Take a Tour';
    tourButton.style.cssText = `
        position: fixed;
        bottom: var(--space-4);
        right: var(--space-4);
        z-index: var(--z-fixed);
        box-shadow: var(--shadow-xl);
    `;

    // Create tour instance
    const tour = new GuidedTour(architectureTourSteps, {
        name: 'BigPlay Architecture',
        storageKey: 'bigplay-architecture-tour',
        autoStart: false,
        showProgress: true,
        allowSkip: true
    });

    // Start tour on button click
    tourButton.addEventListener('click', () => {
        tour.start();
    });

    document.body.appendChild(tourButton);

    // Auto-start on first visit
    if (!tour.isCompleted() && !sessionStorage.getItem('tour-prompt-shown')) {
        sessionStorage.setItem('tour-prompt-shown', 'true');

        // Show prompt after 2 seconds
        setTimeout(() => {
            if (confirm('👋 Welcome! Would you like a guided tour of BigPlay\'s architecture?')) {
                tour.start();
            }
        }, 2000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeArchitectureTour);
} else {
    initializeArchitectureTour();
}
