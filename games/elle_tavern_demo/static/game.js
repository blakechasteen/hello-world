/**
 * The Rusty Mug Tavern - Game Client
 *
 * Client-side game logic for the Elle Tavern demo.
 * Handles UI interaction, API calls, and voice playback.
 *
 * Created: 2025-11-16
 */

class TavernGame {
    constructor() {
        this.sessionId = null;
        this.gameState = null;
        this.currentNPC = null;
        this.voiceEnabled = true;
        this.apiBase = window.location.origin;
        this.isProcessing = false;

        // Bind methods
        this.initialize = this.initialize.bind(this);
        this.handleSendMessage = this.handleSendMessage.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);
        this.handleQuickResponse = this.handleQuickResponse.bind(this);
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    async initialize() {
        console.log('🍺 Initializing The Rusty Mug...');

        // Setup event listeners
        this.setupEventListeners();

        // Start new game
        await this.newGame();

        // Hide loading screen
        this.hideLoading();

        console.log('✅ Game ready!');
    }

    setupEventListeners() {
        // Send message
        document.getElementById('send-button').addEventListener('click', this.handleSendMessage);
        document.getElementById('player-input').addEventListener('keypress', this.handleKeyPress);

        // Voice toggle
        document.getElementById('voice-toggle').addEventListener('click', () => {
            this.voiceEnabled = !this.voiceEnabled;
            this.updateVoiceButton();
        });

        // Quick responses
        document.getElementById('quick-responses').addEventListener('change', this.handleQuickResponse);

        // Action buttons
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleAction(action);
            });
        });

        // Clear conversation
        document.getElementById('clear-conversation').addEventListener('click', () => {
            this.clearConversation();
        });

        // Game controls
        document.getElementById('save-btn').addEventListener('click', () => this.saveGame());
        document.getElementById('load-btn').addEventListener('click', () => this.showLoadModal());
        document.getElementById('new-game-btn').addEventListener('click', () => this.confirmNewGame());

        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').style.display = 'none';
            });
        });

        // Close modals on outside click
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }

    hideLoading() {
        document.getElementById('loading-screen').style.display = 'none';
        document.getElementById('game-container').style.display = 'flex';
    }

    // ========================================================================
    // Game State Management
    // ========================================================================

    async newGame() {
        try {
            const response = await fetch(`${this.apiBase}/api/game/new`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ player_name: 'Traveler' })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.sessionId = data.session_id;
                this.gameState = data.game_state;
                this.render();
                this.addSystemMessage(data.message);
            } else {
                this.showError('Failed to start game');
            }
        } catch (error) {
            console.error('New game error:', error);
            this.showError('Failed to connect to server');
        }
    }

    async saveGame() {
        if (!this.sessionId) return;

        try {
            const saveName = prompt('Enter save name:', 'quicksave');
            if (!saveName) return;

            const response = await fetch(`${this.apiBase}/api/game/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    save_name: saveName
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.addSystemMessage(`Game saved as "${saveName}"`);
            } else {
                this.showError(data.message || 'Save failed');
            }
        } catch (error) {
            console.error('Save error:', error);
            this.showError('Failed to save game');
        }
    }

    async showLoadModal() {
        try {
            const response = await fetch(`${this.apiBase}/api/game/saves`);
            const data = await response.json();

            const modal = document.getElementById('load-modal');
            const saveList = document.getElementById('save-list');

            if (data.saves && data.saves.length > 0) {
                saveList.innerHTML = data.saves.map(save => `
                    <div class="save-item" data-save="${save.name}">
                        <div class="save-name">${save.name}</div>
                        <div class="save-date">${new Date(save.modified * 1000).toLocaleString()}</div>
                    </div>
                `).join('');

                // Add click handlers
                saveList.querySelectorAll('.save-item').forEach(item => {
                    item.addEventListener('click', () => {
                        this.loadGame(item.dataset.save);
                        modal.style.display = 'none';
                    });
                });
            } else {
                saveList.innerHTML = '<p class="empty-message">No saved games found</p>';
            }

            modal.style.display = 'flex';
        } catch (error) {
            console.error('Load modal error:', error);
            this.showError('Failed to load save list');
        }
    }

    async loadGame(saveName) {
        try {
            const response = await fetch(`${this.apiBase}/api/game/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ save_name: saveName })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.sessionId = data.session_id;
                this.gameState = data.game_state;
                this.render();
                this.clearConversation();
                this.addSystemMessage(`Game loaded: ${saveName}`);
            } else {
                this.showError(data.message || 'Load failed');
            }
        } catch (error) {
            console.error('Load error:', error);
            this.showError('Failed to load game');
        }
    }

    confirmNewGame() {
        if (confirm('Start a new game? Current progress will be lost unless saved.')) {
            this.newGame();
            this.clearConversation();
        }
    }

    // ========================================================================
    // Player Actions
    // ========================================================================

    async handleSendMessage() {
        const input = document.getElementById('player-input');
        const message = input.value.trim();

        if (!message || !this.currentNPC || this.isProcessing) return;

        this.isProcessing = true;
        this.setInputEnabled(false);

        // Show player message
        this.addMessage('player', message);

        // Clear input
        input.value = '';

        // Send to server
        try {
            const response = await fetch(`${this.apiBase}/api/game/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    action_type: 'talk',
                    params: {
                        npc_id: this.currentNPC,
                        player_message: message
                    }
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Update game state
                this.gameState = data.game_state;

                // Show NPC response
                const emotion = data.npc_emotion;
                this.addMessage('npc', data.npc_response, data.npc_name, emotion, data.npc_mood);

                // Play voice
                if (this.voiceEnabled && data.audio_url) {
                    this.playVoice(data.audio_url);
                }

                // Show quest offer
                if (data.quest_offered) {
                    this.showQuestOffer(data.quest_offered);
                }

                // Update UI
                this.updateEmotionDisplay();
            } else {
                this.showError(data.message || 'Action failed');
            }
        } catch (error) {
            console.error('Talk error:', error);
            this.showError('Failed to send message');
        } finally {
            this.isProcessing = false;
            this.setInputEnabled(true);
            input.focus();
        }
    }

    handleKeyPress(e) {
        if (e.key === 'Enter') {
            this.handleSendMessage();
        }
    }

    handleQuickResponse(e) {
        const value = e.target.value;
        if (!value) return;

        const responses = {
            hello: "Hello! How are you today?",
            help: "Can you help me with something?",
            quest: "Do you have any quests for me?",
            rumor: "Have you heard any interesting rumors?",
            bye: "I should get going. Farewell!"
        };

        const input = document.getElementById('player-input');
        input.value = responses[value] || '';
        e.target.value = '';
        input.focus();
    }

    async handleAction(action) {
        switch (action) {
            case 'look':
                await this.lookAround();
                break;
            case 'inventory':
                this.showInventory();
                break;
            case 'help':
                this.showHelp();
                break;
        }
    }

    async lookAround() {
        try {
            const response = await fetch(`${this.apiBase}/api/game/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    action_type: 'look',
                    params: {}
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.addSystemMessage(data.description);
                if (data.npcs && data.npcs.length > 0) {
                    this.addSystemMessage(`You see: ${data.npcs.map(npc => npc.name).join(', ')}`);
                }
            }
        } catch (error) {
            console.error('Look error:', error);
        }
    }

    // ========================================================================
    // UI Rendering
    // ========================================================================

    render() {
        if (!this.gameState) return;

        // Update player stats
        const player = this.gameState.player;
        document.getElementById('player-name').textContent = player.name;
        document.getElementById('player-level').textContent = player.level;
        document.getElementById('player-gold').textContent = player.gold;

        // Update location
        const location = this.gameState.location;
        if (location) {
            document.getElementById('location-name').textContent = location.name;
            document.getElementById('location-description').textContent = location.description;
        }

        // Update time of day
        const timeOfDay = this.gameState.world_state.time_of_day;
        document.getElementById('time-of-day').textContent = timeOfDay.charAt(0).toUpperCase() + timeOfDay.slice(1);

        // Render NPCs
        this.renderNPCs();

        // Render quests
        this.renderQuests();

        // Render emotions
        this.updateEmotionDisplay();
    }

    renderNPCs() {
        const npcList = document.getElementById('npc-list');
        const npcs = this.gameState.npcs;

        npcList.innerHTML = Object.values(npcs).map(npc => `
            <div class="npc-card ${this.currentNPC === npc.id ? 'selected' : ''}"
                 data-npc-id="${npc.id}">
                <div class="npc-header">
                    <span class="npc-name">${npc.name}</span>
                    <span class="npc-mood">${this.getEmotionIcon(npc.current_emotion)}</span>
                </div>
                <div class="npc-role">${npc.role}</div>
                <div class="npc-description">${npc.description}</div>
            </div>
        `).join('');

        // Add click handlers
        npcList.querySelectorAll('.npc-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectNPC(card.dataset.npcId);
            });
        });
    }

    selectNPC(npcId) {
        this.currentNPC = npcId;
        const npc = this.gameState.npcs[npcId];

        // Update UI
        document.querySelectorAll('.npc-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.npcId === npcId);
        });

        // Enable input
        this.setInputEnabled(true);

        // Add system message
        this.addSystemMessage(`You approach ${npc.name}. What do you say?`);

        // Focus input
        document.getElementById('player-input').focus();
    }

    renderQuests() {
        const questList = document.getElementById('active-quests');
        const quests = this.gameState.quests;
        const activeQuests = Object.values(quests).filter(q => q.status === 'active');

        if (activeQuests.length === 0) {
            questList.innerHTML = '<p class="empty-message">No active quests</p>';
        } else {
            questList.innerHTML = activeQuests.map(quest => `
                <div class="quest-item">
                    <div class="quest-name">${quest.title}</div>
                    <div class="quest-giver">From: ${this.gameState.npcs[quest.giver_npc_id]?.name}</div>
                </div>
            `).join('');
        }
    }

    updateEmotionDisplay() {
        const emotionList = document.getElementById('npc-emotions');
        const npcs = this.gameState.npcs;

        emotionList.innerHTML = Object.values(npcs).map(npc => {
            const emotion = npc.current_emotion;
            const trust = emotion.trust || 0.5;
            const trustPercent = Math.round(trust * 100);
            const trustColor = trust > 0.7 ? '#4caf50' : trust > 0.4 ? '#ff9800' : '#f44336';

            return `
                <div class="emotion-item">
                    <div class="emotion-header">
                        <span class="emotion-name">${npc.name}</span>
                        <span class="emotion-icon">${this.getEmotionIcon(emotion)}</span>
                    </div>
                    <div class="trust-bar-container">
                        <div class="trust-label">Trust: ${trustPercent}%</div>
                        <div class="trust-bar">
                            <div class="trust-fill" style="width: ${trustPercent}%; background-color: ${trustColor};"></div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // ========================================================================
    // Conversation Display
    // ========================================================================

    addMessage(speaker, text, npcName = null, emotion = null, mood = null) {
        const history = document.getElementById('conversation-history');

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${speaker}-message`;

        if (speaker === 'player') {
            msgDiv.innerHTML = `
                <span class="message-label">You:</span>
                <span class="message-text">${this.escapeHtml(text)}</span>
            `;
        } else if (speaker === 'npc') {
            const emotionIcon = emotion ? this.getEmotionIcon(emotion) : '😐';
            msgDiv.innerHTML = `
                <span class="message-label">${npcName}:</span>
                <span class="emotion-icon">${emotionIcon}</span>
                <span class="message-text">${this.escapeHtml(text)}</span>
            `;
        }

        history.appendChild(msgDiv);
        msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    addSystemMessage(text) {
        const history = document.getElementById('conversation-history');

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system-message';
        msgDiv.innerHTML = `
            <span class="message-icon">ℹ️</span>
            <span class="message-text">${this.escapeHtml(text)}</span>
        `;

        history.appendChild(msgDiv);
        msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    clearConversation() {
        const history = document.getElementById('conversation-history');
        history.innerHTML = `
            <div class="message system-message">
                <span class="message-icon">ℹ️</span>
                <span class="message-text">Conversation cleared.</span>
            </div>
        `;
    }

    // ========================================================================
    // Modals
    // ========================================================================

    showQuestOffer(quest) {
        const modal = document.getElementById('quest-modal');
        document.getElementById('quest-title').textContent = quest.title;
        document.getElementById('quest-description').textContent = quest.description;

        const rewards = quest.rewards;
        let rewardsHtml = '<h4>Rewards:</h4><ul>';
        if (rewards.gold) rewardsHtml += `<li>💰 ${rewards.gold} Gold</li>`;
        if (rewards.xp) rewardsHtml += `<li>⭐ ${rewards.xp} XP</li>`;
        rewardsHtml += '</ul>';
        document.getElementById('quest-rewards').innerHTML = rewardsHtml;

        // Setup buttons
        const acceptBtn = document.getElementById('quest-accept');
        const declineBtn = document.getElementById('quest-decline');

        acceptBtn.onclick = async () => {
            await this.acceptQuest(quest.id);
            modal.style.display = 'none';
        };

        declineBtn.onclick = () => {
            modal.style.display = 'none';
            this.addSystemMessage('You decline the quest for now.');
        };

        modal.style.display = 'flex';
    }

    async acceptQuest(questId) {
        try {
            const response = await fetch(`${this.apiBase}/api/game/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    action_type: 'quest_accept',
                    params: { quest_id: questId }
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.gameState = data.game_state;
                this.renderQuests();
                this.addSystemMessage(data.message);
            }
        } catch (error) {
            console.error('Accept quest error:', error);
        }
    }

    showInventory() {
        const modal = document.getElementById('inventory-modal');
        const inventoryList = document.getElementById('inventory-list');

        const items = this.gameState.player.inventory;

        if (items.length === 0) {
            inventoryList.innerHTML = '<p class="empty-message">Your inventory is empty</p>';
        } else {
            inventoryList.innerHTML = items.map(item => `
                <div class="inventory-item">${item}</div>
            `).join('');
        }

        modal.style.display = 'flex';
    }

    showHelp() {
        document.getElementById('help-modal').style.display = 'flex';
    }

    // ========================================================================
    // Voice Synthesis
    // ========================================================================

    async playVoice(audioUrl) {
        if (!audioUrl) return;

        try {
            const player = document.getElementById('voice-player');
            player.src = audioUrl;
            await player.play();
        } catch (error) {
            console.error('Voice playback error:', error);
        }
    }

    updateVoiceButton() {
        const icon = document.getElementById('voice-icon');
        const label = document.getElementById('voice-label');

        if (this.voiceEnabled) {
            icon.textContent = '🔊';
            label.textContent = 'Voice: ON';
        } else {
            icon.textContent = '🔇';
            label.textContent = 'Voice: OFF';
        }
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    getEmotionIcon(emotion) {
        const valence = emotion.valence || 0;
        const arousal = emotion.arousal || 0;

        if (valence > 0.5 && arousal > 0.3) return '😊';  // excited/happy
        if (valence > 0.3) return '🙂';  // happy
        if (valence < -0.5 && arousal > 0.3) return '😠';  // angry
        if (valence < -0.3) return '😟';  // sad
        if (arousal > 0.5) return '😲';  // surprised
        return '😐';  // neutral
    }

    setInputEnabled(enabled) {
        const input = document.getElementById('player-input');
        const button = document.getElementById('send-button');

        input.disabled = !enabled || !this.currentNPC;
        button.disabled = !enabled || !this.currentNPC;

        if (enabled && this.currentNPC) {
            input.placeholder = 'What do you say?';
        } else {
            input.placeholder = 'Select an NPC first...';
        }
    }

    showError(message) {
        this.addSystemMessage(`❌ Error: ${message}`);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// ============================================================================
// Initialize Game on Load
// ============================================================================

let game;

window.addEventListener('DOMContentLoaded', () => {
    game = new TavernGame();
    game.initialize();
});
