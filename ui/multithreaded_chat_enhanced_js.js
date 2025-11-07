// ========== Breakthrough Notifications ==========

function connectNotificationStream() {
    state.notificationWs = new WebSocket(`${WS_BASE}/ws/notifications`);

    state.notificationWs.onopen = () => {
        console.log('Connected to notification stream');
        updateStatusBar();
    };

    state.notificationWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'breakthrough') {
            handleBreakthrough(data);
        }
    };

    state.notificationWs.onerror = (error) => {
        console.error('Notification WebSocket error:', error);
    };

    state.notificationWs.onclose = () => {
        console.log('Notification stream disconnected');
        setTimeout(connectNotificationStream, 5000);
    };
}

function handleBreakthrough(data) {
    addBreakthroughNotification(data);
    showToast('💡 Breakthrough', `${data.agent_name}: ${data.description}`);

    const thread = state.threads.get(data.thread_id);
    if (thread && thread.id !== state.activeThreadId) {
        thread.hasBreakthrough = true;
        updateThreadTabs();
    }

    if (data.thread_id !== state.activeThreadId) {
        const currentThread = state.threads.get(state.activeThreadId);
        if (currentThread) {
            currentThread.messages.push({
                role: 'breakthrough',
                content: `Breakthrough in ${data.agent_name}: ${data.description}`,
                impact_score: data.impact_score
            });
            renderMessages();
        }
    }
}

function addBreakthroughNotification(data) {
    const list = document.getElementById('notificationsList');

    if (list.children[0]?.textContent.includes('No breakthroughs')) {
        list.innerHTML = '';
    }

    const item = document.createElement('div');
    item.className = 'notification-item';
    item.innerHTML = `
        <div class="title">${data.agent_name}</div>
        <div>${data.description}</div>
        <div class="time">Impact: ${(data.impact_score * 100).toFixed(0)}% • Just now</div>
    `;

    list.insertBefore(item, list.firstChild);

    while (list.children.length > 20) {
        list.removeChild(list.lastChild);
    }
}

// ========== Thread Tabs ==========

function renderThreadTab(thread) {
    const tabsContainer = document.getElementById('threadTabs');
    const newThreadBtn = tabsContainer.querySelector('.new-thread-btn');

    const tab = document.createElement('div');
    tab.className = 'thread-tab';
    tab.id = `tab-${thread.id}`;

    tab.innerHTML = `
        <div class="thread-tab-info">
            <span class="bookmark-star ${thread.bookmarked ? 'bookmarked' : ''}" onclick="event.stopPropagation(); toggleBookmarkFromTab('${thread.id}')">
                ${thread.bookmarked ? '★' : '☆'}
            </span>
            <div class="thread-tab-name">${thread.agentName}</div>
        </div>
        <div class="thread-tab-indicators">
            <span class="unread-badge hidden">0</span>
            <span class="breakthrough-icon hidden">🔔</span>
        </div>
        <button class="thread-tab-close" onclick="event.stopPropagation(); closeThread('${thread.id}')">×</button>
    `;

    tab.onclick = () => switchToThread(thread.id);

    tabsContainer.insertBefore(tab, newThreadBtn);
}

function updateThreadTabs() {
    state.threads.forEach((thread, threadId) => {
        const tab = document.getElementById(`tab-${threadId}`);
        if (!tab) return;

        tab.classList.toggle('active', threadId === state.activeThreadId);

        const unreadBadge = tab.querySelector('.unread-badge');
        if (thread.unreadCount > 0) {
            unreadBadge.textContent = thread.unreadCount;
            unreadBadge.classList.remove('hidden');
        } else {
            unreadBadge.classList.add('hidden');
        }

        const breakthroughIcon = tab.querySelector('.breakthrough-icon');
        breakthroughIcon.classList.toggle('hidden', !thread.hasBreakthrough);

        const bookmarkStar = tab.querySelector('.bookmark-star');
        bookmarkStar.textContent = thread.bookmarked ? '★' : '☆';
        bookmarkStar.classList.toggle('bookmarked', thread.bookmarked);
    });
}

function updateStatusBar() {
    const statusBar = document.getElementById('statusBar');

    if (!state.activeThreadId) {
        statusBar.textContent = 'No active thread';
        statusBar.className = 'status-bar disconnected';
        return;
    }

    const thread = state.threads.get(state.activeThreadId);
    const threadCount = state.threads.size;

    if (thread.status === 'connected') {
        statusBar.textContent = `Connected to ${thread.agentName} | ${threadCount} active thread${threadCount !== 1 ? 's' : ''}`;
        statusBar.className = 'status-bar connected';
    } else {
        statusBar.textContent = `Disconnected | ${threadCount} thread${threadCount !== 1 ? 's' : ''}`;
        statusBar.className = 'status-bar disconnected';
    }
}

// ========== Search & Filtering ==========

async function searchThreads() {
    const query = document.getElementById('searchInput').value;
    if (!query && !hasActiveFilters()) {
        // No search or filters, show all threads
        return;
    }

    await applyFilters();
}

function hasActiveFilters() {
    return document.getElementById('filterAgent').value ||
           document.getElementById('filterFromDate').value ||
           document.getElementById('filterToDate').value ||
           document.getElementById('filterBookmarked').checked ||
           document.getElementById('filterBreakthroughs').checked ||
           document.getElementById('filterConfidence').value > 0;
}

async function applyFilters() {
    const query = document.getElementById('searchInput').value;
    const agent = document.getElementById('filterAgent').value;
    const fromDate = document.getElementById('filterFromDate').value;
    const toDate = document.getElementById('filterToDate').value;
    const bookmarked = document.getElementById('filterBookmarked').checked;
    const breakthroughs = document.getElementById('filterBreakthroughs').checked;
    const confidence = document.getElementById('filterConfidence').value / 100;

    const params = new URLSearchParams();
    if (query) params.append('q', query);
    if (agent) params.append('agent', agent);
    if (fromDate) params.append('from_date', fromDate);
    if (toDate) params.append('to_date', toDate);
    if (bookmarked) params.append('bookmarked', 'true');
    if (breakthroughs) params.append('has_breakthroughs', 'true');
    if (confidence > 0) params.append('min_confidence', confidence);

    try {
        const response = await fetch(`${API_BASE}/threads/search?${params}`);
        const data = await response.json();

        console.log(`Search returned ${data.total_count} results`);
        // In production, would filter visible threads or show results panel

    } catch (error) {
        console.error('Search failed:', error);
        showToast('Error', 'Search failed', true);
    }
}

function clearSearch() {
    document.getElementById('searchInput').value = '';
    document.getElementById('filterAgent').value = '';
    document.getElementById('filterFromDate').value = '';
    document.getElementById('filterToDate').value = '';
    document.getElementById('filterBookmarked').checked = false;
    document.getElementById('filterBreakthroughs').checked = false;
    document.getElementById('filterConfidence').value = 0;
    updateConfidenceLabel();
}

function toggleAdvancedFilters() {
    const filters = document.getElementById('advancedFilters');
    filters.classList.toggle('hidden');
}

function updateConfidenceLabel() {
    const value = document.getElementById('filterConfidence').value;
    document.getElementById('confidenceValue').textContent = `${value}%`;
}

// ========== Bookmarking ==========

async function toggleBookmark() {
    if (!state.activeThreadId) return;
    await toggleBookmarkFromTab(state.activeThreadId);
}

async function toggleBookmarkFromTab(threadId) {
    const thread = state.threads.get(threadId);
    const method = thread.bookmarked ? 'DELETE' : 'POST';

    try {
        await fetch(`${API_BASE}/threads/${threadId}/bookmark`, { method });

        thread.bookmarked = !thread.bookmarked;
        updateThreadTabs();
        updateBookmarkButton();

        showToast(
            thread.bookmarked ? 'Bookmarked' : 'Unbookmarked',
            `Thread ${thread.agentName}`
        );

        // Refresh bookmarks sidebar if open
        if (state.currentSidebarTab === 'bookmarks') {
            await loadBookmarks();
        }

    } catch (error) {
        console.error('Failed to toggle bookmark:', error);
        showToast('Error', 'Failed to update bookmark', true);
    }
}

function updateBookmarkButton() {
    const btn = document.getElementById('bookmarkBtn');
    if (!state.activeThreadId) {
        btn.textContent = '☆ Bookmark';
        return;
    }

    const thread = state.threads.get(state.activeThreadId);
    btn.textContent = thread.bookmarked ? '★ Bookmarked' : '☆ Bookmark';
}

async function loadBookmarks() {
    try {
        const response = await fetch(`${API_BASE}/threads/bookmarked?user_id=${state.userId}`);
        const data = await response.json();

        const container = document.getElementById('sidebarContent');
        container.innerHTML = '';

        if (data.threads.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: var(--text-dim); padding: 20px; font-size: 13px;">No bookmarked threads</div>';
            return;
        }

        data.threads.forEach(thread => {
            const item = document.createElement('div');
            item.className = 'bookmark-item';
            item.onclick = () => {
                const localThread = Array.from(state.threads.values()).find(t => t.id === thread.thread_id);
                if (localThread) {
                    switchToThread(localThread.id);
                }
            };

            item.innerHTML = `
                <div class="bookmark-item-title">
                    <span>${thread.agent_name}</span>
                    <span onclick="event.stopPropagation(); toggleBookmarkFromTab('${thread.thread_id}')">★</span>
                </div>
                <div class="bookmark-item-meta">
                    ${thread.message_count} messages • ${new Date(thread.created_at * 1000).toLocaleDateString()}
                </div>
            `;

            container.appendChild(item);
        });

    } catch (error) {
        console.error('Failed to load bookmarks:', error);
    }
}

// ========== Export ==========

async function exportThread(format) {
    if (!state.activeThreadId) {
        showToast('Error', 'No active thread to export', true);
        return;
    }

    const url = `${API_BASE}/threads/${state.activeThreadId}/export?format=${format}`;

    try {
        const response = await fetch(url);
        const blob = await response.blob();

        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `thread_${state.activeThreadId}.${format === 'markdown' ? 'md' : format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        showToast('Export Complete', `Thread exported as ${format.toUpperCase()}`);

    } catch (error) {
        console.error('Export failed:', error);
        showToast('Error', 'Export failed', true);
    }
}

// ========== Analytics ==========

async function loadAnalytics() {
    try {
        const response = await fetch(`${API_BASE}/stats/agents`);
        const data = await response.json();

        const container = document.getElementById('sidebarContent');
        container.innerHTML = '';

        // Agent comparison chart
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.innerHTML = '<h3>Agent Performance</h3><canvas id="agentChart"></canvas>';
        container.appendChild(chartDiv);

        const ctx = document.getElementById('agentChart').getContext('2d');
        if (state.charts.agentChart) {
            state.charts.agentChart.destroy();
        }

        state.charts.agentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.agents.map(a => a.agent_name),
                datasets: [{
                    label: 'Success Rate',
                    data: data.agents.map(a => a.success_rate * 100),
                    backgroundColor: '#00d4ff'
                }, {
                    label: 'Breakthroughs',
                    data: data.agents.map(a => a.breakthroughs),
                    backgroundColor: '#2a5298'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });

        // Negotiation breakdown
        if (data.agents.length > 0 && data.agents[0].negotiation) {
            const negotiationDiv = document.createElement('div');
            negotiationDiv.className = 'chart-container';
            negotiationDiv.innerHTML = '<h3>Negotiation Breakdown</h3><canvas id="negotiationChart"></canvas>';
            container.appendChild(negotiationDiv);

            const negCtx = document.getElementById('negotiationChart').getContext('2d');
            if (state.charts.negotiationChart) {
                state.charts.negotiationChart.destroy();
            }

            const agent = data.agents[0];
            state.charts.negotiationChart = new Chart(negCtx, {
                type: 'pie',
                data: {
                    labels: ['Creative Wins', 'QC Wins', 'Compromises'],
                    datasets: [{
                        data: [
                            agent.negotiation.creative_win_rate * 100,
                            agent.negotiation.qc_win_rate * 100,
                            agent.negotiation.compromise_rate * 100
                        ],
                        backgroundColor: ['#00d4ff', '#ff3366', '#00ff88']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            labels: { color: '#e0e0e0' }
                        }
                    }
                }
            });
        }

    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

// ========== Voice I/O ==========

function initVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    state.voiceRecognition = new SpeechRecognition();
    state.voiceRecognition.continuous = true;
    state.voiceRecognition.interimResults = true;

    state.voiceRecognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');

        document.getElementById('messageInput').value = transcript;
    };

    state.voiceRecognition.onend = () => {
        state.voiceActive = false;
        document.getElementById('voiceInputBtn').classList.remove('voice-active');
        document.getElementById('voiceInputBtn').textContent = '🎤 Voice';
    };
}

function toggleVoiceInput() {
    if (!state.voiceRecognition) {
        showToast('Error', 'Voice recognition not supported', true);
        return;
    }

    if (state.voiceActive) {
        state.voiceRecognition.stop();
        state.voiceActive = false;
        document.getElementById('voiceInputBtn').classList.remove('voice-active');
        document.getElementById('voiceInputBtn').textContent = '🎤 Voice';
    } else {
        state.voiceRecognition.start();
        state.voiceActive = true;
        document.getElementById('voiceInputBtn').classList.add('voice-active');
        document.getElementById('voiceInputBtn').textContent = '⏹ Stop';
    }
}

function speakMessage(text) {
    if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported');
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    const voices = speechSynthesis.getVoices();
    utterance.voice = voices.find(v => v.lang === 'en-US') || voices[0];

    speechSynthesis.speak(utterance);
}

// ========== Promptly Templates ==========

async function showTemplateSelector() {
    try {
        const response = await fetch(`${API_BASE}/promptly/templates`);
        const data = await response.json();

        const list = document.getElementById('templateList');
        list.innerHTML = '';

        if (data.templates.length === 0) {
            list.innerHTML = '<div style="text-align: center; color: var(--text-dim); padding: 20px;">No templates available</div>';
        } else {
            data.templates.forEach(template => {
                const item = document.createElement('div');
                item.className = 'template-item';
                item.innerHTML = `
                    <h4>${template.name}</h4>
                    <p>${template.description}</p>
                    <div class="template-meta">
                        ${template.step_count} steps | v${template.version}
                    </div>
                    <button onclick="selectTemplate('${template.name}')">Use Template</button>
                `;
                list.appendChild(item);
            });
        }

        document.getElementById('templateSelectorModal').classList.remove('hidden');

    } catch (error) {
        console.error('Failed to load templates:', error);
        showToast('Error', 'Failed to load templates', true);
    }
}

function hideTemplateSelector() {
    document.getElementById('templateSelectorModal').classList.add('hidden');
}

async function selectTemplate(templateName) {
    try {
        const response = await fetch(`${API_BASE}/promptly/templates/${templateName}`);
        const template = await response.json();

        state.selectedTemplate = template;

        hideTemplateSelector();

        const form = document.getElementById('templateInputsForm');
        form.innerHTML = '';

        if (template.inputs && template.inputs.length > 0) {
            template.inputs.forEach(input => {
                const label = document.createElement('label');
                label.textContent = input.name + (input.required ? ' *' : '');

                const inputEl = document.createElement('input');
                inputEl.type = 'text';
                inputEl.id = `input-${input.name}`;
                inputEl.placeholder = input.default || '';
                if (input.required) {
                    inputEl.required = true;
                }

                form.appendChild(label);
                form.appendChild(inputEl);
            });

            document.getElementById('templateInputModal').classList.remove('hidden');
        } else {
            await executeTemplate();
        }

    } catch (error) {
        console.error('Failed to load template:', error);
        showToast('Error', 'Failed to load template', true);
    }
}

function hideTemplateInputModal() {
    document.getElementById('templateInputModal').classList.add('hidden');
}

async function executeTemplate() {
    if (!state.selectedTemplate || !state.activeThreadId) return;

    const inputs = {};

    if (state.selectedTemplate.inputs) {
        for (const input of state.selectedTemplate.inputs) {
            const value = document.getElementById(`input-${input.name}`)?.value;
            if (input.required && !value) {
                showToast('Error', `${input.name} is required`, true);
                return;
            }
            inputs[input.name] = value || input.default || '';
        }
    }

    hideTemplateInputModal();

    try {
        showToast('Executing Template', `Running ${state.selectedTemplate.name}...`);

        const response = await fetch(`${API_BASE}/promptly/execute`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                thread_id: state.activeThreadId,
                template_name: state.selectedTemplate.name,
                inputs: inputs
            })
        });

        const result = await response.json();

        showToast('Template Complete', `${state.selectedTemplate.name} executed successfully`);

        // Display results in messages
        const thread = state.threads.get(state.activeThreadId);
        thread.messages.push({
            role: 'assistant',
            content: `Template "${state.selectedTemplate.name}" completed:\n\n${result.synthesis || 'See thread for details'}`,
            confidence: 1.0,
            mode: 'template'
        });
        renderMessages();

    } catch (error) {
        console.error('Failed to execute template:', error);
        showToast('Error', 'Template execution failed', true);
    }
}

// ========== UI Helpers ==========

function showToast(title, message, isError = false) {
    const container = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = 'toast' + (isError ? ' error' : '');
    toast.innerHTML = `
        <div class="title">${title}</div>
        <div class="message">${message}</div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

function showNewThreadModal() {
    document.getElementById('newThreadModal').classList.remove('hidden');
    document.getElementById('newThreadMessage').value = '';
}

function hideNewThreadModal() {
    document.getElementById('newThreadModal').classList.add('hidden');
}

function toggleLeftSidebar() {
    state.leftSidebarCollapsed = !state.leftSidebarCollapsed;
    const sidebar = document.getElementById('leftSidebar');
    sidebar.classList.toggle('collapsed', state.leftSidebarCollapsed);

    if (!state.leftSidebarCollapsed) {
        if (state.currentSidebarTab === 'bookmarks') {
            loadBookmarks();
        } else {
            loadAnalytics();
        }
    }
}

function toggleRightSidebar() {
    state.rightSidebarCollapsed = !state.rightSidebarCollapsed;
    const sidebar = document.getElementById('rightSidebar');
    sidebar.classList.toggle('collapsed', state.rightSidebarCollapsed);
}

function switchSidebarTab(tab) {
    state.currentSidebarTab = tab;

    const tabs = document.querySelectorAll('.sidebar-tab');
    tabs.forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');

    if (tab === 'bookmarks') {
        loadBookmarks();
    } else {
        loadAnalytics();
    }
}

// ========== Keyboard Shortcuts ==========

document.getElementById('messageInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ========== Initialize ==========

window.addEventListener('load', () => {
    connectNotificationStream();
    initVoiceRecognition();

    // Auto-create first thread
    setTimeout(async () => {
        document.getElementById('newThreadAgent').value = 'budget';
        await createNewThread();
    }, 500);
});
