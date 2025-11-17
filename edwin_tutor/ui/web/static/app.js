// EdWIN Tutor - Web UI JavaScript

const API_BASE = '/api';

// State
let currentLesson = null;
let lessons = {};
let progress = {};
let userAnswers = {};
let currentHintLevels = {};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Set up navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Load initial data
    await loadProgress();
    await loadLessons();
    await loadBadges();

    console.log('✅ EdWIN Tutor initialized');
}

// Navigation
function switchView(viewName) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    // Update views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });

    document.getElementById(`${viewName}-view`).classList.add('active');

    // Load view-specific data
    if (viewName === 'progress') {
        updateProgressView();
    } else if (viewName === 'badges') {
        updateBadgesView();
    }
}

// API Calls
async function apiGet(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
    return response.json();
}

// Load Data
async function loadProgress() {
    progress = await apiGet('/progress');
    updateProgressBar();
}

async function loadLessons() {
    lessons = await apiGet('/lessons');
    renderLessonList();
}

async function loadBadges() {
    const badges = await apiGet('/badges');
    renderBadges(badges);
}

// Update Progress Bar
function updateProgressBar() {
    document.getElementById('learner-name').textContent = progress.learner_name;
    document.getElementById('current-level').textContent = progress.level;
    document.getElementById('current-xp').textContent = progress.total_xp;
    document.getElementById('completed-count').textContent = progress.lessons_completed;

    // Calculate XP progress to next level
    const nextLevelXP = progress.total_xp + progress.xp_to_next_level;
    const currentLevelXP = nextLevelXP - progress.xp_to_next_level;
    const progressPercent = ((progress.total_xp - currentLevelXP) / progress.xp_to_next_level) * 100;

    document.getElementById('xp-progress').style.width = `${Math.min(progressPercent, 100)}%`;
}

// Render Lesson List
function renderLessonList() {
    const container = document.getElementById('lesson-categories');
    container.innerHTML = '';

    const difficulties = ['beginner', 'intermediate', 'advanced'];
    const icons = { beginner: '🌱', intermediate: '🌿', advanced: '🌳' };

    difficulties.forEach(diff => {
        if (lessons[diff] && lessons[diff].length > 0) {
            const category = document.createElement('div');
            category.className = 'lesson-category';

            const title = document.createElement('h3');
            title.textContent = `${icons[diff]} ${diff.charAt(0).toUpperCase() + diff.slice(1)}`;
            category.appendChild(title);

            lessons[diff].forEach(lesson => {
                const item = document.createElement('div');
                item.className = 'lesson-item';

                if (lesson.completed) item.classList.add('completed');
                if (lesson.locked) item.classList.add('locked');

                const status = lesson.completed ? '✓ ' : lesson.locked ? '🔒 ' : '';
                item.textContent = `${status}${lesson.id}: ${lesson.title}`;

                if (!lesson.locked) {
                    item.addEventListener('click', () => loadLesson(lesson.id));
                }

                category.appendChild(item);
            });

            container.appendChild(category);
        }
    });
}

// Load and Display Lesson
async function loadLesson(lessonId) {
    try {
        const lesson = await apiGet(`/lessons/${lessonId}`);
        currentLesson = lesson;
        userAnswers = {};
        currentHintLevels = {};

        // Mark lesson as started
        await apiPost('/progress/start-lesson', { lesson_id: lessonId });

        // Show lesson detail, hide welcome
        document.querySelector('.welcome-card').style.display = 'none';
        document.getElementById('lesson-detail').style.display = 'block';

        // Update active lesson in sidebar
        document.querySelectorAll('.lesson-item').forEach(item => {
            item.classList.toggle('active', item.textContent.includes(lessonId));
        });

        // Render lesson
        renderLesson(lesson);

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
        console.error('Error loading lesson:', error);
        alert('Failed to load lesson. Please try again.');
    }
}

function renderLesson(lesson) {
    // Header
    document.getElementById('lesson-title').textContent = lesson.title;
    document.getElementById('lesson-difficulty').textContent = `📚 ${lesson.difficulty}`;
    document.getElementById('lesson-type').textContent = `📝 ${lesson.lesson_type.replace('_', ' ')}`;
    document.getElementById('lesson-time').textContent = lesson.estimated_time;
    document.getElementById('lesson-xp').textContent = lesson.xp_reward;

    // Content (convert markdown to HTML - simple version)
    const contentHTML = markdownToHTML(lesson.content);
    document.getElementById('lesson-content').innerHTML = contentHTML;

    // Code Examples
    if (lesson.code_examples && lesson.code_examples.length > 0) {
        const section = document.getElementById('code-examples-section');
        section.style.display = 'block';
        const container = document.getElementById('code-examples');
        container.innerHTML = lesson.code_examples.map((code, i) => `
            <div class="code-example">
                <strong>Example ${i + 1}:</strong><br>
                <pre><code>${escapeHTML(code)}</code></pre>
            </div>
        `).join('');
    } else {
        document.getElementById('code-examples-section').style.display = 'none';
    }

    // Quiz
    if (lesson.quiz_questions && lesson.quiz_questions.length > 0) {
        renderQuiz(lesson.quiz_questions);
    } else {
        document.getElementById('quiz-section').style.display = 'none';
    }

    // Challenges
    if (lesson.challenges && lesson.challenges.length > 0) {
        renderChallenges(lesson.challenges);
    } else {
        document.getElementById('challenges-section').style.display = 'none';
    }

    // Completion button
    document.getElementById('completion-section').style.display = 'block';
    document.getElementById('complete-lesson').onclick = () => completeLesson();
}

// Simple Markdown to HTML converter
function markdownToHTML(markdown) {
    let html = markdown;

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Code blocks
    html = html.replace(/```python\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Paragraphs
    html = html.split('\n\n').map(para => {
        if (!para.startsWith('<h') && !para.startsWith('<pre>') && para.trim()) {
            return `<p>${para}</p>`;
        }
        return para;
    }).join('\n');

    return html;
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Quiz
function renderQuiz(questions) {
    const section = document.getElementById('quiz-section');
    section.style.display = 'block';

    const container = document.getElementById('quiz-questions');
    container.innerHTML = questions.map((q, i) => `
        <div class="quiz-question" id="quiz-q${i}">
            <h3>Question ${i + 1}: ${q.question}</h3>
            ${q.options.map((opt, j) => `
                <label class="quiz-option">
                    <input type="radio" name="quiz-q${i}" value="${j}">
                    ${opt}
                </label>
            `).join('')}
            <div class="quiz-result" id="quiz-result-${i}" style="display: none;"></div>
        </div>
    `).join('');

    document.getElementById('submit-quiz').onclick = () => submitQuiz(questions);
    document.getElementById('quiz-results').style.display = 'none';
}

function submitQuiz(questions) {
    let correct = 0;

    questions.forEach((q, i) => {
        const selected = document.querySelector(`input[name="quiz-q${i}"]:checked`);
        const resultDiv = document.getElementById(`quiz-result-${i}`);
        resultDiv.style.display = 'block';

        if (selected && parseInt(selected.value) === q.correct) {
            correct++;
            resultDiv.className = 'quiz-result correct';
            resultDiv.innerHTML = `<strong>✓ Correct!</strong><br>${q.explanation}`;
        } else {
            resultDiv.className = 'quiz-result incorrect';
            const correctAnswer = q.options[q.correct];
            resultDiv.innerHTML = `<strong>✗ Not quite.</strong><br>The correct answer is: <strong>${correctAnswer}</strong><br>${q.explanation}`;
        }
    });

    const score = (correct / questions.length) * 100;
    const resultsDiv = document.getElementById('quiz-results');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `
        <h3>Quiz Score: ${score.toFixed(0)}% (${correct}/${questions.length})</h3>
        ${score === 100 ? '<p>🎉 Perfect score! You really understand this!</p>' :
          score >= 70 ? '<p>✓ Great job! You\'re getting it!</p>' :
          '<p>Keep learning! Review the lesson and try again.</p>'}
    `;

    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Challenges
function renderChallenges(challenges) {
    const section = document.getElementById('challenges-section');
    section.style.display = 'block';

    const container = document.getElementById('challenges');
    container.innerHTML = challenges.map((c, i) => `
        <div class="challenge" id="challenge-${i}">
            <h3>Challenge ${i + 1}</h3>
            <p>${c.description}</p>
            <p><strong>Points:</strong> ${c.points}</p>
            ${c.starter_code ? `
                <div class="challenge-code">
                    <strong>Starter Code:</strong>
                    <pre><code>${escapeHTML(c.starter_code)}</code></pre>
                </div>
            ` : ''}
            <textarea class="challenge-input" id="challenge-input-${i}"
                      placeholder="Enter your code here..."></textarea>
            <div style="margin-top: 1rem;">
                <button class="btn btn-primary" onclick="submitChallenge(${i})">
                    Submit Solution
                </button>
                <button class="btn btn-secondary hint-btn" onclick="showHint(${i})">
                    💡 Show Hint
                </button>
            </div>
            <div id="hint-${i}" style="display: none;"></div>
            <div id="challenge-result-${i}" style="margin-top: 1rem;"></div>
        </div>
    `).join('');

    // Initialize hint levels
    challenges.forEach((c, i) => {
        currentHintLevels[i] = 0;
    });
}

async function submitChallenge(index) {
    const userCode = document.getElementById(`challenge-input-${index}`).value;
    const resultDiv = document.getElementById(`challenge-result-${index}`);

    if (!userCode.trim()) {
        resultDiv.innerHTML = '<p style="color: var(--danger)">Please enter some code first!</p>';
        return;
    }

    // Simple validation (in production, this would be server-side)
    // For demo, we just check if they entered something
    resultDiv.innerHTML = `
        <div class="quiz-result correct">
            <strong>✓ Great job!</strong><br>
            Solution submitted! +${currentLesson.challenges[index].points} points
        </div>
    `;

    // Mark challenge complete
    const challengeId = `${currentLesson.id}_challenge_${index}`;
    await apiPost('/progress/complete-challenge', {
        challenge_id: challengeId,
        points: currentLesson.challenges[index].points
    });

    // Reload progress
    await loadProgress();
}

async function showHint(index) {
    const challenge = currentLesson.challenges[index];
    const hintDiv = document.getElementById(`hint-${index}`);
    const currentLevel = currentHintLevels[index];

    if (currentLevel >= challenge.hints.length) {
        hintDiv.innerHTML = '<div class="hint">No more hints available!</div>';
        return;
    }

    const hint = challenge.hints[currentLevel];
    hintDiv.style.display = 'block';
    hintDiv.innerHTML = `
        <div class="hint">
            <strong>💡 Hint ${hint.level}:</strong><br>
            ${hint.text}
            ${hint.code_snippet ? `<br><pre><code>${escapeHTML(hint.code_snippet)}</code></pre>` : ''}
        </div>
    `;

    currentHintLevels[index]++;

    // Record hint usage
    await apiPost('/progress/use-hint', {});
    await loadProgress();
}

// Complete Lesson
async function completeLesson() {
    if (!currentLesson) return;

    try {
        const result = await apiPost('/progress/complete-lesson', {
            lesson_id: currentLesson.id,
            score: 100
        });

        // Show success message
        alert(`🎉 Lesson Complete! +${currentLesson.xp_reward} XP`);

        // Check for level up
        if (result.leveled_up) {
            showLevelUpModal(result.new_level);
        }

        // Reload data
        await loadProgress();
        await loadLessons();

        // Show next lesson suggestion
        if (currentLesson.next_lessons && currentLesson.next_lessons.length > 0) {
            const nextLessonId = currentLesson.next_lessons[0];
            const proceed = confirm(`Great job! Ready to start ${nextLessonId}?`);
            if (proceed) {
                loadLesson(nextLessonId);
            }
        }
    } catch (error) {
        console.error('Error completing lesson:', error);
        alert('Failed to complete lesson. Please try again.');
    }
}

// Level Up Modal
function showLevelUpModal(newLevel) {
    const modal = document.getElementById('level-up-modal');
    document.getElementById('new-level').textContent = newLevel;
    modal.classList.add('active');
}

function closeLevelUpModal() {
    document.getElementById('level-up-modal').classList.remove('active');
}

// Progress View
function updateProgressView() {
    document.getElementById('stats-level').textContent = progress.level;
    document.getElementById('stats-xp').textContent = progress.total_xp;
    document.getElementById('stats-lessons').textContent = progress.lessons_completed;
    document.getElementById('stats-challenges').textContent = progress.challenges_completed;
    document.getElementById('stats-badges').textContent = progress.badges_earned;
    document.getElementById('stats-completion').textContent = `${progress.completion_rate.toFixed(0)}%`;

    // Show recent badges (earned ones)
    const recentBadgesDiv = document.getElementById('recent-badges');
    if (progress.badges && progress.badges.length > 0) {
        // This will be populated when we render all badges
        recentBadgesDiv.innerHTML = '<p>See all your badges in the Badges tab!</p>';
    } else {
        recentBadgesDiv.innerHTML = '<p>Complete lessons to earn badges!</p>';
    }
}

// Badges View
function renderBadges(badges) {
    const container = document.getElementById('all-badges');
    container.innerHTML = badges.map(badge => `
        <div class="badge-card ${badge.earned ? 'earned' : ''}">
            <div class="badge-icon">${badge.icon}</div>
            <div class="badge-name">${badge.name}</div>
            <div class="badge-description">${badge.description}</div>
            ${badge.earned ? '<div style="margin-top: 1rem; color: var(--success); font-weight: bold;">✓ EARNED</div>' : ''}
        </div>
    `).join('');
}

function updateBadgesView() {
    // Badges are already rendered, just ensure they're showing earned status
    loadBadges();
}

// Make functions available globally for onclick handlers
window.submitChallenge = submitChallenge;
window.showHint = showHint;
window.closeLevelUpModal = closeLevelUpModal;
