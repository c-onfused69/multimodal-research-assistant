const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('chat-messages');

// Initialize markdown renderer
const md = window.markdownit({
    html: true,
    linkify: true,
    typographer: true
});

let chatHistory = [];

// Auto-resize textarea
input.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim() === '';
});

// Handle Enter key for submit
input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (this.value.trim() !== '') {
            form.dispatchEvent(new Event('submit'));
        }
    }
});

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function appendMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="ph-fill ph-user"></i>' : '<i class="ph-fill ph-robot"></i>';
    
    const msgContent = document.createElement('div');
    msgContent.className = 'message-content';
    
    if (role === 'assistant') {
        msgContent.innerHTML = md.render(content);
    } else {
        // Just text for user
        msgContent.textContent = content;
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(msgContent);
    messagesContainer.appendChild(msgDiv);
    
    scrollToBottom();
}

function appendThinking() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message thinking-message';
    msgDiv.id = 'thinking-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="ph-fill ph-robot"></i>';
    
    const msgContent = document.createElement('div');
    msgContent.className = 'message-content';
    msgContent.innerHTML = `
        <div class="thinking-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(msgContent);
    messagesContainer.appendChild(msgDiv);
    
    scrollToBottom();
}

function removeThinking() {
    const indicator = document.getElementById('thinking-indicator');
    if (indicator) {
        indicator.remove();
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = input.value.trim();
    if (!query) return;
    
    // Get selected mode
    const mode = document.querySelector('input[name="search-mode"]:checked').value;
    
    // Clear input
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    
    // Add User message
    appendMessage('user', query);
    appendThinking();
    
    try {
        const response = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                history: chatHistory,
                mode: mode
            })
        });
        
        removeThinking();
        
        if (!response.ok) {
            let errorText = await response.text();
            try {
                const errorJson = JSON.parse(errorText);
                errorText = errorJson.detail || errorText;
            } catch (e) {}
            appendMessage('assistant', `**Error:** ${errorText}`);
            return;
        }
        
        const data = await response.json();
        let answer = data.answer || "Sorry, I couldn't generate an answer.";
        const citations = data.citations || [];
        
        // Add citations to markdown
        if (citations.length > 0) {
            answer += '\n\n---\n**Sources:**\n';
            citations.forEach(c => {
                answer += `- [${c.index}] ${c.source} _(Score: ${c.score.toFixed(2)})_\n`;
            });
        }
        
        appendMessage('assistant', answer);
        
        // Update history
        chatHistory.push({"role": "user", "content": query});
        chatHistory.push({"role": "assistant", "content": answer});
        
    } catch (error) {
        removeThinking();
        appendMessage('assistant', `**Error:** ${error.message}`);
    }
});
