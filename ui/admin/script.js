const API_KEY = 'secret-admin-key';
const HEADERS = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
};

// DOM Elements
const statTextChunks = document.getElementById('stat-text-chunks');
const statVisualPages = document.getElementById('stat-visual-pages');
const statTables = document.getElementById('stat-tables');
const statCacheSize = document.getElementById('stat-cache-size');

const refreshBtn = document.getElementById('refresh-btn');
const ingestBtn = document.getElementById('ingest-btn');
const notificationsArea = document.getElementById('notifications');
const statusDot = document.querySelector('.status-dot');

function showNotification(message, isError = false) {
    const notif = document.createElement('div');
    notif.className = `notification ${isError ? 'error' : ''}`;
    notif.innerHTML = message;
    
    notificationsArea.appendChild(notif);
    
    setTimeout(() => {
        notif.style.opacity = '0';
        setTimeout(() => notif.remove(), 300);
    }, 4000);
}

async function fetchStats() {
    try {
        refreshBtn.style.transform = 'rotate(180deg)';
        refreshBtn.style.transition = 'transform 0.5s ease';
        
        const res = await fetch('/api/v1/admin/stats', { headers: HEADERS });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const data = await res.json();
        
        // Update DOM
        statTextChunks.textContent = data.qdrant.text_chunks.toLocaleString();
        statVisualPages.textContent = data.qdrant.visual_pages.toLocaleString();
        statTables.textContent = data.qdrant.tables.toLocaleString();
        statCacheSize.textContent = data.redis.cache_size.toLocaleString();
        
        statusDot.className = 'status-dot healthy';
        
        setTimeout(() => {
            refreshBtn.style.transform = 'rotate(0deg)';
        }, 500);
        
    } catch (e) {
        console.error(e);
        statusDot.className = 'status-dot error';
        statusDot.style.backgroundColor = 'var(--color-error)';
        showNotification(`Failed to load stats: ${e.message}`, true);
    }
}

async function triggerIngestion() {
    try {
        ingestBtn.disabled = true;
        ingestBtn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Starting...';
        
        const res = await fetch('/api/v1/admin/ingest', { 
            method: 'POST',
            headers: HEADERS 
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        showNotification(`<strong>Success!</strong><br/>${data.message}`);
        
    } catch (e) {
        console.error(e);
        showNotification(`Ingestion Failed: ${e.message}`, true);
    } finally {
        setTimeout(() => {
            ingestBtn.disabled = false;
            ingestBtn.innerHTML = '<i class="ph-bold ph-play"></i> Run Ingestion';
        }, 2000);
    }
}

// Event Listeners
refreshBtn.addEventListener('click', fetchStats);
ingestBtn.addEventListener('click', triggerIngestion);

// Auto-refresh every 30 seconds
setInterval(fetchStats, 30000);

// Initial Load
document.addEventListener('DOMContentLoaded', fetchStats);
