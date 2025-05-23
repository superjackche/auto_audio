// 建立WebSocket连接
const socket = io();

// 初始化图表
let riskGauge = null;
let transcriptChart = null;
let keywordChart = null;

// DOM元素缓存
const elements = {
    transcriptContainer: document.getElementById('transcript-container'),
    riskMeter: document.getElementById('risk-meter'),
    statusIndicator: document.getElementById('status-indicator'),
    recordButton: document.getElementById('record-button')
};

// 风险等级配置
const riskLevels = {
    low: { color: '#28a745', threshold: 30 },
    medium: { color: '#ffc107', threshold: 60 },
    high: { color: '#dc3545', threshold: 90 }
};

// 初始化函数
function initializeSystem() {
    setupWebSocket();
    initializeCharts();
    setupEventListeners();
    updateStatus('已连接');
}

// 设置WebSocket事件监听
function setupWebSocket() {
    socket.on('connect', () => {
        updateStatus('已连接');
        elements.recordButton?.removeAttribute('disabled');
    });

    socket.on('disconnect', () => {
        updateStatus('已断开', 'danger');
        elements.recordButton?.setAttribute('disabled', 'disabled');
    });

    socket.on('transcript', (data) => {
        updateTranscript(data);
    });

    socket.on('risk_assessment', (data) => {
        updateRiskGauge(data.score);
        updateRiskIndicators(data);
    });

    socket.on('error', (error) => {
        showNotification(error.message, 'danger');
    });
}

// 初始化图表
function initializeCharts() {
    if (elements.riskMeter) {
        riskGauge = new Chart(elements.riskMeter, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#1e3c72', '#ecf0f1']
                }]
            },
            options: {
                circumference: 180,
                rotation: -90,
                cutout: '80%',
                plugins: {
                    tooltip: { enabled: false },
                    legend: { display: false }
                }
            }
        });
    }
}

// 更新实时转录内容
function updateTranscript(data) {
    if (!elements.transcriptContainer) return;

    const item = document.createElement('div');
    item.className = 'transcript-item';
    
    const time = document.createElement('div');
    time.className = 'transcript-time';
    time.textContent = new Date().toLocaleTimeString();
    
    const text = document.createElement('div');
    text.className = 'transcript-text';
    text.innerHTML = highlightKeywords(data.text);
    
    item.appendChild(time);
    item.appendChild(text);
    
    elements.transcriptContainer.appendChild(item);
    elements.transcriptContainer.scrollTop = elements.transcriptContainer.scrollHeight;
}

// 高亮关键词
function highlightKeywords(text) {
    // 这里应该从后端获取关键词列表
    const keywords = window.KEYWORDS || [];
    let highlightedText = text;
    
    keywords.forEach(keyword => {
        const regex = new RegExp(keyword, 'gi');
        highlightedText = highlightedText.replace(regex, 
            match => `<span class="highlight">${match}</span>`);
    });
    
    return highlightedText;
}

// 更新风险仪表
function updateRiskGauge(score) {
    if (!riskGauge) return;

    const level = getRiskLevel(score);
    riskGauge.data.datasets[0].data = [score, 100 - score];
    riskGauge.data.datasets[0].backgroundColor = [level.color, '#ecf0f1'];
    riskGauge.update();
}

// 获取风险等级
function getRiskLevel(score) {
    if (score <= riskLevels.low.threshold) return { color: riskLevels.low.color, name: '低' };
    if (score <= riskLevels.medium.threshold) return { color: riskLevels.medium.color, name: '中' };
    return { color: riskLevels.high.color, name: '高' };
}

// 更新状态指示器
function updateStatus(message, type = 'success') {
    if (!elements.statusIndicator) return;
    
    elements.statusIndicator.textContent = message;
    elements.statusIndicator.className = `status-indicator ${type}`;
}

// 显示通知
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// 设置事件监听器
function setupEventListeners() {
    elements.recordButton?.addEventListener('click', function() {
        const isRecording = this.classList.toggle('recording');
        this.innerHTML = isRecording ? 
            '<i class="bi bi-stop-circle"></i> 停止录制' : 
            '<i class="bi bi-record-circle"></i> 开始录制';
        socket.emit('toggle_recording', { recording: isRecording });
    });
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initializeSystem);

// 导出公共函数供其他模块使用
window.audioSystem = {
    updateStatus,
    showNotification,
    getRiskLevel
};
