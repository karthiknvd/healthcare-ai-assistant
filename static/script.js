const chatArea = document.getElementById('chatArea');
    const emptyState = document.getElementById('emptyState');
    const imageSection = document.getElementById('imageSection');
    const typingWrap = document.getElementById('typingWrap');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const modeBtn = document.getElementById('modeBtn');
    const modeMenu = document.getElementById('modeMenu');
    const clearBtn = document.getElementById('clearBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const quickChips = document.getElementById('quickChips');
    const dropZone = document.getElementById('dropZone');
    const browseBtn = document.getElementById('browseBtn');
    const fileInput = document.getElementById('fileInput');
    const imgResults = document.getElementById('imgResults');

    let currentMode = 'chat';
    let isUploading = false;
    let hasMessages = false;

    function getTime() {
        return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
    }

    // ── MODE TOGGLE ──────────────────────────────────────────────────────────
    modeBtn.addEventListener('click', e => {
        e.stopPropagation();
        modeMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', e => {
        if (!modeMenu.contains(e.target) && e.target !== modeBtn) {
            modeMenu.classList.add('hidden');
        }
    });

    document.querySelectorAll('.mode-opt').forEach(opt => {
        opt.addEventListener('click', () => {
            const mode = opt.dataset.mode;
            document.querySelectorAll('.mode-opt').forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            currentMode = mode;
            modeMenu.classList.add('hidden');
            switchMode(mode);
        });
    });

    function switchMode(mode) {
        if (mode === 'chat') {
            imageSection.classList.add('hidden');
            quickChips.style.display = 'flex';
            chatInput.disabled = false;
            chatInput.placeholder = 'Ask any health question…';
            if (hasMessages) { chatArea.classList.remove('hidden'); emptyState.classList.add('hidden'); }
            else { chatArea.classList.add('hidden'); emptyState.classList.remove('hidden'); }
        } else {
            chatArea.classList.add('hidden');
            emptyState.classList.add('hidden');
            imageSection.classList.remove('hidden');
            quickChips.style.display = 'none';
            chatInput.disabled = true;
            chatInput.placeholder = 'Switch to Health Chat mode to ask questions…';
            imgResults.classList.add('hidden');
        }
    }

    // ── TYPING ANIMATION ─────────────────────────────────────────────────────
    function typeMessage(bubble, html, onDone) {
        let i = 0;
        const SPEED = 6;

        function next() {
            if (i >= html.length) {
                bubble.innerHTML = html;
                if (onDone) onDone();
                return;
            }

            if (html[i] === '<') {
                const closeIdx = html.indexOf('>', i);
                if (closeIdx !== -1) {
                    i = closeIdx + 1;
                    bubble.innerHTML = html.slice(0, i) + '<span class="typing-cursor"></span>';
                    requestAnimationFrame(next);
                    return;
                }
            }

            if (html[i] === '&') {
                const semiIdx = html.indexOf(';', i);
                if (semiIdx !== -1 && semiIdx - i <= 8) {
                    i = semiIdx + 1;
                    bubble.innerHTML = html.slice(0, i) + '<span class="typing-cursor"></span>';
                    requestAnimationFrame(next);
                    return;
                }
            }

            bubble.innerHTML = html.slice(0, i + 1) + '<span class="typing-cursor"></span>';
            i++;
            setTimeout(next, SPEED);
        }

        next();
    }

    // ── SEND MESSAGE ─────────────────────────────────────────────────────────
    async function sendMessage(text) {
        const msg = text || chatInput.value.trim();
        if (!msg) return;

        if (!hasMessages) {
            hasMessages = true;
            emptyState.classList.add('hidden');
            chatArea.classList.remove('hidden');
        }

        addMessage(msg, 'user');
        chatInput.value = '';
        chatInput.style.height = 'auto';
        chatInput.disabled = true;
        sendBtn.disabled = true;

        typingWrap.classList.remove('hidden');

        try {
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await res.json();
            typingWrap.classList.add('hidden');
            addMessage(data.reply, 'bot', true);
        } catch (err) {
            typingWrap.classList.add('hidden');
            addMessage('❌ <b>Connection error.</b> Please check the server is running and try again.', 'bot', false);
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    function addMessage(text, role, animate = false) {
        const wrap = document.createElement('div');
        wrap.className = `message ${role === 'bot' ? 'bot' : 'user'}`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = role === 'bot' ? '⚕️' : '👤';

        const body = document.createElement('div');
        body.className = 'msg-body';

        const meta = document.createElement('div');
        meta.className = 'msg-meta';
        meta.innerHTML = `
            <span class="msg-name">${role === 'bot' ? 'Healthcare AI' : 'You'}</span>
            <span class="msg-time">${getTime()}</span>
        `;

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        body.appendChild(meta);
        body.appendChild(bubble);
        wrap.appendChild(avatar);
        wrap.appendChild(body);
        chatArea.appendChild(wrap);

        setTimeout(() => {
            wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);

        if (role === 'bot' && animate) {
            typeMessage(bubble, text, () => {});
        } else {
            bubble.innerHTML = text;
        }
    }

    // ── SCROLL BUTTON ────────────────────────────────────────────────────────
    const scrollBtn = document.createElement('button');
    scrollBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>`;
    scrollBtn.style.cssText = `position:fixed;bottom:110px;right:28px;width:38px;height:38px;border-radius:50%;background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.4);color:#38bdf8;cursor:pointer;display:none;align-items:center;justify-content:center;z-index:999;backdrop-filter:blur(10px);transition:all 0.2s ease;box-shadow:0 4px 12px rgba(0,0,0,0.3);`;
    scrollBtn.addEventListener('click', () => chatArea.scrollTop = chatArea.scrollHeight);
    document.body.appendChild(scrollBtn);

    chatArea.addEventListener('scroll', () => {
        const isNearBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 60;
        scrollBtn.style.display = isNearBottom ? 'none' : 'flex';
    });

    // ── INPUT HANDLERS ───────────────────────────────────────────────────────
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 130) + 'px';
    });

    chatInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey && currentMode === 'chat') {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', () => { if (currentMode === 'chat') sendMessage(); });

    document.querySelectorAll('.chip').forEach(btn => btn.addEventListener('click', () => sendMessage(btn.dataset.q)));
    document.querySelectorAll('.quick-ask-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (currentMode !== 'chat') { currentMode = 'chat'; switchMode('chat'); }
            sendMessage(btn.dataset.q);
        });
    });
    document.querySelectorAll('.example-q').forEach(btn => btn.addEventListener('click', () => sendMessage(btn.dataset.q)));

    function clearChat() {
        chatArea.innerHTML = '';
        hasMessages = false;
        emptyState.classList.remove('hidden');
        chatArea.classList.add('hidden');
        fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: 'reset' }) });
    }

    clearBtn.addEventListener('click', clearChat);
    newChatBtn.addEventListener('click', clearChat);

    // ── IMAGE UPLOAD ─────────────────────────────────────────────────────────
    const dropZoneContent = document.getElementById('dropZoneContent');
    const idleHTML = dropZoneContent.innerHTML;

    // All listeners attached ONCE to static elements — never duplicated
    browseBtn.addEventListener('click', e => { e.stopPropagation(); if (!isUploading) fileInput.click(); });
    fileInput.addEventListener('change', e => { const f = e.target.files[0]; if (f) { uploadImage(f); e.target.value = ''; } });
    dropZone.addEventListener('click', e => { if (!e.target.closest('#browseBtn') && !isUploading) fileInput.click(); });
    dropZone.addEventListener('dragover', e => { e.preventDefault(); if (!isUploading) dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) uploadImage(f);
    });

    async function uploadImage(file) {
        if (isUploading) return;
        isUploading = true;
        imgResults.classList.add('hidden');

        // Only swap inner content — fileInput lives outside and keeps its listener
        dropZoneContent.innerHTML = `
            <div style="display:flex;flex-direction:column;align-items:center;gap:16px">
                <div style="display:flex;gap:6px">
                    <div class="td"></div>
                    <div class="td" style="animation-delay:.2s"></div>
                    <div class="td" style="animation-delay:.4s"></div>
                </div>
                <p style="color:var(--text-secondary);font-size:13px">Processing ${file.name}…</p>
            </div>`;

        const formData = new FormData();
        formData.append('image', file);

        try {
            const res = await fetch('/analyze-image', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.success) {
                document.getElementById('edgeImg').src     = data.processed_images.edge_map;
                document.getElementById('contrastImg').src = data.processed_images.contrast_enhanced;
                document.getElementById('heatmapImg').src  = data.processed_images.attention_heatmap;
                document.getElementById('anomalyImg').src  = data.processed_images.anomaly_detection;

                if (data.descriptions) {
                    document.getElementById('edgeDesc').innerHTML    = data.descriptions.edge_map          || '';
                    document.getElementById('contrastDesc').innerHTML = data.descriptions.contrast_enhanced || '';
                    document.getElementById('heatmapDesc').innerHTML  = data.descriptions.attention_heatmap || '';
                    document.getElementById('anomalyDesc').innerHTML  = data.descriptions.anomaly_detection || '';
                }

                imgResults.classList.remove('hidden');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            alert('Upload failed: ' + err.message);
        } finally {
            isUploading = false;
            // Restore idle content — no new listeners added
            dropZoneContent.innerHTML = idleHTML;
        }
    }

    chatInput.focus();