const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 3000;

const html = `<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>S2ST 语音翻译 Demo</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 600px; margin: 40px auto; padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    h1 { text-align: center; color: white; margin-bottom: 8px; }
    .subtitle { text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 24px; }
    .card {
      background: white; border-radius: 16px; padding: 24px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px;
    }
    label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
    button {
      width: 100%; padding: 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; border: none; border-radius: 12px; font-size: 16px; 
      cursor: pointer; margin-top: 12px; font-weight: 600;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
    button:disabled { background: #ccc; transform: none; box-shadow: none; cursor: not-allowed; }
    .record-btn { background: linear-gradient(135deg, #f44336 0%, #e91e63 100%); }
    .record-btn.recording { animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
    audio { width: 100%; margin-top: 12px; border-radius: 8px; }
    .info { background: #e8f5e9; padding: 16px; border-radius: 12px; margin-top: 16px; }
    .info.error { background: #ffebee; color: #c62828; }
    .transcript { background: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 14px; }
    .transcript .en { color: #1976d2; }
    .transcript .zh { color: #388e3c; font-weight: 500; }
    .row { display: flex; gap: 10px; margin-top: 12px; }
    .row button { flex: 1; margin-top: 0; }
    .sample-btn { background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%); }
    .hidden { display: none; }
    .status { text-align: center; color: #666; font-size: 14px; margin-top: 8px; }
    input[type="file"] { width: 100%; padding: 12px; border: 2px dashed #ddd; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>🌏 S2ST 语音翻译</h1>
  <p class="subtitle">英语 → 中文 实时翻译 Demo</p>
  
  <div class="card">
    <label>🎤 录音或上传英语语音</label>
    
    <button id="recordBtn" class="record-btn" onclick="toggleRecord()">🎤 点击开始录音</button>
    <div id="recordStatus" class="status hidden"></div>
    
    <div class="row">
      <input type="file" id="audioFile" accept="audio/*" onchange="onFileSelected()" style="flex:2">
      <button class="sample-btn" onclick="useSample()" style="flex:1">📢 示例</button>
    </div>
    
    <audio id="inputAudio" controls class="hidden"></audio>
    
    <button id="translateBtn" onclick="translate()">🚀 翻译</button>
  </div>
  
  <div id="result" class="card hidden">
    <label>🔊 翻译结果</label>
    <audio id="outputAudio" controls></audio>
    <div id="transcript" class="transcript hidden"></div>
    <div id="info" class="info"></div>
  </div>

<script>
let audioBlob = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function initRecorder() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    
    mediaRecorder.onstop = () => {
      audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];
      
      document.getElementById('inputAudio').src = URL.createObjectURL(audioBlob);
      document.getElementById('inputAudio').classList.remove('hidden');
      document.getElementById('recordStatus').textContent = '✅ 录音完成';
    };
    
    return true;
  } catch (e) {
    alert('无法访问麦克风: ' + e.message);
    return false;
  }
}

async function toggleRecord() {
  const btn = document.getElementById('recordBtn');
  const status = document.getElementById('recordStatus');
  
  if (!mediaRecorder) {
    const ok = await initRecorder();
    if (!ok) return;
  }
  
  if (!isRecording) {
    audioChunks = [];
    mediaRecorder.start();
    isRecording = true;
    btn.classList.add('recording');
    btn.textContent = '🔴 录音中... 点击停止';
    status.textContent = '正在录音...';
    status.classList.remove('hidden');
  } else {
    mediaRecorder.stop();
    isRecording = false;
    btn.classList.remove('recording');
    btn.textContent = '🎤 点击开始录音';
  }
}

function onFileSelected() {
  const fileInput = document.getElementById('audioFile');
  if (fileInput.files[0]) {
    audioBlob = fileInput.files[0];
    document.getElementById('inputAudio').src = URL.createObjectURL(audioBlob);
    document.getElementById('inputAudio').classList.remove('hidden');
    document.getElementById('recordStatus').classList.add('hidden');
  }
}

async function useSample() {
  const res = await fetch('/sample/source');
  audioBlob = await res.blob();
  document.getElementById('inputAudio').src = URL.createObjectURL(audioBlob);
  document.getElementById('inputAudio').classList.remove('hidden');
  document.getElementById('recordStatus').textContent = '✅ 已加载示例';
  document.getElementById('recordStatus').classList.remove('hidden');
}

async function translate() {
  if (!audioBlob) {
    alert('请先录音或上传音频');
    return;
  }
  
  const btn = document.getElementById('translateBtn');
  const result = document.getElementById('result');
  const info = document.getElementById('info');
  const transcript = document.getElementById('transcript');
  
  btn.disabled = true;
  btn.textContent = '⏳ 翻译中...';
  result.classList.add('hidden');
  
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'input.wav');
    
    const start = Date.now();
    const res = await fetch('/translate', { method: 'POST', body: formData });
    const elapsed = Date.now() - start;
    
    if (!res.ok) throw new Error(await res.text());
    
    const data = await res.json();
    
    document.getElementById('outputAudio').src = '/output/' + data.output;
    
    // 显示转写和翻译
    if (data.text_en && data.text_zh) {
      transcript.innerHTML = '<span class="en">🇺🇸 ' + data.text_en + '</span><br><span class="zh">🇨🇳 ' + data.text_zh + '</span>';
      transcript.classList.remove('hidden');
    }
    
    info.className = 'info';
    info.innerHTML = '✅ 翻译完成！<br>ASR: ' + (data.timings?.asr?.toFixed(2) || '?') + 's | MT: ' + (data.timings?.mt?.toFixed(2) || '?') + 's | TTS: ' + (data.timings?.tts?.toFixed(2) || '?') + 's<br>总耗时: ' + (elapsed/1000).toFixed(2) + 's';
    result.classList.remove('hidden');
    document.getElementById('outputAudio').play();
    
  } catch (e) {
    info.className = 'info error';
    info.textContent = '❌ ' + e.message;
    result.classList.remove('hidden');
  }
  
  btn.disabled = false;
  btn.textContent = '🚀 翻译';
}
</script>
</body>
</html>`;

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://' + req.headers.host);
  
  if (url.pathname === '/' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
    return;
  }
  
  if (url.pathname.startsWith('/sample/')) {
    const type = url.pathname.split('/')[2];
    const file = type === 'source' ? '/tmp/s2st-distill/test_source.wav' : '/tmp/s2st-distill/test_target.wav';
    if (fs.existsSync(file)) {
      res.writeHead(200, { 'Content-Type': 'audio/wav' });
      fs.createReadStream(file).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
    return;
  }
  
  if (url.pathname.startsWith('/output/')) {
    const file = '/tmp/s2st-distill/output/' + path.basename(url.pathname);
    if (fs.existsSync(file)) {
      res.writeHead(200, { 'Content-Type': 'audio/wav' });
      fs.createReadStream(file).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
    return;
  }
  
  if (url.pathname === '/translate' && req.method === 'POST') {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', async () => {
      try {
        const buffer = Buffer.concat(chunks);
        const contentType = req.headers['content-type'] || '';
        
        if (!contentType.includes('multipart/form-data')) {
          res.writeHead(400);
          res.end('Bad request');
          return;
        }
        
        const boundary = contentType.split('boundary=')[1];
        if (!boundary) {
          res.writeHead(400);
          res.end('No boundary');
          return;
        }
        
        const crlfcrlf = Buffer.from('\r\n\r\n');
        let headerEnd = buffer.indexOf(crlfcrlf);
        if (headerEnd < 0) {
          res.writeHead(400);
          res.end('Invalid format');
          return;
        }
        
        const contentStart = headerEnd + 4;
        const endBoundary = Buffer.from('\r\n--' + boundary);
        let contentEnd = buffer.indexOf(endBoundary, contentStart);
        if (contentEnd < 0) contentEnd = buffer.length;
        
        const audioData = buffer.slice(contentStart, contentEnd);
        
        if (audioData.length < 100) {
          res.writeHead(400);
          res.end('Audio too small');
          return;
        }
        
        const ts = Date.now();
        const inputPath = '/tmp/s2st-distill/upload_' + ts + '.webm';
        const wavPath = '/tmp/s2st-distill/upload_' + ts + '.wav';
        const outputName = 'translated_' + ts + '.wav';
        const outputPath = '/tmp/s2st-distill/output/' + outputName;
        
        fs.writeFileSync(inputPath, audioData);
        
        const runTranslate = () => {
          const py = spawn('/tmp/s2st-distill/s2st_env/bin/python', [
            '/tmp/s2st-distill/cascade/translate_cascade_fast.py',
            wavPath,
            outputPath
          ]);
          
          let stdout = '', stderr = '';
          py.stdout.on('data', d => stdout += d);
          py.stderr.on('data', d => stderr += d);
          
          py.on('close', code => {
            try { fs.unlinkSync(inputPath); } catch {}
            try { fs.unlinkSync(wavPath); } catch {}
            
            if (code === 0 && fs.existsSync(outputPath)) {
              try {
                const result = JSON.parse(stdout);
                result.output = outputName;
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(result));
              } catch {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ output: outputName }));
              }
            } else {
              res.writeHead(500);
              res.end('Translation failed: ' + stderr);
            }
          });
        };
        
        // 检查格式并转换
        const header = audioData.slice(0, 12).toString();
        if (header.includes('RIFF') || header.includes('WAVE')) {
          fs.renameSync(inputPath, wavPath);
          runTranslate();
        } else {
          const ffmpeg = spawn('ffmpeg', ['-y', '-i', inputPath, '-ar', '16000', '-ac', '1', wavPath]);
          ffmpeg.on('close', code => {
            if (code === 0) {
              runTranslate();
            } else {
              res.writeHead(500);
              res.end('Audio conversion failed');
            }
          });
        }
        
      } catch (e) {
        res.writeHead(500);
        res.end('Error: ' + e.message);
      }
    });
    return;
  }
  
  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log('🌏 S2ST Demo: http://localhost:' + PORT);
});
