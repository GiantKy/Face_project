/**
 * ============================================================================
 * NODE.JS RECEIVER SERVER - TIẾP NHẬN KẾT QUẢ TỪ FASTAPI AI SERVER
 * ============================================================================
 * 
 * Máy chủ này lắng nghe Webhook được đẩy tự động từ FastAPI AI Server sau khi
 * ESP32-CAM chụp và gửi ảnh lên.
 * 
 * TÍNH NĂNG:
 * 1. Chạy thuần bằng Node.js tiêu chuẩn (Không cần cài bất kỳ thư viện npm nào!).
 * 2. Tiếp nhận kết quả tại POST /api/ekyc/result.
 * 3. Hiển thị Dashboard kết quả trực quan trên Terminal với mã màu.
 * 4. Tự động trích xuất Base64 lưu ảnh khuôn mặt nhận diện vào thư mục `captured_faces/`.
 * 5. Cung cấp Web Dashboard trực tiếp tại http://localhost:3000 để giám sát trực tiếp.
 * 
 * HƯỚNG DẪN KHỞI ĐỘNG:
 *   node server_module/nodejs_server_receiver.js
 * ============================================================================
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const SAVE_DIR = path.join(__dirname, 'captured_faces');

// Tạo thư mục lưu ảnh khuôn mặt nếu chưa có
if (!fs.existsSync(SAVE_DIR)) {
  fs.mkdirSync(SAVE_DIR, { recursive: true });
}

// Lưu trữ 20 kết quả gần nhất trong RAM để hiển thị trên Web Dashboard
let recentVerifications = [];

// Mã màu ANSI cho Terminal
const Colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  blue: '\x1b[34m',
  gray: '\x1b[90m',
};

const server = http.createServer((req, res) => {
  // Bật CORS cho mọi nguồn
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // --------------------------------------------------------------------------
  // 1. ROUTE WEBHOOK: POST /api/ekyc/result (Nhận kết quả từ FastAPI AI Server)
  // --------------------------------------------------------------------------
  if (req.method === 'POST' && req.url === '/api/ekyc/result') {
    let bodyChunks = [];

    req.on('data', chunk => {
      bodyChunks.push(chunk);
    });

    req.on('end', () => {
      try {
        const rawBody = Buffer.concat(bodyChunks).toString('utf-8');
        const payload = JSON.parse(rawBody);

        // Xử lý dữ liệu nhận diện
        const result = processEKYCResult(payload);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          status: 'SUCCESS',
          message: 'Node.js Server đã tiếp nhận kết quả xác thực thành công!',
          received_at: new Date().toISOString(),
          record_id: result.id
        }));
      } catch (err) {
        console.error(`${Colors.red}[LỖI PARSE JSON]: ${err.message}${Colors.reset}`);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ERROR', message: 'Payload JSON không hợp lệ' }));
      }
    });
    return;
  }

  // --------------------------------------------------------------------------
  // 2. ROUTE API LỊCH SỬ: GET /api/ekyc/history
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && req.url === '/api/ekyc/history') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      total: recentVerifications.length,
      history: recentVerifications
    }));
    return;
  }

  // --------------------------------------------------------------------------
  // 3. ROUTE WEB DASHBOARD: GET / (Giao diện giám sát Realtime)
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(generateDashboardHTML());
    return;
  }

  // --------------------------------------------------------------------------
  // 4. PHỤC VỤ ẢNH ĐÃ LƯU: GET /images/:filename
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && req.url.startsWith('/images/')) {
    const filename = path.basename(req.url);
    const filePath = path.join(SAVE_DIR, filename);

    if (fs.existsSync(filePath)) {
      res.writeHead(200, { 'Content-Type': 'image/jpeg' });
      fs.createReadStream(filePath).pipe(res);
      return;
    }
  }

  // Route 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Endpoint không tồn tại' }));
});

/**
 * Xử lý dữ liệu xác thực nhận từ FastAPI AI Server
 */
function processEKYCResult(data) {
  const timestamp = data.timestamp || new Date().toLocaleString('vi-VN');
  const deviceId = data.device_id || 'UNKNOWN_DEVICE';
  const approved = data.approved === true;
  const verdict = data.verdict || (approved ? 'REAL' : 'SPOOF');
  const isReal = data.is_real === true;
  const confidence = typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(1) : 'N/A';
  const procTime = data.processing_time_ms || 'N/A';

  const recordId = `REC_${Date.now()}`;
  let savedImagePath = null;
  let savedImageRelUrl = null;

  // Tự động lưu ảnh khuôn mặt nếu có chuỗi base64
  if (data.crop_face_base64) {
    try {
      let b64Data = data.crop_face_base64;
      if (b64Data.includes(',')) {
        b64Data = b64Data.split(',')[1];
      }
      const filename = `face_${Date.now()}_${verdict}.jpg`;
      savedImagePath = path.join(SAVE_DIR, filename);
      fs.writeFileSync(savedImagePath, Buffer.from(b64Data, 'base64'));
      savedImageRelUrl = `/images/${filename}`;
    } catch (e) {
      console.error(`Không thể lưu file ảnh: ${e.message}`);
    }
  }

  // Lưu vào bộ nhớ RAM phục vụ Web Dashboard
  const record = {
    id: recordId,
    timestamp,
    device_id: deviceId,
    approved,
    verdict,
    is_real: isReal,
    confidence,
    reasons: data.reasons || [],
    image_url: savedImageRelUrl,
    processing_time_ms: procTime
  };

  recentVerifications.unshift(record);
  if (recentVerifications.length > 25) {
    recentVerifications.pop();
  }

  // In thông tin ra Terminal Dashboard
  printTerminalDashboard(record);

  return record;
}

/**
 * In giao diện Terminal Dashboard
 */
function printTerminalDashboard(r) {
  const isPass = r.approved;
  const statusColor = isPass ? Colors.green : Colors.red;
  const icon = isPass ? '✅' : '❌';
  const border = '═'.repeat(65);

  console.log(`\n${statusColor}${border}${Colors.reset}`);
  console.log(` ${icon} ${Colors.bright}SỰ KIỆN eKYC ESP32-CAM: [${r.verdict}]${Colors.reset}  (${r.timestamp})`);
  console.log(`${statusColor}${border}${Colors.reset}`);
  console.log(` • Thiết bị gửi:  ${Colors.cyan}${r.device_id}${Colors.reset}`);
  console.log(` • Kết luận:      ${statusColor}${r.verdict} (${isPass ? 'ĐƯỢC DUYỆT' : 'BỊ TỪ CHỐI'})${Colors.reset}`);
  console.log(` • Độ tin cậy:    ${Colors.yellow}${r.confidence}%${Colors.reset}`);
  console.log(` • Thời gian AI:  ${Colors.gray}${r.processing_time_ms} ms${Colors.reset}`);
  if (r.reasons && r.reasons.length > 0) {
    console.log(` • Chi tiết:      ${r.reasons.join(', ')}`);
  }
  if (r.image_url) {
    console.log(` • Ảnh khuôn mặt: ${Colors.blue}${r.image_url}${Colors.reset}`);
  }
  console.log(`${statusColor}${border}${Colors.reset}\n`);
}

/**
 * Tạo giao diện HTML Web Dashboard đơn giản
 */
function generateDashboardHTML() {
  return `
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>eKYC Access Control - Node.js Monitor</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131b2e;
      --border: #222f4c;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --green: #10b981;
      --red: #ef4444;
      --accent: #38bdf8;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }
    .container {
      max-width: 1000px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    h1 {
      font-size: 22px;
      margin: 0;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-live {
      background: rgba(16, 185, 129, 0.2);
      color: var(--green);
      border: 1px solid var(--green);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      display: flex;
      gap: 14px;
    }
    .card.pass { border-left: 4px solid var(--green); }
    .card.fail { border-left: 4px solid var(--red); }
    .avatar {
      width: 80px;
      height: 80px;
      border-radius: 8px;
      object-fit: cover;
      background: #020617;
      border: 1px solid var(--border);
    }
    .info {
      flex: 1;
    }
    .verdict {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .pass .verdict { color: var(--green); }
    .fail .verdict { color: var(--red); }
    .meta {
      font-size: 12px;
      color: var(--text-dim);
      margin-bottom: 3px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🛡️ eKYC Gate Access Monitor (Node.js)</h1>
      <div class="badge-live">● SẴN SÀNG</div>
    </header>

    <div id="cardsList" class="grid">
      <div style="color: var(--text-dim); font-size: 14px;">Đang tải dữ liệu...</div>
    </div>
  </div>

  <script>
    async function loadData() {
      try {
        const res = await fetch('/api/ekyc/history');
        const data = await res.json();
        const list = document.getElementById('cardsList');

        if (!data.history || data.history.length === 0) {
          list.innerHTML = '<div style="color: var(--text-dim); font-size: 14px;">Chưa có lượt xác thực nào từ ESP32-CAM. Hãy bấm chụp trên camera!</div>';
          return;
        }

        list.innerHTML = data.history.map(item => \`
          <div class="card \${item.approved ? 'pass' : 'fail'}">
            \${item.image_url ? \`<img class="avatar" src="\${item.image_url}" alt="Face">\` : '<div class="avatar" style="display:flex;align-items:center;justify-content:center;font-size:24px;">👤</div>'}
            <div class="info">
              <div class="verdict">\${item.approved ? '✅' : '❌'} \${item.verdict}</div>
              <div class="meta">📍 \${item.device_id}</div>
              <div class="meta">🎯 Độ tin cậy: \${item.confidence}%</div>
              <div class="meta">⏱️ \${item.processing_time_ms} ms</div>
              <div class="meta">🕒 \${item.timestamp}</div>
            </div>
          </div>
        \`).join('');
      } catch (err) {
        console.error(err);
      }
    }

    loadData();
    setInterval(loadData, 2000);
  </script>
</body>
</html>
  `;
}

server.listen(PORT, () => {
  console.log(`\n${Colors.bright}${Colors.green}===============================================================${Colors.reset}`);
  console.log(`${Colors.bright} [NODE.JS RECEIVER SERVER ĐÃ SẴN SÀNG]${Colors.reset}`);
  console.log(`${Colors.green}===============================================================${Colors.reset}`);
  console.log(` • Lắng nghe Webhook tại:  ${Colors.cyan}http://127.0.0.1:${PORT}/api/ekyc/result${Colors.reset}`);
  console.log(` • Web Realtime Dashboard:  ${Colors.yellow}http://localhost:${PORT}/${Colors.reset}`);
  console.log(` • Thư mục lưu ảnh:        ${Colors.blue}${SAVE_DIR}${Colors.reset}`);
  console.log(`${Colors.green}===============================================================${Colors.reset}\n`);
});
