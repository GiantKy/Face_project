/**
 * ============================================================================
 * NODE.JS RECEIVER & STREAM RELAY HUB (:3000)
 * ============================================================================
 * 
 * Máy chủ trung tâm đóng 3 vai trò:
 * 1. STREAM INGESTION HUB: Tiếp nhận luồng frame JPEG liên tục từ ESP32-CAM qua POST /api/stream/frame.
 * 2. STREAM BROADCASTER: Phát luồng video MJPEG tại GET /stream và GET /api/stream/latest cho
 *    Trình duyệt Web và FastAPI AI Server quét tự động qua localhost.
 * 3. WEB eKYC CONTROLLER: Phục vụ giao diện người dùng static/index.html tại http://localhost:3000
 *    và đóng vai trò Reverse Proxy chuyển tiếp các API eKYC sang FastAPI AI Server (:8000).
 * 
 * ĐẶC BIỆT: Chạy thuần 100% bằng thư viện tiêu chuẩn của Node.js (Không cần npm install!).
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

// ============================================================================
// BỘ NHỚ ĐỆM STREAM TRONG RAM
// ============================================================================
let latestFrame = null;            // Buffer JPEG của frame mới nhất
let latestFrameTime = 0;           // Thời điểm nhận frame gần nhất (ms)
let esp32DeviceIp = '';            // IP của thiết bị ESP32-CAM
const streamClients = new Set();   // Danh sách client đang kết nối luồng MJPEG GET /stream

// Lưu trữ 25 kết quả gần nhất trong RAM
let recentVerifications = [];

// Thống kê luồng ESP32 Ingestion
let frameReceiveCounter = 0;
let lastLogTimestamp = 0;

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

/**
 * Phát frame mới nhất tới tất cả các client đang kết nối MJPEG (/stream)
 * CÓ CƠ CHẾ DROP FRAME CHỐNG NGHẼN BỘ ĐỆM TCP (ANTI BUFFER-BLOAT):
 * Nếu client chưa gửi xong frame trước, bỏ qua frame này để giữ độ trễ 0ms!
 */
function broadcastFrame(frameBuffer) {
  if (streamClients.size === 0) return;
  const boundary = '--mjpeg_frame\r\n';
  const header = `Content-Type: image/jpeg\r\nContent-Length: ${frameBuffer.length}\r\n\r\n`;
  const footer = '\r\n';

  for (const client of streamClients) {
    // Nếu client socket còn dữ liệu đang xếp hàng trong buffer (chưa gửi xong ra mạng),
    // BỎ QUA frame này đối với client này! Không nhồi thêm vào hàng đợi TCP!
    // Trình duyệt sẽ luôn nhận frame mới nhất với độ trễ 0ms!
    if (client.writableLength > 0 || (client.socket && client.socket.bufferSize > 0)) {
      continue;
    }

    try {
      client.write(boundary);
      client.write(header);
      client.write(frameBuffer);
      client.write(footer);
    } catch (err) {
      streamClients.delete(client);
    }
  }
}

const server = http.createServer((req, res) => {
  // Bật CORS cho mọi nguồn
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Device-ID, X-ESP32-IP');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = new URL(req.url, 'http://localhost');
  const pathname = parsedUrl.pathname;

  // --------------------------------------------------------------------------
  // 1. INGESTION ROUTE: POST /api/stream/frame (ESP32 đẩy frame JPEG liên tục)
  // --------------------------------------------------------------------------
  if (req.method === 'POST' && pathname === '/api/stream/frame') {
    if (req.socket) req.socket.setNoDelay(true);
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const buffer = Buffer.concat(chunks);
      if (buffer.length > 500) {
        latestFrame = buffer;
        latestFrameTime = Date.now();

        // Ghi nhận IP ESP32
        const rawIp = req.headers['x-esp32-ip'] || req.socket.remoteAddress || '';
        esp32DeviceIp = rawIp.replace(/^::ffff:/, '');

        frameReceiveCounter++;
        const now = Date.now();
        if (now - lastLogTimestamp >= 5000) {
          const fps = Math.round((frameReceiveCounter * 1000) / Math.max(1, now - lastLogTimestamp));
          console.log(`${Colors.cyan}[ESP32 STREAM INGEST] Đang nhận frame từ ${esp32DeviceIp || 'ESP32'} (~${fps} FPS, ${buffer.length} bytes, viewers: ${streamClients.size})${Colors.reset}`);
          frameReceiveCounter = 0;
          lastLogTimestamp = now;
        }

        // Broadcast ngay lập tức cho các client đang xem
        broadcastFrame(buffer);
      }

      res.writeHead(200, {
        'Content-Type': 'text/plain',
        'Content-Length': '2',
        'Connection': 'keep-alive'
      });
      res.end('OK');
    });
    req.on('error', (err) => {
      // Bỏ qua lỗi ngắt kết nối tạm thời từ ESP32
    });
    return;
  }

  // --------------------------------------------------------------------------
  // 2. BROADCAST ROUTE: GET /stream (Phát luồng MJPEG thời gian thực)
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname === '/stream') {
    res.writeHead(200, {
      'Content-Type': 'multipart/x-mixed-replace; boundary=mjpeg_frame',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Connection': 'close',
      'Access-Control-Allow-Origin': '*'
    });

    if (res.socket) {
      res.socket.setNoDelay(true); // Gửi gói tin TCP ngay tức thì, tắt thuật toán Nagle
      res.socket.setKeepAlive(true, 1000);
    }

    streamClients.add(res);

    // Gửi ngay frame mới nhất trong RAM nếu có
    if (latestFrame) {
      try {
        res.write('--mjpeg_frame\r\n');
        res.write(`Content-Type: image/jpeg\r\nContent-Length: ${latestFrame.length}\r\n\r\n`);
        res.write(latestFrame);
        res.write('\r\n');
      } catch (e) {}
    }

    req.on('close', () => {
      streamClients.delete(res);
    });
    return;
  }

  // --------------------------------------------------------------------------
  // 3. SINGLE FRAME ROUTE: GET /api/stream/latest (Trả về 1 frame JPEG mới nhất)
  //    Dành cho AI Server chụp snapshot với độ trễ 0ms qua localhost
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname === '/api/stream/latest') {
    if (!latestFrame) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Chưa nhận được frame nào từ ESP32-CAM' }));
      return;
    }
    res.writeHead(200, {
      'Content-Type': 'image/jpeg',
      'Content-Length': latestFrame.length,
      'Cache-Control': 'no-cache, no-store',
      'Access-Control-Allow-Origin': '*'
    });
    res.end(latestFrame);
    return;
  }

  // --------------------------------------------------------------------------
  // 4. DEVICE INFO ROUTE: GET /api/device/info (Trạng thái kết nối ESP32)
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname === '/api/device/info') {
    const isLive = latestFrame !== null && (latestFrameTime > 0) && ((Date.now() - latestFrameTime) < 15000);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      is_live: isLive,
      has_frame: latestFrame !== null,
      esp32_ip: esp32DeviceIp || '192.168.137.1',
      last_frame_ago_ms: latestFrameTime > 0 ? (Date.now() - latestFrameTime) : -1,
      active_viewers: streamClients.size
    }));
    return;
  }

  // --------------------------------------------------------------------------
  // 5. WEBHOOK ROUTE: POST /api/ekyc/result (Nhận kết quả từ FastAPI AI Server)
  // --------------------------------------------------------------------------
  if (req.method === 'POST' && pathname === '/api/ekyc/result') {
    let bodyChunks = [];
    req.on('data', chunk => bodyChunks.push(chunk));
    req.on('end', () => {
      try {
        const rawBody = Buffer.concat(bodyChunks).toString('utf-8');
        const payload = JSON.parse(rawBody);

        const result = processEKYCResult(payload);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          status: 'SUCCESS',
          message: 'Node.js Server đã lưu kết quả xác thực thành công!',
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
  // 6. ROUTE API LỊCH SỬ: GET /api/ekyc/history
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname === '/api/ekyc/history') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      total: recentVerifications.length,
      history: recentVerifications
    }));
    return;
  }

  // --------------------------------------------------------------------------
  // 6.5. PROXY ROUTE: /api/v1/* -> Chuyển tiếp tới FastAPI AI Server (:8000)
  // --------------------------------------------------------------------------
  if (pathname.startsWith('/api/v1/')) {
    const aiServerHost = process.env.AI_SERVER_HOST || '127.0.0.1';
    const aiServerPort = parseInt(process.env.AI_SERVER_PORT || '8000', 10);

    const proxyOptions = {
      hostname: aiServerHost,
      port: aiServerPort,
      path: req.url,
      method: req.method,
      headers: { ...req.headers, host: `${aiServerHost}:${aiServerPort}` }
    };

    const proxyReq = http.request(proxyOptions, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (err) => {
      console.error(`${Colors.red}[PROXY ERROR]: Không thể kết nối tới FastAPI AI Server (${aiServerHost}:${aiServerPort}): ${err.message}${Colors.reset}`);
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        detail: `Không thể kết nối tới FastAPI AI Server tại ${aiServerHost}:${aiServerPort}. Hãy kiểm tra xem server AI đã bật chưa (python app.py). Lỗi: ${err.message}`
      }));
    });

    req.pipe(proxyReq);
    return;
  }

  // --------------------------------------------------------------------------
  // 7. ROUTE WEB DASHBOARD: GET / và GET /index.html (Giao diện static/index.html)
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    const htmlPath = path.join(__dirname, 'static', 'index.html');
    if (fs.existsSync(htmlPath)) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      fs.createReadStream(htmlPath).pipe(res);
      return;
    } else {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Lỗi: Không tìm thấy file static/index.html');
      return;
    }
  }

  // --------------------------------------------------------------------------
  // 7.1. PHỤC VỤ STATIC ASSETS: GET /static/*
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname.startsWith('/static/')) {
    const staticBase = path.join(__dirname, 'static');
    const cleanUrl = pathname.replace(/^\/static\//, '');
    const safePath = path.normalize(cleanUrl).replace(/^(\.\.[\/\\])+/, '');
    const targetFile = path.join(staticBase, safePath);

    if (targetFile.startsWith(staticBase) && fs.existsSync(targetFile) && fs.statSync(targetFile).isFile()) {
      const ext = path.extname(targetFile).toLowerCase();
      const mimeTypes = {
        '.html': 'text/html; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon'
      };
      res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
      fs.createReadStream(targetFile).pipe(res);
      return;
    }
  }

  // --------------------------------------------------------------------------
  // 8. PHỤC VỤ ẢNH ĐÃ LƯU: GET /images/:filename
  // --------------------------------------------------------------------------
  if (req.method === 'GET' && pathname.startsWith('/images/')) {
    const filename = path.basename(pathname);
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
  const deviceId = data.device_id || 'ESP32_S3_GATE_01';
  const approved = data.approved === true;
  const verdict = data.verdict || (approved ? 'APPROVED' : 'REJECTED');
  const isReal = data.is_real === true;
  const confidence = typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(1) : 'N/A';
  const procTime = data.processing_time_ms || 'N/A';

  const recordId = `REC_${Date.now()}`;
  let savedImagePath = null;
  let savedImageRelUrl = null;
  let savedDualPath = null;
  let savedDualRelUrl = null;

  // Lưu ảnh Dual Window Side-by-Side nếu có
  if (data.dual_window_image_base64) {
    try {
      let b64Dual = data.dual_window_image_base64;
      if (b64Dual.includes(',')) b64Dual = b64Dual.split(',')[1];
      const dualFilename = `dual_${Date.now()}_${verdict}.jpg`;
      savedDualPath = path.join(SAVE_DIR, dualFilename);
      fs.writeFileSync(savedDualPath, Buffer.from(b64Dual, 'base64'));
      savedDualRelUrl = `/images/${dualFilename}`;
    } catch (e) {
      console.error(`Không thể lưu file Dual Window: ${e.message}`);
    }
  }

  // Lưu ảnh khuôn mặt crop/captured
  if (data.crop_face_base64 || data.captured_image_base64) {
    try {
      let b64Data = data.crop_face_base64 || data.captured_image_base64;
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
    dual_window_url: savedDualRelUrl,
    processing_time_ms: procTime
  };

  recentVerifications.unshift(record);
  if (recentVerifications.length > 25) {
    recentVerifications.pop();
  }

  printTerminalDashboard(record);
  return record;
}

function printTerminalDashboard(r) {
  const isPass = r.approved;
  const statusColor = isPass ? Colors.green : Colors.red;
  const icon = isPass ? '✅' : '❌';
  const border = '═'.repeat(65);

  console.log(`\n${statusColor}${border}${Colors.reset}`);
  console.log(` ${icon} ${Colors.bright}SỰ KIỆN eKYC: [${r.verdict}]${Colors.reset}  (${r.timestamp})`);
  console.log(`${statusColor}${border}${Colors.reset}`);
  console.log(` • Thiết bị gửi:     ${Colors.cyan}${r.device_id}${Colors.reset}`);
  console.log(` • Kết luận:         ${statusColor}${r.verdict} (${isPass ? 'ĐƯỢC DUYỆT' : 'BỊ TỪ CHỐI'})${Colors.reset}`);
  console.log(` • Độ tin cậy:       ${Colors.yellow}${r.confidence}%${Colors.reset}`);
  console.log(` • Thời gian AI:     ${Colors.gray}${r.processing_time_ms} ms${Colors.reset}`);
  if (r.image_url) {
    console.log(` • Ảnh khuôn mặt:    ${Colors.blue}${r.image_url}${Colors.reset}`);
  }
  if (r.dual_window_url) {
    console.log(` • Bảng Dual-Window: ${Colors.green}${r.dual_window_url}${Colors.reset}`);
  }
  console.log(`${statusColor}${border}${Colors.reset}\n`);
}

// ============================================================================
// KHỞI ĐỘNG SERVER
// ============================================================================
server.listen(PORT, () => {
  console.log(`\n${Colors.bright}${Colors.green}===============================================================${Colors.reset}`);
  console.log(`${Colors.bright} [NODE.JS STREAM RELAY & eKYC HUB ĐÃ SẴN SÀNG]${Colors.reset}`);
  console.log(`${Colors.green}===============================================================${Colors.reset}`);
  console.log(` • Cổng Ingestion (ESP32 đẩy frame): ${Colors.cyan}POST http://127.0.0.1:${PORT}/api/stream/frame${Colors.reset}`);
  console.log(` • Cổng phát MJPEG Stream:          ${Colors.cyan}GET  http://127.0.0.1:${PORT}/stream${Colors.reset}`);
  console.log(` • Cổng lấy 1 frame mới nhất:       ${Colors.cyan}GET  http://127.0.0.1:${PORT}/api/stream/latest${Colors.reset}`);
  console.log(` • Web eKYC Dashboard:              ${Colors.yellow}http://localhost:${PORT}/${Colors.reset}`);
  console.log(` • Webhook kết quả:                 ${Colors.cyan}POST http://127.0.0.1:${PORT}/api/ekyc/result${Colors.reset}`);
  console.log(` • Thư mục lưu ảnh:                 ${Colors.blue}${SAVE_DIR}${Colors.reset}`);
  console.log(`${Colors.green}===============================================================${Colors.reset}\n`);
});
