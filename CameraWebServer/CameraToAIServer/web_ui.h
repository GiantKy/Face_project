#ifndef WEB_UI_H
#define WEB_UI_H

#include <Arduino.h>

static const char INDEX_HTML[] PROGMEM = R"rawliteral(<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32-S3 Camera AI eKYC</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; text-align: center; margin: 0; padding: 16px; }
    .card { max-width: 500px; margin: 0 auto; background: #161e2e; padding: 22px; border-radius: 20px; border: 1px solid #1e293b; box-shadow: 0 15px 35px rgba(0,0,0,0.5); }
    h1 { font-size: 20px; color: #38bdf8; margin: 0 0 4px 0; }
    p.sub { color: #94a3b8; font-size: 13px; margin: 0 0 14px 0; }
    .ip-box { background: #0f172a; padding: 10px 14px; border-radius: 10px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; font-size: 13px; border: 1px solid #334155; }
    .ip-box input { background: #1e293b; border: 1px solid #475569; color: #fff; padding: 6px 10px; border-radius: 6px; width: 140px; font-family: monospace; font-size: 13px; }
    .ip-box button { background: #3b82f6; color: #fff; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }
    
    /* Khung Ảnh Vuông vắn giữ nguyên như cũ (280x280), hiển thị ảnh chụp 640x480 sắc nét */
    .stream-container { position: relative; width: 280px; height: 280px; margin: 0 auto 16px auto; border-radius: 20px; border: 2px solid #38bdf8; overflow: hidden; background: #020617; box-shadow: 0 0 25px rgba(56, 189, 248, 0.25); transition: border-color 0.3s; }
    .stream-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .stream-badge { position: absolute; top: 10px; left: 10px; background: rgba(16, 185, 129, 0.85); color: #fff; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 10px; letter-spacing: 0.5px; z-index: 4; }
    .stream-reload-btn { position: absolute; top: 8px; right: 8px; background: rgba(15,23,42,0.75); color: #fff; border: 1px solid #334155; border-radius: 8px; cursor: pointer; padding: 4px 8px; font-size: 11px; z-index: 4; }
    .stream-reload-btn:hover { background: #1e293b; }

    /* Overlay hướng dẫn trực quan ngay trên luồng Stream */
    .stream-challenge-overlay { position: absolute; bottom: 10px; left: 12px; right: 12px; background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(4px); border: 1px solid #38bdf8; border-radius: 10px; padding: 7px 10px; display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 12px; font-weight: 700; color: #38bdf8; pointer-events: none; z-index: 5; transition: all 0.3s; }
    .stream-challenge-overlay.blink { border-color: #10b981; color: #34d399; animation: pulse 1.2s infinite; }
    .stream-challenge-overlay.turn-left { border-color: #f59e0b; color: #fbbf24; animation: slideLeft 1s infinite alternate; }
    .stream-challenge-overlay.turn-right { border-color: #f59e0b; color: #fbbf24; animation: slideRight 1s infinite alternate; }
    @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.04); opacity: 0.85; } }
    @keyframes slideLeft { 0% { transform: translateX(0); } 100% { transform: translateX(-6px); } }
    @keyframes slideRight { 0% { transform: translateX(0); } 100% { transform: translateX(6px); } }

    .step-hud { background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 12px; margin-bottom: 14px; text-align: left; }
    .step-title { font-size: 13px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .step-desc { font-size: 12px; color: #cbd5e1; line-height: 1.4; }

    .btn-main { background: linear-gradient(135deg, #0284c7, #2563eb); color: #fff; border: none; padding: 13px 18px; font-size: 15px; font-weight: 700; border-radius: 12px; cursor: pointer; width: 100%; box-shadow: 0 4px 15px rgba(37,99,235,0.4); transition: all 0.2s; margin-bottom: 8px; }
    .btn-main:hover { transform: translateY(-1px); filter: brightness(1.1); }
    .btn-main:disabled { background: #475569; cursor: not-allowed; box-shadow: none; transform: none; }
    .btn-step { background: #10b981; }

    #resultBox { margin-top: 14px; padding: 14px; border-radius: 10px; font-size: 13px; font-weight: 600; line-height: 1.5; display: none; }
    .pass { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
    .fail { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; }
    .load { background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #60a5fa; }
    .node-link { margin-top: 14px; font-size: 12px; color: #64748b; }
    .node-link a { color: #38bdf8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📷 ESP32-S3 Camera eKYC</h1>
    <p class="sub">Chụp ảnh khuôn mặt 640x480 & Thử thách Liveness đa bước</p>

    <div class="ip-box">
      <span>⚙️ Máy chủ AI:</span>
      <div>
        <input type="text" id="aiIp" value="%AI_SERVER_IP%">
        <button onclick="saveAiIp()">💾 Lưu</button>
      </div>
    </div>

    <!-- Màn hình Xem Trước (Ảnh Chụp Trực Tiếp, Không Cần Stream) -->
    <div id="streamBox" class="stream-container">
      <span class="stream-badge">PHOTO 640x480</span>
      <button class="stream-reload-btn" onclick="takeSnapshot()" title="Chụp lại ảnh xem trước">📸</button>
      <img id="camStream" src="/capture" alt="Camera Snapshot" onerror="handleImageError(this)">
      <div id="streamChallengeOverlay" class="stream-challenge-overlay">
        <span id="overlayIcon">👀</span>
        <span id="overlayText">NHÌN THẲNG VÀO CAMERA</span>
      </div>
    </div>

    <!-- Hướng dẫn từng bước -->
    <div id="stepHud" class="step-hud">
      <div class="step-title">
        <span id="stepTitle">BƯỚC 1/3: KHỞI TẠO XÁC THỰC</span>
        <span id="stepBadge" style="color:#38bdf8">1/3</span>
      </div>
      <div id="stepDesc" class="step-desc">Đứng cách camera 35-50cm, nhìn thẳng tự nhiên và bấm "Bắt Đầu".</div>
    </div>

    <button id="actionBtn" class="btn-main" onclick="onActionClick()">🚀 BẮT ĐẦU XÁC THỰC eKYC</button>
    <button id="singleBtn" style="background:#334155;color:#94a3b8;border:none;padding:8px;border-radius:8px;font-size:12px;cursor:pointer;width:100%;margin-top:4px;" onclick="triggerSingleShot()">📸 Hoặc Chụp 1 Shot Nhanh (Single Shot)</button>
    <div id="resultBox"></div>

    <div class="node-link">
      🖥️ Kết quả đẩy tự động sang Node.js Server: 
      <a href="http://localhost:3000" target="_blank">http://localhost:3000</a>
    </div>
  </div>

  <script>
    let currentSessionId = '';
    let currentStep = 'start';
    let currentPrompt = '';

    // Tải ảnh xem trước trực tiếp từ camera qua endpoint /capture
    window.addEventListener('DOMContentLoaded', () => {
      takeSnapshot();
    });

    function takeSnapshot() {
      const streamImg = document.getElementById('camStream');
      streamImg.src = '/capture?t=' + Date.now();
    }

    function handleImageError(img) {
      setTimeout(() => { takeSnapshot(); }, 2000);
    }

    function saveAiIp() {
      const ip = document.getElementById('aiIp').value.trim();
      if (!ip) return;
      fetch('/set-ai-ip?ip=' + encodeURIComponent(ip))
        .then(() => alert('Đã lưu IP Máy chủ AI: ' + ip));
    }

    function updateOverlay(step, action) {
      const overlay = document.getElementById('streamChallengeOverlay');
      const icon = document.getElementById('overlayIcon');
      const text = document.getElementById('overlayText');
      const box = document.getElementById('streamBox');

      if (step === 'start') {
        icon.innerText = '👀';
        text.innerText = 'NHÌN THẲNG VÀO CAMERA';
        overlay.className = 'stream-challenge-overlay';
        box.style.borderColor = '#38bdf8';
      } else if (step === 'eye_blink') {
        icon.innerText = '👁️';
        text.innerText = 'CHỚP MẮT HOẶC NHẮM NHẸ';
        overlay.className = 'stream-challenge-overlay blink';
        box.style.borderColor = '#10b981';
      } else if (step === 'head_movement') {
        if (action === 'TURN_LEFT') {
          icon.innerText = '⬅️';
          text.innerText = 'QUAY NHẸ SANG TRÁI (~5°-10°)';
          overlay.className = 'stream-challenge-overlay turn-left';
        } else {
          icon.innerText = '➡️';
          text.innerText = 'QUAY NHẸ SANG PHẢI (~5°-10°)';
          overlay.className = 'stream-challenge-overlay turn-right';
        }
        box.style.borderColor = '#f59e0b';
      } else if (step === 'completed') {
        icon.innerText = '🎉';
        text.innerText = 'XÁC THỰC THÀNH CÔNG (REAL)!';
        overlay.className = 'stream-challenge-overlay';
        box.style.borderColor = '#10b981';
      }
    }

    // Hàm gọi API và parse JSON an toàn tuyệt đối chống lỗi Unexpected end of JSON
    async function fetchJSON(url) {
      const res = await fetch(url);
      const text = await res.text();
      if (!text || text.trim() === '') {
        throw new Error('Máy chủ không phản hồi dữ liệu (Empty Response). Hãy kiểm tra IP máy chủ AI và đảm bảo start_ai_server.bat đang chạy!');
      }
      try {
        return JSON.parse(text);
      } catch (e) {
        throw new Error('Dữ liệu phản hồi không đúng định dạng JSON: ' + text.substring(0, 100));
      }
    }

    function renderCapturedPreview(data) {
      if (!data) return '';
      const b64 = data.captured_image_base64 || data.crop_face_base64;
      if (!b64) return '';
      const src = b64.startsWith('data:') ? b64 : ('data:image/jpeg;base64,' + b64);
      const isReal = !!data.is_real;
      const borderCol = isReal ? '#10b981' : '#ef4444';
      const label = isReal ? '✅ MẶT THẬT (REAL)' : ('❌ ' + (data.verdict || 'THẤT BẠI'));
      return '<div style="margin-top:12px;padding-top:10px;border-top:1px dashed rgba(255,255,255,0.15);">' +
               '<div style="font-size:11px;color:#94a3b8;margin-bottom:6px;font-weight:600;">📸 ẢNH VỪA CHỤP ĐƯỢC TỪ CAMERA:</div>' +
               '<img src="' + src + '" alt="Ảnh camera vừa chụp" style="width:180px;height:180px;border-radius:14px;border:2px solid ' + borderCol + ';object-fit:cover;display:block;margin:0 auto;box-shadow:0 4px 16px rgba(0,0,0,0.6);">' +
               '<div style="font-size:12px;font-weight:700;color:' + borderCol + ';margin-top:6px;">' + label + '</div>' +
             '</div>';
    }

    async function onActionClick() {
      const btn = document.getElementById('actionBtn');
      const box = document.getElementById('resultBox');
      const hudTitle = document.getElementById('stepTitle');
      const hudDesc = document.getElementById('stepDesc');
      const hudBadge = document.getElementById('stepBadge');
      const streamBox = document.getElementById('streamBox');

      btn.disabled = true;
      box.style.display = 'block';
      box.className = 'load';

      try {
        // =====================================================================
        // BƯỚC 1/3: CHỤP ẢNH TĨNH -> FACE DETECT & ENSEMBLE ANTI-SPOOFING
        // =====================================================================
        box.innerHTML = '⏳ <b>BƯỚC 1/3:</b> Đang chụp ảnh & duyệt Anti-Spoofing AI (YOLO + RF-DETR)...';
        hudTitle.innerText = 'BƯỚC 1/3: DUYỆT ANTI-SPOOFING';
        hudBadge.innerText = '1/3';
        hudDesc.innerText = 'Đang kiểm tra khuôn mặt và xác thực người thật...';

        const data = await fetchJSON('/challenge-start');

        // Hiển thị ngay bức ảnh đã chụp ở bước Face Detect lên khung hình chính
        if (data.captured_image_base64) {
          const src = data.captured_image_base64.startsWith('data:') ? data.captured_image_base64 : ('data:image/jpeg;base64,' + data.captured_image_base64);
          document.getElementById('camStream').src = src;
        }

        // Nếu giả mạo (SPOOF) hoặc không phát hiện mặt -> FAIL-FAST ngay & HIỂN THỊ ẢNH ĐÃ CHỤP
        if (!data.success || !data.passed || !data.is_real) {
          btn.disabled = false;
          box.className = 'fail';
          box.innerHTML = '❌ <b>TỪ CHỐI XÁC THỰC:</b> ' + (data.message || 'Phát hiện giả mạo (SPOOF) hoặc góc mặt không hợp lệ!') + renderCapturedPreview(data);
          streamBox.style.borderColor = '#ef4444';
          updateOverlay('start');
          hudTitle.innerText = 'BƯỚC 1/3: THẤT BẠI';
          hudBadge.innerText = '✕';
          btn.innerText = '🚀 THỬ LẠI (NHÌN THẲNG)';
          return;
        }

        // =====================================================================
        // BƯỚC 1/3 ĐÃ ĐẠT (REAL): Chuyển sang BƯỚC 2/3: THỬ THÁCH CHỚP MẮT
        // =====================================================================
        currentSessionId = data.session_id;
        const promptText = data.action_prompt || 'Hãy quay đầu';
        const challengeAction = data.challenge_action || 'TURN_LEFT';

        box.className = 'pass';
        box.innerHTML = '✅ <b>BƯỚC 1/3 ĐẠT!</b> Mặt thật REAL (' + (data.confidence * 100).toFixed(1) + '%)<br>' +
                        '👁️ <b>BƯỚC 2/3: THỬ THÁCH CHỚP MẮT (EYE BLINK)</b><br>' +
                        '<i>Hãy nhìn vào camera và chớp mắt tự nhiên 1-2 lần. AI đang đọc stream...</i>' +
                        renderCapturedPreview(data);
        hudTitle.innerText = 'BƯỚC 2/3: THỬ THÁCH CHỚP MẮT';
        hudBadge.innerText = '2/3';
        hudDesc.innerText = 'Hãy nhìn thẳng vào camera và CHỚP MẮT TỰ NHIÊN 1-2 lần. AI đang tự động theo dõi cử động mắt!';
        btn.innerText = '👁️ ĐANG THEO DÕI CHỚP MẮT (BƯỚC 2/3)...';
        btn.className = 'btn-main btn-step';
        updateOverlay('eye_blink');

        // =====================================================================
        // BƯỚC 2/3: PUSH-IMAGE LIÊN TỤC KIỂM TRA CHỚP MẮT (eye_blink)
        // =====================================================================
        let blinkPassed = false;
        let blinkRes = null;
        const blinkStartTime = Date.now();
        const STEP_TIMEOUT_MS = 25000;

        while (Date.now() - blinkStartTime < STEP_TIMEOUT_MS) {
          try {
            blinkRes = await fetchJSON('/challenge-step?session_id=' + encodeURIComponent(currentSessionId) + '&step=eye_blink');
            if (blinkRes && blinkRes.success) {
              if (blinkRes.captured_image_base64) {
                const src = blinkRes.captured_image_base64.startsWith('data:') ? blinkRes.captured_image_base64 : ('data:image/jpeg;base64,' + blinkRes.captured_image_base64);
                document.getElementById('camStream').src = src;
              }
              if (blinkRes.ear) {
                const earVal = blinkRes.ear.current !== undefined ? blinkRes.ear.current : 0;
                const stateStr = blinkRes.blink_state ? ' (ĐANG NHẮM)' : '';
                btn.innerText = '👁️ ĐANG THEO DÕI CHỚP MẮT...' + stateStr + ' (EAR: ' + earVal.toFixed(2) + ')';
              }
              if (blinkRes.passed) {
                blinkPassed = true;
                break;
              }
            }
          } catch (e) {
            console.warn('Lỗi step blink:', e);
          }
          await new Promise(r => setTimeout(r, 40));
        }

        if (!blinkPassed) {
          btn.disabled = false;
          box.className = 'fail';
          box.innerHTML = '❌ <b>BƯỚC 2/3 THẤT BẠI:</b> ' + ((blinkRes && blinkRes.message) || 'Chưa phát hiện chớp mắt hoặc hết thời gian (25s)!') + renderCapturedPreview(blinkRes);
          streamBox.style.borderColor = '#ef4444';
          btn.innerText = '🚀 THỬ LẠI TỪ ĐẦU';
          btn.className = 'btn-main';
          updateOverlay('start');
          currentSessionId = '';
          return;
        }

        // =====================================================================
        // BƯỚC 2/3 ĐÃ ĐẠT: Chuyển sang BƯỚC 3/3: THỬ THÁCH QUAY ĐẦU (head_movement)
        // =====================================================================
        const headPrompt = (blinkRes && blinkRes.head_prompt) || promptText;
        const headAction = (blinkRes && blinkRes.target_head_action) || challengeAction;
        const actionIcon = (headAction === 'TURN_LEFT') ? '⬅️' : '➡️';

        box.className = 'pass';
        box.innerHTML = '✅ <b>BƯỚC 2/3 ĐẠT!</b> Đã xác nhận chớp mắt thành công!<br>' +
                        actionIcon + ' <b>BƯỚC 3/3: ' + headPrompt + '</b><br>' +
                        '<i>Đang chụp ảnh phân tích góc quay mặt...</i>';
        hudTitle.innerText = 'BƯỚC 3/3: THỬ THÁCH QUAY ĐẦU';
        hudBadge.innerText = '3/3';
        hudDesc.innerText = headPrompt + '. Hãy giữ tư thế trong 1-2 giây để AI bắt cử động!';
        btn.innerText = actionIcon + ' ĐANG THEO DÕI QUAY ĐẦU (BƯỚC 3/3)...';
        btn.className = 'btn-main btn-step';
        updateOverlay('head_movement', headAction);

        // =====================================================================
        // BƯỚC 3/3: PUSH-IMAGE LIÊN TỤC KIỂM TRA QUAY ĐẦU (head_movement)
        // =====================================================================
        let headApproved = false;
        let headRes = null;
        const headStartTime = Date.now();

        while (Date.now() - headStartTime < STEP_TIMEOUT_MS) {
          try {
            headRes = await fetchJSON('/challenge-step?session_id=' + encodeURIComponent(currentSessionId) + '&step=head_movement');
            if (headRes && headRes.success) {
              if (headRes.captured_image_base64) {
                const src = headRes.captured_image_base64.startsWith('data:') ? headRes.captured_image_base64 : ('data:image/jpeg;base64,' + headRes.captured_image_base64);
                document.getElementById('camStream').src = src;
              }
              const prog = headRes.progress !== undefined ? Math.round(headRes.progress * 100) : 0;
              const deltaYaw = headRes.delta ? headRes.delta.yaw : 0;
              btn.innerText = actionIcon + ' ' + headPrompt + ' (' + prog + '% - ΔYaw: ' + deltaYaw + '°)';
              if (headRes.approved || (headRes.passed && headRes.step === 'completed')) {
                headApproved = true;
                break;
              }
            }
          } catch (e) {
            console.warn('Lỗi step head:', e);
          }
          await new Promise(r => setTimeout(r, 40));
        }

        btn.disabled = false;

        if (headApproved && headRes && headRes.approved) {
          if (headRes.captured_image_base64) {
            const src = headRes.captured_image_base64.startsWith('data:') ? headRes.captured_image_base64 : ('data:image/jpeg;base64,' + headRes.captured_image_base64);
            document.getElementById('camStream').src = src;
          }
          box.className = 'pass';
          box.innerHTML = '🎉 <b>XÁC THỰC TOÀN DIỆN THÀNH CÔNG (REAL)!</b><br>' +
                          'Đã vượt qua toàn bộ 3 bước: Anti-Spoof + Chớp Mắt + Quay Đầu!<br>' +
                          '<i>🔓 Cửa đã mở & dữ liệu đã chuyển tới Node.js!</i>' +
                          renderCapturedPreview(headRes);
          hudTitle.innerText = 'HOÀN TẤT 3/3 BƯỚC (REAL)';
          hudBadge.innerText = '3/3';
          hudDesc.innerText = 'Người thật (REAL) - Độ tin cậy: ' + ((headRes.confidence || 1.0) * 100).toFixed(1) + '%';
          btn.innerText = '🔄 BẮT ĐẦU PHIÊN MỚI';
          btn.className = 'btn-main';
          updateOverlay('completed');
          currentSessionId = '';
        } else {
          box.className = 'fail';
          box.innerHTML = '❌ <b>BƯỚC 3/3 THẤT BẠI:</b> ' + ((headRes && headRes.message) || 'Góc quay đầu chưa đạt yêu cầu hoặc hết thời gian (25s)!') + renderCapturedPreview(headRes);
          streamBox.style.borderColor = '#ef4444';
          btn.innerText = '🚀 THỬ LẠI TỪ ĐẦU';
          btn.className = 'btn-main';
          updateOverlay('start');
          currentSessionId = '';
        }
      } catch (err) {
        btn.disabled = false;
        box.className = 'fail';
        box.innerHTML = '❌ <b>Lỗi kết nối:</b> ' + err.message;
        streamBox.style.borderColor = '#ef4444';
        btn.innerText = '🚀 THỬ LẠI';
      }
    }

    async function triggerSingleShot() {
      const box = document.getElementById('resultBox');
      box.style.display = 'block';
      box.className = 'load';
      box.innerHTML = '⏳ Đang chụp 1 shot và gửi sang AI Server...';

      try {
        const data = await fetchJSON('/send-to-ai');
        if (data.captured_image_base64) {
          const src = data.captured_image_base64.startsWith('data:') ? data.captured_image_base64 : ('data:image/jpeg;base64,' + data.captured_image_base64);
          document.getElementById('camStream').src = src;
        }
        if (data.approved) {
          box.className = 'pass';
          box.innerHTML = '✅ <b>XÁC THỰC THÀNH CÔNG (REAL)</b> - ' + (data.confidence * 100).toFixed(1) + '%' + renderCapturedPreview(data);
        } else {
          box.className = 'fail';
          box.innerHTML = '❌ <b>TỪ CHỐI</b>: ' + (data.message || data.verdict) + renderCapturedPreview(data);
        }
      } catch (e) {
        box.className = 'fail';
        box.innerHTML = '❌ <b>Lỗi kết nối:</b> ' + e.message;
      }
    }
  </script>
</body>
</html>)rawliteral";

#endif // WEB_UI_H
