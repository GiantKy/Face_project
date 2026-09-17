/**
 * Node.js Client Example for E-KYC FastAPI AI Server.
 * 
 * Minh họa cách Backend Node.js (Express, NestJS, Fastify, v.v.)
 * gửi ảnh sang FastAPI AI Server để quét và xác thực khuôn mặt,
 * sau đó tiếp nhận kết quả để lưu Database và hiển thị cho người dùng.
 * 
 * HỖ TRỢ NODE.JS 18+ (Dùng Native fetch & FormData, không cần cài thêm thư viện)
 * Hoặc có thể dùng với Axios / Form-Data.
 */

const fs = require('fs');
const path = require('path');

// Cấu hình địa chỉ AI Server (FastAPI)
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://127.0.0.1:8000';

/**
 * Cách 1: Gửi file ảnh qua Multipart Form-Data (Khuyến nghị khi nhận file từ Multer)
 * @param {string} filePath - Đường dẫn file ảnh trên ổ cứng
 * @param {object} options - Các thông số liveness tùy chọn
 */
async function verifyFaceViaMultipart(filePath, options = {}) {
  console.log(`\n[Node.js Client] Đang gửi ảnh (Multipart): ${filePath} tới ${AI_SERVER_URL}/api/v1/verify`);

  if (!fs.existsSync(filePath)) {
    throw new Error(`File không tồn tại: ${filePath}`);
  }

  // Đọc file thành Buffer và tạo Blob trong Node.js 18+
  const fileBuffer = fs.readFileSync(filePath);
  const fileBlob = new Blob([fileBuffer], { type: 'image/jpeg' });

  const formData = new FormData();
  formData.append('file', fileBlob, path.basename(filePath));
  formData.append('img_id', options.imgId || `TRAN_${Date.now()}`);
  formData.append('user_id', options.userId || 'USER_12345');
  formData.append('blink_passed', options.blinkPassed !== undefined ? String(options.blinkPassed) : 'true');
  formData.append('head_passed', options.headPassed !== undefined ? String(options.headPassed) : 'true');
  formData.append('head_action', options.headAction || 'TURN_LEFT');
  formData.append('return_crop_image', 'true');      // Lấy ảnh crop 224x224 để lưu DB
  formData.append('return_annotated_image', 'true'); // Lấy ảnh HUD để hiển thị nếu cần

  const tStart = Date.now();
  const response = await fetch(`${AI_SERVER_URL}/api/v1/verify`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`AI Server trả về lỗi HTTP ${response.status}: ${errorText}`);
  }

  const result = await response.json();
  const tElapsed = Date.now() - tStart;
  console.log(`[Node.js Client] Hoàn tất nhận phản hồi sau ${tElapsed} ms!`);

  return result;
}

/**
 * Cách 2: Gửi ảnh dạng chuỗi Base64 qua JSON
 * @param {string} imageBase64 - Chuỗi Base64 của ảnh (data:image/jpeg;base64,...)
 * @param {object} options - Các thông số liveness tùy chọn
 */
async function verifyFaceViaBase64Json(imageBase64, options = {}) {
  console.log(`\n[Node.js Client] Đang gửi ảnh (JSON Base64) tới ${AI_SERVER_URL}/api/v1/verify`);

  const payload = {
    image_base64: imageBase64,
    img_id: options.imgId || `TRAN_${Date.now()}`,
    user_id: options.userId || 'USER_12345',
    blink_passed: options.blinkPassed !== undefined ? options.blinkPassed : true,
    head_passed: options.headPassed !== undefined ? options.headPassed : true,
    head_action: options.headAction || 'TURN_LEFT',
    return_crop_image: true,
    return_annotated_image: true
  };

  const response = await fetch(`${AI_SERVER_URL}/api/v1/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`AI Server trả về lỗi HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Kiểm tra góc mặt và khoảng cách (Pre-capture check)
 * Thường dùng khi client stream frame lên để Node.js hỏi AI xem mặt đã thẳng chưa
 */
async function validateFacePose(filePath) {
  const fileBuffer = fs.readFileSync(filePath);
  const fileBlob = new Blob([fileBuffer], { type: 'image/jpeg' });

  const formData = new FormData();
  formData.append('file', fileBlob, path.basename(filePath));

  const response = await fetch(`${AI_SERVER_URL}/api/v1/validate-pose`, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

/**
 * Xử lý kết quả trả về từ AI Server và lưu trữ vào hệ thống
 */
function handleAIResult(result) {
  console.log('====================================================');
  console.log('            KẾT QUẢ XÁC THỰC TỪ AI SERVER           ');
  console.log('====================================================');
  console.log(`- Trạng thái chung: ${result.approved ? '✅ HỢP LỆ (APPROVED)' : '❌ TỪ CHỐI (REJECTED)'}`);
  console.log(`- Phân loại: ${result.is_real ? 'NGƯỜI THẬT (REAL)' : 'GIẢ MẠO (SPOOF)'}`);
  console.log(`- Độ tin cậy Anti-Spoof: ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`- Thời gian xử lý AI: ${result.processing_time_ms} ms`);
  console.log(`- Số khuôn mặt trong ảnh: ${result.face_detection.num_faces}`);
  console.log(`- Tư thế 3D Pose: Yaw=${result.pose_3d.yaw}°, Pitch=${result.pose_3d.pitch}° -> ${result.pose_3d.is_valid ? 'ĐẠT' : 'KHÔNG ĐẠT'}`);

  if (!result.approved) {
    console.log('\n[!] LÝ DO BỊ TỪ CHỐI:');
    result.reasons.forEach((reason, idx) => {
      console.log(`   ${idx + 1}. ${reason}`);
    });
  }

  // 1. Lưu ảnh crop khuôn mặt 224x224 (Dùng để lưu Database hoặc so khớp Face Match)
  if (result.crop_face_base64) {
    const base64Data = result.crop_face_base64.replace(/^data:image\/\w+;base64,/, '');
    const cropBuffer = Buffer.from(base64Data, 'base64');
    const saveCropPath = path.join(__dirname, `crop_${result.image_id}.jpg`);
    fs.writeFileSync(saveCropPath, cropBuffer);
    console.log(`\n[DB/Storage] Đã lưu ảnh khuôn mặt crop 224x224 vào: ${saveCropPath}`);
  }

  // 2. Lưu kết quả audit log (JSON)
  console.log('\n[Tiêu chuẩn eKYC 7 Bước]:');
  console.table(result.criteria);

  return {
    isSuccess: result.approved,
    verdict: result.verdict,
    confidence: result.confidence,
    reasons: result.reasons
  };
}

// =============================================================================
// DEMO CHẠY THỬ NGHIỆM
// =============================================================================
async function runDemo() {
  try {
    // Tìm ảnh test có sẵn trong data_raw/
    const sampleImagePath = path.join(__dirname, '..', 'data_raw', '0.jpg');

    if (fs.existsSync(sampleImagePath)) {
      // 1. Chạy xác thực bằng Multipart
      console.log('--- TEST 1: GỬI MULTIPART FORM-DATA ---');
      const resMultipart = await verifyFaceViaMultipart(sampleImagePath, {
        imgId: 'TEST_NODEJS_01',
        blinkPassed: true,
        headPassed: true
      });
      handleAIResult(resMultipart);

      // 2. Chạy kiểm tra Pose
      console.log('\n--- TEST 2: KIỂM TRA TƯ THẾ (PRE-CAPTURE) ---');
      const resPose = await validateFacePose(sampleImagePath);
      console.log('Kết quả kiểm tra tư thế:', resPose);
    } else {
      console.log(`[Lưu ý] Không tìm thấy ảnh mẫu ${sampleImagePath}. Vui lòng truyền đường dẫn ảnh thực tế để chạy thử.`);
    }
  } catch (err) {
    console.error('[Node.js Client Error]:', err.message);
  }
}

// Chạy trực tiếp nếu gọi node nodejs_client_example.js
if (require.main === module) {
  runDemo();
}

module.exports = {
  verifyFaceViaMultipart,
  verifyFaceViaBase64Json,
  validateFacePose,
  handleAIResult
};
