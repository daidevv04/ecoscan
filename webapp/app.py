import os
import json
import io
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

app = Flask(__name__)
# Giới hạn dung lượng upload tối đa 5MB (ảnh đã được nén phía client)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# ── Cấu hình đường dẫn model ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LABEL_PATH = os.path.join(BASE_DIR, "..", "models", "label_mapping.json")

# ── Lazy-load model cache ────────────────────────────────────────────────────
_models = {}
_labels = None
tflite = None

def get_model(model_type="mobilenet_v2"):
    global _models, _labels, tflite
    if model_type not in _models:
        # Ưu tiên sử dụng TFLite (.tflite) để tiết kiệm bộ nhớ RAM trên hosting
        tflite_path = os.path.join(BASE_DIR, "..", "models", f"{'cnn' if model_type == 'cnn' else 'mobilenetv2'}_garbage_best.tflite")
        keras_path = os.path.join(BASE_DIR, "..", "models", f"{'cnn' if model_type == 'cnn' else 'mobilenetv2'}_garbage_best.keras")
        
        if os.path.exists(tflite_path):
            if tflite is None:
                try:
                    import tflite_runtime.interpreter as tflite_mod
                    tflite = tflite_mod
                except ImportError:
                    try:
                        from tensorflow import lite as tflite_mod
                        tflite = tflite_mod
                    except ImportError:
                        raise ImportError("Không thể import tflite_runtime hoặc tensorflow.lite")
            
            # Sử dụng XNNPACK delegate đa luồng để tăng tốc dự đoán trên CPU
            num_threads = min(os.cpu_count() or 2, 4)
            interpreter = tflite.Interpreter(
                model_path=tflite_path,
                num_threads=num_threads
            )
            interpreter.allocate_tensors()
            _models[model_type] = {
                "type": "tflite",
                "interpreter": interpreter,
                "input_details": interpreter.get_input_details(),
                "output_details": interpreter.get_output_details()
            }
        else:
            # Fallback về model Keras (yêu cầu cài đặt thư viện tensorflow đầy đủ)
            if not os.path.exists(keras_path):
                raise FileNotFoundError(f"Không tìm thấy file model tại: {keras_path}")
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_path)
            _models[model_type] = {
                "type": "keras",
                "model": model
            }
        
    if _labels is None:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            _labels = json.load(f)
            
    return _models[model_type], _labels

# ── Pre-load model MobileNetV2 ngay khi khởi động để request đầu tiên nhanh ──
try:
    print("[EcoScan] Loading MobileNetV2 model...")
    get_model("mobilenet_v2")
    print("[EcoScan] Model MobileNetV2 ready!")
except Exception as e:
    print(f"[EcoScan] Warning: Could not preload model: {e}")

# ── Mapping nhãn Tiếng Việt & Phân nhóm rác thải ──────────────────────────────
VI_LABEL = {
    "battery":     {"vi": "Pin / Rác nguy hại",          "icon": "⚡", "color": "#ff4444", "cat": "NGUY HẠI"},
    "biological":  {"vi": "Rác hữu cơ dễ phân hủy",      "icon": "🌱", "color": "#4caf50", "cat": "HỮU CƠ"},
    "brown-glass": {"vi": "Thủy tinh màu nâu",           "icon": "🍶", "color": "#a0784a", "cat": "TÁI CHẾ"},
    "cardboard":   {"vi": "Bìa carton",                   "icon": "📦", "color": "#ff9800", "cat": "TÁI CHẾ"},
    "clothes":     {"vi": "Quần áo, vải vóc cũ",         "icon": "👕", "color": "#ab47bc", "cat": "TÁI SỬ DỤNG"},
    "green-glass": {"vi": "Thủy tinh màu xanh",          "icon": "🍵", "color": "#00bfa5", "cat": "TÁI CHẾ"},
    "metal":       {"vi": "Lon/Vật dụng kim loại",        "icon": "🔩", "color": "#78909c", "cat": "TÁI CHẾ"},
    "paper":       {"vi": "Giấy báo/vở cũ",               "icon": "📄", "color": "#ffeb3b", "cat": "TÁI CHẾ"},
    "plastic":     {"vi": "Hộp/Chai nhựa",                "icon": "🧴", "color": "#2196f3", "cat": "TÁI CHẾ"},
    "shoes":       {"vi": "Giày dép cũ",                  "icon": "👟", "color": "#ec407a", "cat": "TÁI SỬ DỤNG"},
    "trash":       {"vi": "Rác thải sinh hoạt chung",     "icon": "🗑️", "color": "#bdbdbd", "cat": "RÁC THƯỜNG"},
    "white-glass": {"vi": "Thủy tinh trong suốt",        "icon": "🥃", "color": "#e0f7fa", "cat": "TÁI CHẾ"},
}

ECO_TIPS = {
    "battery": "⚡ PIN LÀ RÁC NGUY HẠI! Chứa các kim loại nặng độc hại như chì, cadmium và thủy ngân. Tuyệt đối KHÔNG vứt chung với rác sinh hoạt thông thường hoặc đốt. Hãy đem gom và mang trực tiếp đến các thùng thu gom pin cũ công cộng gần nhất để xử lý an toàn.",
    "biological": "🌱 RÁC HỮU CƠ: Dễ dàng phân hủy sinh học! Bạn có thể ủ làm phân bón compost hữu cơ tại nhà để bón rau/cây cảnh rất tốt, hoặc đựng vào túi phân hủy sinh học rồi vứt vào thùng rác hữu cơ riêng biệt.",
    "brown-glass": "🍶 THỦY TINH MÀU NÂU: Tái chế vô hạn! Vui lòng tráng sạch cặn bẩn/nước ngọt bên trong chai, tháo bỏ nắp nhựa hoặc vòng đệm kim loại cổ chai trước khi vứt vào thùng phân loại chai lọ thủy tinh.",
    "cardboard": "📦 BÌA CARTON: Tái chế dễ dàng! Gấp dẹp phẳng phiu các hộp bìa carton để tiết kiệm tối đa diện tích thùng rác. Lưu ý giữ bìa luôn khô ráo, tránh thấm nước hoặc dính dầu mỡ thực phẩm.",
    "clothes": "👕 QUẦN ÁO CŨ: Hãy cố gắng kéo dài vòng đời sản phẩm! Nếu quần áo còn tốt, hãy giặt sạch và đem quyên góp cho hội từ thiện hoặc các chương trình thu đổi quần áo cũ. Không nên vứt trực tiếp ra bãi chôn lấp.",
    "green-glass": "🍵 THỦY TINH MÀU XANH: Tái chế 100%! Tráng rửa sạch qua nước lã và gỡ bỏ phần nắp chai thiếc/nhựa trước khi đưa vào khâu gom tái chế thủy tinh màu.",
    "metal": "🔩 KIM LOẠI TÁI CHẾ: Có giá trị tài nguyên rất cao! Tráng sạch lon nước ngọt, lon bia hoặc hộp thiếc đựng thực phẩm, ép bẹp chúng để giảm thể tích trước khi bỏ vào thùng gom phế liệu tái chế.",
    "paper": "📄 GIẤY THƯỜNG: Báo cũ, vở học sinh, giấy văn phòng. Hãy xếp ngăn nắp và giữ khô ráo. Không tái chế giấy đã dính chất lỏng hóa chất hoặc dính đầy dầu mỡ ăn uống.",
    "plastic": "🧴 RÁC THẢI NHỰA: Cực kỳ khó phân hủy trong tự nhiên! Tráng sạch sữa, dầu gội, nước ngọt bám trong chai nhựa. Bóp xẹp chai và vứt vào thùng rác tái chế để bảo vệ động vật hoang dã và đại dương.",
    "shoes": "👟 GIÀY DÉP CŨ: Hãy tái sử dụng hoặc quyên góp! Giày dép cũ có thể vệ sinh sạch sẽ rồi tặng cho các chương trình thu gom đồ cũ hỗ trợ trẻ em vùng cao hoặc người có hoàn cảnh khó khăn.",
    "trash": "🗑️ RÁC THẢI KHÔNG TÁI CHẾ: Khăn ướt, tã giấy, đầu lọc thuốc lá, khẩu trang y tế... Hãy buộc kín túi rác sinh hoạt chung này để công nhân vệ sinh mang đi chôn lấp hoặc thiêu hủy tập trung.",
    "white-glass": "🥃 THỦY TINH TRONG SUỐT: Tái chế vô hạn! Vui lòng súc rửa sạch cặn bẩn bám bên trong và gỡ nắp đậy nhựa/nhôm trước khi bỏ vào thùng phân loại rác tái chế thủy tinh."
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HTML / CSS / JS  ─  toàn bộ frontend nằm ở đây
# ═══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EcoScan AI · Phân Loại Rác Thải</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap"/>
<style>
/* ── Reset & Vars ─────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:       #060c07;
  --surface:  #0a180e;
  --glass:    rgba(14,35,19,.7);
  --border:   rgba(46,204,113,.2);
  --accent:   #2ecc71;
  --accent2:  #a8ff78;
  --text:     #e2f3e5;
  --muted:    #678c6b;
  --danger:   #e74c3c;
  --warn:     #f39c12;
  --r:        16px;
}

/* ── Body & Canvas ────────────────────────────────────────────── */
html,body{height:100%;overflow-x:hidden}
body{
  background:var(--bg);
  color:var(--text);
  font-family:'Syne',sans-serif;
  min-height:100vh;
  position:relative;
}

/* animated grid background */
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(rgba(46,204,113,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(46,204,113,.03) 1px, transparent 1px);
  background-size:40px 40px;
  pointer-events:none;
  z-index:0;
}
/* ambient glow blobs */
body::after{
  content:'';
  position:fixed;
  top:-20%;left:50%;
  transform:translateX(-50%);
  width:70vw;height:70vw;
  background:radial-gradient(ellipse, rgba(46,204,113,.06) 0%, transparent 70%);
  pointer-events:none;z-index:0;
  animation:breathe 8s ease-in-out infinite;
}
@keyframes breathe{0%,100%{opacity:.6;transform:translateX(-50%) scale(1)}50%{opacity:1;transform:translateX(-50%) scale(1.1)}}

/* ── Layout ───────────────────────────────────────────────────── */
.wrap{
  position:relative;z-index:1;
  max-width:900px;margin:0 auto;
  padding:48px 24px 80px;
}

/* ── Header ───────────────────────────────────────────────────── */
header{text-align:center;margin-bottom:48px;animation:fadeDown .8s ease both}
.logo{
  display:inline-flex;align-items:center;gap:10px;
  font-family:'Space Mono',monospace;font-size:.78rem;
  color:var(--accent);letter-spacing:.2em;text-transform:uppercase;
  margin-bottom:18px;
}
.logo-dot{
  width:8px;height:8px;border-radius:50%;background:var(--accent);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(46,204,113,.7)}50%{box-shadow:0 0 0 8px rgba(46,204,113,0)}}
h1{
  font-family:'Syne',sans-serif;
  font-size:clamp(2.2rem,6.5vw,3.8rem);
  font-weight:800;
  line-height:1.05;
  letter-spacing:-.03em;
  background:linear-gradient(135deg,#a8ff78,#2ecc71 45%,#00bfa5);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.sub{
  margin-top:14px;font-size:1rem;color:var(--muted);
  font-family:'Space Mono',monospace;letter-spacing:.04em;
}

/* ── Upload Zone ─────────────────────────────────────────────── */
.upload-card{
  background:var(--glass);
  border:1.5px dashed var(--border);
  border-radius:var(--r);
  backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
  padding:56px 32px;
  text-align:center;
  cursor:pointer;
  transition:border-color .3s,background .3s,transform .2s;
  animation:fadeUp .8s .15s ease both;
  position:relative;overflow:hidden;
}
.upload-card:hover,.upload-card.dragover{
  border-color:var(--accent);
  background:rgba(46,204,113,.06);
  transform:translateY(-3px);
}
.upload-card::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 0%, rgba(46,204,113,.05) 0%, transparent 70%);
  opacity:0;transition:opacity .4s;
}
.upload-card:hover::before,.upload-card.dragover::before{opacity:1}

.upload-icon{
  font-size:3.5rem;margin-bottom:16px;
  display:block;animation:float 3s ease-in-out infinite;
}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.upload-title{font-size:1.2rem;font-weight:600;margin-bottom:8px;color:var(--text)}
.upload-hint{font-size:.82rem;color:var(--muted);font-family:'Space Mono',monospace}
.upload-hint span{color:var(--accent)}
input[type=file]{
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0,0,0,0);
  border: 0;
  opacity: 0;
}

/* ── Preview & Controls ──────────────────────────────────────── */
.preview-wrap{
  display:none;
  gap:24px;
  margin-top:24px;
  animation:fadeUp .5s ease both;
}
.preview-wrap.visible{display:grid;grid-template-columns:1.2fr 1fr}
@media(max-width:768px){.preview-wrap.visible{grid-template-columns:1fr}}

.preview-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--r);
  backdrop-filter:blur(18px);
  overflow:hidden;
}
.preview-card img{
  width:100%;height:250px;object-fit:cover;display:block;
}
.preview-label{
  padding:12px 16px;
  font-size:.75rem;color:var(--muted);
  font-family:'Space Mono',monospace;letter-spacing:.06em;
}

/* Dropdown styling */
.model-select-dropdown {
  width: 100%;
  padding: 12px 14px;
  background: #060e08;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: 'Space Mono', monospace;
  font-size: 0.82rem;
  margin-top: 6px;
  margin-bottom: 12px;
  outline: none;
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  transition: border-color .25s, box-shadow .25s;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232ecc71' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  background-size: 14px;
}
.model-select-dropdown:focus {
  border-color: var(--accent);
  box-shadow: 0 0 10px rgba(46,204,113,.25);
}

/* analyse button */
.btn-analyse{
  width:100%;margin-top:8px;
  padding:16px;
  background:linear-gradient(135deg,#2ecc71,#00bfa5);
  color:#040c06;
  font-family:'Syne',sans-serif;font-weight:700;font-size:1.05rem;
  letter-spacing:.04em;
  border:none;border-radius:var(--r);cursor:pointer;
  transition:opacity .25s,transform .2s,box-shadow .25s;
  box-shadow:0 0 0 rgba(46,204,113,0);
  display:flex;align-items:center;justify-content:center;gap:8px;
}
.btn-analyse:hover:not(:disabled){
  opacity:.95;transform:translateY(-2px);
  box-shadow:0 8px 32px rgba(46,204,113,.3);
}
.btn-analyse:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn-analyse .btn-icon{font-size:1.1rem;transition:transform .4s}
.btn-analyse:hover .btn-icon{transform:rotate(20deg)}

.ctrl-panel{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--r);
  backdrop-filter:blur(18px);
  padding:24px;
  display:flex;flex-direction:column;gap:12px;justify-content:center;
}
.ctrl-info{
  font-size:.78rem;color:var(--muted);font-family:'Space Mono',monospace;
  line-height:1.6;
}
.ctrl-info b{color:var(--accent)}

/* ── Scan Overlay ────────────────────────────────────────────── */
.scan-wrap{display:none;position:relative;border-radius:var(--r);overflow:hidden}
.scan-wrap.active{display:block}
.scan-line{
  position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  box-shadow:0 0 12px var(--accent);
  animation:scan 1.6s linear infinite;
  z-index:5;
}
@keyframes scan{0%{top:0}100%{top:100%}}
.scan-corners::before,.scan-corners::after,
.scan-corners>span::before,.scan-corners>span::after{
  content:'';position:absolute;width:18px;height:18px;
  border-color:var(--accent);border-style:solid;
}
.scan-corners::before{top:8px;left:8px;border-width:2px 0 0 2px}
.scan-corners::after{top:8px;right:8px;border-width:2px 2px 0 0}
.scan-corners>span::before{bottom:8px;left:8px;border-width:0 0 2px 2px}
.scan-corners>span::after{bottom:8px;right:8px;border-width:0 2px 2px 0}

/* ── Result Card ─────────────────────────────────────────────── */
.result-wrap{
  margin-top:28px;
  display:none;
  animation:fadeUp .5s ease both;
}
.result-wrap.visible{display:block}

.result-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--r);
  backdrop-filter:blur(18px);
  overflow:hidden;
  position:relative;
}
.result-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),var(--accent2),#00bfa5);
}

.result-header{
  padding:28px 28px 0;
  display:flex;align-items:center;gap:16px;
}
.result-icon{
  font-size:2.8rem;
  animation:popIn .4s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes popIn{from{transform:scale(.3);opacity:0}to{transform:scale(1);opacity:1}}
.result-meta{}
.result-cat{
  font-size:.65rem;font-family:'Space Mono',monospace;
  letter-spacing:.14em;font-weight:700;
  padding:4px 10px;border-radius:4px;
  background:rgba(46,204,113,.15);color:var(--accent);
  display:inline-block;margin-bottom:6px;
}
.result-name{
  font-size:1.8rem;font-weight:800;letter-spacing:-.01em;
  line-height:1.1;
}

/* progress bar */
.conf-wrap{padding:20px 28px 20px}
.conf-label{
  display:flex;justify-content:space-between;
  font-size:.75rem;font-family:'Space Mono',monospace;
  color:var(--muted);margin-bottom:8px;
}
.conf-label span:last-child{color:var(--text);font-weight:700}
.conf-bar-bg{
  height:6px;border-radius:3px;
  background:rgba(46,204,113,.1);
  overflow:hidden;
}
.conf-bar{
  height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  width:0;transition:width 1.2s cubic-bezier(.22,1,.36,1);
  box-shadow:0 0 8px rgba(168,255,120,.4);
}

/* tip */
.result-tip{
  margin:0 28px 24px;
  padding:16px 20px;border-radius:10px;
  font-size:.85rem;color:var(--text);line-height:1.6;
}
.result-tip.ok{background:rgba(46,204,113,.08);border-left:4px solid var(--accent)}
.result-tip.warn{background:rgba(243,156,18,.08);border-left:4px solid var(--warn);color:var(--warn)}
.result-tip.err{background:rgba(231,76,60,.08);border-left:4px solid var(--danger);color:var(--danger)}

/* top-k table */
.topk{padding:0 28px 28px}
.topk-title{font-size:.65rem;letter-spacing:.12em;color:var(--muted);margin-bottom:10px;font-family:'Space Mono',monospace}
.topk-row{
  display:flex;align-items:center;gap:10px;
  padding:8px 0;border-bottom:1px solid rgba(46,204,113,.06);
  font-size:.82rem;
}
.topk-row:last-child{border:none}
.topk-icon{width:22px;text-align:center}
.topk-name{flex:1;color:var(--text)}
.topk-pct{font-family:'Space Mono',monospace;color:var(--muted);min-width:55px;text-align:right}
.topk-mini{flex:1;height:3px;border-radius:2px;background:rgba(46,204,113,.08)}
.topk-mini-fill{height:100%;border-radius:2px;background:var(--accent);opacity:.5;transition:width .8s ease}

/* ── Reset button ────────────────────────────────────────────── */
.btn-reset{
  display:block;width:fit-content;
  margin:18px auto 0;
  padding:9px 24px;
  background:transparent;
  color:var(--muted);
  font-family:'Space Mono',monospace;font-size:.75rem;
  border:1px solid var(--border);border-radius:50px;cursor:pointer;
  transition:color .2s,border-color .2s;
}
.btn-reset:hover{color:var(--accent);border-color:var(--accent)}

/* ── Footer ───────────────────────────────────────────────────── */
footer{
  text-align:center;
  margin-top:60px;
  font-family:'Space Mono',monospace;
  font-size:.7rem;color:var(--muted);
  letter-spacing:.1em;
  animation:fadeUp .8s .4s ease both;
}
footer a{color:var(--accent);text-decoration:none}

/* ── Animations ──────────────────────────────────────────────── */
@keyframes fadeDown{from{opacity:0;transform:translateY(-24px)}to{opacity:1;transform:none}}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}

/* ── Spinner ─────────────────────────────────────────────────── */
.spinner{
  width:20px;height:20px;
  border:2px solid rgba(0,0,0,.2);
  border-top-color:#000;
  border-radius:50%;
  animation:spin .7s linear infinite;
  display:none;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Toast ───────────────────────────────────────────────────── */
.toast{
  position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(100px);
  background:rgba(10,30,12,.96);border:1px solid var(--border);
  backdrop-filter:blur(16px);
  border-radius:50px;padding:10px 24px;
  font-size:.82rem;font-family:'Space Mono',monospace;color:var(--text);
  transition:transform .4s cubic-bezier(.34,1.56,.64,1);
  z-index:100;white-space:nowrap;
}
.toast.show{transform:translateX(-50%) translateY(0)}

/* ── Camera UI ────────────────────────────────────────────────── */
.camera-wrap {
  margin-top: 24px;
  animation: fadeUp .5s ease both;
}
.camera-card {
  position: relative;
  background: #000;
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  height: min(650px, 75vh);
  display: flex;
  flex-direction: column;
}
#cameraVideo {
  width: 100%;
  height: 100%;
  object-fit: cover;
  background: #000;
}
.camera-overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
}
.camera-guide-box {
  width: min(450px, 85vw);
  height: min(450px, 85vw);
  border: 3px dashed var(--accent);
  border-radius: 24px;
  box-shadow: 0 0 0 9999px rgba(0,0,0,0.55);
  animation: guidePulse 2.5s ease-in-out infinite;
}
@keyframes guidePulse {
  0%, 100% { border-color: var(--accent); opacity: 0.8; }
  50% { border-color: var(--accent2); opacity: 1; filter: drop-shadow(0 0 10px rgba(46,204,113,0.6)); }
}
.camera-controls {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 24px;
  background: linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.4) 70%, transparent 100%);
  display: flex;
  justify-content: space-around;
  align-items: center;
  z-index: 3;
}
.btn-camera {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: rgba(46,204,113,0.12);
  color: var(--accent);
  border: 1px solid var(--border);
  border-radius: 50px;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.25s ease;
  margin-top: 16px;
  outline: none;
}
.btn-camera:hover {
  background: var(--accent);
  color: #040c06;
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(46,204,113,0.3);
}
.btn-capture {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  border: 4px solid #fff;
  background: transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s;
  padding: 0;
  outline: none;
}
.btn-capture:active {
  transform: scale(0.9);
}
.btn-capture-inner {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  background: var(--accent);
}
.btn-close-camera {
  background: transparent;
  border: 1px solid rgba(231,76,60,0.5);
  color: var(--danger);
  padding: 8px 18px;
  border-radius: 50px;
  font-family: 'Space Mono', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
  outline: none;
}
.btn-close-camera:hover {
  background: var(--danger);
  color: #fff;
  transform: translateY(-1px);
}

/* ── Library Button ───────────────────────────────────────────── */
.btn-library {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: rgba(33,150,243,0.12);
  color: #2196f3;
  border: 1px solid rgba(33,150,243,0.2);
  border-radius: 50px;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.25s ease;
  margin-top: 16px;
  outline: none;
}
.btn-library:hover {
  background: #2196f3;
  color: #040c06;
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(33,150,243,0.3);
}

/* ── Consent Modal ────────────────────────────────────────────── */
.consent-modal {
  position: fixed;
  inset: 0;
  background: rgba(4,10,6,0.85);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
  animation: fadeIn 0.3s ease both;
}
.consent-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 32px;
  max-width: 480px;
  width: 100%;
  text-align: center;
  box-shadow: 0 10px 40px rgba(0,0,0,0.6);
  animation: scaleUp 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.consent-icon {
  font-size: 3rem;
  display: block;
  margin-bottom: 16px;
  animation: iconPulse 2s ease-in-out infinite;
}
.consent-card h3 {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  color: var(--accent);
  margin-bottom: 12px;
}
.consent-card p {
  font-size: 0.9rem;
  color: var(--text);
  line-height: 1.6;
  margin-bottom: 12px;
}
.consent-card p.consent-sub {
  font-size: 0.78rem;
  color: var(--muted);
  font-style: italic;
  margin-bottom: 24px;
}
.consent-buttons {
  display: flex;
  gap: 12px;
  justify-content: center;
}
.btn-consent-decline {
  padding: 12px 24px;
  background: rgba(231,76,60,0.12);
  color: var(--danger);
  border: 1px solid rgba(231,76,60,0.2);
  border-radius: 50px;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  flex: 1;
  transition: all 0.25s ease;
}
.btn-consent-decline:hover {
  background: var(--danger);
  color: #fff;
  transform: translateY(-2px);
}
.btn-consent-accept {
  padding: 12px 24px;
  background: var(--accent);
  color: #040c06;
  border: none;
  border-radius: 50px;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  flex: 1.5;
  transition: all 0.25s ease;
}
.btn-consent-accept:hover {
  background: var(--accent2);
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(46,204,113,0.3);
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes scaleUp {
  from { transform: scale(0.9); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}
@keyframes iconPulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); filter: drop-shadow(0 0 8px rgba(46,204,113,0.4)); }
}

/* ── Supported Wastes list ───────────────────────────────────── */
.supported-wastes-container {
  margin-top: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  transition: border-color 0.25s;
}
.supported-wastes-container:hover {
  border-color: rgba(46,204,113,0.4);
}
.btn-toggle-supported {
  width: 100%;
  padding: 14px 20px;
  background: transparent;
  border: none;
  color: var(--accent);
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  outline: none;
}
.btn-toggle-supported span.arrow-icon {
  transition: transform 0.3s ease;
  font-size: 0.9rem;
}
.btn-toggle-supported.active span.arrow-icon {
  transform: rotate(90deg);
}
.supported-list {
  background: rgba(0,0,0,0.2);
}
.supported-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  padding: 20px;
}
.supported-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: rgba(14,35,19,0.4);
  border: 1px solid rgba(46,204,113,0.08);
  border-left: 3px solid var(--item-color);
  border-radius: 8px;
  transition: all 0.2s ease;
}
.supported-item:hover {
  background: rgba(14,35,19,0.8);
  transform: translateY(-1px);
  border-color: rgba(46,204,113,0.2);
}
.item-icon {
  font-size: 1.1rem;
}
.item-name {
  font-size: 0.78rem;
  color: var(--text);
  font-weight: 600;
}
</style>
</head>
<body>

<div class="wrap">

  <!-- ── Header ─────────────────────────────────────────────── -->
  <header>
    <div class="logo"><div class="logo-dot"></div>EcoScan · AI v1.2</div>
    <h1>Phân Loại<br/>Rác Thải AI</h1>
    <p class="sub">// Deep Learning · Green Environment Solution</p>
  </header>

  <!-- ── Upload Zone ───────────────────────────────────────── -->
  <div class="upload-card" id="dropZone">
    <label for="fileInput" style="cursor:pointer;display:block">
      <span class="upload-icon">🍃</span>
      <div class="upload-title">Kéo thả ảnh rác thải vào đây</div>
      <div class="upload-hint">hoặc <span>nhấn để duyệt file</span> · JPG, PNG, WEBP</div>
    </label>
    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp"/>
  </div>

  <!-- Buttons separated from upload-card to avoid label click conflict -->
  <div style="margin-top: 16px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
    <button type="button" class="btn-camera" id="btnOpenCamera">
      <span>📷</span> Chụp ảnh từ Camera
    </button>
    <button type="button" class="btn-library" id="btnOpenLibrary">
      <span>📁</span> Chọn ảnh từ Thư viện
    </button>
    <!-- Hidden file input exclusively for Library button, avoids label[for] conflict -->
    <input type="file" id="libraryFileInput" accept="image/jpeg,image/png,image/webp"/>
  </div>

  <!-- Collapsible Supported Waste List -->
  <div class="supported-wastes-container">
    <button type="button" class="btn-toggle-supported" id="btnToggleSupported">
      <span>🔍 Xem danh mục rác nhận diện được (12 nhóm)</span> <span class="arrow-icon">▸</span>
    </button>
    <div class="supported-list" id="supportedList" style="max-height: 0px; overflow: hidden; transition: max-height 0.3s ease-out;">
      <div class="supported-grid">
        <div class="supported-item" style="--item-color: #ff4444">
          <span class="item-icon">⚡</span>
          <span class="item-name">Pin / Rác nguy hại</span>
        </div>
        <div class="supported-item" style="--item-color: #4caf50">
          <span class="item-icon">🌱</span>
          <span class="item-name">Rác hữu cơ phân hủy</span>
        </div>
        <div class="supported-item" style="--item-color: #a0784a">
          <span class="item-icon">🍶</span>
          <span class="item-name">Thủy tinh màu nâu</span>
        </div>
        <div class="supported-item" style="--item-color: #ff9800">
          <span class="item-icon">📦</span>
          <span class="item-name">Bìa carton</span>
        </div>
        <div class="supported-item" style="--item-color: #ab47bc">
          <span class="item-icon">👕</span>
          <span class="item-name">Quần áo, vải vóc cũ</span>
        </div>
        <div class="supported-item" style="--item-color: #00bfa5">
          <span class="item-icon">🍵</span>
          <span class="item-name">Thủy tinh màu xanh</span>
        </div>
        <div class="supported-item" style="--item-color: #78909c">
          <span class="item-icon">🔩</span>
          <span class="item-name">Lon/Vật dụng kim loại</span>
        </div>
        <div class="supported-item" style="--item-color: #ffeb3b">
          <span class="item-icon">📄</span>
          <span class="item-name">Giấy báo/vở cũ</span>
        </div>
        <div class="supported-item" style="--item-color: #2196f3">
          <span class="item-icon">🧴</span>
          <span class="item-name">Hộp/Chai nhựa</span>
        </div>
        <div class="supported-item" style="--item-color: #ec407a">
          <span class="item-icon">👟</span>
          <span class="item-name">Giày dép cũ</span>
        </div>
        <div class="supported-item" style="--item-color: #bdbdbd">
          <span class="item-icon">🗑️</span>
          <span class="item-name">Rác sinh hoạt chung</span>
        </div>
        <div class="supported-item" style="--item-color: #e0f7fa">
          <span class="item-icon">🥃</span>
          <span class="item-name">Thủy tinh trong suốt</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Camera Container: Only visible when camera permission is granted and camera is active -->
  <div id="cameraWrap" class="camera-wrap" style="display:none;">
    <div class="camera-card">
      <video id="cameraVideo" autoplay playsinline muted></video>
      <div class="camera-overlay">
        <div class="camera-guide-box"></div>
      </div>
      <div class="camera-controls">
        <button type="button" class="btn-capture" id="btnCapture" title="Chụp ảnh">
          <div class="btn-capture-inner"></div>
        </button>
        <button type="button" class="btn-close-camera" id="btnCloseCamera">❌ Đóng Camera</button>
      </div>
    </div>
  </div>

  <!-- ── Preview + Controls ────────────────────────────────── -->
  <div class="preview-wrap" id="previewWrap">

    <!-- image preview with scan overlay -->
    <div style="position:relative;">
      <div class="preview-card" id="previewCard">
        <img id="previewImg" src="" alt="preview"/>
        <div class="preview-label">▸ ẢNH VẬT THỂ</div>
      </div>
      <!-- scan animation shown while processing -->
      <div class="scan-wrap" id="scanOverlay" style="position:absolute;inset:0;border-radius:var(--r)">
        <div class="scan-line"></div>
        <div class="scan-corners"><span></span></div>
      </div>
    </div>

    <!-- control panel -->
    <div class="ctrl-panel">
      <div class="ctrl-info">
        <b>ĐỘ PHÂN GIẢI XỬ LÝ</b><br/>224 × 224 px<br/><br/>
        
        <b>CHỌN MÔ HÌNH PHÂN LOẠI</b><br/>
        <select id="modelSelect" class="model-select-dropdown">
          <option value="mobilenet_v2">MobileNetV2 (Transfer Learning - Tốt nhất)</option>
          <option value="cnn">CNN Baseline (Mạng tự xây dựng)</option>
        </select>
        
        <b>THƯ MỤC LƯU MODEL</b><br/>
        /models/
      </div>
      
      <button class="btn-analyse" id="btnAnalyse" disabled>
        <span class="upload-icon" style="font-size:1.1rem;animation:none;margin:0">🔍</span>
        <span id="btnText">Chờ ảnh...</span>
        <div class="spinner" id="spinner"></div>
      </button>
      <button class="btn-reset" id="btnReset" style="display:none">↺ Đổi ảnh khác</button>
    </div>

  </div>

  <!-- ── Result ─────────────────────────────────────────────── -->
  <div class="result-wrap" id="resultWrap">
    <div class="result-card">
      <div class="result-header">
        <div class="result-icon" id="resIcon">🗑️</div>
        <div class="result-meta">
          <div class="result-cat" id="resCat">PHÂN TÍCH</div>
          <div class="result-name" id="resName">—</div>
        </div>
      </div>

      <div class="conf-wrap">
        <div class="conf-label">
          <span>ĐỘ TIN CẬY DỰ ĐOÁN</span>
          <span id="confPct">0%</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar" id="confBar"></div>
        </div>
      </div>

      <div class="result-tip" id="resTip"></div>

      <div class="topk" id="topkWrap">
        <div class="topk-title">▸ TOP 3 PHÂN LỚP PHÙ HỢP NHẤT</div>
        <div id="topkRows"></div>
      </div>
    </div>
  </div>

</div><!-- /wrap -->

<footer>
  EcoScan AI v1.2 · Hỗ trợ bảo vệ môi trường và phân loại rác thải thông minh 🌍
</footer>

<div class="toast" id="toast"></div>

<!-- Privacy Consent Modal -->
<div id="consentModal" class="consent-modal" style="display:none;">
  <div class="consent-card">
    <span class="consent-icon">🛡️</span>
    <h3>Quyền Phân Tích Hình Ảnh</h3>
    <p>Để thực hiện phân tích phân loại bằng công nghệ AI, EcoScan cần tải ảnh của bạn lên máy chủ để xử lý.</p>
    <p class="consent-sub">Chúng tôi cam kết hình ảnh chỉ sử dụng để phân tích phân loại rác và tự động xóa sau khi xử lý.</p>
    <div class="consent-buttons">
      <button type="button" class="btn-consent-decline" id="btnConsentDecline">Từ chối</button>
      <button type="button" class="btn-consent-accept" id="btnConsentAccept">Tôi đồng ý & Cho phép</button>
    </div>
  </div>
</div>

<!-- ── Scripts ───────────────────────────────────────────────── -->
<script>
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const previewWrap= document.getElementById('previewWrap');
const previewImg = document.getElementById('previewImg');
const btnAnalyse = document.getElementById('btnAnalyse');
const btnText    = document.getElementById('btnText');
const spinner    = document.getElementById('spinner');
const btnReset   = document.getElementById('btnReset');
const resultWrap = document.getElementById('resultWrap');
const scanOverlay= document.getElementById('scanOverlay');
const toast      = document.getElementById('toast');
const modelSelect= document.getElementById('modelSelect');

const btnOpenLibrary    = document.getElementById('btnOpenLibrary');
const libraryFileInput = document.getElementById('libraryFileInput');
const consentModal   = document.getElementById('consentModal');
const btnConsentDecline = document.getElementById('btnConsentDecline');
const btnConsentAccept  = document.getElementById('btnConsentAccept');

let currentFile = null;
let isFromCamera = false;
let consentResolve = null;

// ── Open Photo Library ──────────────────────────────────────────
btnOpenLibrary.addEventListener('click', e => {
  e.preventDefault();
  e.stopPropagation();
  libraryFileInput.value = ''; // Reset giá trị để có thể chọn lại cùng file
  libraryFileInput.click();
});

// ── Consent Handling ────────────────────────────────────────────
function askUserConsent() {
  consentModal.style.display = 'flex';
  return new Promise((resolve) => {
    consentResolve = resolve;
  });
}

btnConsentAccept.addEventListener('click', () => {
  consentModal.style.display = 'none';
  if (consentResolve) {
    consentResolve(true);
    consentResolve = null;
  }
});

btnConsentDecline.addEventListener('click', () => {
  consentModal.style.display = 'none';
  if (consentResolve) {
    consentResolve(false);
    consentResolve = null;
  }
});

// ── Supported List Toggle ───────────────────────────────────────
const btnToggleSupported = document.getElementById('btnToggleSupported');
const supportedList      = document.getElementById('supportedList');

btnToggleSupported.addEventListener('click', () => {
  btnToggleSupported.classList.toggle('active');
  if (supportedList.style.maxHeight === '0px' || !supportedList.style.maxHeight) {
    supportedList.style.maxHeight = supportedList.scrollHeight + 'px';
  } else {
    supportedList.style.maxHeight = '0px';
  }
});

// ── Toast ───────────────────────────────────────────────────────
function showToast(msg, dur=2800){
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(()=>toast.classList.remove('show'), dur);
}

// ── Drag & Drop ─────────────────────────────────────────────────
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('dragover')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if(f) {
    isFromCamera = false;
    loadFile(f);
  }
});
fileInput.addEventListener('change',e=>{
  if(e.target.files[0]) {
    isFromCamera = false;
    loadFile(e.target.files[0]);
  }
});
libraryFileInput.addEventListener('change',e=>{
  if(e.target.files[0]) {
    isFromCamera = false;
    loadFile(e.target.files[0]);
  }
});

// ── Paste ────────────────────────────────────────────────────────
document.addEventListener('paste',e=>{
  const items = e.clipboardData?.items;
  if(!items) return;
  for(const item of items){
    if(item.type.startsWith('image/')){
      isFromCamera = false;
      loadFile(item.getAsFile());
      break;
    }
  }
});

// ── Load file ────────────────────────────────────────────────────
function loadFile(file){
  if(!file.type.match(/^image\//)){showToast('⚠ Chỉ hỗ trợ ảnh dạng JPG, PNG hoặc WEBP');return}
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewWrap.classList.add('visible');
    dropZone.style.display='none';
    resultWrap.classList.remove('visible');
    btnAnalyse.disabled = false;
    btnText.textContent  = 'Phân tích phân loại';
    btnReset.style.display='block';
    showToast('✓ Ảnh rác thải đã nạp thành công');
  };
  reader.readAsDataURL(file);
}

// ── Compress Image (Giảm dung lượng ảnh trước khi tải lên server) ────────
function compressImage(fileOrBlob, maxSize = 800, quality = 0.8) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      let w = img.width, h = img.height;
      // Thu nhỏ ảnh nếu quá lớn (giữ tỉ lệ khung hình)
      if (w > maxSize || h > maxSize) {
        if (w > h) { h = Math.round(h * maxSize / w); w = maxSize; }
        else       { w = Math.round(w * maxSize / h); h = maxSize; }
      }
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      canvas.getContext('2d').drawImage(img, 0, 0, w, h);
      canvas.toBlob(blob => {
        URL.revokeObjectURL(img.src); // Giải phóng bộ nhớ
        resolve(blob || fileOrBlob); // Fallback về file gốc nếu nén thất bại
      }, 'image/jpeg', quality);
    };
    img.onerror = () => resolve(fileOrBlob); // Fallback
    // Đọc ảnh từ file hoặc blob
    if (fileOrBlob instanceof Blob) {
      img.src = URL.createObjectURL(fileOrBlob);
    } else {
      resolve(fileOrBlob);
    }
  });
}

// ── Analyse ──────────────────────────────────────────────────────
btnAnalyse.addEventListener('click', async ()=>{
  if(!currentFile) return;

  // Yêu cầu sự cho phép sử dụng hình ảnh từ người dùng
  const consented = await askUserConsent();
  if (!consented) {
    showToast('⚠ Bạn đã từ chối cấp quyền phân tích hình ảnh.');
    return;
  }

  // UI loading
  btnAnalyse.disabled = true;
  btnText.style.display='none';
  spinner.style.display='block';
  scanOverlay.classList.add('active');
  resultWrap.classList.remove('visible');

  // Nén ảnh trước khi gửi lên server để tránh timeout
  const compressed = await compressImage(currentFile, 800, 0.8);

  const formData = new FormData();
  const filename = (currentFile && currentFile.name) ? currentFile.name : 'camera_capture.jpg';
  formData.append('image', compressed, filename);
  formData.append('model_type', modelSelect.value);

  try{
    const res  = await fetch('/predict', {method:'POST', body:formData});
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      throw new Error(`Máy chủ phản hồi lỗi nhưng không đúng định dạng JSON`);
    }

    if(!res.ok || data.error){
      showToast('❌ '+(data.error || `Lỗi máy chủ HTTP ${res.status}`));
      return;
    }

    // render result
    renderResult(data);
    resultWrap.classList.add('visible');
    resultWrap.scrollIntoView({behavior:'smooth',block:'nearest'});
  } catch(err){
    showToast('❌ '+(err.message || 'Không thể kết nối máy chủ'));
    console.error(err);
  } finally{
    btnAnalyse.disabled = false;
    btnText.style.display='';
    spinner.style.display='none';
    scanOverlay.classList.remove('active');
  }
});

// ── Render result ────────────────────────────────────────────────
function renderResult(data){
  const {label_en, label_vi, icon, cat, color, confidence, top3, eco_tip} = data;
  const pct = Math.round(confidence);
  const isGarbage = (pct >= 45);

  if (isGarbage) {
    document.getElementById('resIcon').textContent = icon;
    document.getElementById('resCat').textContent  = cat;
    document.getElementById('resCat').style.color  = color;
    document.getElementById('resCat').style.background = color+'22';
    document.getElementById('resName').textContent = label_vi;
    document.getElementById('resName').style.color = color;
  } else {
    // Không nhận diện được hoặc không phải rác thải
    document.getElementById('resIcon').textContent = "⚠️";
    document.getElementById('resCat').textContent  = "KHÔNG RÕ / KHÔNG PHẢI RÁC THẢI";
    document.getElementById('resCat').style.color  = "var(--danger)";
    document.getElementById('resCat').style.background = "rgba(231,76,60,0.12)";
    document.getElementById('resName').textContent = "Vật thể không xác định";
    document.getElementById('resName').style.color = "var(--danger)";
  }

  document.getElementById('confPct').textContent = pct + '%';

  // animate bar after short delay
  requestAnimationFrame(()=>{
    requestAnimationFrame(()=>{
      document.getElementById('confBar').style.width = pct+'%';
      document.getElementById('confBar').style.background =
        isGarbage ? (pct>=70 ? 'linear-gradient(90deg,#2ecc71,#a8ff78)' : 'linear-gradient(90deg,#f39c12,#f1c40f)')
        : 'linear-gradient(90deg,#e74c3c,#ff8a80)';
    });
  });

  // tip
  const tip = document.getElementById('resTip');
  if(isGarbage){
    tip.className='result-tip ok';
    tip.innerHTML = `<div style="font-weight:800;color:var(--accent);margin-bottom:6px;font-size:0.9rem">🌿 HƯỚNG DẪN TÁI CHẾ & XỬ LÝ:</div>${eco_tip}`;
  } else {
    tip.className='result-tip err';
    tip.innerHTML = `<div style="font-weight:800;color:var(--danger);margin-bottom:6px;font-size:0.9rem">⚠️ CẢNH BÁO KHÔNG NHẬN DIỆN ĐƯỢC:</div>Độ tin cậy dự đoán quá thấp (${pct}%). Vật thể trong ảnh có thể không phải là rác thải nằm trong danh mục phân loại tiêu chuẩn của hệ thống, hoặc ảnh chụp bị mờ/thiếu sáng. Vui lòng chụp rõ nét đối tượng rác thải cần phân tích.`;
  }

  // top-3
  const rows = document.getElementById('topkRows');
  rows.innerHTML = '';
  const maxConf = top3[0].conf;
  top3.forEach((item,i)=>{
    const row = document.createElement('div');
    row.className='topk-row';
    row.style.opacity='0';
    row.style.transform='translateX(-10px)';
    row.style.transition=`opacity .35s ${i*.1}s ease, transform .35s ${i*.1}s ease`;
    row.innerHTML=`
      <div class="topk-icon">${item.icon}</div>
      <div class="topk-name">${item.label_vi}</div>
      <div class="topk-mini"><div class="topk-mini-fill" style="width:0%" data-w="${(item.conf/maxConf*100).toFixed(1)}"></div></div>
      <div class="topk-pct">${item.conf.toFixed(1)}%</div>`;
    rows.appendChild(row);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      row.style.opacity='1'; row.style.transform='none';
      row.querySelector('.topk-mini-fill').style.width = (item.conf/maxConf*100).toFixed(1)+'%';
    }));
  });
}

// ── Camera Elements & Handlers ────────────────────────────────────
const btnOpenCamera = document.getElementById('btnOpenCamera');
const cameraWrap    = document.getElementById('cameraWrap');
const cameraVideo   = document.getElementById('cameraVideo');
const btnCapture    = document.getElementById('btnCapture');
const btnCloseCamera= document.getElementById('btnCloseCamera');
let cameraStream    = null;

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => {
      try {
        track.stop();
      } catch (e) {
        console.error("Lỗi khi dừng track camera:", e);
      }
    });
    cameraStream = null;
  }
  if (cameraVideo) {
    try {
      cameraVideo.pause();
      cameraVideo.srcObject = null;
      cameraVideo.load(); // Buộc trình duyệt giải phóng phần cứng camera (Đặc biệt trên iOS Safari)
    } catch (e) {
      console.error("Lỗi khi giải phóng thẻ video:", e);
    }
  }
  cameraWrap.style.display = 'none';
}

async function startCamera() {
  // Trước khi mở camera mới, giải phóng hoàn toàn camera cũ để tránh xung đột phần cứng
  stopCamera();
  
  // Trì hoãn 120ms để phần cứng camera (Đặc biệt trên iOS và Android) kịp giải phóng hẳn trước khi tạo luồng mới
  await new Promise(resolve => setTimeout(resolve, 120));

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('⚠ Trình duyệt chặn quyền hoặc không hỗ trợ Camera (Cần kết nối HTTPS/Localhost)');
    return;
  }
  try {
    // Yêu cầu luồng camera mới
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Ưu tiên camera sau trên điện thoại
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    
    // Gán luồng dữ liệu vào thẻ video
    cameraVideo.srcObject = cameraStream;
    
    // Cập nhật giao diện ẩn vùng kéo thả và hiện camera
    dropZone.style.display = 'none';
    previewWrap.classList.remove('visible');
    resultWrap.classList.remove('visible');
    cameraWrap.style.display = 'block';
    
    // Phát video và bắt lỗi autoplay/interrupted của trình duyệt
    cameraVideo.play().catch(err => {
      console.warn("Lỗi tự động phát video:", err);
    });
    
    showToast('✓ Đã bật Camera thành công');
  } catch (err) {
    console.error('Lỗi truy cập camera:', err);
    showToast(`⚠ Lỗi Camera: ${err.message || 'Không thể mở nguồn camera'}`);
  }
}

btnOpenCamera.addEventListener('click', e => {
  e.preventDefault();
  e.stopPropagation();
  startCamera();
});

btnCloseCamera.addEventListener('click', e => {
  e.preventDefault();
  e.stopPropagation();
  stopCamera();
  isFromCamera = false; // Đóng hẳn thì hủy chế độ camera tự động
  dropZone.style.display = '';
});

btnCapture.addEventListener('click', e => {
  e.preventDefault();
  e.stopPropagation();
  if (!cameraVideo.srcObject) return;
  
  const canvas = document.createElement('canvas');
  canvas.width = cameraVideo.videoWidth || 640;
  canvas.height = cameraVideo.videoHeight || 480;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  
  canvas.toBlob(blob => {
    if (blob) {
      isFromCamera = true; // Đánh dấu ảnh này chụp từ camera
      stopCamera();
      loadFile(blob);
    } else {
      showToast('❌ Không thể trích xuất ảnh từ camera');
    }
  }, 'image/jpeg', 0.95);
});

// Kiểm tra kết nối bảo mật khi tải trang (Camera chỉ mở khi người dùng nhấn nút)
window.addEventListener('DOMContentLoaded', () => {
  if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(window.location.hostname)) {
    showToast('⚠ Trình duyệt yêu cầu kết nối bảo mật (HTTPS) để mở Camera');
  }
  // Hiển thị vùng upload mặc định, camera chỉ mở khi nhấn nút "Chụp ảnh từ Camera"
  dropZone.style.display = '';
});

// ── Reset ────────────────────────────────────────────────────────
btnReset.addEventListener('click', e => {
  e.preventDefault();
  e.stopPropagation();
  currentFile=null;
  fileInput.value='';
  previewImg.src='';
  previewWrap.classList.remove('visible');
  resultWrap.classList.remove('visible');
  btnAnalyse.disabled=true;
  btnText.textContent='Chờ ảnh...';
  btnReset.style.display='none';
  document.getElementById('confBar').style.width='0';
  stopCamera();
  
  // Nếu ảnh cũ chụp từ camera, tự động mở lại camera để chụp phát nữa. 
  // Ngược lại nếu là ảnh tải lên, hiện lại vùng kéo thả dropZone.
  if (isFromCamera) {
    startCamera();
  } else {
    dropZone.style.display='';
  }
});
</script>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh trong request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file."}), 400

    model_type = request.form.get("model_type", "mobilenet_v2")

    try:
        # Lấy thông tin mô hình và nhãn tương ứng động
        model_info, labels_dict = get_model(model_type)

        # Đọc & tiền xử lý ảnh (Dùng PIL và Numpy, hoàn toàn độc lập với TensorFlow)
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        # Sử dụng LANCZOS (chất lượng cao nhất) để resize, giúp giữ chi tiết ảnh tốt hơn
        img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tiền xử lý theo từng loại mô hình
        if model_type == "mobilenet_v2":
            # MobileNetV2 preprocess_input: chuyển thang màu pixel về [-1, 1]
            img_array = (img_array / 127.5) - 1.0
        else:
            # CNN Baseline chạy ở range [0, 1]
            img_array = img_array / 255.0

        # Tiến hành dự đoán dựa trên định dạng mô hình đã tải
        if model_info["type"] == "tflite":
            interpreter = model_info["interpreter"]
            input_details = model_info["input_details"]
            output_details = model_info["output_details"]
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            # Fallback Keras
            model = model_info["model"]
            preds = model.predict(img_array)[0]  # shape (num_classes,)

        top_indices = np.argsort(preds)[::-1][:3]

        pred_idx   = int(top_indices[0])
        confidence = float(preds[pred_idx]) * 100
        label_en   = labels_dict.get(str(pred_idx), "trash")

        meta = VI_LABEL.get(label_en, {"vi": label_en, "icon": "❓", "color": "#aaa", "cat": "KHÔNG RÕ"})
        eco_tip = ECO_TIPS.get(label_en, "Vui lòng vứt rác đúng nơi quy định để bảo vệ hành tinh xanh của chúng ta.")

        # Top-3
        top3 = []
        for idx in top_indices:
            en  = labels_dict.get(str(int(idx)), "trash")
            m   = VI_LABEL.get(en, {"vi": en, "icon": "❓", "color": "#aaa", "cat": ""})
            top3.append({
                "label_en": en,
                "label_vi": m["vi"],
                "icon":     m["icon"],
                "color":    m["color"],
                "conf":     float(preds[int(idx)]) * 100,
            })

        return jsonify({
            "label_en":   label_en,
            "label_vi":   meta["vi"],
            "icon":       meta["icon"],
            "color":      meta["color"],
            "cat":        meta["cat"],
            "confidence": confidence,
            "top3":       top3,
            "eco_tip":    eco_tip
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
