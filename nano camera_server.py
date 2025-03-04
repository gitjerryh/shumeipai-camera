from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time
import threading
import numpy as np

app = Flask(__name__)
picam2 = None
camera_lock = threading.Lock()

def init_camera():
    global picam2
    if picam2 is None:
        picam2 = Picamera2()
        
        config = picam2.create_video_configuration(
            main={
                "size": (640, 480),
                "format": "RGB888"
            },
            controls={
                "FrameDurationLimits": (33333, 33333),  # 30fps
                "AwbEnable": True,
                "AwbMode": 1,  # 日光模式
                "ColourGains": (1.4, 1.2),  # 红绿增益
                "Brightness": 0.1,
                "Contrast": 1.0,
                "Saturation": 1.1,
                "ExposureValue": 0.2,
                "Sharpness": 1.0
            }
        )
        
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

def adjust_colors(frame):
    # 转换为浮点数类型进行处理
    frame = frame.astype(np.float32)
    
    # 分离通道
    b, g, r = cv2.split(frame)
    
    # 调整颜色通道强度
    r = r * 1.15  # 红色
    g = g * 1.0   # 绿色
    b = b * 0.75  # 蓝色
    
    # 检测高亮区域
    brightness = (r + g + b) / 3
    white_mask = (brightness > 200).astype(np.float32)
    
    # 在高亮区域调整颜色
    r = r - white_mask * 10
    g = g - white_mask * 5
    b = b + white_mask * 15
    
    # 整体提亮
    r = r + 15
    g = g + 15
    b = b + 15
    
    # 确保值在有效范围内
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    
    # 合并通道并转回uint8类型
    frame_adjusted = cv2.merge([b, g, r])
    frame_adjusted = frame_adjusted.astype(np.uint8)
    
    return frame_adjusted

def generate_frames():
    global picam2
    last_frame_time = time.time()
    
    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1/30.0:
            continue
            
        with camera_lock:
            if picam2 is None:
                init_camera()
            try:
                frame = picam2.capture_array()
                frame = adjust_colors(frame)
                
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.1)

@app.route('/')
def index():
    return """
    <h1>Camera Stream</h1>
    <img src="/video_feed" width="640" height="480" />
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_camera()
    app.run(host='0.0.0.0', port=8000, threaded=True)
