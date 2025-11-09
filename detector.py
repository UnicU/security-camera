import os
import io
import time
import threading
import logging
import signal
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


try:
    import pyaudio
except Exception:
    pyaudio = None

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("motion_recorder.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


TELEGRAM_TOKEN = "8280495216:AAE55SeWmR9Hi1-rzxjOhbBo18VyJul6Ff0"
TELEGRAM_CHAT_ID = "2132792365"


PRE_POST_MIN_SECONDS = 60


class TelegramNotifier:

    def __init__(self, token: str, chat_id: str, cooldown_seconds: int = 30):
        self.token = token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_time = None

    def _can_send(self) -> bool:
        if self._last_time is None:
            return True
        return (datetime.now() - self._last_time).total_seconds() >= self.cooldown_seconds

    def send_message(self, text: str) -> bool:
        if not self._can_send():
            logger.debug("Telegram: cooldown, пропускаю сообщение")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=10)
            if r.status_code == 200:
                self._last_time = datetime.now()
                return True
            logger.error("Telegram send_message error: %s %s", r.status_code, r.text)
        except Exception as e:
            logger.exception("Telegram send_message exception")
        return False

    def send_photo_bytes(self, jpg_bytes: bytes, caption: str = "") -> bool:
        if not self._can_send():
            logger.debug("Telegram: cooldown, пропускаю фото")
            return False
        if not jpg_bytes:
            logger.error("Telegram: пустые байты фото")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            files = {"photo": ("motion.jpg", io.BytesIO(jpg_bytes), "image/jpeg")}
            data = {"chat_id": self.chat_id, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=20)
            if r.status_code == 200:
                self._last_time = datetime.now()
                return True
            logger.error("Telegram send_photo error: %s %s", r.status_code, r.text)
        except Exception:
            logger.exception("Telegram send_photo exception")
        return False

    def send_video_file(self, path: str, caption: str = "") -> bool:
        if not self._can_send():
            logger.debug("Telegram: cooldown, пропускаю видео")
            return False
        if not os.path.exists(path):
            logger.error("Telegram: видео не найдено %s", path)
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendVideo"
            with open(path, "rb") as f:
                files = {"video": (os.path.basename(path), f, "video/mp4")}
                data = {"chat_id": self.chat_id, "caption": caption}
                r = requests.post(url, data=data, files=files, timeout=60)
            if r.status_code == 200:
                self._last_time = datetime.now()
                return True
            logger.error("Telegram send_video error: %s %s", r.status_code, r.text)
        except Exception:
            logger.exception("Telegram send_video exception")
        return False


class MotionDetector:

    def __init__(self, settings: dict):
        self.settings = settings
        self.bg = None  
        self.motion_history = deque(maxlen=20)
        self.light_history = deque(maxlen=6)

    def update_background(self, frame_gray: np.ndarray):
        if self.bg is None:
            self.bg = frame_gray.astype(np.float32)
            return
        lr = float(self.settings.get("bg_alpha", 0.01))
        if any(self.motion_history):
            lr *= 0.05
        cv2.accumulateWeighted(frame_gray, self.bg, lr)

    def _is_light_burst(self, diff: np.ndarray, thresh_mask: np.ndarray) -> bool:
        mean_diff = float(np.mean(diff))
        self.light_history.append(mean_diff)
        avg = float(np.mean(self.light_history)) if self.light_history else mean_diff
        ratio = (np.count_nonzero(thresh_mask) / (diff.shape[0] * diff.shape[1])) if diff.size else 0
        if avg > self.settings.get("light_change_threshold", 25.0) and ratio > self.settings.get(
            "light_change_ratio_threshold", 0.4
        ):
            logger.debug("Засвет: avg=%.1f ratio=%.2f", avg, ratio)
            return True
        return False

    def detect(self, frame_gray: np.ndarray):
        if self.bg is None:
            return False, []

        bg_uint8 = cv2.convertScaleAbs(self.bg)
        diff = cv2.absdiff(frame_gray, bg_uint8)
        _, thresh = cv2.threshold(diff, int(self.settings.get("motion_threshold", 25)), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        if self._is_light_burst(diff, thresh):
            self.motion_history.append(False)
            return False, []

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        min_area = int(self.settings.get("min_area", 5000))
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(1.0, h)
            if 0.2 < ar < 5.0:
                rects.append((x, y, w, h))

        motion = len(rects) > 0
        self.motion_history.append(bool(motion))

        
        min_frames = int(self.settings.get("min_motion_frames", 3))
        if sum(list(self.motion_history)[-min_frames:]) < max(1, min_frames // 2):
            motion = False

        return motion, rects


class MotionRecorder:

    def __init__(self, preview_callback, settings):
        self.preview_callback = preview_callback
        self.settings = settings.copy()
        self.cap = None
        self.frame_queue = None 
        self.fps = float(self.settings.get("fallback_fps", 20))
        self.frame_size = None
        self.writer = None
        self.writer_lock = threading.Lock()
        self.recording = False
        self.current_video_path = None
        self._stop_event = threading.Event()
        self.thread = None
        self._watchdog = None
        self._watchdog_interval = 2.0

   
        self.audio_enabled = pyaudio is not None and self.settings.get("enable_audio", True)
        self.p = None
        self.audio_stream = None
        self.audio_queue = None

    
        self.detector = MotionDetector(self.settings)
        self.telegram = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

    def open_camera(self, index: int):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_FFMPEG, None]
        last_err = None
        for b in backends:
            try:
                cap = cv2.VideoCapture(index) if b is None else cv2.VideoCapture(index, b)
                time.sleep(0.05)
                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    continue
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1]), int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0]
                )
                fps = cap.get(cv2.CAP_PROP_FPS) or self.settings.get("fallback_fps", 20)
                self.cap = cap
                self.frame_size = (w, h)
                self.fps = float(fps) if fps > 0 else float(self.settings.get("fallback_fps", 20))
                maxlen = max(1, int(self.settings.get("prebuffer_seconds", PRE_POST_MIN_SECONDS) * self.fps * 1.2))
                self.frame_queue = deque(maxlen=maxlen)
                logger.info("Камера открыта idx=%s backend=%s fps=%.1f size=%s queue=%d", index, b, self.fps, self.frame_size, maxlen)
                return
            except Exception as e:
                last_err = e
                try:
                    cap.release()
                except Exception:
                    pass
                continue
        raise RuntimeError(f"Не удалось открыть камеру {index}: {last_err}")


    def start_audio(self):
        if not self.audio_enabled:
            return False
        try:
            self.p = pyaudio.PyAudio()
            chunk = int(self.settings.get("audio_chunk", 1024))
            rate = int(self.settings.get("audio_rate", 44100))
            channels = int(self.settings.get("audio_channels", 1))
            self.audio_stream = self.p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
            maxlen = int(self.settings.get("prebuffer_seconds", PRE_POST_MIN_SECONDS) * rate / chunk * 1.2)
            self.audio_queue = deque(maxlen=maxlen)
            logger.info("Аудио инициализировано")
            return True
        except Exception:
            logger.exception("Не удалось инициализировать аудио")
            self.audio_stream = None
            self.p = None
            return False

    def stop_audio(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.p:
                self.p.terminate()
        except Exception:
            logger.exception("Ошибка при остановке аудио")
        self.audio_stream = None
        self.p = None

    def read_audio_once(self):
        if not self.audio_stream:
            return False, 0.0
        try:
            chunk = int(self.settings.get("audio_chunk", 1024))
            data = self.audio_stream.read(chunk, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16)
            if arr.size == 0:
                return False, 0.0
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            self.audio_queue.append((datetime.now(), data, rms))
            return rms > float(self.settings.get("audio_threshold", 500)), rms
        except Exception:
            logger.exception("Ошибка чтения аудио")
            return False, 0.0

  
    def _start_writer(self):
        os.makedirs(self.settings.get("output_dir", "recordings"), exist_ok=True)
        ts = datetime.now().strftime("motion_%Y%m%d_%H%M%S_video.mp4")
        path = os.path.join(self.settings.get("output_dir", "recordings"), ts)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(self.fps), self.frame_size)
        if not writer.isOpened():
            logger.error("Не открылся VideoWriter")
            return False, None
        with self.writer_lock:
            self.writer = writer
            self.recording = True
            self.current_video_path = path
            cutoff = datetime.now() - timedelta(seconds=int(self.settings.get("prebuffer_seconds", PRE_POST_MIN_SECONDS)))
            if self.frame_queue:
                for ts_item, frame_item, rects in list(self.frame_queue):
                    if ts_item >= cutoff:
                        frame_to_write = frame_item.copy()
                        self._draw_rects(frame_to_write, rects)
                        try:
                            self.writer.write(frame_to_write)
                        except Exception:
                            logger.exception("Ошибка записи предзаписи")
        logger.info("Start recording -> %s", path)
        return True, path

    def _stop_writer(self):
        with self.writer_lock:
            path = self.current_video_path
            if self.writer:
                try:
                    self.writer.release()
                except Exception:
                    logger.exception("Ошибка при закрытии writer")
                self.writer = None
            self.recording = False
            self.current_video_path = None
        logger.info("Stopped recording %s", path)
        return path

    def _draw_rects(self, frame, rects):
        if not rects:
            return
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def start(self):
        if self.cap is None or not getattr(self.cap, "isOpened", lambda: False)():
            raise RuntimeError("Камера не открыта")
        if self.thread and self.thread.is_alive():
            return
        self._stop_event.clear()
        if self.audio_enabled:
            if not self.start_audio():
                logger.warning("Аудио не инициализировано, продолжаем без него")
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        if self._watchdog is None or not self._watchdog.is_alive():
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog.start()

    def stop(self):
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

        try:
            self._stop_writer()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.stop_audio()

    def _watchdog_loop(self):
        while not self._stop_event.is_set():
            if self.thread and not self.thread.is_alive():
                logger.warning("Capture thread died, перезапуск")
                try:
                    self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                    self.thread.start()
                except Exception:
                    logger.exception("Не удалось перезапустить capture thread")
            time.sleep(self._watchdog_interval)

    def _send_notifications_async(self, frame_bgr, video_path=None):

        def job():
            try:
                success, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not success:
                    logger.error("JPEG encode failed for notification")
                    return
                jpg_bytes = buf.tobytes()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                text = f"Движение: {ts}"
                self.telegram.send_message(text)
                self.telegram.send_photo_bytes(jpg_bytes, caption=f"Движение {ts}")
                if video_path and os.path.exists(video_path):
                    self.telegram.send_video_file(video_path, caption=f"Видео {ts}")
            except Exception:
                logger.exception("Ошибка в отправке уведомления")
        threading.Thread(target=job, daemon=True).start()

    def _capture_loop(self):
        logger.info("Capture loop started")
        last_motion_time = None
        episode_notification_time = None
        min_notification_interval = 5.0
        frame_interval = 1.0 / max(1.0, self.fps)
        post_seconds = int(self.settings.get("post_motion_seconds", PRE_POST_MIN_SECONDS))

        while not self._stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                ts = datetime.now()
                if self.frame_size is None:
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)

               
                audio_motion = False
                if self.audio_enabled and self.audio_stream:
                    audio_motion, _ = self.read_audio_once()

                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_k = tuple(self.settings.get("blur_kernel", (11, 11)))
                gray_blurred = cv2.GaussianBlur(gray, blur_k, 0)

                
                self.detector.update_background(gray_blurred)
                motion, rects = self.detector.detect(gray_blurred)

                # буферизация
                try:
                    self.frame_queue.append((ts, frame.copy(), list(rects)))
                except Exception:
                    logger.exception("frame_queue.append failed")

                # логика записи
                if motion or audio_motion:
                    last_motion_time = ts
                    if not self.recording:
                        ok, path = self._start_writer()
                        if not ok:
                            logger.error("Не удалось стартовать запись")
                    if self.recording:
                        
                        with self.writer_lock:
                            if self.writer:
                                try:
                                    fw = frame.copy()
                                    self._draw_rects(fw, rects)
                                    self.writer.write(fw)
                                except Exception:
                                    logger.exception("Ошибка записи кадра")
                    # уведомление при старте эпизода 
                    if episode_notification_time is None or (ts - episode_notification_time).total_seconds() >= min_notification_interval:
                        episode_notification_time = ts
                        preview = frame.copy()
                        self._draw_rects(preview, rects)
                        self._send_notifications_async(preview, None)
                else:
                    
                    if self.recording and last_motion_time:
                        elapsed = (datetime.now() - last_motion_time).total_seconds()
                        if elapsed >= post_seconds:
                            video_path = self._stop_writer()
                            if video_path and os.path.exists(video_path):
                                
                                last_frame = None
                                if self.frame_queue:
                                    try:
                                        last_frame = self.frame_queue[-1][1].copy()
                                        last_rects = self.frame_queue[-1][2] if len(self.frame_queue[-1]) > 2 else []
                                        self._draw_rects(last_frame, last_rects)
                                    except Exception:
                                        last_frame = None
                                if last_frame is not None:
                                    self._send_notifications_async(last_frame, video_path)
                                else:
                                    
                                    self._send_notifications_async(frame.copy(), video_path)
                            episode_notification_time = None
                            last_motion_time = None

                
                try:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.preview_callback(pil, rects)
                except Exception:
                    logger.exception("Preview callback failed")

                time.sleep(frame_interval)
            except Exception:
                logger.exception("Unhandled exception in capture loop; подождём и продолжим")
                time.sleep(0.5)

        logger.info("Capture loop stopping, finalizing")
        try:
            self._stop_writer()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.stop_audio()
        logger.info("Capture loop exited")


class App:

    def __init__(self, root):
        self.root = root
        root.title("Motion Recorder")
        self.settings = {
            "prebuffer_seconds": PRE_POST_MIN_SECONDS,
            "post_motion_seconds": PRE_POST_MIN_SECONDS,
            "bg_alpha": 0.01,
            "motion_threshold": 25,
            "min_area": 5000,
            "min_motion_frames": 5,
            "output_dir": os.path.join(os.getcwd(), "recordings"),
            "fallback_fps": 30,
            "blur_kernel": (25, 25),
            "audio_channels": 1,
            "audio_rate": 44100,
            "audio_chunk": 1024,
            "audio_threshold": 500,
            "enable_audio": True,
        }
        self.available_cams = []
        self._scan_cameras()
        self.recorder = MotionRecorder(self.update_preview, self.settings)
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        signal.signal(signal.SIGINT, self._signal_handler)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _scan_cameras(self):
        self.available_cams = []
        for i in range(6):
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.available_cams.append(i)
            except Exception:
                pass
            finally:
                try:
                    if cap and cap.isOpened():
                        cap.release()
                except Exception:
                    pass

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(sticky="nsew")
        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Камера:").grid(row=0, column=0, sticky="w")
        self.cam_var = tk.StringVar()
        cam_choices = [str(i) for i in self.available_cams] or ["0"]
        self.cam_combo = ttk.Combobox(top, values=cam_choices, textvariable=self.cam_var, width=6)
        self.cam_combo.set(cam_choices[0])
        self.cam_combo.grid(row=0, column=1)
        ttk.Button(top, text="Обновить", command=self._refresh).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Открыть", command=self._open_cam).grid(row=0, column=3, padx=4)
        self.btn_start = ttk.Button(top, text="Старт", command=self._start)
        self.btn_start.grid(row=0, column=4, padx=4)
        self.btn_stop = ttk.Button(top, text="Стоп", command=self._stop)
        self.btn_stop.grid(row=0, column=5, padx=4)

        self.preview_w = 640
        self.preview_h = 480
        self.preview_canvas = tk.Canvas(frm, width=self.preview_w, height=self.preview_h, bg="black")
        self.preview_canvas.grid(row=1, column=0, pady=8)
        self.preview_photo = None
        sfrm = ttk.LabelFrame(frm, text="Настройки", padding=8)
        sfrm.grid(row=2, column=0, sticky="ew", pady=6)

        self.pre_var = tk.IntVar(value=self.settings["prebuffer_seconds"])
        ttk.Label(sfrm, text="Предзапись (сек, мин 60):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(sfrm, from_=PRE_POST_MIN_SECONDS, to=600, textvariable=self.pre_var, width=6).grid(row=0, column=1)

        self.post_var = tk.IntVar(value=self.settings["post_motion_seconds"])
        ttk.Label(sfrm, text="Поддержка (сек, мин 60):").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(sfrm, from_=PRE_POST_MIN_SECONDS, to=600, textvariable=self.post_var, width=6).grid(row=1, column=1)

        self.minarea_var = tk.IntVar(value=self.settings["min_area"])
        ttk.Label(sfrm, text="Мин площадь:").grid(row=2, column=0, sticky="w")
        ttk.Entry(sfrm, textvariable=self.minarea_var, width=8).grid(row=2, column=1)

        self.threshold_var = tk.IntVar(value=self.settings["motion_threshold"])
        ttk.Label(sfrm, text="Порог (0-255):").grid(row=3, column=0, sticky="w")
        ttk.Entry(sfrm, textvariable=self.threshold_var, width=8).grid(row=3, column=1)

        self.audio_threshold_var = tk.IntVar(value=self.settings["audio_threshold"])
        ttk.Label(sfrm, text="Порог аудио:").grid(row=4, column=0, sticky="w")
        ttk.Entry(sfrm, textvariable=self.audio_threshold_var, width=8).grid(row=4, column=1)

        ttk.Label(sfrm, text="Папка записей:").grid(row=5, column=0, sticky="w")
        self.outdir_var = tk.StringVar(value=self.settings["output_dir"])
        ttk.Entry(sfrm, textvariable=self.outdir_var, width=40).grid(row=5, column=1, columnspan=3, sticky="w")
        ttk.Button(sfrm, text="Выбрать", command=self._choose_out).grid(row=5, column=4, padx=4)

        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(frm, textvariable=self.status_var).grid(row=3, column=0, sticky="w", pady=6)

    def _refresh(self):
        self._scan_cameras()
        cam_choices = [str(i) for i in self.available_cams] or ["0"]
        self.cam_combo["values"] = cam_choices
        self.cam_combo.set(cam_choices[0])
        messagebox.showinfo("Камеры", f"Найдено: {len(self.available_cams)}")

    def _open_cam(self):
        idx = int(self.cam_var.get())
        try:
            self.recorder.open_camera(idx)
            self.status_var.set(f"Камера {idx} открыта. FPS~{self.recorder.fps:.1f} размер {self.recorder.frame_size}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _apply_settings(self):
        pre = max(int(self.pre_var.get()), PRE_POST_MIN_SECONDS)
        post = max(int(self.post_var.get()), PRE_POST_MIN_SECONDS)
        self.settings["prebuffer_seconds"] = pre
        self.settings["post_motion_seconds"] = post
        self.settings["min_area"] = int(self.minarea_var.get())
        self.settings["motion_threshold"] = int(self.threshold_var.get())
        self.settings["audio_threshold"] = int(self.audio_threshold_var.get())
        self.settings["output_dir"] = self.outdir_var.get()
        self.recorder.settings.update(self.settings)
        self.recorder.settings["prebuffer_seconds"] = pre
        self.recorder.settings["post_motion_seconds"] = post

    def _start(self):
        try:
            self._apply_settings()
            self.recorder.start()
            self.status_var.set("Запись: активна")
            self.btn_start.state(["disabled"])
            self.btn_stop.state(["!disabled"])
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _stop(self):
        self.recorder.stop()
        self.status_var.set("Остановлено")
        self.btn_start.state(["!disabled"])
        self.btn_stop.state(["disabled"])

    def _choose_out(self):
        p = filedialog.askdirectory()
        if p:
            self.outdir_var.set(p)

    def update_preview(self, pil_image: Image.Image, rects=None):
        try:
            img_copy = pil_image.copy()
            rects_copy = list(rects) if rects else []
            self.root.after(0, lambda img=img_copy, rects=rects_copy: self._update_preview_main(img, rects))
        except Exception:
            logger.exception("Preview update failed")

    def _update_preview_main(self, pil_image: Image.Image, rects=None):
        try:
            pw, ph = self.preview_w, self.preview_h

            iw, ih = pil_image.size
            canvas_ratio = pw / ph if ph else 1.0
            img_ratio = iw / ih if ih else 1.0
            rotated = False
            if abs(img_ratio - canvas_ratio) > abs((ih / iw) - canvas_ratio):
                pil_image = pil_image.rotate(90, expand=True)
                iw, ih = pil_image.size
                rotated = True

            scale = min(pw / iw, ph / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            img = pil_image.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (iw, ih) else pil_image
            photo = ImageTk.PhotoImage(img)

            self.preview_canvas.delete("img")
            x = (pw - nw) // 2
            y = (ph - nh) // 2
            self.preview_canvas.create_image(x, y, image=photo, anchor="nw", tags="img")

            self.preview_canvas.delete("motion")
            if rects:
                for (rx, ry, rw, rh) in rects:
                    if rotated:

                        sx = int(ry * scale)
                        sy = int((iw - (rx + rw)) * scale)
                        sw = max(1, int(rh * scale))
                        sh = max(1, int(rw * scale))
                    else:
                        sx = int(rx * scale)
                        sy = int(ry * scale)
                        sw = max(1, int(rw * scale))
                        sh = max(1, int(rh * scale))
                    self.preview_canvas.create_rectangle(x + sx, y + sy, x + sx + sw, y + sy + sh, outline="red", width=2, tags="motion")

            self.preview_photo = photo
        except Exception:
            logger.exception("Preview update failed")


    def _on_close(self):
        if messagebox.askokcancel("Выход", "Завершить работу и остановить запись?"):
            try:
                self.recorder.stop()
            except Exception:
                pass
            self.root.destroy()

    def _signal_handler(self, signum, frame):
        logger.info("Signal %s received, stopping...", signum)
        try:
            self.recorder.stop()
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
