#!/usr/bin/env python3
import json
import time
import threading
import argparse
from queue import Queue, Empty

import paho.mqtt.client as mqtt

# ポップアップ用 GUI（Tkinter）を遅延インポートして例外処理
try:
    import tkinter as tk
    from tkinter import font as tkfont
    HAS_TK = True
except ImportError:
    HAS_TK = False

POPUP_DURATION_DEFAULT = 3.0  # 秒

class MQTTPopup:
    def __init__(self, broker, port, topic, qos, duration, client_id=None, username=None, password=None, echo=False):
        self.topic = topic
        self.duration = duration
        self.echo = echo
        self.queue = Queue()
        self.last_summary = ""
        self.hide_after_id = None

        # MQTT クライアントセットアップ
        self.client = mqtt.Client(client_id=client_id or "")
        if username or password:
            self.client.username_pw_set(username or "", password or "")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect(broker, port, keepalive=60)
        self.client.loop_start()

        # GUI
        self.root = None
        if HAS_TK:
            try:
                self._init_gui()
            except Exception as e:
                print(f"[WARN] GUI unavailable ({e}), falling back to console output.")
                self.root = None

    def _on_connect(self, cli, userdata, flags, rc, properties=None):
        print(f"[MQTT] connected rc={rc}, subscribing to '{self.topic}'")
        cli.subscribe(self.topic, qos=self.client._userdata if False else qos)  # workaround; will set qos below
        # subscribe with correct qos
        cli.subscribe(self.topic, qos=qos)

    def _on_message(self, cli, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore")
        except Exception:
            return
        # 期待する JSON に summary があれば抽出
        summary = None
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict) and "summary" in obj:
                summary = str(obj["summary"]).strip()
        except Exception:
            pass
        if not summary:
            summary = payload.strip()
        if self.echo:
            print(f"[IN ] {summary}")
        self.queue.put(summary)

    def _init_gui(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # 枠なし
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)
        # 右上に置く（後で配置）
        self.root.configure(bg="black")
        # テキストラベル
        f = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.label = tk.Label(
            self.root,
            text="",
            font=f,
            fg="white",
            bg="black",
            justify="left",
            anchor="w",
            wraplength=400,
        )
        self.label.pack(padx=10, pady=6)
        # 初回位置調整（画面右上）
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        screen_w = self.root.winfo_screenwidth()
        # マージン少し
        x = screen_w - w - 20
        y = 20
        self.root.geometry(f"+{x}+{y}")

    def _show_popup(self, text: str):
        if self.root is None:
            # GUI 使えないならコンソール出力
            print(f"[POPUP] {text}")
            return

        # 更新
        self.label.config(text=text)
        self.root.update_idletasks()
        # 再配置（サイズ変化対応）
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - w - 20
        y = 20
        self.root.geometry(f"+{x}+{y}")

        # 表示（すでに表示されていればタイマーをリセット）
        if self.hide_after_id:
            self.root.after_cancel(self.hide_after_id)
        self.root.deiconify()
        self.hide_after_id = self.root.after(int(self.duration * 1000), self._hide_popup)

    def _hide_popup(self):
        if self.root:
            self.root.withdraw()
        self.hide_after_id = None

    def run(self):
        if self.root:
            self._hide_popup()  # 最初は隠す
        try:
            while True:
                try:
                    summary = self.queue.get(timeout=0.1)
                except Empty:
                    summary = None
                if summary:
                    # 重複抑制（直前と同じなら短縮する）
                    if summary != self.last_summary:
                        self.last_summary = summary
                        self._show_popup(summary)
                    else:
                        # 同文なら再表示タイマーだけリセット
                        self._show_popup(summary)
                if self.root:
                    self.root.update()
                # 微小スリープで CPU 型負荷を下げる
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Subscribe to scene summary MQTT and pop up on screen.")
    ap.add_argument("--broker", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--input-topic", default="yolo/scene_summary", help="MQTT topic to subscribe summary")
    ap.add_argument("--qos", type=int, default=0, choices=[0,1,2], help="MQTT QoS")
    ap.add_argument("--duration", type=float, default=POPUP_DURATION_DEFAULT, help="Popup display seconds")
    ap.add_argument("--client-id", default=None, help="MQTT client id")
    ap.add_argument("--username", default=None, help="MQTT username")
    ap.add_argument("--password", default=None, help="MQTT password")
    ap.add_argument("--echo", action="store_true", help="Also print incoming summary to stdout")
    args = ap.parse_args()

    # グローバル qos 参照を整備（内部の subscribe 回避用）
    qos = args.qos

    popup = MQTTPopup(
        broker=args.broker,
        port=args.port,
        topic=args.input_topic,
        qos=args.qos,
        duration=args.duration,
        client_id=args.client_id,
        username=args.username,
        password=args.password,
        echo=args.echo,
    )
    popup.run()
