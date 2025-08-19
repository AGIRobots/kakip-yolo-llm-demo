#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict

import paho.mqtt.client as mqtt


def open_fifo_reader(path: str):
    while True:
        try:
            return open(path, "r")
        except FileNotFoundError:
            os.mkfifo(path, 0o666)   
        except Exception as e:
            print(f"[WARN] open FIFO failed: {e}; retry in 0.5s")
            time.sleep(0.5)


def make_mqtt(broker: str, port: int, username: str | None, password: str | None, client_id: str | None) -> mqtt.Client:
    client = mqtt.Client(client_id=client_id or "", clean_session=True)
    if username or password:
        client.username_pw_set(username or "", password or "")
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    return client


def publish_per_detection(cli: mqtt.Client, topic: str, dets: List[Dict], qos: int, retain: bool):
    ts = int(time.time() * 1000)
    for d in dets:
        if not all(k in d for k in ("label", "x_norm", "y_norm", "w_norm", "h_norm")):
            continue
        payload = {
            "ts": ts,
            "label": d["label"],
            "x_norm": float(d["x_norm"]),
            "y_norm": float(d["y_norm"]),
            "w_norm": float(d["w_norm"]),
            "h_norm": float(d["h_norm"]),
        }
        if "prob" in d:
            try:
                payload["prob"] = float(d["prob"])
            except Exception:
                pass
        cli.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos, retain=retain)


def publish_per_frame(cli: mqtt.Client, topic: str, dets: List[Dict], qos: int, retain: bool):
    ts = int(time.time() * 1000)
    slim = []
    for d in dets:
        if not all(k in d for k in ("label", "x_norm", "y_norm", "w_norm", "h_norm")):
            continue
        one = {
            "label": d["label"],
            "x_norm": float(d["x_norm"]),
            "y_norm": float(d["y_norm"]),
            "w_norm": float(d["w_norm"]),
            "h_norm": float(d["h_norm"]),
        }
        if "prob" in d:
            try:
                one["prob"] = float(d["prob"])
            except Exception:
                pass
        slim.append(one)
    payload = {"ts": ts, "detections": slim}
    cli.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos, retain=retain)


def main():
    ap = argparse.ArgumentParser(description="Read normalized detections from FIFO and publish to MQTT.")
    ap.add_argument("--fifo", default="/tmp/yolo_fifo", help="FIFO path (default: /tmp/yolo_fifo)")
    ap.add_argument("--broker", default="localhost", help="MQTT broker (default: localhost)")
    ap.add_argument("--port", type=int, default=1883, help="MQTT port (default: 1883)")
    ap.add_argument("--topic", default="yolo/detections_norm", help="MQTT topic (default: yolo/detections_norm)")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2], help="MQTT QoS")
    ap.add_argument("--retain", action="store_true", help="Retain flag")
    ap.add_argument("--per-detection", action="store_true",
                    help="Publish one message per detection (default: per frame array)")
    ap.add_argument("--username", default=None, help="MQTT username")
    ap.add_argument("--password", default=None, help="MQTT password")
    ap.add_argument("--client-id", default=None, help="MQTT client id")
    args = ap.parse_args()

    cli = make_mqtt(args.broker, args.port, args.username, args.password, args.client_id)
    print(f"[INFO] Publishing to mqtt://{args.broker}:{args.port} topic='{args.topic}' "
          f"(per_detection={args.per_detection}, qos={args.qos}, retain={args.retain})")

    while True:
        with open_fifo_reader(args.fifo) as fifo:
            print(f"[INFO] Listening on {args.fifo} ... (Ctrl+C to quit)")
            for line in fifo:
                line = line.strip()
                if not line:
                    continue
                try:
                    dets = json.loads(line)
                    if not isinstance(dets, list):
                        continue
                except json.JSONDecodeError:
                    continue

                if args.per_detection:
                    publish_per_detection(cli, args.topic, dets, args.qos, args.retain)
                else:
                    publish_per_frame(cli, args.topic, dets, args.qos, args.retain)

            print("[INFO] Writer closed FIFO; reopening ...")
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exit.")
