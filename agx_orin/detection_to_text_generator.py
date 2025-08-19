# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import json
# import time
# import argparse
# import threading
# from collections import defaultdict, Counter
# from typing import Any, Dict, List, Optional, Tuple

# import paho.mqtt.client as mqtt

# def _clip01(v: float) -> float:
#     return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

# def _xyxy_from_norm(d: Dict[str, Any]) -> Optional[Tuple[float,float,float,float]]:
#     if isinstance(d.get("xyxy_norm"), list) and len(d["xyxy_norm"]) == 4:
#         xmin, ymin, xmax, ymax = map(float, d["xyxy_norm"])
#         return (_clip01(xmin), _clip01(ymin), _clip01(xmax), _clip01(ymax))
#     need = ("x_norm","y_norm","w_norm","h_norm")
#     if all(k in d for k in need):
#         x, y, w, h = (float(d[k]) for k in need)
#         xmin = x - w/2.0
#         ymin = y - h/2.0
#         xmax = x + w/2.0
#         ymax = y + h/2.0
#         return (_clip01(xmin), _clip01(ymin), _clip01(xmax), _clip01(ymax))
#     return None

# def normalize_detection(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     label = d.get("label") or d.get("class") or d.get("cls")
#     if label is None:
#         return None
#     xyxy = _xyxy_from_norm(d)
#     if xyxy is None:
#         return None
#     xmin, ymin, xmax, ymax = xyxy
#     cx = (xmin + xmax)/2.0
#     cy = (ymin + ymax)/2.0
#     w  = max(0.0, xmax - xmin)
#     h  = max(0.0, ymax - ymin)
#     out = {
#         "label": str(label),
#         "x_norm": _clip01(cx),
#         "y_norm": _clip01(cy),
#         "w_norm": _clip01(w),
#         "h_norm": _clip01(h),
#         "xmin": _clip01(xmin), "ymin": _clip01(ymin),
#         "xmax": _clip01(xmax), "ymax": _clip01(ymax),
#     }
#     if "prob" in d:
#         out["prob"] = float(d["prob"])
#     elif "confidence" in d:
#         out["prob"] = float(d["confidence"])
#     return out

# def parse_payload(payload: str, arrival_ms: int) -> Tuple[int, List[Dict[str, Any]]]:
#     try:
#         obj = json.loads(payload)
#     except json.JSONDecodeError:
#         return arrival_ms, []

#     ts = arrival_ms
#     dets_raw: List[Dict[str, Any]] = []

#     if isinstance(obj, dict) and "detections" in obj and isinstance(obj["detections"], list):
#         ts = int(obj.get("ts", arrival_ms))
#         dets_raw = [d for d in obj["detections"] if isinstance(d, dict)]
#     elif isinstance(obj, dict):
#         ts = int(obj.get("ts", arrival_ms))
#         dets_raw = [obj]
#     elif isinstance(obj, list):
#         dets_raw = [d for d in obj if isinstance(d, dict)]

#     dets = []
#     for d in dets_raw:
#         nd = normalize_detection(d)
#         if nd:
#             nd["ts"] = ts
#             dets.append(nd)
#     return ts, dets

# def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#     ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
#     inter = iw * ih
#     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
#     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
#     union = area_a + area_b - inter
#     return 0.0 if union <= 0 else inter / union

# def region_name(x: float, y: float) -> str:
#     xr = "左" if x < 1/3 else ("中央" if x < 2/3 else "右")
#     yr = "上" if y < 1/3 else ("中央" if y < 2/3 else "下")
#     return f"{yr}{xr}"

# class Track:
#     __slots__ = ("tid","label","xmin","ymin","xmax","ymax","prob","last_seen","hits")
#     def __init__(self, tid: int, label: str, xyxy: Tuple[float,float,float,float], prob: float, ts: int):
#         self.tid = tid
#         self.label = label
#         self.xmin, self.ymin, self.xmax, self.ymax = xyxy
#         self.prob = prob
#         self.last_seen = ts
#         self.hits = 1

#     def xyxy(self) -> Tuple[float,float,float,float]:
#         return (self.xmin, self.ymin, self.xmax, self.ymax)

#     def center_wh(self) -> Tuple[float,float,float,float]:
#         x = (self.xmin + self.xmax)/2.0
#         y = (self.ymin + self.ymax)/2.0
#         w = max(0.0, self.xmax - self.xmin)
#         h = max(0.0, self.ymax - self.ymin)
#         return x,y,w,h

#     def update(self, xyxy_new, prob_new, ts, alpha=0.3):
#         nx1, ny1, nx2, ny2 = xyxy_new
#         self.xmin = (1-alpha)*self.xmin + alpha*nx1
#         self.ymin = (1-alpha)*self.ymin + alpha*ny1
#         self.xmax = (1-alpha)*self.xmax + alpha*nx2
#         self.ymax = (1-alpha)*self.ymax + alpha*ny2
#         self.prob = max(self.prob, prob_new)
#         self.last_seen = ts
#         self.hits += 1

# class SimpleTracker:
#     def __init__(self, iou_match=0.5, ttl_ms=1000, nms_iou=0.7, min_prob=0.0):
#         self.iou_match = float(iou_match)
#         self.ttl_ms = int(ttl_ms)
#         self.nms_iou = float(nms_iou)
#         self.min_prob = float(min_prob)
#         self.tracks: Dict[int, Track] = {}
#         self._next_id = 1
#         self._lock = threading.Lock()

#     def _pre_nms(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         if not dets:
#             return dets
#         dets_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
#         for d in dets:
#             if self.min_prob and float(d.get("prob", 1.0)) < self.min_prob:
#                 continue
#             dets_by_label[d["label"]].append(d)

#         out: List[Dict[str, Any]] = []
#         for lab, arr in dets_by_label.items():
#             arr_sorted = sorted(arr, key=lambda x: float(x.get("prob", 0.0)), reverse=True)
#             kept: List[Dict[str, Any]] = []
#             for d in arr_sorted:
#                 xy = (d["xmin"], d["ymin"], d["xmax"], d["ymax"])
#                 if any(iou_xyxy(xy, (k["xmin"],k["ymin"],k["xmax"],k["ymax"])) >= self.nms_iou for k in kept):
#                     continue
#                 kept.append(d)
#             out.extend(kept)
#         return out

#     def update_with_dets(self, dets: List[Dict[str, Any]], ts: int):
#         if not dets:
#             self._gc(ts)
#             return

#         dets = self._pre_nms(dets)  

#         with self._lock:
#             for d in dets:
#                 xy = (d["xmin"], d["ymin"], d["xmax"], d["ymax"])
#                 prob = float(d.get("prob", 0.0))
#                 label = d["label"]

#                 best_tid = None
#                 best_iou = 0.0
#                 for tid, tr in self.tracks.items():
#                     if tr.label != label:
#                         continue
#                     iou = iou_xyxy(xy, tr.xyxy())
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_tid = tid

#                 if best_tid is not None and best_iou >= self.iou_match:
#                     self.tracks[best_tid].update(xy, prob, ts)
#                 else:
#                     tid = self._next_id
#                     self._next_id += 1
#                     self.tracks[tid] = Track(tid, label, xy, prob, ts)

#             self._gc(ts)

#     def _gc(self, now_ms: int):
#         dead = [tid for tid, tr in self.tracks.items() if now_ms - tr.last_seen > self.ttl_ms]
#         for tid in dead:
#             self.tracks.pop(tid, None)

#     def snapshot(self, now_ms: int) -> List[Track]:
#         with self._lock:
#             self._gc(now_ms)
#             return list(self.tracks.values())

# def summarize_tracks(tracks: List[Track]) -> Tuple[str, Dict[str,int]]:
#     if not tracks:
#         return "直近の1秒間に検出はありません。", {}

#     counts = Counter(t.label for t in tracks)
#     regions: Dict[str, Counter] = defaultdict(Counter)
#     for t in tracks:
#         x,y,_,_ = t.center_wh()
#         regions[t.label][region_name(x,y)] += 1

#     top = counts.most_common(2)
#     parts = []
#     for lab, cnt in top:
#         reg = regions[lab]
#         if reg:
#             best_region = reg.most_common(1)[0][0]
#             if cnt == 1:
#                 parts.append(f"{lab}が{best_region}に1件見えます")
#             else:
#                 parts.append(f"{lab}が{best_region}を中心に{cnt}件見えます")
#         else:
#             parts.append(f"{lab}が{cnt}件見えます")
#     return "、".join(parts) + "。", dict(counts)

# class App:
#     def __init__(self, args):
#         self.args = args
#         self.tracker = SimpleTracker(
#             iou_match=args.iou,
#             ttl_ms=int(args.ttl*1000),
#             nms_iou=args.nms,
#             min_prob=args.min_prob,
#         )
#         self.last_pub = 0
#         self.cli = self._make_mqtt()

#     def _make_mqtt(self) -> mqtt.Client:
#         try:
#             client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.args.client_id or "", clean_session=True)
#         except Exception:
#             client = mqtt.Client(client_id=self.args.client_id or "", clean_session=True)

#         if self.args.username or self.args.password:
#             client.username_pw_set(self.args.username or "", self.args.password or "")
#         client.on_connect = self._on_connect
#         client.on_message = self._on_message
#         client.reconnect_delay_set(min_delay=1, max_delay=30)
#         client.connect(self.args.broker, self.args.port, keepalive=60)
#         client.loop_start()
#         return client

#     def _on_connect(self, cli, userdata, flags, rc, properties=None):
#         print(f"[MQTT] Connected rc={rc}; subscribing '{self.args.input_topic}'")
#         cli.subscribe(self.args.input_topic, qos=self.args.qos)

#     def _on_message(self, cli, userdata, msg):
#         now = int(time.time() * 1000)
#         ts, dets = parse_payload(msg.payload.decode("utf-8", errors="ignore"), now)
#         if dets:
#             self.tracker.update_with_dets(dets, ts)

#     def run(self):
#         print(f"[INFO] Subscribing from '{self.args.input_topic}', publishing to '{self.args.output_topic}'")
#         try:
#             while True:
#                 now = int(time.time() * 1000)
#                 if now - self.last_pub >= int(self.args.interval * 1000):
#                     self.last_pub = now
#                     tracks = self.tracker.snapshot(now)
#                     text, counts = summarize_tracks(tracks)
#                     payload = {"ts": now, "summary": text, "counts": counts}
#                     self.cli.publish(self.args.output_topic, json.dumps(payload, ensure_ascii=False), qos=self.args.qos, retain=False)
#                     if self.args.echo:
#                         print("[OUT]", text)
#                 time.sleep(0.02)
#         except KeyboardInterrupt:
#             print("\n[INFO] Exit.")
#         finally:
#             try:
#                 self.cli.loop_stop()
#                 self.cli.disconnect()
#             except Exception:
#                 pass

# def build_argparser():
#     ap = argparse.ArgumentParser(description="Track dedup over 1s and summarize scene in Japanese.")
#     ap.add_argument("--broker", default="localhost")
#     ap.add_argument("--port", type=int, default=1883)
#     ap.add_argument("--input-topic", default="yolo/detections_norm")
#     ap.add_argument("--output-topic", default="yolo/scene_summary")
#     ap.add_argument("--qos", type=int, default=0, choices=[0,1,2])

#     ap.add_argument("--interval", type=float, default=1.0, help="Publish interval seconds")
#     ap.add_argument("--ttl", type=float, default=1.0, help="Track TTL seconds (no detection -> remove)")
#     ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold to associate detection to track")
#     ap.add_argument("--nms", type=float, default=0.7, help="In-frame NMS IoU (same label)")
#     ap.add_argument("--min-prob", type=float, default=0.0, help="Filter detections with low prob before tracking")
#     ap.add_argument("--username", default=None)
#     ap.add_argument("--password", default=None)
#     ap.add_argument("--client-id", default=None)
#     ap.add_argument("--echo", action="store_true")
#     return ap

# def main():
#     args = build_argparser().parse_args()
#     App(args).run()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT scene summarizer with optional Llama (llama.cpp) natural-language generation.

- Subscribes YOLO-style normalized detections on --input-topic
- Performs simple tracking / de-dup over a TTL window
- Publishes a concise Japanese summary to --output-topic every --interval seconds
- If --llama is enabled, uses a local GGUF model via llama-cpp-python to generate the summary
  from compact JSON stats (counts + region histogram). Falls back to rule-based summary on errors.

Run example (rule-based):
  python mqtt_scene_summarizer_with_llama.py --broker localhost --echo

Run example (with Llama):
  python mqtt_scene_summarizer_with_llama.py \
    --broker localhost --echo --llama \
    --llama-model ~/models/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
    --llama-gpu-layers 60 --llama-threads 4 --llama-max-tokens 96

Notes for Jetson Orin (CUDA build):
  # Build llama-cpp-python with CUDA (cuBLAS) acceleration
  # (Adjust versions as needed; JetPack 6.x typically ships CUDA 12.x)
  # sudo apt-get install -y build-essential cmake
  # CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-build-isolation llama-cpp-python==0.2.90
"""

import os
import json
import time
import argparse
import threading
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

# -----------------------------
# Utility functions (detections)
# -----------------------------

def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _xyxy_from_norm(d: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(d.get("xyxy_norm"), list) and len(d["xyxy_norm"]) == 4:
        xmin, ymin, xmax, ymax = map(float, d["xyxy_norm"]) 
        return (_clip01(xmin), _clip01(ymin), _clip01(xmax), _clip01(ymax))

    need = ("x_norm", "y_norm", "w_norm", "h_norm")
    if all(k in d for k in need):
        x, y, w, h = (float(d[k]) for k in need)
        xmin = x - w / 2.0
        ymin = y - h / 2.0
        xmax = x + w / 2.0
        ymax = y + h / 2.0
        return (_clip01(xmin), _clip01(ymin), _clip01(xmax), _clip01(ymax))

    return None


def normalize_detection(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = d.get("label") or d.get("class") or d.get("cls")
    if label is None:
        return None
    xyxy = _xyxy_from_norm(d)
    if xyxy is None:
        return None
    xmin, ymin, xmax, ymax = xyxy
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    out = {
        "label": str(label),
        "x_norm": _clip01(cx),
        "y_norm": _clip01(cy),
        "w_norm": _clip01(w),
        "h_norm": _clip01(h),
        "xmin": _clip01(xmin),
        "ymin": _clip01(ymin),
        "xmax": _clip01(xmax),
        "ymax": _clip01(ymax),
    }
    if "prob" in d:
        out["prob"] = float(d["prob"])
    elif "confidence" in d:
        out["prob"] = float(d["confidence"])
    return out


def parse_payload(payload: str, arrival_ms: int) -> Tuple[int, List[Dict[str, Any]]]:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return arrival_ms, []

    ts = arrival_ms
    dets_raw: List[Dict[str, Any]] = []

    if isinstance(obj, dict) and "detections" in obj and isinstance(obj["detections"], list):
        ts = int(obj.get("ts", arrival_ms))
        dets_raw = [d for d in obj["detections"] if isinstance(d, dict)]
    elif isinstance(obj, dict):
        ts = int(obj.get("ts", arrival_ms))
        dets_raw = [obj]
    elif isinstance(obj, list):
        dets_raw = [d for d in obj if isinstance(d, dict)]

    dets = []
    for d in dets_raw:
        nd = normalize_detection(d)
        if nd:
            nd["ts"] = ts
            dets.append(nd)
    return ts, dets


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


# ---------------
# Tracking helpers
# ---------------

def region_name(x: float, y: float) -> str:
    xr = "左" if x < 1 / 3 else ("中央" if x < 2 / 3 else "右")
    yr = "上" if y < 1 / 3 else ("中央" if y < 2 / 3 else "下")
    return f"{yr}{xr}"


class Track:
    __slots__ = ("tid", "label", "xmin", "ymin", "xmax", "ymax", "prob", "last_seen", "hits")

    def __init__(self, tid: int, label: str, xyxy: Tuple[float, float, float, float], prob: float, ts: int):
        self.tid = tid
        self.label = label
        self.xmin, self.ymin, self.xmax, self.ymax = xyxy
        self.prob = prob
        self.last_seen = ts
        self.hits = 1

    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def center_wh(self) -> Tuple[float, float, float, float]:
        x = (self.xmin + self.xmax) / 2.0
        y = (self.ymin + self.ymax) / 2.0
        w = max(0.0, self.xmax - self.xmin)
        h = max(0.0, self.ymax - self.ymin)
        return x, y, w, h

    def update(self, xyxy_new, prob_new, ts, alpha=0.3):
        nx1, ny1, nx2, ny2 = xyxy_new
        self.xmin = (1 - alpha) * self.xmin + alpha * nx1
        self.ymin = (1 - alpha) * self.ymin + alpha * ny1
        self.xmax = (1 - alpha) * self.xmax + alpha * nx2
        self.ymax = (1 - alpha) * self.ymax + alpha * ny2
        self.prob = max(self.prob, prob_new)
        self.last_seen = ts
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_match=0.5, ttl_ms=1000, nms_iou=0.7, min_prob=0.0):
        self.iou_match = float(iou_match)
        self.ttl_ms = int(ttl_ms)
        self.nms_iou = float(nms_iou)
        self.min_prob = float(min_prob)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def _pre_nms(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not dets:
            return dets
        dets_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for d in dets:
            if self.min_prob and float(d.get("prob", 1.0)) < self.min_prob:
                continue
            dets_by_label[d["label"]].append(d)

        out: List[Dict[str, Any]] = []
        for lab, arr in dets_by_label.items():
            arr_sorted = sorted(arr, key=lambda x: float(x.get("prob", 0.0)), reverse=True)
            kept: List[Dict[str, Any]] = []
            for d in arr_sorted:
                xy = (d["xmin"], d["ymin"], d["xmax"], d["ymax"])
                if any(iou_xyxy(xy, (k["xmin"], k["ymin"], k["xmax"], k["ymax"])) >= self.nms_iou for k in kept):
                    continue
                kept.append(d)
            out.extend(kept)
        return out

    def update_with_dets(self, dets: List[Dict[str, Any]], ts: int):
        if not dets:
            self._gc(ts)
            return

        dets = self._pre_nms(dets)

        with self._lock:
            for d in dets:
                xy = (d["xmin"], d["ymin"], d["xmax"], d["ymax"])
                prob = float(d.get("prob", 0.0))
                label = d["label"]

                best_tid = None
                best_iou = 0.0
                for tid, tr in self.tracks.items():
                    if tr.label != label:
                        continue
                    iou = iou_xyxy(xy, tr.xyxy())
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

                if best_tid is not None and best_iou >= self.iou_match:
                    self.tracks[best_tid].update(xy, prob, ts)
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self.tracks[tid] = Track(tid, label, xy, prob, ts)

            self._gc(ts)

    def _gc(self, now_ms: int):
        dead = [tid for tid, tr in self.tracks.items() if now_ms - tr.last_seen > self.ttl_ms]
        for tid in dead:
            self.tracks.pop(tid, None)

    def snapshot(self, now_ms: int) -> List[Track]:
        with self._lock:
            self._gc(now_ms)
            return list(self.tracks.values())


# ---------------------------
# Stats + rule-based summary
# ---------------------------

def scene_stats(tracks: List[Track]):
    """Return (counts, regions_by_label) for current tracks."""
    counts = Counter(t.label for t in tracks)
    regions: Dict[str, Counter] = defaultdict(Counter)
    for t in tracks:
        x, y, _, _ = t.center_wh()
        regions[t.label][region_name(x, y)] += 1
    return counts, regions


def summarize_tracks_rule_based(tracks: List[Track]) -> Tuple[str, Dict[str, int]]:
    if not tracks:
        return "直近の1秒間に検出はありません。", {}

    counts, regions = scene_stats(tracks)
    top = counts.most_common(2)
    parts = []
    for lab, cnt in top:
        reg = regions[lab]
        if reg:
            best_region = reg.most_common(1)[0][0]
            if cnt == 1:
                parts.append(f"{lab}が{best_region}に1件見えます")
            else:
                parts.append(f"{lab}が{best_region}を中心に{cnt}件見えます")
        else:
            parts.append(f"{lab}が{cnt}件見えます")
    return "、".join(parts) + "。", dict(counts)


# ----------------
# Llama integration
# ----------------

class LlamaSummarizer:
    """Thin wrapper around llama_cpp.Llama for short Japanese summaries.

    We feed compact JSON stats to reduce token count and latency. The prompt
    requests a single concise sentence (~40-60 chars), plain Japanese.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        max_tokens: int = 96,
        temperature: float = 0.6,
    ):
        from llama_cpp import Llama  # local import; optional dependency

        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=int(n_gpu_layers),
            n_threads=int(n_threads),
        )

    @staticmethod
    def _build_prompt(stats_json: str) -> str:
        return (
            "あなたは監視カメラのオペレーターです。\n"
            "以下のJSONは直近の検出件数の要約です。\n"
            "・日本語で1文だけ、簡潔に状況を説明してください。\n"
            "・重要な対象と位置（上/中央/下×左/中央/右）を優先し、重複表現は避けてください。\n"
            "・40～60文字程度、語尾は『。』で終えること。\n"
            "JSON:\n" + stats_json + "\n出力："
        )

    def summarize(self, counts: Dict[str, int], regions: Dict[str, Counter]) -> str:
        # Convert Counter values to plain dicts with stable ordering
        stats = {
            "objects": [
                {
                    "label": lab,
                    "count": int(cnt),
                    "regions": dict(regions.get(lab, {})),
                }
                for lab, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            ]
        }
        prompt = self._build_prompt(json.dumps(stats, ensure_ascii=False))

        with self._lock:
            try:
                # Use non-stream for simplicity (we need the whole sentence for MQTT publish)
                out = self._llm.create_completion(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                text = (out.get("choices") or [{}])[0].get("text", "").strip()
                # Basic sanitization
                text = text.replace("\n", " ")
                if not text.endswith("。"):
                    text += "。"
                # Keep very short degenerate outputs from breaking UX
                return text[:140]
            except Exception as e:
                return f"要約生成に失敗しました（LLMエラー: {type(e).__name__}）。"


# -------------
# Main MQTT app
# -------------

class App:
    def __init__(self, args):
        self.args = args
        self.tracker = SimpleTracker(
            iou_match=args.iou,
            ttl_ms=int(args.ttl * 1000),
            nms_iou=args.nms,
            min_prob=args.min_prob,
        )
        self.last_pub = 0
        self.cli = self._make_mqtt()
        self.llm = None
        if args.llama:
            try:
                self.llm = LlamaSummarizer(
                    model_path=args.llama_model,
                    n_gpu_layers=args.llama_gpu_layers,
                    n_threads=args.llama_threads,
                    max_tokens=args.llama_max_tokens,
                    temperature=args.llama_temperature,
                )
                print("[LLM] Llama initialized.")
            except Exception as e:
                print(f"[LLM] Init failed: {e}. Falling back to rule-based summary.")
                self.llm = None

    # ---- MQTT ----
    def _make_mqtt(self) -> mqtt.Client:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.args.client_id or "", clean_session=True)
        except Exception:
            client = mqtt.Client(client_id=self.args.client_id or "", clean_session=True)

        if self.args.username or self.args.password:
            client.username_pw_set(self.args.username or "", self.args.password or "")
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect(self.args.broker, self.args.port, keepalive=60)
        client.loop_start()
        return client

    def _on_connect(self, cli, userdata, flags, rc, properties=None):
        print(f"[MQTT] Connected rc={rc}; subscribing '{self.args.input_topic}'")
        cli.subscribe(self.args.input_topic, qos=self.args.qos)

    def _on_message(self, cli, userdata, msg):
        now = int(time.time() * 1000)
        ts, dets = parse_payload(msg.payload.decode("utf-8", errors="ignore"), now)
        if dets:
            self.tracker.update_with_dets(dets, ts)

    # ---- summarization ----
    def _summarize(self, tracks: List[Track]) -> Tuple[str, Dict[str, int]]:
        if not tracks:
            return "直近の1秒間に検出はありません。", {}

        if self.llm is None:
            return summarize_tracks_rule_based(tracks)

        counts, regions = scene_stats(tracks)
        text = self.llm.summarize(counts, regions)
        return text, dict(counts)

    # ---- main loop ----
    def run(self):
        print(f"[INFO] Subscribing from '{self.args.input_topic}', publishing to '{self.args.output_topic}'")
        try:
            while True:
                now = int(time.time() * 1000)
                if now - self.last_pub >= int(self.args.interval * 1000):
                    self.last_pub = now
                    tracks = self.tracker.snapshot(now)
                    text, counts = self._summarize(tracks)
                    payload = {
                        "ts": now,
                        "summary": text,
                        "counts": counts,
                        "llama": bool(self.llm is not None),
                    }
                    self.cli.publish(self.args.output_topic, json.dumps(payload, ensure_ascii=False), qos=self.args.qos, retain=False)
                    if self.args.echo:
                        print("[OUT]", text)
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[INFO] Exit.")
        finally:
            try:
                self.cli.loop_stop()
                self.cli.disconnect()
            except Exception:
                pass


# -------------
# Argparse / main
# -------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Track de-dup over 1s and summarize scene in Japanese (rule-based or Llama).")
    # MQTT
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--input-topic", default="yolo/detections_norm")
    ap.add_argument("--output-topic", default="yolo/scene_summary")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    # Timing / tracker params
    ap.add_argument("--interval", type=float, default=1.0, help="Publish interval seconds")
    ap.add_argument("--ttl", type=float, default=1.0, help="Track TTL seconds (no detection -> remove)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold to associate detection to track")
    ap.add_argument("--nms", type=float, default=0.7, help="In-frame NMS IoU (same label)")
    ap.add_argument("--min-prob", type=float, default=0.0, help="Filter detections with low prob before tracking")
    # MQTT auth
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--echo", action="store_true")
    # Llama options
    ap.add_argument("--llama", action="store_true", help="Enable Llama-based natural language summary")
    ap.add_argument("--llama-model", default=os.path.expanduser("~/models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"))
    ap.add_argument("--llama-gpu-layers", type=int, default=60)
    ap.add_argument("--llama-threads", type=int, default=4)
    ap.add_argument("--llama-max-tokens", type=int, default=96)
    ap.add_argument("--llama-temperature", type=float, default=0.6)
    return ap


def main():
    args = build_argparser().parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
