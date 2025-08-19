python3 mqtt_summary_popup.py \
  --broker 192.168.0.7 \
  --port 1883 \
  --input-topic yolo/scene_summary \
  --duration 2.5 \
  --echo


xhost +SI:localuser:root
sudo DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY python3 mqtt_summary_popup.py \
  --broker 192.168.0.7 \
  --port 1883 \
  --input-topic yolo/scene_summary \
  --duration 2.5 \
  --echo