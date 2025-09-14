# YOLOv5 Rock Paper Scissors


### CPU Training
```
python3 -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirements.txt
(.venv) python train.py \
  --data <path_to_dataset>/data.yaml \
  --weights yolov5m.pt \
  --epochs 10 --patience 0 --img 640 --batch 16 \
  --name rps_yolov5m
```

### XPU Training
```
python3 -m venv .venv_xpu
source .venv_xpu/bin/activate
(.venv_xpu) pip install -r requirements.txt
(.venv_xpu) python train_xpu.py \
  --data <path_to_dataset>/data.yaml \
  --weights yolov5m.pt \
  --epochs 10 --patience 0 --img 640 --batch 30 \
  --name rps_yolov5m_xpu
```