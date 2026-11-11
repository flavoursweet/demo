from pathlib import Path
import hashlib, json, time, threading
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from models.unet import UNet

ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = 20_000_000
LOCK = threading.Lock()
MODEL = None
DEVICE = 'cpu'
CHECKSUM = ''

def load_model():
    global MODEL, DEVICE, CHECKSUM
    if MODEL is None:
        torch.set_num_threads(4)
        checkpoint = ROOT / 'checkpoints/unet_best.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        model = UNet()
        model.load_state_dict(state, strict=True)
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        MODEL = model.to(DEVICE).eval()
        CHECKSUM = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

def infer(path, outdir, threshold, gt=None, gsd=None):
    start = time.perf_counter()
    with LOCK:
        load_model()
        raw = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        if max(raw.size) > 4096:
            raise ValueError('演示支持最长边不超过 4096 像素的影像，请先切片。')
        raw.save(outdir / 'original.png')
        # 与原 predict.py 一致：PIL 双线性缩放到512，ImageNet归一化。
        arr = np.asarray(raw.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        arr = (arr - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
        x = torch.from_numpy(arr.transpose(2,0,1).copy()).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            prob512 = torch.sigmoid(MODEL.infer(x))[0,0].cpu().numpy()
        # 概率恢复原尺寸后阈值化，像素统计始终对应输入影像。
        prob = np.asarray(Image.fromarray(prob512).resize(raw.size, Image.Resampling.BILINEAR))
        mask = prob > threshold
        Image.fromarray(mask.astype('uint8')*255).save(outdir / 'mask.png')
        Image.fromarray((prob*65535).astype('uint16')).save(outdir / 'probability.png')
        rgb = np.array(raw).copy()
        rgb[mask] = (rgb[mask]*.45 + np.array([255,80,60])*.55).astype('uint8')
        Image.fromarray(rgb).save(outdir / 'overlay.png')
        _, regions = ndimage.label(mask, structure=np.ones((3,3)))
        metrics = None
        if gt:
            truth = np.asarray(Image.open(gt).convert('L').resize(raw.size, Image.Resampling.NEAREST)) > 127
            Image.fromarray(truth.astype('uint8')*255).save(outdir / 'ground_truth.png')
            tp, fp, fn = (int(np.sum(mask & truth)),int(np.sum(mask & ~truth)),int(np.sum(~mask & truth)))
            ratio = lambda a,b: round(a/b,6) if b else None
            metrics = dict(iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),tp=tp,fp=fp,fn=fn)
        return dict(width=raw.width,height=raw.height,foreground_pixels=int(mask.sum()),total_pixels=int(mask.size),
                    coverage=round(float(mask.mean())*100,3),regions=int(regions),threshold=threshold,
                    area_m2=round(float(mask.sum())*gsd*gsd,3) if gsd else None,gsd=gsd,
                    metrics=metrics,device=DEVICE,seconds=round(time.perf_counter()-start,3),
                    model='UNet',checkpoint_sha256=CHECKSUM,input_size=512)
