"""按原训练代码的 seed=42 / 30% 划分抽取前 16 个验证样本，不依据模型效果挑选。"""
from pathlib import Path
import json, shutil
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
source = root.parent / '地灾模型' / 'data'
files = sorted(p.name for p in (source / 'image').glob('*.png'))
indices = np.arange(len(files))
np.random.RandomState(42).shuffle(indices)
items = []
for i, index in enumerate(indices[:16]):
    name = files[index]
    sid = f'sample-{i+1:02}'
    shutil.copy2(source / 'image' / name, root / 'samples' / f'{sid}.png')
    shutil.copy2(source / 'mask' / name, root / 'samples' / f'{sid}-gt.png')
    items.append(dict(id=sid, name=f'资兴验证样本 {i+1:02}', source=name,
                      image=f'/samples/{sid}.png', gt=f'/samples/{sid}-gt.png',
                      size=list(Image.open(source / 'image' / name).size)))
(root / 'samples' / 'manifest.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(items, ensure_ascii=False))
