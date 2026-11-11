import base64, json, time, urllib.request, urllib.error
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8765'
def get(path):
    with urllib.request.urlopen(BASE+path) as r:return json.load(r)
def run(**body):
    req=urllib.request.Request(BASE+'/api/agent',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as r: jid=json.load(r)['id']
    for _ in range(240):
        j=get('/api/jobs/'+jid)
        if j['status'] not in ('queued','running'):return j
        time.sleep(.25)
    raise AssertionError('timeout')
checks=[]
a=run(message='识别当前影像并生成报告',sample_id='sample-01',threshold=.5)
assert a['status']=='done',a
r=a['result']; s=r['stats']
mask=np.array(Image.open(ROOT/r['files']['mask.png'].lstrip('/')))>0
assert int(mask.sum())==s['foreground_pixels'] and mask.shape==(64,64)
checks.append('真实推理：掩码像素计数与统计一致')
b=run(message='将阈值改为0.8重新识别',sample_id='sample-01',threshold=.5)
assert b['status']=='done' and b['result']['stats']['threshold']==.8
assert b['result']['stats']['foreground_pixels']<=s['foreground_pixels']
checks.append('对话修改阈值：0.8候选像素不多于0.5')
c=run(message='总结当前结果',result_id=r['id'])
assert c['status']=='done' and c['result']['id']==r['id']
checks.append('有状态总结：读取同一任务真实结果')
d=run(message='识别上传影像',image_base64=base64.b64encode((ROOT/'samples/sample-02.png').read_bytes()).decode(),filename='upload.png',gsd=2)
assert d['status']=='done',d
assert d['result']['stats']['metrics'] is None
assert d['result']['stats']['area_m2']==d['result']['stats']['foreground_pixels']*4
checks.append('上传推理与可选面积换算；无标注不虚构精度')
for body in [dict(message='开始识别',sample_id='missing'),dict(message='开始识别',sample_id='sample-01',threshold=1.2),dict(message='开始识别',image_base64='invalid')]:
    assert run(**body)['status']=='error'
checks.append('无效样本、越界阈值、损坏上传均返回错误')
for file,url in r['files'].items():
    with urllib.request.urlopen(BASE+url) as response:assert response.status==200 and len(response.read())>0
checks.append('所有结果文件可下载')
assert len(get('/api/samples'))==8 and get('/api/history')
checks.append('样本列表与历史记录正常')
(ROOT/'verification.json').write_text(json.dumps(dict(checks=checks,example=s),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=len(checks),example=s),ensure_ascii=False))
