from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor
import base64, csv, io, json, mimetypes, os, threading, uuid, traceback, webbrowser
from datetime import datetime
import engine, agent

ROOT = Path(__file__).resolve().parent
for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines() if (ROOT / '.env').exists() else []:
    if '=' in line and not line.lstrip().startswith('#'):
        k,v = line.split('=',1)
        os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
SAMPLES = json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)
MAX_BODY = 16*1024*1024

def read_result(rid):
    if not isinstance(rid,str) or not re_id(rid):
        raise ValueError('无效结果编号。')
    return json.loads((ROOT/'outputs'/rid/'result.json').read_text(encoding='utf-8'))

def re_id(value):
    return len(value)==32 and all(c in '0123456789abcdef' for c in value)

def run_job(jid, body):
    job = JOBS[jid]
    def step(tool, detail):
        job['steps'].append(dict(tool=tool,detail=detail,time=datetime.now().strftime('%H:%M:%S')))
    try:
        job['status']='running'
        message = str(body.get('message','开始识别'))[:2000]
        threshold = float(body.get('threshold',.5))
        step('plan','解析任务意图与参数')
        # Reload the small planner so a local rule update takes effect without
        # forcing the user to restart the already-loaded UNet process.
        import importlib
        importlib.reload(agent)
        tool,threshold,mode = agent.plan(message,threshold,bool(body.get('result_id')),history=body.get('history'))
        job['mode']=mode
        job['planner']=agent.LAST_PLAN
        if job['planner']:
            trace=job['planner']
            provider_label='云端API' if trace['provider']=='cloud' else '云端API'
            step('llm_plan',f"{provider_label}真实调用 · {trace['model']} · {trace['seconds']}秒 · 调用编号 {trace['call_id']}")
        step('plan',f'{mode} → {tool}；阈值 {threshold}')
        if tool=='help':
            job['answer']='我可以调用 UNet 进行真实滑坡分割、调整阈值重新识别、对比标注并生成报告。\n请先在左侧选择样本或上传 RGB 影像，再说“识别当前影像并生成报告”。\n工作流：输入校验 → ImageNet归一化 → UNet推理 → 概率阈值化 → 连通域与指标统计 → 报告。\n当前模式：'+mode+'。未配置云端大模型时停止规划；本地模式由云端大模型规划工具。本Demo的帮助回复仍采用固定说明。'
        elif tool=='summarize':
            if not body.get('result_id'):
                job['answer']='当前还没有识别结果。请选择影像并发送“开始识别”。'
            else:
                result=read_result(body['result_id'])
                step('summarize','读取该次真实推理记录')
                job.update(result=result,answer=agent.summarize(result))
        else:
            out = ROOT/'outputs'/jid
            out.mkdir()
            gt=None
            sid=body.get('sample_id')
            if body.get('image_base64'):
                data=base64.b64decode(body['image_base64'],validate=True)
                if len(data)>10*1024*1024: raise ValueError('上传图片不能超过10MB。')
                path=out/'input.upload'
                path.write_bytes(data)
                name=Path(str(body.get('filename','上传影像'))).name[:180]
                source='用户上传；无标注'
            else:
                sample=next((s for s in SAMPLES if s['id']==sid),None)
                if not sample: raise ValueError('请先选择有效样本或上传影像。')
                path=ROOT/sample['image'].lstrip('/')
                gt=ROOT/sample['gt'].lstrip('/')
                name=sample['name']; source=sample['source']
            gsd=body.get('gsd')
            gsd=float(gsd) if gsd not in (None,'') else None
            if gsd is not None and not 0<gsd<=1000: raise ValueError('地面分辨率需在0至1000米/像素之间。')
            step('validate_image','读取RGB影像，检查尺寸与参数')
            step('unet_segment','加载训练权重，执行512×512前向推理')
            stats=engine.infer(path,out,threshold,gt,gsd)
            step('calculate_statistics','原图尺度像素统计、8邻域连通域'+('、标注对比' if gt else ''))
            result=dict(id=jid,name=name,source=source,created=datetime.now().isoformat(timespec='seconds'),stats=stats,
                        files={f: f'/outputs/{jid}/{f}' for f in ['original.png','mask.png','overlay.png','probability.png','report.md','statistics.csv','result.json']})
            if gt: result['files']['ground_truth.png']=f'/outputs/{jid}/ground_truth.png'
            step('export_report','生成掩码、叠加图、Markdown报告和CSV统计')
            result['steps']=job['steps'].copy(); result['mode']=mode
            result['planner']=job['planner']
            (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            report='# 滑坡语义分割识别报告\n\n'+agent.summarize(result)+'\n\n'
            report+=f"- 任务编号：{jid}\n- 时间：{result['created']}\n- 数据来源：{source}\n- 规划模式：{mode}\n- 权重 SHA256：{stats['checkpoint_sha256']}\n\n"
            report+='## 可视化\n\n![原图](original.png)\n\n![预测叠加](overlay.png)\n\n![二值掩码](mask.png)\n\n'
            report+='## 方法与限制\n\n输入双线性缩放到512×512，使用ImageNet均值/标准差归一化，UNet输出经sigmoid，概率双线性恢复原尺寸后阈值化。二值掩码前景255、背景0；16位概率PNG除以65535得到概率。连通区域为8邻域，不是灾害事件数量。样本指标仅针对本次输入，不代表全验证集。普通PNG无坐标，不可据此生成真实地理定位。\n'
            (out/'report.md').write_text(report,encoding='utf-8')
            with (out/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f); w.writerow(['metric','value'])
                for k,v in stats.items():
                    if k!='metrics': w.writerow([k,v])
                for k,v in (stats['metrics'] or {}).items(): w.writerow([k,v])
            job.update(result=result,answer=agent.summarize(result))
        if job.get('planner') and job['planner'].get('provider')=='cloud':
            import responder
            importlib.reload(responder)
            context_result=job.get('result')
            if context_result is None and body.get('result_id'):
                context_result=read_result(body['result_id'])
            step('generate_answer','结合当前问题、最近对话与真实工具结果生成回复')
            try:
                answer,trace=responder.reply(message,tool,context_result,body.get('history'))
                job.update(answer=answer,responder=trace)
                step('generate_answer',f"云端回复完成 · {trace['seconds']}秒 · 调用编号 {trace['call_id']}")
            except ValueError as exc:
                job['answer_warning']=str(exc)
                job['answer']=str(exc)+' 以下为已完成工具的固定摘要（并非大模型回复）：\n'+job['answer']
                step('generate_answer','回复生成失败；保留已完成工具结果')
        job['status']='done'
    except Exception as exc:
        traceback.print_exc()
        job.update(status='error',error=str(exc))

class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data):
        content=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if path=='/api/health':
            return self.respond(200,dict(ok=True,model='UNet',device=engine.DEVICE,loaded=engine.MODEL is not None,
                                        mode=('云端大模型规划' if os.getenv('LLM_API_KEY') else '需配置云端大模型')))
        if path=='/api/samples': return self.respond(200,SAMPLES)
        if path.startswith('/api/jobs/'):
            job=JOBS.get(path.rsplit('/',1)[-1])
            return self.respond(200 if job else 404,job or {'error':'任务不存在'})
        if path=='/api/history':
            results=[]
            for p in sorted((ROOT/'outputs').glob('*/result.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
                results.append(json.loads(p.read_text(encoding='utf-8')))
            return self.respond(200,results)
        base=ROOT if path.startswith(('/samples/','/outputs/')) else ROOT/'web'
        target=(base/(path.lstrip('/') or 'index.html')).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file() or target.suffix not in ['.html','.css','.js','.png','.json','.md','.csv']:
            return self.respond(404,{'error':'文件不存在'})
        content=target.read_bytes()
        self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(len(content))); self.send_header('Cache-Control','no-cache')
        self.end_headers(); self.wfile.write(content)
    def do_POST(self):
        if self.path!='/api/agent': return self.respond(404,{'error':'接口不存在'})
        # 同源本地接口，避免外部网页向本地工具提交任务。
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}']:
            return self.respond(403,{'error':'仅允许本地同源请求'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=MAX_BODY: return self.respond(413,{'error':'请求为空或超过16MB'})
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict): raise ValueError('请求必须为JSON对象')
            if sum(j['status'] in ('queued','running') for j in JOBS.values())>=4:
                return self.respond(429,{'error':'任务队列已满，请等待当前任务完成'})
            jid=uuid.uuid4().hex
            JOBS[jid]=dict(id=jid,status='queued',steps=[])
            POOL.submit(run_job,jid,body)
            self.respond(202,{'id':jid})
        except Exception as exc: self.respond(400,{'error':str(exc)})

PORT=int(os.getenv('PORT','8765'))
if __name__=='__main__':
    engine.load_model()
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    print(f'UNet ready on {engine.DEVICE}: http://127.0.0.1:{PORT}',flush=True)
    if '--open' in __import__('sys').argv: webbrowser.open(f'http://127.0.0.1:{PORT}')
    server.serve_forever()