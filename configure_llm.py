"""本机配置窗口：密钥不进入聊天、不在日志显示；测试成功后写入.env。"""
import os, threading, tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from urllib.parse import urlparse
import agent

ROOT=Path(__file__).resolve().parent

def main():
    win=tk.Tk()
    win.title('滑坡智能体 · 大模型连接配置')
    win.geometry('760x560')
    win.resizable(False,False)
    frame=ttk.Frame(win,padding=24); frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='连接云端大模型',font=('Microsoft YaHei',16)).pack(anchor='w')
    ttk.Label(frame,text='填写支持工具调用的兼容接口地址、模型名称和 API Key。密钥只保存在本机。').pack(anchor='w',pady=(8,18))
    values={}
    existing={'LLM_BASE_URL':'https://api.deepseek.com','LLM_MODEL':'deepseek-flash'}
    path=ROOT/'.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1); existing[k.strip()]=v.strip().strip('"').strip("'")
    for key,label in [('LLM_BASE_URL','接口前缀（不包含 /chat/completions）'),('LLM_MODEL','模型名称'),('LLM_API_KEY','API Key')]:
        ttk.Label(frame,text=label).pack(anchor='w')
        values[key]=tk.StringVar(value=existing.get(key,''))
        ttk.Entry(frame,textvariable=values[key],show='*' if key=='LLM_API_KEY' else '').pack(fill='x',pady=(3,10))
    status=tk.StringVar(value='默认填写 DeepSeek 官方接口；其他平台请修改地址和模型。测试将发送6条固定任务，可能产生少量API费用。影像仍由本机U-Net处理。')
    ttk.Label(frame,textvariable=status,wraplength=590).pack(anchor='w',pady=8)
    def finish(ok,text):
        button.configure(state='normal'); status.set(text)
        if ok: messagebox.showinfo('连接成功',text)
    def test():
        config={k:v.get().strip() for k,v in values.items()}
        if any(not v or '\n' in v or '\r' in v for v in config.values()):
            return messagebox.showerror('配置不完整','三个字段都需要填写，且不能包含换行。')
        address=urlparse(config['LLM_BASE_URL'])
        if address.scheme!='https' and not(address.scheme=='http' and address.hostname in ('localhost','127.0.0.1')):
            return messagebox.showerror('接口地址错误','云端接口需要HTTPS；本地接口可使用http://127.0.0.1。')
        if address.username or address.password or address.query or address.fragment:
            return messagebox.showerror('接口地址错误','接口地址不能包含账号、密码、查询参数或片段。')
        if config['LLM_BASE_URL'].rstrip('/').endswith('/chat/completions'):
            config['LLM_BASE_URL']=config['LLM_BASE_URL'].rstrip('/')[:-len('/chat/completions')]
# 云端测试前显式切换提供方，确保测试使用云端配置。
        config['LLM_PROVIDER']='cloud'
        button.configure(state='disabled'); status.set('正在验证大模型是否能正确选择UNet工具，请等待…')
        def worker():
            previous={k:os.environ.get(k) for k in config}
            try:
                os.environ.update(config)
                cases=[('识别当前影像并生成报告','segment',.5,False),
                       ('把阈值设为0.7，再识别一次','segment',.7,False),
                       ('总结刚才的识别结果','summarize',.5,True),
                       ('解释识别流程','help',.5,False),
                       ('先不要识别，解释一下工作流程','help',.5,True),
                       ('你好','help',.5,False)]
                results=[]
                import json
                from datetime import datetime
                for i,(message,expected,value,has_result) in enumerate(cases,1):
                    win.after(0,status.set,f'正在测试 {i}/{len(cases)}：{message}')
                    tool,threshold,mode=agent.plan(message,.5,has_result)
                    passed=tool==expected and abs(threshold-value)<1e-6
                    results.append(dict(message=message,expected=expected,tool=tool,threshold=threshold,passed=passed,planner=agent.LAST_PLAN))
                    (ROOT/'cloud_llm_verification.json').write_text(json.dumps(dict(time=datetime.now().isoformat(),results=results),ensure_ascii=False,indent=2),encoding='utf-8')
                    if not passed:
                        raise ValueError(f'第{i}项意图或参数测试失败，未切换云端。详情见cloud_llm_verification.json。')
                lines=path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
                lines=[line for line in lines if line.split('=',1)[0].strip() not in config]
                lines.extend(k+'='+v for k,v in config.items())
                temp=ROOT/'.env.tmp'; temp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); temp.replace(path)
                win.after(0,finish,True,'6项真实云端工具规划验证通过，已保存到.env。请回到聊天告知“配置好了”，继续完成服务切换和U-Net端到端验收。')
            except Exception as exc:
                message=str(exc)
                if config['LLM_API_KEY']: message=message.replace(config['LLM_API_KEY'],'[已隐藏]')
                win.after(0,finish,False,'连接失败：'+message)
            finally:
                for k,v in previous.items():
                    if v is None: os.environ.pop(k,None)
                    else: os.environ[k]=v
        threading.Thread(target=worker,daemon=True).start()
    button=ttk.Button(frame,text='测试连接并保存',command=test); button.pack(anchor='e',pady=8)
    win.mainloop()

if __name__=='__main__':main()
