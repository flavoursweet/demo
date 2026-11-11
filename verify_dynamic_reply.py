"""验收按问题生成回复、追问上下文、真实统计约束和工具执行隔离。"""
import json
from pathlib import Path
from datetime import datetime
from verify_llm_integration import run

ROOT=Path(__file__).resolve().parent

def main():
    report=dict(time=datetime.now().isoformat(),passed=False,cases=[])
    def ask(message,tool,**kwargs):
        job=run(message,**kwargs)
        trace=job.get('responder') or {}
        assert job['status']=='done',job.get('error')
        allowed=(tool,) if isinstance(tool,str) else tool
        assert job['planner']['tool'] in allowed,job['planner']
        assert not job.get('answer_warning'),job.get('answer_warning')
        assert trace.get('provider')=='cloud' and trace.get('response_id')
        if job['planner']['tool']!='segment':
            assert not any(s['tool']=='unet_segment' for s in job['steps'])
        report['cases'].append(dict(message=message,tool=tool,id=job['id'],answer=job['answer'],responder=trace))
        print(message+'\n'+job['answer']+'\n',flush=True)
        return job
    try:
        hello=ask('你好','help')
        concept=ask('U-Net的跳跃连接有什么作用？请解释清楚，不要执行识别。','help')
        history=[dict(role='user',content='U-Net的跳跃连接有什么作用？'),dict(role='assistant',content=concept['answer'])]
        followup=ask('那它为什么能帮助恢复边界细节？','help',history=history)
        assert len({hello['answer'],concept['answer'],followup['answer']})==3
        assert any(w in followup['answer'] for w in ['边界','空间','细节'])
        result=ask('识别当前影像并生成报告','segment',sample_id='sample-01',threshold=.5)['result']
        count=ask('这次候选像素到底有多少？只回答这个数，不要重新识别。','summarize',result_id=result['id'])
        normalized=count['answer'].replace(',','').replace('，','')
        assert str(result['stats']['foreground_pixels']) in normalized
        area=ask('没有提供地面分辨率，能知道真实面积是多少平方米吗？只解释这个问题。',('summarize','help'),result_id=result['id'])
        assert any(w in area['answer'] for w in ['不能','无法','不知道','不足','不可以'])
        assert count['result']['id']==result['id']
        if area.get('result'):
            assert area['result']['id']==result['id']
        report['passed']=True
    finally:
        (ROOT/'dynamic_reply_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('动态回复验收通过：'+str(len(report['cases']))+'项')

if __name__=='__main__':
    main()
