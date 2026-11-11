"""通过实际Demo接口验证云端规划和本机U-Net，记录成功与失败。"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from verify_llm_integration import BASE, OPENER, request, run

ROOT=Path(__file__).resolve().parent

def main():
    report=dict(time=datetime.now().isoformat(),base=BASE,jobs=[],checks=[],passed=False)
    try:
        report['health']=request('/api/health')
        def checked(message,expected,**kwargs):
            job=run(message,**kwargs)
            trace=job.get('planner') or {}
            passed=(job['status']=='done' and trace.get('provider')=='cloud'
                    and trace.get('tool')==expected and bool(trace.get('response_id')))
            report['jobs'].append(dict(message=message,expected=expected,id=job['id'],
                passed=passed,status=job['status'],planner=trace,error=job.get('error')))
            assert passed, job.get('error') or f'工具选择不符：{message}'
            print(message,expected,trace['seconds'],flush=True)
            return job
        first=checked('识别当前影像并生成报告','segment',sample_id='sample-01',threshold=.5)
        result=first['result']
        mask=np.array(Image.open(ROOT/result['files']['mask.png'].lstrip('/')))>0
        assert int(mask.sum())==result['stats']['foreground_pixels']
        assert tuple(mask.shape)==(result['stats']['height'],result['stats']['width'])
        report['checks'].append('云端选择分割，本机真实U-Net推理；掩码尺寸和像素统计一致')
        second=checked('把阈值设为0.7，再识别一次','segment',sample_id='sample-01',threshold=.5)
        assert second['result']['stats']['threshold']==.7
        assert second['result']['stats']['foreground_pixels']<=result['stats']['foreground_pixels']
        report['checks'].append('云端解析阈值0.7并重跑，候选像素单调性通过')
        summary=checked('总结刚才的识别结果','summarize',result_id=result['id'])
        assert summary['result']['id']==result['id']
        assert not any(s['tool']=='unet_segment' for s in summary['steps'])
        report['checks'].append('总结读取原结果，无额外模型推理')
        for message,expected in [('解释识别流程','help'),('你好','help'),
                                 ('先不要识别，解释一下工作流程','help'),
                                 ('请重新分割这张图片','segment'),
                                 ('这张图的结果是多少','summarize')]:
            job=checked(message,expected,sample_id='sample-01',result_id=result['id'])
            if expected=='help':
                assert not any(s['tool']=='unet_segment' for s in job['steps'])
        report['checks'].append('原本误判的5种表达全部正确路由；帮助和否定请求不启动分割')
        for url in result['files'].values():
            with OPENER.open(BASE+url,timeout=10) as response:
                assert response.status==200 and response.read()
        report['checks'].append('全部结果文件可下载')
        report['example']=result['stats']
        report['passed']=True
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        (ROOT/'cloud_integration_verification.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],jobs=len(report['jobs']),checks=len(report['checks'])),ensure_ascii=False))

if __name__=='__main__':
    main()
