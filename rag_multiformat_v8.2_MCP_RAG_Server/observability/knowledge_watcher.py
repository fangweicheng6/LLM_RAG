"""知识库文件变更监视（目前未被 ingest 主流程接入，作为独立工具保留）。

scan(folder) 比较目录内文件 MD5 与上次快照，返回新增/修改与删除的文件。
注意：只比较「本次扫描到的文件」会漏掉「上次有、这次已删除」的文件，
因此必须同时报告 deleted，否则删除文件不会触发重索引。
"""
import hashlib,json
from pathlib import Path

STATE="knowledge_state.json"

def md5(path):
    h=hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()

def scan(folder):
    now={}
    for f in Path(folder).rglob("*"):
        if f.is_file():
            now[str(f)]=md5(f)
    old={}
    if Path(STATE).exists():
        old=json.loads(Path(STATE).read_text())
    changed=[x for x in now if now[x]!=old.get(x)]
    # 报告被删除/消失的文件：仅比较 now 无法发现「上次有、这次没有」的文件。
    deleted=[x for x in old if x not in now]
    Path(STATE).write_text(json.dumps(now,ensure_ascii=False,indent=2))
    return {"changed": changed, "deleted": deleted}
