# -*- coding: utf-8 -*-
"""
知识库增量更新管理

流程:
新增文档
 -> hash检测
 -> 解析
 -> chunk
 -> embedding
 -> 更新向量库
"""

import hashlib,json
from pathlib import Path

STATE=Path("data_index_only/update_state.json")

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def need_update(path):
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    h=file_hash(path)
    return state.get(str(path))!=h

def commit(path):
    STATE.parent.mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    state[str(path)]=file_hash(path)
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2))

if __name__=="__main__":
    print("增量更新管理模块")
