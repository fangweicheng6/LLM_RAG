# -*- coding: utf-8 -*-
"""HTTP client for Industrial RAG FastAPI service."""
import requests

class RAGHttpClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=120):
        self.base_url=base_url.rstrip("/")
        self.timeout=timeout

    def chat(self, question):
        r=requests.post(self.base_url+"/chat",json={"question":question},timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self):
        r=requests.get(self.base_url+"/health",timeout=self.timeout)
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    import sys,json
    c=RAGHttpClient()
    print(json.dumps(c.chat(" ".join(sys.argv[1:])),ensure_ascii=False,indent=2))
