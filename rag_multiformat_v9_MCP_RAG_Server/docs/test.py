from pathlib import Path
from docling.document_converter import DocumentConverter

pdf = Path(
r"D:\RAG\rag_multiformat_v9_MCP_RAG_Server\docs\typical knowledge\control\arxiv_1206.5033.pdf"
)

converter = DocumentConverter()

result = converter.convert(pdf)

print(result.document.export_to_markdown()[:500])