from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ParsedOutput:
    source_path: str
    raw_markdown: str = ""
    enriched_markdown: str = ""
    # Danh sách ảnh trích xuất được (để debug hoặc lưu file nếu cần)
    # Key: Tên file ảnh (image_0.png), Value: PIL Image
    images: Dict[str, Any] = field(default_factory=dict)
    # Metadata bổ sung (số trang, title...)
    metadata: Dict[str, Any] = field(default_factory=dict)  