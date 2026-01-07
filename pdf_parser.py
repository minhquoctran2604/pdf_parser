import torch
import gc
import os
from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional
from PIL import Image as PILImage

# PDF and AI model libraries
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Local modules
from models import ParsedOutput
import utils

class PDFParser:
    def __init__(self, device: str = None):
        print(">>> Initializing AI Engine (Marker + Qwen2-VL)...")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Using device: {self.device}")

        self.converter = None
        self.vision_model = None
        self.vision_processor = None

        self._load_marker()
        self._load_qwen_vl()

    def _load_marker(self):
        print("   - [1/2] Loading Marker Converter...")
        try:
            marker_models = create_model_dict()
            self.converter = PdfConverter(artifact_dict=marker_models)
            print("     ✅ Marker loaded successfully!")
        except Exception as e:
            print(f"     ❌ Critical Error loading Marker: {e}")
            raise

    def _load_qwen_vl(self):
        print("   - [2/2] Loading Qwen2-VL-2B-Instruct...")
        try:
            torch.cuda.empty_cache()
            self.vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else self.device,
            )
            self.vision_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            print("     ✅ Qwen2-VL-2B loaded successfully!")
        except Exception as e:
            print(f"     ❌ Critical Error loading Qwen2-VL: {e}")
            raise

    def _extract_structure(self, pdf_path: str) -> Tuple[str, Dict[str, PILImage.Image]]:
        print(f"   -> [Step 1] Analyzing PDF structure with Marker: {Path(pdf_path).name}")
        try:
            rendered = self.converter(pdf_path)
            print(f"      - Extracted {len(rendered.images)} images/tables.")
            return rendered.markdown, rendered.images
        except Exception as e:
            print(f"      ❌ ERROR in structure extraction: {e}")
            return None, None

    def _enrich_content(self, full_text: str, images: Dict[str, PILImage.Image]) -> str:
        if not images:
            return full_text

        print(f"   -> [Step 2] Generating descriptions for {len(images)} images with Qwen2-VL...")
        caption_cache = {}

        for img_name, pil_image in images.items():
            try:
                # Skip if image is not referenced in the markdown
                if not re.search(r"!\\[[^\\]*\\]\(" + re.escape(img_name) + r"\\)", full_text):
                    continue

                if pil_image.width < 32 or pil_image.height < 32:
                    continue

                if img_name not in caption_cache:
                    prompt_text = (
                        "Mô tả hình ảnh này bằng tiếng Việt, ngắn gọn và trung tính. "
                        "Nếu là màn hình ứng dụng: nêu tên màn hình và các nút/chức năng chính. "
                        "Nếu là bảng/biểu đồ: mô tả nội dung bảng và các số liệu/điểm nổi bật. "
                        "Nếu ảnh là CÔNG THỨC TOÁN HỌC: CHỈ TRẢ VỀ LaTeX (dùng $$...$$), KHÔNG GIẢI THÍCH."
                    )
                    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                    text = self.vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.vision_processor(text=[text], images=[pil_image.convert("RGB")], padding=True, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        generated_ids = self.vision_model.generate(
                            **inputs, max_new_tokens=256, do_sample=False, num_beams=3,
                            repetition_penalty=1.12, no_repeat_ngram_size=4
                        )
                    
                    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                    out_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    is_latex = (out_text.strip().startswith("$") or bool(re.search(r"\\(frac|sum|int|sqrt|begin{|[a-zA-Z]+\\_{|\\)", out_text)))
                    
                    if is_latex:
                        caption = out_text
                        print(f"      + {img_name}: [FORMULA] {caption[:60]}...")
                    else:
                        caption = utils.clean_caption(out_text)
                        caption = utils.shorten_logo_caption(caption)
                        print(f"      + {img_name}: {caption[:80]}...")
                    
                    caption_cache[img_name] = caption

                caption = caption_cache[img_name]
                if not caption:
                    continue

                if caption.lstrip().startswith("$"):
                    first_occurrence = f"\n\n{caption}\n\n"
                    later_occurrence = first_occurrence
                else:
                    first_occurrence = f"\n> **[Mô tả ảnh AI]** ({img_name}): {caption}\n"
                    later_occurrence = f"\n> **[Ảnh]** ({img_name}): (đã mô tả ở trên)\n"

                full_text = utils.replace_md_image_refs(full_text, img_name, first=first_occurrence, later=later_occurrence)

            except Exception as e:
                print(f"      ! Warning: Failed to process image {img_name}: {e}")

        return full_text

    def parse(self, pdf_path: str) -> Optional[ParsedOutput]:
        print(f"🚀 Starting processing for: {Path(pdf_path).name}")
        torch.cuda.empty_cache()
        gc.collect()

        raw_md, images = self._extract_structure(pdf_path)
        if raw_md is None:
            return None

        torch.cuda.empty_cache()
        gc.collect()

        final_md = self._enrich_content(raw_md, images)
        final_md = utils.clean_garbage_text(final_md)

        print(f"✅ Finished: {Path(pdf_path).name}")
        
        # Trả về ParsedOutput khớp với models.py
        return ParsedOutput(
            source_path=pdf_path,
            raw_markdown=raw_md,
            enriched_markdown=final_md,
            images=images if images else {},
            metadata={"processed_by": "marker+qwen2vl"}
        )

def process_all_pdfs_in_directory(root_folder: str, parser: PDFParser):
    pdf_files = list(Path(root_folder).rglob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in: {root_folder}")
        return

    print(f"🔍 Found {len(pdf_files)} PDF files.")
    print("=" * 60)

    for idx, pdf_path in enumerate(pdf_files, 1):
        pdf_path_str = str(pdf_path)
        output_path = pdf_path_str.replace(".pdf", "_parser_output.md")

        if os.path.exists(output_path):
            print(f"[{idx}/{len(pdf_files)}] ⏭️ SKIPPING: {pdf_path.name}")
            continue

        try:
            result = parser.parse(pdf_path_str)
            if result and result.enriched_markdown:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.enriched_markdown)
                print(f"   -> ✅ Saved: {Path(output_path).name}\n")
        except Exception as e:
            print(f"   ❌ ERROR on {pdf_path.name}: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    PDF_ROOT_FOLDER = "D:\\Document\\Intern\\DocumentParsers\\PDFParser\\raw documents\\raw documents"
    
    if not Path(PDF_ROOT_FOLDER).exists():
        print(f"Error: Folder not found: '{PDF_ROOT_FOLDER}'")
    else:
        pdf_parser = PDFParser()
        process_all_pdfs_in_directory(PDF_ROOT_FOLDER, pdf_parser)