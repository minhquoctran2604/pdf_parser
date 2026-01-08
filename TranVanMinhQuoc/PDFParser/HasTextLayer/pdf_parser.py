import torch
import gc
import os
from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional
from PIL import Image as PILImage
from ..Shared import MarkerConverter, QwenCaptioner

from models import ParsedOutput
import utils


class PDFParser:
    def __init__(self, marker_converter: MarkerConverter, qwen_captioner: QwenCaptioner):
        self._marker = marker_converter
        self._qwen_model = qwen_captioner.model
        self._qwen_processor = qwen_captioner.processor
        self._device = qwen_captioner.device
        
        if self._qwen_model is not None:
            self._device = next(self._qwen_model.parameters()).device
            
    def _extract_structure(self, pdf_path: str) -> Tuple[str, Dict[str, PILImage.Image]]:
        print(f"-> [Step 1] Analyzing PDF structure with Marker: {Path(pdf_path).name}")
        try:
            rendered = self._marker.convert(pdf_path)
            print(f"      - Extracted {len(rendered.images)} images/tables.")
            return rendered.markdown, rendered.images
        except Exception as e:
            print(f" ❌ ERROR in structure extraction: {e}")
            return None, None

    def _enrich_content(self, full_text: str, images: Dict[str, PILImage.Image]) -> str:
        if not images:
            return full_text

        print(f"   -> [Step 2] Generating descriptions for {len(images)} images with Qwen2-VL...")
        caption_cache = {}

        for img_name, pil_image in images.items():
            try:
                # Skip if image is not referenced in the markdown
                if not re.search(r"!\[[^\]]*\]\(" + re.escape(img_name) + r"\)", full_text):
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
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    text = self._qwen_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._qwen_processor(
                        text=[text],
                        images=[pil_image.convert("RGB")],
                        padding=True,
                        return_tensors="pt",
                    ).to(self._device)

                    with torch.no_grad():
                        generated_ids = self._qwen_model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            num_beams=3,
                            repetition_penalty=1.12,
                            no_repeat_ngram_size=4,
                        )

                    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
                    out_text = self._qwen_processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()

                    is_latex = out_text.strip().startswith("$") or bool(
                        re.search(
                            r"\\(frac|sum|int|sqrt|begin\{|[a-zA-Z]+\s*_\{)", out_text
                        )
                    )

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
                    first_occurrence = (
                        f"\n> **[Mô tả ảnh AI]** ({img_name}): {caption}\n"
                    )
                    later_occurrence = (
                        f"\n> **[Ảnh]** ({img_name}): (đã mô tả ở trên)\n"
                    )

                full_text = utils.replace_md_image_refs(
                    full_text, img_name, first=first_occurrence, later=later_occurrence
                )

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

        return ParsedOutput(
            source_path=pdf_path,
            raw_markdown=raw_md,
            enriched_markdown=final_md,
            images=images if images else {},
            metadata={"processed_by": "marker+qwen2vl"},
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


if __name__ == "__main__":
    PDF_ROOT_FOLDER = ("D:\\Document\\Intern\\DocumentParsers\\PDFParser\\raw documents\\raw documents")

    if not Path(PDF_ROOT_FOLDER).exists():
        print(f"Error: Folder not found: '{PDF_ROOT_FOLDER}'")
        
    print("Loading Marker...")
    marker = MarkerConverter()
    
    print("Loading Qwen2-VL model...")
    qwen_captioner = QwenCaptioner()
    
    pdf_parser = PDFParser(marker_converter=marker, qwen_captioner=qwen_captioner)
    process_all_pdfs_in_directory(PDF_ROOT_FOLDER, pdf_parser)
