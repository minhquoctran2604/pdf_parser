import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class QwenCaptioner:
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
    
    def load(self) -> None:
        if self._model is not None:
            return
        
        print(f"   Loading Qwen2-VL model: {self.model_name}...")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._device = next(self._model.parameters()).device
        print(f"   ✅ Qwen2-VL loaded!")
    
    @property
    def model(self):
        """Trả về model, tự động load nếu chưa load"""
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def processor(self):
        if self._processor is None:
            self.load()
        return self._processor
    
    @property
    def device(self):
        if self._device is None:
            self.load()
        return self._device