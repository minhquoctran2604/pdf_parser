import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


class MarkerConverter:
    def __init__(self):
        self._model_dict = None
        self._converter = None

    def load(self, model_name=None, device="cpu"):
        if self._converter:
            return
        if model_name:
            self._model_dict = create_model_dict(model_name, device)
        else:
            self._model_dict = create_model_dict()
        self._converter = PdfConverter(self._model_dict)
        print("Model loaded successfully.")

    @property
    def converter(self):
        if self._converter is None:
            self.load()
        return self._converter

    @property
    def model_dict(self):
        if self._model_dict is None:
            self.load()
        return self._model_dict

    def convert(self, pdf_path):
        if self._converter is None:
            self.load()
        return self._converter(pdf_path)
