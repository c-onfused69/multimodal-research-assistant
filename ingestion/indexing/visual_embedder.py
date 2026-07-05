"""ColPali late-interaction visual embeddings for page screenshots."""
import torch

from config.settings import settings


class VisualEmbedder:
    _model = None
    _processor = None

    @classmethod
    def load(cls):
        if cls._model is None:
            from colpali_engine.models import ColPali, ColPaliProcessor
            cls._model = ColPali.from_pretrained(
                settings.visual_embedding_model,
                torch_dtype=torch.bfloat16,
                device_map="cpu"  # switch to "cuda" if GPU is available
            ).eval()
            cls._processor = ColPaliProcessor.from_pretrained(
                settings.visual_embedding_model)
        return cls._processor, cls._model

    async def embed_images(self, images: list[bytes]) -> list[list[list[float]]]:
        """Returns multivector representations per image: shape (patches, 128)"""
        import io
        from PIL import Image
        processor, model = self.load()
        pil_imgs = [Image.open(io.BytesIO(b)) for b in images]
        inputs = processor.process_images(pil_imgs).to(model.device)

        with torch.no_grad():
            embs = model(**inputs)
        return embs.tolist()

    async def embed_query(self, query: str) -> list[list[float]]:
        processor, model = self.load()
        inputs = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            embs = model(**inputs)
        return embs[0].tolist()
