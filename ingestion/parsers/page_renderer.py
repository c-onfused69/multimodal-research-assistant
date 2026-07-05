"""Converts document pages to base64 images for ColPali visual embeddings."""
import pypdfium2 as pdfium


class PageRenderer:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def render_pdf(self, file_path: str) -> list[bytes]:
        pdf = pdfium.PdfDocument(file_path)
        images = []
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            bitmap = page.render(scale=self.dpi / 72.0)
            pil_img = bitmap.to_pil()
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            images.append(buf.getvalue())
        return images
