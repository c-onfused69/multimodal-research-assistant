"""Layout-aware PDF/DOCX parsing via Docling.

Emits three element streams: text blocks, tables, figures — matching
the multimodal ingestion flow diagram.
"""
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter

from ingestion.connectors.base import RawDocument


@dataclass
class ParsedElement:
    element_type: str            # text | table | figure
    content: str                 # text / markdown table / figure placeholder
    page: int | None = None
    section: str | None = None
    structured: dict | None = None      # tables: structured JSON
    image_bytes: bytes | None = None    # figures: raw image for VLM/ColPali
    metadata: dict = field(default_factory=dict)


class DoclingParser:
    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, doc: RawDocument) -> list[ParsedElement]:
        result = self._converter.convert(doc.source_uri)
        dl = result.document
        elements: list[ParsedElement] = []
        current_section = None

        for item, _level in dl.iterate_items():
            label = getattr(item, "label", "")
            page = self._page_of(item)

            if label in ("section_header", "title"):
                current_section = item.text
                elements.append(ParsedElement("text", item.text, page, current_section))
            elif label == "table":
                md = item.export_to_markdown(dl)
                structured = item.export_to_dataframe(dl).to_dict(orient="records")
                elements.append(ParsedElement(
                    "table", md, page, current_section, structured=structured))
            elif label in ("picture", "figure"):
                img = item.get_image(dl)
                img_bytes = None
                if img is not None:
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                elements.append(ParsedElement(
                    "figure", f"[FIGURE p.{page}]", page, current_section,
                    image_bytes=img_bytes))
            elif getattr(item, "text", "").strip():
                elements.append(ParsedElement("text", item.text, page, current_section))

        return elements

    @staticmethod
    def _page_of(item) -> int | None:
        prov = getattr(item, "prov", None)
        if prov:
            return prov[0].page_no
        return None
