import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from rich.console import Console

# Create console instance
console = Console()


class MarkdownProcessor:
    def __init__(self, base_path: str):
        """
        Initialize Markdown processor

        Args:
            base_path: Base path where Markdown files are located
        """
        from easyocr import Reader

        self.base_path = Path(base_path)
        console.print("[blue]Initializing OCR engine...[/]")
        self.reader = Reader(["ch_sim", "en"], gpu=False)

    def process_markdown_file(self, input_path: str, output_path: str) -> None:
        """Process Markdown file"""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        content = input_path.read_text(encoding="utf-8")
        processed_content = self.replace_images_with_text(content)

        Path(output_path).write_text(processed_content, encoding="utf-8")
        console.print(f"[green]✓ Markdown file processing completed: {output_path}[/]")

    def replace_images_with_text(self, content: str) -> str:
        """Replace images with text"""
        # Process Markdown format images
        content = re.sub(
            r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)(?P<caption>.*?)(?=\n|$)",
            self._process_markdown_image,
            content,
        )

        # Process HTML format images
        content = re.sub(
            r'<img\s+[^>]*?src=["\']([^"\']+)["\'][^>]*>',
            lambda m: self._process_html_image(m.group(1)),
            content,
        )

        return content

    def _process_markdown_image(self, match) -> str:
        """Process Markdown format image match"""
        alt = match.group("alt")
        img_path = match.group("path")
        caption = match.group("caption")

        # Get text from image
        extracted_text = self._extract_text_from_image(img_path)

        if not extracted_text:
            # If no text is extracted, keep the original image tag
            console.print(
                f"[yellow]Warning: Unable to extract text from image: {img_path}[/]"
            )
            return f"![{alt}]({img_path}){caption}"

        # Build new text block, including original image info and extracted text
        result = [
            f"<!-- Original picture: ![{alt}]({img_path}){caption} -->",
            "",
            "```",
            f"Picture description: {alt if alt else 'None'}",
        ]

        if caption:
            result.append(f"Caption: {caption.strip()}")

        result.extend(["Extracted text:", extracted_text, "```", ""])

        return "\n".join(result)

    def _process_html_image(self, img_path: str) -> str:
        """Process HTML format image"""
        extracted_text = self._extract_text_from_image(img_path)
        if not extracted_text:
            console.print(
                f"[yellow]Warning: Unable to extract text from HTML image: {img_path}[/]"
            )
            return f'<img src="{img_path}">'

        return f"""
        <!-- Original picture: <img src="{img_path}"> -->
        <div class="image-text-block">
        <details>
        <summary>Extracted text</summary>

        {extracted_text}
        </details>
        </div>
        """

    def _extract_text_from_image(self, img_path: str) -> Optional[str]:
        """Extract text from image"""
        full_path = self.base_path / img_path
        if not full_path.exists():
            console.print(f"[yellow]Warning: Image file not found: {full_path}[/]")
            return None

        try:
            image = Image.open(full_path)
            image = np.array(image)
            result = self.reader.readtext(image)

            if not result:
                return None

            text = ""
            for detection in result:
                text += detection[1] + "\n"

            return text.strip()
        except Exception as e:
            console.print(
                f"[red]Error: Failed to process image {img_path}: {str(e)}[/]"
            )
            return None
