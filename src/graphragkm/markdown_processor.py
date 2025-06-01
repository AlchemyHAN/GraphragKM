import re
from pathlib import Path
from typing import Optional

import pytesseract
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

        self.base_path = Path(base_path)
        console.print(
            "[yellow]Info: MarkdownProcessor now uses Tesseract OCR. "
            "Ensure Tesseract is installed and in your PATH, "
            "or set pytesseract.tesseract_cmd.[/]"
        )

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
        """Extract text from image with robustness using Tesseract"""
        full_path = self.base_path / img_path

        if not full_path.exists():
            console.print(f"[yellow]Warning: Image file not found: {full_path}[/]")
            return None

        try:
            console.print(f"[cyan]Processing image with Tesseract: {full_path.name}[/]")

            image = Image.open(full_path).convert("RGB")

            # Resize large image to prevent memory issues and improve OCR
            image.thumbnail(
                (2000, 2000), Image.Resampling.LANCZOS
            )  # Increased size for potentially better OCR

            # Use Tesseract to extract text
            # You might need to specify the language, e.g., lang='eng' for English
            # Add --psm X to specify page segmentation mode if needed
            text = pytesseract.image_to_string(image, lang="eng")

            if not text or text.isspace():
                console.print(
                    f"[yellow]Warning: No text found in {full_path.name} using Tesseract[/]"
                )
                return None

            return text.strip()

        except pytesseract.TesseractNotFoundError:
            console.print(
                "[red]Error: Tesseract is not installed or not in your PATH. "
                "Please install Tesseract and make sure it's accessible."
            )
            return None
        except Exception as e:
            console.print(
                f"[red]Error processing image {img_path} with Tesseract: {e}[/]"
            )
            return None
