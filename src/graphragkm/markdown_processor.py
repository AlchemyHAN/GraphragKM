import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from rich.console import Console

# 创建控制台实例
console = Console()


class MarkdownProcessor:
    def __init__(self, base_path: str):
        """
        初始化Markdown处理器

        Args:
            base_path: Markdown文件所在的基础路径
        """
        from easyocr import Reader

        self.base_path = Path(base_path)
        console.print("[blue]正在初始化OCR引擎...[/]")
        self.reader = Reader(["ch_sim", "en"], gpu=False)

    def process_markdown_file(self, input_path: str, output_path: str) -> None:
        """处理Markdown文件"""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"找不到文件: {input_path}")

        content = input_path.read_text(encoding="utf-8")
        processed_content = self.replace_images_with_text(content)

        Path(output_path).write_text(processed_content, encoding="utf-8")
        console.print(f"[green]✓ Markdown文件处理完成: {output_path}[/]")

    def replace_images_with_text(self, content: str) -> str:
        """替换图片为文字"""
        # 处理Markdown格式的图片
        content = re.sub(
            r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)(?P<caption>.*?)(?=\n|$)",
            self._process_markdown_image,
            content,
        )

        # 处理HTML格式的图片
        content = re.sub(
            r'<img\s+[^>]*?src=["\']([^"\']+)["\'][^>]*>',
            lambda m: self._process_html_image(m.group(1)),
            content,
        )

        return content

    def _process_markdown_image(self, match) -> str:
        """处理Markdown格式的图片匹配"""
        alt = match.group("alt")
        img_path = match.group("path")
        caption = match.group("caption")

        # 获取图片中的文字
        extracted_text = self._extract_text_from_image(img_path)

        if not extracted_text:
            # 如果没有提取到文字，保留原始图片标记
            console.print(f"[yellow]警告: 无法从图片提取文字: {img_path}[/]")
            return f"![{alt}]({img_path}){caption}"

        # 构建新的文本块，包含原始图片信息和提取的文字
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
        """处理HTML格式的图片"""
        extracted_text = self._extract_text_from_image(img_path)
        if not extracted_text:
            console.print(f"[yellow]警告: 无法从HTML图片提取文字: {img_path}[/]")
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
        """从图片中提取文字"""
        full_path = self.base_path / img_path
        if not full_path.exists():
            console.print(f"[yellow]警告: 找不到图片文件: {full_path}[/]")
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
            console.print(f"[red]错误: 处理图片失败 {img_path}: {str(e)}[/]")
            return None
