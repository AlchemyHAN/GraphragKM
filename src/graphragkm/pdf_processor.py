import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console

from .config.config import Config

console = Console()


class PDFProcessor:
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化PDF处理器

        Args:
            config_path: 配置文件路径，默认为当前目录下的config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config.yaml"

        self.config = Config.from_yaml(str(config_path))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.mineru_token}",
        }

    def process_pdf(self, file_path: str, output_dir: str) -> None:
        """处理PDF文件的主函数"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]开始处理PDF文件: {os.path.basename(file_path)}[/]")

        batch_id = self._request_upload_url(file_path)
        if not batch_id:
            return

        if not self._wait_for_processing(batch_id):
            return

        self._download_and_extract_results(batch_id, output_dir)
        console.print(f"[green]✓ PDF处理完成，结果保存在: {output_dir}[/]")

    def _request_upload_url(self, file_path: str) -> Optional[str]:
        """请求上传URL"""
        upload_request_data = {
            "enable_formula": True,
            "language": self.config.doc_language,
            "layout_model": "doclayout_yolo",
            "enable_table": True,
            "files": [
                {"name": os.path.basename(file_path), "is_ocr": True, "data_id": "abcd"}
            ],
        }

        try:
            console.print("[blue]正在请求上传URL...[/]")
            response = requests.post(
                self.config.mineru_upload_url,
                headers=self.headers,
                json=upload_request_data,
            )
            result = response.json()

            if response.status_code != 200 or result["code"] != 0:
                raise Exception(f"申请上传URL失败: {result.get('msg', '未知错误')}")

            batch_id = result["data"]["batch_id"]
            file_url = result["data"]["file_urls"][0]

            # 上传PDF文件
            console.print("[blue]正在上传PDF文件...[/]")
            with open(file_path, "rb") as f:
                res_upload = requests.put(file_url, data=f)

            if res_upload.status_code != 200:
                raise Exception(f"文件上传失败: {res_upload.status_code}")

            console.print("[green]✓ 文件上传成功[/]")
            return batch_id

        except Exception as e:
            console.print(f"[red]错误: {e}[/]")
            return None

    def _wait_for_processing(self, batch_id: str) -> bool:
        """等待处理完成"""
        result_url = self.config.mineru_results_url_template.format(batch_id)
        console.print("[blue]正在等待服务器处理文件...[/]")

        while True:
            try:
                response = requests.get(result_url, headers=self.headers)
                if response.status_code != 200:
                    raise Exception(f"查询状态失败: {response.status_code}")

                result = response.json()
                extract_results = result["data"]["extract_result"]

                if self._check_processing_complete(extract_results):
                    console.print("[green]✓ 服务器处理完成[/]")
                    return True

            except Exception as e:
                console.print(f"[red]错误: 查询处理状态时出错: {e}[/]")
                return False

            time.sleep(5)

    def _check_processing_complete(self, extract_results: list) -> bool:
        """检查处理是否完成"""
        for result in extract_results:
            state = result.get("state", "unknown")
            if state == "failed":
                console.print(
                    f"[red]错误: 处理失败: {result.get('err_msg', '未知错误')}[/]"
                )
                return False
            elif state != "done":
                return False
        return True

    def _download_and_extract_results(self, batch_id: str, output_dir: str) -> None:
        """下载并解压结果"""
        result_url = self.config.mineru_results_url_template.format(batch_id)
        console.print("[blue]正在获取处理结果...[/]")
        response = requests.get(result_url, headers=self.headers)
        result = response.json()

        file_url = result["data"]["extract_result"][0].get("full_zip_url")
        if not file_url:
            console.print("[red]错误: 未找到下载URL[/]")
            return

        output_zip_path = Path(output_dir) / "result.zip"

        # 下载文件
        console.print("[blue]正在下载结果文件...[/]")
        response = requests.get(file_url, headers=self.headers, stream=True)
        if response.status_code == 200:
            with open(output_zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 解压文件
            console.print("[blue]正在解压文件...[/]")
            with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # 删除zip文件
            output_zip_path.unlink()
            console.print("[green]✓ 结果文件解压完成[/]")
        else:
            console.print(f"[red]错误: 下载失败，状态码: {response.status_code}[/]")
