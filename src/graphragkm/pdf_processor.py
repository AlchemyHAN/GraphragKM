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
        Initialize PDF processor

        Args:
            config_path: Configuration file path, defaults to config.yaml in the current directory
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config.yaml"

        self.config = Config.from_yaml(str(config_path))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.mineru_token}",
        }

    def process_pdf(self, file_path: str, output_dir: str) -> None:
        """Main function for processing PDF files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        console.print(
            f"[blue]Starting PDF processing: {os.path.basename(file_path)}[/]"
        )

        batch_id = self._request_upload_url(file_path)
        if not batch_id:
            return

        if not self._wait_for_processing(batch_id):
            return

        self._download_and_extract_results(batch_id, output_dir)
        console.print(
            f"[green]✓ PDF processing completed, results saved to: {output_dir}[/]"
        )

    def _request_upload_url(self, file_path: str) -> Optional[str]:
        """Request upload URL"""
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
            console.print("[blue]Requesting upload URL...[/]")
            response = requests.post(
                self.config.mineru_upload_url,
                headers=self.headers,
                json=upload_request_data,
            )
            result = response.json()

            if response.status_code != 200 or result["code"] != 0:
                raise Exception(
                    f"Failed to request upload URL: {result.get('msg', 'Unknown error')}"
                )

            batch_id = result["data"]["batch_id"]
            file_url = result["data"]["file_urls"][0]

            # Upload PDF file
            console.print("[blue]Uploading PDF file...[/]")
            with open(file_path, "rb") as f:
                res_upload = requests.put(file_url, data=f)

            if res_upload.status_code != 200:
                raise Exception(f"File upload failed: {res_upload.status_code}")

            console.print("[green]✓ File upload successful[/]")
            return batch_id

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            return None

    def _wait_for_processing(self, batch_id: str) -> bool:
        """Wait for processing to complete"""
        result_url = self.config.mineru_results_url_template.format(batch_id)
        console.print("[blue]Waiting for server to process file...[/]")

        while True:
            try:
                response = requests.get(result_url, headers=self.headers)
                if response.status_code != 200:
                    raise Exception(f"Failed to query status: {response.status_code}")

                result = response.json()
                extract_results = result["data"]["extract_result"]

                if self._check_processing_complete(extract_results):
                    console.print("[green]✓ Server processing completed[/]")
                    return True

            except Exception as e:
                console.print(f"[red]Error: Failed to query processing status: {e}[/]")
                return False

            time.sleep(5)

    def _check_processing_complete(self, extract_results: list) -> bool:
        """Check if processing is completed"""
        for result in extract_results:
            state = result.get("state", "unknown")
            if state == "failed":
                console.print(
                    f"[red]Error: Processing failed: {result.get('err_msg', 'Unknown error')}[/]"
                )
                return False
            elif state != "done":
                return False
        return True

    def _download_and_extract_results(self, batch_id: str, output_dir: str) -> None:
        """Download and extract results"""
        result_url = self.config.mineru_results_url_template.format(batch_id)
        console.print("[blue]Retrieving processing results...[/]")
        response = requests.get(result_url, headers=self.headers)
        result = response.json()

        file_url = result["data"]["extract_result"][0].get("full_zip_url")
        if not file_url:
            console.print("[red]Error: Download URL not found[/]")
            return

        output_zip_path = Path(output_dir) / "result.zip"

        # Download file
        console.print("[blue]Downloading result file...[/]")
        response = requests.get(file_url, headers=self.headers, stream=True)
        if response.status_code == 200:
            with open(output_zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract file
            console.print("[blue]Extracting file...[/]")
            with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # Delete zip file
            output_zip_path.unlink()
            console.print("[green]✓ Result file extraction completed[/]")
        else:
            console.print(
                f"[red]Error: Download failed, status code: {response.status_code}[/]"
            )
