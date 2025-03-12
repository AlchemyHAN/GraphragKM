"""
配置模块
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

import yaml


@dataclass
class Config:
    """配置类"""

    # Mineru API settings
    mineru_upload_url: str
    mineru_results_url_template: str
    mineru_token: str

    # Chat model settings
    chat_model_api_key: str
    chat_model_api_base: str
    chat_model_name: str

    # Embedding model settings
    embedding_model_api_key: str
    embedding_model_api_base: str
    embedding_model_name: str

    # OWL settings
    owl_namespace: str

    # App settings
    max_concurrent_requests: int

    doc_language: str

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """从YAML文件加载配置"""
        # 确保config_path是Path对象
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        api_config = config.get("api", {})
        app_config = config.get("app", {})

        return cls(
            # Mineru API settings
            mineru_upload_url=api_config.get("mineru_upload_url", ""),
            mineru_results_url_template=api_config.get(
                "mineru_results_url_template", ""
            ),
            mineru_token=api_config.get("mineru_token", ""),
            # Chat model settings
            chat_model_api_key=api_config.get("chat_model_api_key", ""),
            chat_model_api_base=api_config.get("chat_model_api_base", ""),
            chat_model_name=api_config.get("chat_model_name", ""),
            # Embedding model settings
            embedding_model_api_key=api_config.get("embedding_model_api_key", ""),
            embedding_model_api_base=api_config.get("embedding_model_api_base", ""),
            embedding_model_name=api_config.get("embedding_model_name", ""),
            # OWL settings
            owl_namespace=app_config.get("owl_namespace", "https://example.com/"),
            # App settings
            max_concurrent_requests=app_config.get("max_concurrent_requests", 25),
            doc_language=app_config.get("doc_language", "en"),
        )

    @property
    def required_fields(self) -> list[str]:
        """获取所有必需的配置字段"""
        return [
            "mineru_upload_url",
            "mineru_results_url_template",
            "mineru_token",
            "chat_model_api_key",
            "chat_model_api_base",
            "chat_model_name",
            "embedding_model_api_key",
            "embedding_model_api_base",
            "embedding_model_name",
            "owl_namespace",
            "max_concurrent_requests",
            "doc_language",
        ]

    def validate(self) -> Tuple[bool, Optional[str]]:
        """验证配置是否完整

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        for field in self.required_fields:
            if not getattr(self, field, None):
                return False, f"缺少必需的配置项: {field}"
        return True, None
