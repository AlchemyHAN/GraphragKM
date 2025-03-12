import json
import os
import re

from rich.console import Console

# 创建控制台实例
console = Console()


class PlantUMLGenerator:
    def __init__(self, input_path, output_file_name="uml_model.puml"):
        """
        初始化 PlantUML 生成器
        :param input_path: 输入 JSON 文件的路径
        :param output_file_name: 生成的 UML 模型文件名（puml 格式）
        """
        self.input_path = input_path
        self.output_file_name = output_file_name
        self.attributes_file = os.path.join(input_path, "inferred_attributes.json")
        self.relations_file = os.path.join(input_path, "inferred_relations.json")
        self.clusters_file = os.path.join(input_path, "clustered_entities.json")

    def load_json(self, file_path):
        """加载 JSON 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]错误: 找不到文件: {file_path}[/]")
            raise
        except json.JSONDecodeError:
            console.print(f"[red]错误: JSON 格式错误: {file_path}[/]")
            raise

    def safe_name(self, name):
        """将类名转换为安全格式（去除特殊字符，仅保留字母、数字和下划线）"""
        return re.sub(r"\W+", "_", name)

    def generate_puml(self):
        """
        生成 PlantUML 格式的文本：
        1. 根据 clustered_entities.json 将实体归类到 package 中；
        2. 根据 inferred_attributes.json 生成 UML 类（包含属性及注释）；
        3. 根据 inferred_relations.json 生成类之间的关联关系。
        """
        attributes_data = self.load_json(self.attributes_file)
        relations_data = self.load_json(self.relations_file)
        clusters_data = self.load_json(self.clusters_file)

        entity_names = {entity["name"] for entity in attributes_data}

        # 保存实体信息：使用安全名称作为 key
        entity_definitions = {}
        # package 映射： package 名称 -> 包含的实体（安全名称）列表
        package_map = {}

        # 遍历所有实体（attributes_data）
        for entity in attributes_data:
            original_name = entity["name"]
            safe_name = self.safe_name(original_name)  # 确保类名安全
            description = entity.get("description", "")
            attributes = entity.get("attr", {})  # 属性为字典 {属性名: 类型}

            # 通过 clusters_data 判断该实体是否属于某个 package
            package_name = None
            for cluster in clusters_data:
                if cluster.get("title") == original_name:
                    package_name = cluster.get("cluster_name")
                    break

            # 保存实体定义信息
            entity_definitions[safe_name] = {
                "original_name": original_name,
                "description": description,
                "attributes": attributes,
                "package": package_name,
            }

            if package_name:
                package_map.setdefault(package_name, []).append(safe_name)

        # 开始生成 puml 内容
        console.print("[blue]正在生成 PlantUML 模型...[/]")
        lines = ["@startuml"]
        lines.append("skinparam classAttributeIconSize 0")

        # 1. 生成 package 中的类
        defined_in_package = set()
        for package_name, class_list in package_map.items():
            lines.append(f'package "{package_name}" {{')
            for safe_name in class_list:
                entity = entity_definitions[safe_name]
                if entity["description"]:
                    lines.append(f'  \' {entity["description"]}')
                lines.append(f"  class {safe_name} {{")
                for attr, attr_type in entity["attributes"].items():
                    lines.append(f"    + {attr} : {attr_type}")
                lines.append("  }")
                defined_in_package.add(safe_name)
            lines.append("}")

        # 2. 生成未归入 package 的类（顶层类）
        for safe_name, entity in entity_definitions.items():
            if safe_name in defined_in_package:
                continue
            if entity["description"]:
                lines.append(f'// {entity["description"]}')
            lines.append(f"class {safe_name} {{")
            for attr, attr_type in entity["attributes"].items():
                lines.append(f"  + {attr} : {attr_type}")
            lines.append("}")

        # 3. 生成类之间的关联关系
        console.print("[blue]正在处理实体关系...[/]")

        # 过滤出有效的关系
        valid_relations = [
            relation
            for relation in relations_data
            if relation["source"] in entity_names and relation["target"] in entity_names
        ]

        console.print(
            f"[blue]处理 {len(valid_relations)} 个有效关系（共 {len(relations_data)} 个关系）[/]"
        )

        for relation in valid_relations:
            source_safe = self.safe_name(relation["source"])
            target_safe = self.safe_name(relation["target"])
            relation_name = relation["relation"]
            lines.append(f"{source_safe} --> {target_safe} : {relation_name}")

        lines.append("@enduml")
        return "\n".join(lines)

    def save_puml(self):
        """将生成的 puml 文本保存到文件"""
        puml_text = self.generate_puml()
        output_path = os.path.join(self.input_path, self.output_file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(puml_text)
        console.print(f"[green]✓ PlantUML 文件已保存: {output_path}[/]")

    def run(self):
        """执行 PlantUML 模型生成全过程"""
        console.print("[blue]开始生成 PlantUML 模型...[/]")
        self.save_puml()
        console.print("[green]✓ PlantUML 模型生成完成！[/]")
