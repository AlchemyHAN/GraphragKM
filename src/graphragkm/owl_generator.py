import json
from pathlib import Path

from rdflib import Graph, URIRef, Literal, RDF, RDFS
from rich.console import Console

from .config.config import Config

console = Console()


class OWLGenerator:
    def __init__(self, config: Config, input_path, output_file_name="ontology.owl"):
        """
        初始化 OWL 生成器
        :param input_path: 输入 JSON 文件的路径
        :param config: 配置对象
        :param output_file_name: 生成的 OWL 文件名
        """
        self.input_path = input_path
        self.output_file_name = output_file_name
        self.attributes_file = f"{input_path}/inferred_attributes.json"
        self.relations_file = f"{input_path}/inferred_relations.json"
        self.clusters_file = f"{input_path}/clustered_entities.json"
        self.config = config

        self.graph = Graph()

        # OWL 命名空间 - 从配置中获取
        self.owl_ns = config.owl_namespace if config else "https://example.com/"

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

    def generate_ontology(self):
        """生成 OWL 本体"""
        attributes_data = self.load_json(self.attributes_file)
        relations_data = self.load_json(self.relations_file)
        clusters_data = self.load_json(self.clusters_file)

        entity_names = {entity["name"] for entity in attributes_data}

        # 1️⃣ 创建 Cluster 父类
        cluster_classes = {}
        for entity in clusters_data:
            cluster_id = entity["cluster"]
            cluster_name = entity["cluster_name"]
            cluster_uri = URIRef(f"{self.owl_ns}{cluster_name}")

            if cluster_name not in cluster_classes:
                self.graph.add(
                    (
                        cluster_uri,
                        RDF.type,
                        URIRef("http://www.w3.org/2002/07/owl#Class"),
                    )
                )
                self.graph.add((cluster_uri, RDFS.label, Literal(cluster_name)))
                cluster_classes[cluster_name] = cluster_uri

        # 2️⃣ 处理实体（Class）
        entity_map = {}
        for entity in attributes_data:
            entity_name = entity["name"].replace(" ", "_")
            entity_uri = URIRef(f"{self.owl_ns}{entity_name}")
            entity_map[entity["name"]] = entity_uri

            # 设定为 OWL 类
            self.graph.add(
                (entity_uri, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
            )
            self.graph.add((entity_uri, RDFS.label, Literal(entity["name"])))

            # 添加描述
            if "description" in entity:
                self.graph.add(
                    (entity_uri, RDFS.comment, Literal(entity["description"]))
                )

            cluster_info = next(
                (e for e in clusters_data if e["title"] == entity["name"]), None
            )
            if cluster_info:
                cluster_name = cluster_info["cluster_name"]
                if cluster_name in cluster_classes:
                    self.graph.add(
                        (entity_uri, RDFS.subClassOf, cluster_classes[cluster_name])
                    )

            # 3️⃣ 处理属性（DatatypeProperty）
            for attr, attr_type in entity["attr"].items():
                attr_uri = URIRef(f"{self.owl_ns}{attr.replace(' ', '_')}")
                self.graph.add(
                    (
                        attr_uri,
                        RDF.type,
                        URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"),
                    )
                )
                self.graph.add((attr_uri, RDFS.domain, entity_uri))
                self.graph.add(
                    (
                        attr_uri,
                        RDFS.range,
                        URIRef(f"http://www.w3.org/2001/XMLSchema#{attr_type}"),
                    )
                )
                self.graph.add((attr_uri, RDFS.label, Literal(attr)))

        # 4️⃣ 处理关系（ObjectProperty）
        valid_relations = [
            relation
            for relation in relations_data
            if relation["source"] in entity_names and relation["target"] in entity_names
        ]

        console.print(
            f"[blue]处理 {len(valid_relations)} 个有效关系（共 {len(relations_data)} 个关系）[/]"
        )

        for relation in valid_relations:
            source = relation["source"].replace(" ", "_")
            target = relation["target"].replace(" ", "_")
            relation_name = relation["relation"].replace(" ", "_")

            source_uri = entity_map.get(
                relation["source"], URIRef(f"{self.owl_ns}{source}")
            )
            target_uri = entity_map.get(
                relation["target"], URIRef(f"{self.owl_ns}{target}")
            )
            relation_uri = URIRef(f"{self.owl_ns}{relation_name}")

            self.graph.add(
                (
                    relation_uri,
                    RDF.type,
                    URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
                )
            )
            self.graph.add((relation_uri, RDFS.domain, source_uri))
            self.graph.add((relation_uri, RDFS.range, target_uri))
            self.graph.add((relation_uri, RDFS.label, Literal(relation_name)))

            # 添加描述
            if "description" in relation:
                self.graph.add(
                    (relation_uri, RDFS.comment, Literal(relation["description"]))
                )

    def save_ontology(self):
        """保存 OWL 文件"""
        output_path = Path(self.input_path) / self.output_file_name
        self.graph.serialize(str(output_path), format="xml")
        console.print(f"[green]✓ OWL 文件已保存: {output_path}[/]")

    def run(self):
        """执行 OWL 生成全过程"""
        console.print("[blue]开始生成 OWL 本体...[/]")
        self.generate_ontology()
        self.save_ontology()
