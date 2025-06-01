import json
from pathlib import Path

from rdflib import Graph, URIRef, Literal, RDF, RDFS
from rich.console import Console

from .config.config import Config

console = Console()


class OWLGenerator:
    def __init__(self, config: Config, input_path, output_file_name="ontology.owl"):
        """
        Initialize OWL generator
        :param input_path: Path to input JSON file
        :param config: Configuration object
        :param output_file_name: Generated OWL filename
        """
        self.input_path = input_path
        self.output_file_name = output_file_name
        self.attributes_file = f"{input_path}/inferred_attributes.json"
        self.relations_file = f"{input_path}/merged_relations.json"
        self.clusters_file = f"{input_path}/clustered_entities.json"
        self.config = config

        self.graph = Graph()

        # OWL namespace - get from configuration
        self.owl_ns = config.owl_namespace if config else "https://example.com/"

    def load_json(self, file_path):
        """Load JSON file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {file_path}[/]")
            raise
        except json.JSONDecodeError:
            console.print(f"[red]Error: JSON format error: {file_path}[/]")
            raise

    def generate_ontology(self):
        """Generate OWL ontology"""
        attributes_data = self.load_json(self.attributes_file)
        relations_data = self.load_json(self.relations_file)
        clusters_data = self.load_json(self.clusters_file)

        entity_names = {entity["name"] for entity in attributes_data}

        # 1️⃣ Create Cluster parent class
        cluster_classes = {}
        for entity in clusters_data:
            cluster_id = entity["cluster"]
            cluster_name = entity["cluster_name"]
            cluster_uri = URIRef(f"{self.owl_ns}{cluster_name}")

            if (
                cluster_name
                and cluster_name != "None"
                and cluster_name not in cluster_classes
            ):
                self.graph.add(
                    (
                        cluster_uri,
                        RDF.type,
                        URIRef("http://www.w3.org/2002/07/owl#Class"),
                    )
                )
                self.graph.add((cluster_uri, RDFS.label, Literal(cluster_name)))
                cluster_classes[cluster_name] = cluster_uri

        # 2️⃣ Process entities (Class)
        entity_map = {}
        for entity in attributes_data:
            entity_name = entity["name"].replace(" ", "_")
            entity_uri = URIRef(f"{self.owl_ns}{entity_name}")
            entity_map[entity["name"]] = entity_uri

            # Set as OWL class
            self.graph.add(
                (entity_uri, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
            )
            self.graph.add((entity_uri, RDFS.label, Literal(entity["name"])))

            # Add description
            if "description" in entity:
                self.graph.add(
                    (entity_uri, RDFS.comment, Literal(entity["description"]))
                )

            cluster_info = next(
                (e for e in clusters_data if e["title"] == entity["name"]), None
            )
            cluster_name = cluster_info.get("cluster_name") if cluster_info else None

            if cluster_name and cluster_name != "None":
                parent_class_uri = cluster_classes.get(cluster_name)

                if parent_class_uri:
                    self.graph.add((entity_uri, RDFS.subClassOf, parent_class_uri))

            # 3️⃣ Process attributes (DatatypeProperty)
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

        # 4️⃣ Process relationships (ObjectProperty)
        relations_data = relations_data["relations"]
        valid_relations = [
            relation
            for relation in relations_data
            if relation["new_source"] in entity_names
            and relation["new_target"] in entity_names
        ]

        console.print(
            f"[blue]Processing {len(valid_relations)} valid relationships (out of {len(relations_data)} total relationships)[/]"
        )

        for relation in valid_relations:
            source = relation["new_source"].replace(" ", "_")
            target = relation["new_target"].replace(" ", "_")
            relation_name = relation["relation"].replace(" ", "_")

            source_uri = entity_map.get(
                relation["new_source"], URIRef(f"{self.owl_ns}{source}")
            )
            target_uri = entity_map.get(
                relation["new_target"], URIRef(f"{self.owl_ns}{target}")
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

            # Add description
            if "description" in relation:
                self.graph.add(
                    (relation_uri, RDFS.comment, Literal(relation["description"]))
                )

    def save_ontology(self):
        """Save OWL file"""
        output_path = Path(self.input_path) / self.output_file_name
        self.graph.serialize(str(output_path), format="xml")
        console.print(f"[green]✓ OWL file saved: {output_path}[/]")

    def run(self):
        """Execute the entire OWL generation process"""
        console.print("[blue]Starting OWL ontology generation...[/]")
        self.generate_ontology()
        self.save_ontology()
