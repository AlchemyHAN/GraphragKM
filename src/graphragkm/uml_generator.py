import json
import os
import re

from rich.console import Console

# Create console instance
console = Console()


class PlantUMLGenerator:
    def __init__(self, input_path, output_file_name="uml_model.puml"):
        """
        Initialize PlantUML generator
        :param input_path: Path to input JSON file
        :param output_file_name: Generated UML model filename (puml format)
        """
        self.input_path = input_path
        self.output_file_name = output_file_name
        self.attributes_file = os.path.join(input_path, "inferred_attributes.json")
        self.relations_file = os.path.join(input_path, "inferred_relations.json")
        self.clusters_file = os.path.join(input_path, "clustered_entities.json")

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

    def safe_name(self, name):
        """Convert class name to safe format (remove special characters, only retain letters, numbers and underscores)"""
        return re.sub(r"\W+", "_", name)

    def generate_puml(self):
        """
        Generate PlantUML format text:
        1. Categorize entities into packages based on clustered_entities.json;
        2. Generate UML classes with attributes and comments based on inferred_attributes.json;
        3. Generate associations between classes based on inferred_relations.json.
        """
        attributes_data = self.load_json(self.attributes_file)
        relations_data = self.load_json(self.relations_file)
        clusters_data = self.load_json(self.clusters_file)

        entity_names = {entity["name"] for entity in attributes_data}

        # Save entity information: use safe name as key
        entity_definitions = {}
        # Package mapping: package name -> list of contained entities (safe names)
        package_map = {}

        # Iterate through all entities (attributes_data)
        for entity in attributes_data:
            original_name = entity["name"]
            safe_name = self.safe_name(original_name)  # Ensure class name is safe
            description = entity.get("description", "")
            attributes = entity.get(
                "attr", {}
            )  # Attributes as dictionary {attribute name: type}

            # Determine if the entity belongs to a package via clusters_data
            package_name = None
            for cluster in clusters_data:
                if cluster.get("title") == original_name:
                    package_name = cluster.get("cluster_name")
                    break

            # Save entity definition information
            entity_definitions[safe_name] = {
                "original_name": original_name,
                "description": description,
                "attributes": attributes,
                "package": package_name,
            }

            if package_name:
                package_map.setdefault(package_name, []).append(safe_name)

        # Start generating puml content
        console.print("[blue]Generating PlantUML model...[/]")
        lines = ["@startuml"]
        lines.append("skinparam classAttributeIconSize 0")

        # 1. Generate classes in packages
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

        # 2. Generate classes not in any package (top-level classes)
        for safe_name, entity in entity_definitions.items():
            if safe_name in defined_in_package:
                continue
            if entity["description"]:
                lines.append(f'// {entity["description"]}')
            lines.append(f"class {safe_name} {{")
            for attr, attr_type in entity["attributes"].items():
                lines.append(f"  + {attr} : {attr_type}")
            lines.append("}")

        # 3. Generate associations between classes
        console.print("[blue]Processing entity relationships...[/]")

        # Filter valid relationships
        valid_relations = [
            relation
            for relation in relations_data
            if relation["source"] in entity_names and relation["target"] in entity_names
        ]

        console.print(
            f"[blue]Processing {len(valid_relations)} valid relationships (out of {len(relations_data)} total relationships)[/]"
        )

        for relation in valid_relations:
            source_safe = self.safe_name(relation["source"])
            target_safe = self.safe_name(relation["target"])
            relation_name = relation["relation"]
            lines.append(f"{source_safe} --> {target_safe} : {relation_name}")

        lines.append("@enduml")
        return "\n".join(lines)

    def save_puml(self):
        """Save the generated puml text to file"""
        puml_text = self.generate_puml()
        output_path = os.path.join(self.input_path, self.output_file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(puml_text)
        console.print(f"[green]✓ PlantUML file saved: {output_path}[/]")

    def run(self):
        """Execute the entire PlantUML model generation process"""
        console.print("[blue]Starting PlantUML model generation...[/]")
        self.save_puml()
        console.print("[green]✓ PlantUML model generation completed![/]")
