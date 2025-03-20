import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openai import AsyncOpenAI
from pandas import DataFrame
from rich.console import Console
from rich.progress import Progress, TaskID
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config.config import Config

console = Console()


class InferenceProcessor:
    def __init__(
        self, config: Config, parquet_input_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the inference processor and load data files.
        """
        if parquet_input_path is None:
            parquet_input_path = Path(__file__).resolve().parents[2] / "output"

        self.config = config
        self.parquet_input_path = Path(parquet_input_path)
        self.entities_path = self.parquet_input_path / "entities.parquet"
        self.relations_path = self.parquet_input_path / "relationships.parquet"

        if not os.path.exists(self.entities_path) or not os.path.exists(
            self.relations_path
        ):
            raise FileNotFoundError(
                "Entities or Relationships parquet files not found."
            )

        # Load data
        console.print("[blue]Loading entities and relationships data...[/]")
        self.entities_df = self._load_entities()
        self.relationships_df = self._load_relationships()
        console.print(
            f"[green]✓ Loaded {len(self.entities_df)} entities and {len(self.relationships_df)} relationships[/]"
        )

        self.chat_client = AsyncOpenAI(
            api_key=self.config.chat_model_api_key,
            base_url=self.config.chat_model_api_base,
            max_retries=5,
        )
        self.embedding_client = AsyncOpenAI(
            api_key=self.config.embedding_model_api_key,
            base_url=self.config.embedding_model_api_base,
            max_retries=5,
        )

    def _load_entities(self) -> DataFrame:
        """Read entity data and clean it"""
        df = pd.read_parquet(self.entities_path)
        df = df[["title", "description"]].copy()
        df["description"] = df["description"].astype(str).apply(self._clean_text)
        df = df[df["description"].str.len() > 0].reset_index(drop=True)
        return df

    def _load_relationships(self) -> DataFrame:
        """Read relationship data"""
        df = pd.read_parquet(self.relations_path)
        df = self._process_relationships(df)
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text data, remove line breaks, special characters, etc."""
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)  # Remove invisible characters
        return text.strip()

    def _process_relationships(self, df):
        """Parse relationship data"""
        if "description" in df.columns:
            df["description"] = df["description"].astype(str).apply(self._clean_text)
        return df

    async def _infer_entity_attributes(self, title, desc, progress, task):
        """Use Chat Model to generate entity attributes and update progress bar"""
        prompt = f"""
        Given an entity with its description:
        Entity Title: "{title}"
        Description: "{desc}"
        Identify the key attributes this entity should have, along with their data types.
        Return the result in the format: "attributeName:dataType", separated by commas.
        Only use the following data types: "boolean", "string", "integer", "double", "datetime".
        Ensure that attribute names are in camelCase.
        Do not include any explanation or additional text.
        """

        response = await self.chat_client.chat.completions.create(
            model=self.config.chat_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        else:
            content = ""

        progress.update(task, advance=1)

        return {
            "name": title,
            "description": desc,
            "attr": {
                item.split(":")[0].strip(): item.split(":")[1].strip()
                for item in content.split(",")
            },
        }

    async def infer_all_attributes(self):
        """Concurrently infer all entity attributes"""
        console.print("[blue]Starting entity attribute inference...[/]")

        # Get concurrent request limit from config
        max_concurrent = self.config.max_concurrent_requests

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Inferring entity attributes...", total=len(self.entities_df)
            )

            semaphore = asyncio.Semaphore(max_concurrent)

            async def infer_entity_with_semaphore(title, desc):
                async with semaphore:
                    return await self._infer_entity_attributes(
                        title, desc, progress, task
                    )

            tasks = [
                infer_entity_with_semaphore(row["title"], row["description"])
                for _, row in self.entities_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)

        output_path = self.parquet_input_path / "inferred_attributes.json"
        self._save_to_json(results, output_path)
        console.print(
            f"[green]✓ Entity attribute inference completed, results saved to: {output_path}[/]"
        )

    async def _infer_relationships(
        self, source, target, description, progress: Progress, task: TaskID
    ):
        """Use Chat Model to generate relationships"""
        prompt = f"""
        Given the following relationship:
        Source Entity: "{source}"
        Target Entity: "{target}"
        Relationship Description: "{description}"
    
        Generate a concise object property name in camelCase for this relationship, following the verb-noun-preposition format.
        Return only the property name, without any explanation or additional text.
        """

        response = await self.chat_client.chat.completions.create(
            model=self.config.chat_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        else:
            content = ""

        progress.update(task, advance=1)

        return {
            "source": source,
            "target": target,
            "description": description,
            "relation": content,
        }

    async def infer_all_relationships(self):
        """Concurrently infer all relationships"""
        console.print("[blue]Starting relationship inference...[/]")

        # Get concurrent request limit from config
        max_concurrent = self.config.max_concurrent_requests

        with Progress() as progress:
            task = progress.add_task(
                "[magenta]Inferring relationships...", total=len(self.relationships_df)
            )

            semaphore = asyncio.Semaphore(max_concurrent)

            async def infer_relationship_with_semaphore(source, target, description):
                async with semaphore:
                    return await self._infer_relationships(
                        source, target, description, progress, task
                    )

            tasks = [
                infer_relationship_with_semaphore(
                    row["source"], row["target"], row["description"]
                )
                for _, row in self.relationships_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)

        output_path = self.parquet_input_path / "inferred_relations.json"
        self._save_to_json(results, output_path)
        console.print(
            f"[green]✓ Relationship inference completed, results saved to: {output_path}[/]"
        )

    async def get_embeddings(self, text_list, batch_size=20):
        """Get text embeddings, supporting batch requests"""
        embeddings = []

        with Progress() as progress:
            task = progress.add_task(
                "[green]Getting embeddings...", total=len(text_list)
            )

            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]

                response = await self.embedding_client.embeddings.create(
                    model=self.config.embedding_model_name, input=batch, dimensions=512
                )
                data = response.data

                if data:
                    batch_embeddings = [item.embedding for item in data]
                    embeddings.extend(batch_embeddings)
                else:
                    console.print("[red]Error: Embedding API call failed[/]")
                    return None

                progress.update(task, advance=len(batch))

        return embeddings

    async def compute_all_embeddings(self):
        """Compute embeddings for all entities"""
        console.print("[blue]Starting entity embedding computation...[/]")

        # Get embeddings for entity names
        entity_texts = self.entities_df["title"].tolist()
        entity_embeddings = await self.get_embeddings(entity_texts)

        if entity_embeddings is None:
            console.print(
                "[red]Error: Unable to get entity embeddings, please check the API[/]"
            )
            raise RuntimeError("Unable to get entity embeddings, please check the API")

        self.entities_df["embedding"] = entity_embeddings
        output_path = self.parquet_input_path / "entity_embeddings.npy"
        np.save(output_path, np.array(entity_embeddings))
        console.print(
            f"[green]✓ Entity embedding computation completed, results saved to: {output_path}[/]"
        )

    def _optimal_kmeans(self, X, max_k=10):
        """Use the Elbow Method and Silhouette Score to select the optimal K"""
        console.print("[blue]Calculating optimal number of clusters...[/]")
        distortions = []
        silhouette_scores = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        # Plot Elbow Method and Silhouette Score charts
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(K, distortions, "bx-")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Distortion (Inertia)")
        plt.title("Elbow Method for Optimal K")

        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, "rx-")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score for Clustering")

        # plt.show()
        best_k = silhouette_scores.index(max(silhouette_scores)) + 2
        console.print(f"[green]✓ Optimal number of clusters: {best_k}[/]")
        return best_k

    async def cluster_entities(self):
        """Use KMeans for clustering"""
        console.print("[blue]Starting entity clustering...[/]")

        # Read pre-computed embeddings
        embeddings_path = self.parquet_input_path / "entity_embeddings.npy"
        if not embeddings_path.exists():
            console.print(
                "[red]Error: Embeddings file not found, please compute embeddings first[/]"
            )
            return

        embeddings = np.load(embeddings_path)

        # Select optimal K value
        optimal_k = self._optimal_kmeans(embeddings)

        # Run KMeans clustering
        console.print(f"[blue]Clustering with K={optimal_k}...[/]")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        # Bind clustering results to DataFrame
        self.entities_df["cluster"] = clusters

        selected_entities = await self.select_random_entities_per_cluster()

        selected_entities_records = selected_entities[
            ["title", "description", "cluster"]
        ].to_dict(orient="records")

        console.print("[blue]Generating cluster names...[/]")
        clustered_entities_prompt = f"""
        Given a list of entities with their descriptions and cluster assignments:
        {selected_entities_records}
        
        Generate names for these clusters.
        Output the cluster names in the format: "Cluster_X:ClusterName", separated by commas.
        
        For example, if the clusters are named "People", "Places", and "Things", the output should be:
        "Cluster_0:People,Cluster_1:Places,Cluster_2:Things"
        
        Ensure that the cluster names are descriptive and representative of the entities in each cluster.
        Do not include any explanation or additional text.
        """

        with Progress() as progress:
            task = progress.add_task("[green]Generating cluster names...", total=1)
            # Call LLM API to generate cluster names
            content = await self.chat_client.chat.completions.create(
                model=self.config.chat_model_name,
                messages=[{"role": "user", "content": clustered_entities_prompt}],
            )
            # Process cluster names
            cluster_names = content.choices[0].message.content.strip()

            cluster_names_dict = {
                item.split(":")[0]
                .replace("Cluster_", "")
                .strip(): item.split(":")[1]
                .strip()
                for item in cluster_names.split(",")
            }

            self.entities_df["cluster_name"] = (
                self.entities_df["cluster"].astype(str).map(cluster_names_dict)
            )

            # Save clustering results
            output_path = self.parquet_input_path / "clustered_entities.json"
            self.entities_df.to_json(
                output_path, orient="records", force_ascii=False, indent=4
            )

            progress.update(task, advance=1)

        console.print(
            f"[green]✓ Clustering completed, results saved to: {output_path}[/]"
        )

    async def select_random_entities_per_cluster(
        self, max_entities_per_cluster=10
    ) -> DataFrame:
        """Select up to 10 random entities from each cluster"""

        # Group by cluster
        clustered_entities = self.entities_df.groupby("cluster")

        selected_entities = []

        for cluster_id, group in clustered_entities:
            # If there are fewer entities in the current cluster than max_entities_per_cluster, take all of them
            num_entities_to_select = min(len(group), max_entities_per_cluster)

            selected_entities_for_cluster = group.sample(
                n=num_entities_to_select, random_state=42
            )

            selected_entities.append(selected_entities_for_cluster)

        # Merge all selected entities
        selected_entities_df = pd.concat(selected_entities, ignore_index=True)

        return selected_entities_df

    @staticmethod
    def _save_to_json(data, filename):
        """Save data to JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
