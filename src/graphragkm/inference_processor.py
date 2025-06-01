import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional, Union
import faiss
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openai import AsyncOpenAI
from pandas import DataFrame
from rich.console import Console
from rich.progress import Progress, TaskID
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

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

    def clean_uri_fragment(self, s: str) -> str:
        s = re.sub(r"[^\w\s-]", "", s).strip()
        return s.replace(" ", "_")

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
        # 将results中的relation字段写回relationships_df
        self.relationships_df["relation"] = [r["relation"] for r in results]
        # 保存更新后的relationships_df
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

    def _merge_synonyms(self, embeddings, threshold=0.9, target_entity_count=None):
        """
        合并同义词实体，基于互为最近邻的动态阈值聚类

        Args:
            embeddings: 实体嵌入向量
            threshold: 初始相似度阈值
            target_entity_count: 目标实体数量（若提供）

        Returns:
            grouped: dict(主实体索引 → 成员索引列表)
        """
        console.print("[blue]Starting to merge synonyms...[/]")

        mode = "threshold" if target_entity_count is None else "count"

        n = len(embeddings)
        emb = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        thr = threshold

        while True:
            D, I = index.search(emb, 5)  # 查找每个向量的5个最近邻
            neighbor_sets = [set(row) for row in I]
            parent = list(range(n))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # 双向最近邻合并
            for i in range(n):
                for j_pos, j in enumerate(I[i]):
                    if j <= i or j >= n:
                        continue
                    if D[i, j_pos] < thr or i not in neighbor_sets[j]:
                        continue
                    j_to_i = list(I[j]).index(i)
                    if D[j, j_to_i] >= thr:
                        union(i, j)

            # 聚簇收集
            clusters = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            console.print(
                f"[blue]Threshold={thr:.2f}, merged into {len(clusters)} clusters[/]"
            )
            if (
                mode == "count" and len(clusters) <= target_entity_count
            ) or mode == "threshold":
                break
            thr -= 0.05
            if thr <= 0.1:
                break

        # 按 degree 选主实体
        grouped = {}
        for members in clusters.values():
            main_idx = max(
                members, key=lambda i: self.entities_df.iloc[i].get("degree", 0)
            )
            grouped[main_idx] = members

        console.print(f"[green]✓ Synonyms merged into {len(grouped)} clusters[/]")
        return grouped

    def _build_merged_entities(self, grouped):
        """
        构建合并后的实体列表

        Args:
            grouped: {主实体行号 → 成员行号列表}

        Returns:
            merged_entities_df: 合并后的实体DataFrame
            name_to_title_map: 实体名称到主实体标题的映射
        """
        console.print("[blue]Building merged entities...[/]")

        new_entities = []
        name_to_title_map = {}

        for main_idx, members in grouped.items():
            main = self.entities_df.iloc[main_idx]
            main_title = main["title"]

            # 聚合 degree 和 frequency
            total_deg = int(main.get("degree", 0)) + sum(
                int(self.entities_df.iloc[i].get("degree", 0))
                for i in members
                if i != main_idx
            )
            total_freq = int(main.get("frequency", 0)) + sum(
                int(self.entities_df.iloc[i].get("frequency", 0))
                for i in members
                if i != main_idx
            )

            # 创建新实体
            ent = {
                "id": main.get("id", str(main_idx)),
                "title": main_title,
                "description": main["description"],
                "frequency": total_freq,
                "degree": total_deg,
                "embedding": main.get("embedding", None),
                "related_entities": [],
            }

            # related_entities 中包含所有成员（包括自身）
            for idx in members:
                row = self.entities_df.iloc[idx]
                row_title = row["title"]
                name_to_title_map[row_title] = main_title  # 添加到映射

                ent["related_entities"].append(
                    {
                        "id": row.get("id", str(idx)),
                        "name": row_title,
                        "description": row["description"],
                        "frequency": int(row.get("frequency", 0)),
                        "degree": int(row.get("degree", 0)),
                    }
                )

            new_entities.append(ent)

        # 创建新的DataFrame
        merged_entities_df = pd.DataFrame(new_entities)

        # 保存合并后的实体
        merged_entities_path = self.parquet_input_path / "merged_entities.json"
        self._save_to_json({"entities": new_entities}, merged_entities_path)
        console.print(f"[green]✓ Merged entities saved to: {merged_entities_path}[/]")

        return merged_entities_df, name_to_title_map

    def _update_relationships(self, name_to_title_map):
        """
        更新关系，将实体名称替换为主实体标题

        Args:
            name_to_title_map: 实体名称到主实体标题的映射

        Returns:
            new_relationships_df: 更新后的关系DataFrame
        """
        console.print("[blue]Updating relationships with merged entities...[/]")

        new_relations = []

        for idx, row in self.relationships_df.iterrows():
            src, tgt = row["source"], row["target"]
            src_str, tgt_str = str(src), str(tgt)

            # 匹配替换
            new_src = name_to_title_map.get(src_str, src_str)
            new_tgt = name_to_title_map.get(tgt_str, tgt_str)

            # 跳过自环
            if new_src == new_tgt:
                continue

            # 创建新关系
            rel = row.to_dict()
            rel["new_source"] = new_src
            rel["new_target"] = new_tgt
            rel["relation"] = row["relation"]
            if "text_unit_ids" in rel:
                rel.pop("text_unit_ids", None)

            new_relations.append(rel)

        # 保存更新后的关系
        new_relationships_df = pd.DataFrame(new_relations)
        merged_relations_path = self.parquet_input_path / "merged_relations.json"
        self._save_to_json({"relations": new_relations}, merged_relations_path)
        console.print(
            f"[green]✓ Updated relationships saved to: {merged_relations_path}[/]"
        )

        return new_relationships_df

    def _build_features_for_clustering(self, entities_df, relationships_df):
        """
        为聚类构建特征向量

        Args:
            entities_df: 实体DataFrame
            relationships_df: 关系DataFrame

        Returns:
            feats: 特征向量矩阵
        """
        console.print("[blue]Building features for clustering...[/]")

        # 参数设置
        TITLE_DIM, DESC_DIM, NEIGHBOR_DIM = 64, 20, 50
        TITLE_WEIGHT, DESC_WEIGHT = 1.0, 1.0
        NEIGHBOR_WEIGHT, NUMERIC_WEIGHT = 1.0, 1.0
        RANDOM_STATE = 42

        titles = entities_df["title"].tolist()
        descs = entities_df["description"].tolist()
        descs = [
            d if len(str(d).split()) >= 3 else titles[i] for i, d in enumerate(descs)
        ]
        deg = np.array(entities_df.get("degree", [0] * len(entities_df)), dtype=float)
        freq = np.array(
            entities_df.get("frequency", [0] * len(entities_df)), dtype=float
        )
        name2idx = {t: i for i, t in enumerate(titles)}

        # 构建标题向量
        try:
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            title_vec = st_model.encode(titles, convert_to_numpy=True)
            if title_vec.shape[1] > TITLE_DIM:
                title_vec = TruncatedSVD(
                    TITLE_DIM, random_state=RANDOM_STATE
                ).fit_transform(title_vec)
        except:
            console.print(
                "[yellow]Warning: SentenceTransformer not available, falling back to TF-IDF[/]"
            )
            tf = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(
                titles
            )
            title_vec = TruncatedSVD(
                TITLE_DIM, random_state=RANDOM_STATE
            ).fit_transform(tf)

        # 构建描述向量
        tf = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(
            descs
        )
        desc_vec = TruncatedSVD(DESC_DIM, random_state=RANDOM_STATE).fit_transform(tf)

        # 构建邻居向量
        n, d = desc_vec.shape
        neigh = np.zeros((n, d), dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)

        for _, r in relationships_df.iterrows():
            s = r.get("new_source") or r.get("source")
            t = r.get("new_target") or r.get("target")
            w = float(r.get("weight", 1.0))

            if s in name2idx and t in name2idx:
                i, j = name2idx[s], name2idx[t]
                neigh[i] += w * desc_vec[j]
                neigh[j] += w * desc_vec[i]
                wsum[i] += w
                wsum[j] += w

        mask = wsum > 0
        neigh[mask] /= wsum[mask, None]

        if neigh.shape[1] > NEIGHBOR_DIM:
            neigh = TruncatedSVD(NEIGHBOR_DIM, random_state=RANDOM_STATE).fit_transform(
                neigh
            )

        # 构建数值特征
        numeric_vec = np.vstack(
            [
                np.log1p(deg) / np.log1p(deg.max() or 1),
                np.log1p(freq) / np.log1p(freq.max() or 1),
            ]
        ).T

        # 合并所有特征
        title_vec = title_vec * TITLE_WEIGHT
        desc_vec = desc_vec * DESC_WEIGHT
        neigh = neigh * NEIGHBOR_WEIGHT
        numeric_vec = numeric_vec * NUMERIC_WEIGHT

        feats = StandardScaler().fit_transform(
            np.hstack([title_vec, desc_vec, neigh, numeric_vec])
        )

        console.print(f"[green]✓ Built features with shape {feats.shape}[/]")
        return feats

    def _choose_optimal_k(self, feats):
        """
        自动选择最佳的聚类数K

        Args:
            feats: 特征向量矩阵

        Returns:
            best_k: 最佳的聚类数K
        """
        console.print("[blue]Choosing optimal number of clusters (K)...[/]")

        # 参数设置
        K_MIN, K_MAX, K_STEP = 5, 30, 5
        DBI_PENALTY = 0.3
        K_PENALTY = 0.02
        RANDOM_STATE = 42

        best_k, best_score = None, -1e9
        records = []

        for k in range(K_MIN, K_MAX + 1, K_STEP):
            try:
                labels = KMeans(
                    n_clusters=k, random_state=RANDOM_STATE, n_init="auto"
                ).fit_predict(feats)
                sil = silhouette_score(feats, labels)
                dbi = davies_bouldin_score(feats, labels)

                # 计算得分: 轮廓系数/log(k+1) - DBI惩罚 - K惩罚
                score = sil / np.log(k + 1) - DBI_PENALTY * dbi - K_PENALTY * k

                records.append((k, sil, dbi, score))
                if score > best_score:
                    best_k, best_score = k, score

            except Exception as e:
                console.print(
                    f"[yellow]Warning: Error calculating metrics for k={k}: {str(e)}[/]"
                )

        # 打印结果
        console.print(" k\tSil\tDBI\tScore")
        for k, sil, dbi, sc in records:
            console.print(f"{k}\t{sil:.4f}\t{dbi:.4f}\t{sc:.4f}")

        console.print(f"[green]✓ Optimal number of clusters: {best_k}[/]")
        return best_k if best_k is not None else 10  # 默认值为10

    async def cluster_entities(self):
        """
        Entity clustering:
        1. First, merge synonyms
        2. Then, perform clustering
        """
        console.print("[blue]Starting entity clustering with synonym merging...[/]")

        # Load precomputed embedding vectors
        embeddings_path = self.parquet_input_path / "entity_embeddings.npy"
        if not embeddings_path.exists():
            console.print(
                "[red]Error: Embeddings file not found, please compute embeddings first[/]"
            )
            return

        embeddings = np.load(embeddings_path)

        # Step 1: Merge synonyms
        grouped = self._merge_synonyms(embeddings, threshold=0.9)

        # Build merged entities
        merged_entities_df, name_to_title_map = self._build_merged_entities(grouped)

        # Update relationships
        merged_relationships_df = self._update_relationships(name_to_title_map)

        # Step 2: Build features for merged entities
        feats = self._build_features_for_clustering(
            merged_entities_df, merged_relationships_df
        )

        # Determine the optimal number of clusters
        optimal_k = self._choose_optimal_k(feats)

        # Perform clustering using KMeans
        console.print(f"[blue]Clustering merged entities with K={optimal_k}...[/]")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(feats)

        # Bind clustering results to DataFrame
        merged_entities_df["cluster"] = clusters

        # Build cluster samples to generate cluster names
        cluster_samples = {}
        for cluster_id, group in merged_entities_df.groupby("cluster"):
            # Select up to 10 samples per cluster
            num_samples = min(10, len(group))
            samples = group.sample(n=num_samples, random_state=42)
            cluster_samples[str(cluster_id)] = samples[
                ["title", "description"]
            ].to_dict(orient="records")

        # Generate cluster names
        console.print("[blue]Generating cluster names...[/]")
        clustered_entities_prompt = f"""
        Given a list of entities grouped into clusters:
        {json.dumps(cluster_samples, ensure_ascii=False)}
        
        Generate names for these clusters.
        Output the cluster names in the format: "Cluster_X:ClusterName", separated by commas.
        
        For example, if the clusters are named "People", "Places", and "Things", the output should be:
        "Cluster_0:People,Cluster_1:Places,Cluster_2:Things"
        
        Ensure that the cluster names are descriptive and representative of the entities in each cluster.
        Do not include any explanation or additional text.
        """

        with Progress() as progress:
            task = progress.add_task("[green]Generating cluster names...", total=1)
            # Call language model to generate cluster names
            content = await self.chat_client.chat.completions.create(
                model=self.config.chat_model_name,
                messages=[{"role": "user", "content": clustered_entities_prompt}],
            )

            # Process cluster names
            cluster_names = content.choices[0].message.content.strip()
            cluster_names_dict = {}

            for item in cluster_names.split(","):
                if ":" in item:
                    cluster_id, name = item.split(":", 1)
                    cluster_id = cluster_id.replace("Cluster_", "").strip()
                    name_clean = self.clean_uri_fragment(name.strip())
                    cluster_names_dict[cluster_id] = name_clean

            merged_entities_df["cluster_name"] = (
                merged_entities_df["cluster"].astype(str).map(cluster_names_dict)
            )

            # Save clustering results
            output_path = self.parquet_input_path / "clustered_entities.json"
            merged_entities_df.to_json(
                output_path, orient="records", force_ascii=False, indent=4
            )

            progress.update(task, advance=1)

        # If need to save as original format entity DataFrame
        self.entities_df = merged_entities_df

        console.print(
            f"[green]✓ Enhanced clustering (synonym merge + clustering) completed, results saved to: {output_path}[/]"
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
