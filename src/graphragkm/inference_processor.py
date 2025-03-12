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
        初始化推理处理器，加载数据文件。
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

        # 读取数据
        console.print("[blue]正在加载实体和关系数据...[/]")
        self.entities_df = self._load_entities()
        self.relationships_df = self._load_relationships()
        console.print(
            f"[green]✓ 已加载 {len(self.entities_df)} 个实体和 {len(self.relationships_df)} 个关系[/]"
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
        """读取实体数据并清理"""
        df = pd.read_parquet(self.entities_path)
        df = df[["title", "description"]].copy()
        df["description"] = df["description"].astype(str).apply(self._clean_text)
        df = df[df["description"].str.len() > 0].reset_index(drop=True)
        return df

    def _load_relationships(self) -> DataFrame:
        """读取关系数据"""
        df = pd.read_parquet(self.relations_path)
        df = self._process_relationships(df)
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本数据，去除换行符、特殊字符等"""
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)  # 删除不可见字符
        return text.strip()

    def _process_relationships(self, df):
        """解析关系数据"""
        if "description" in df.columns:
            df["description"] = df["description"].astype(str).apply(self._clean_text)
        return df

    async def _infer_entity_attributes(self, title, desc, progress, task):
        """使用 Chat Model 生成实体属性，并更新进度条"""
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
        """并发推理所有实体属性"""
        console.print("[blue]开始推理实体属性...[/]")

        # 从配置中获取并发请求限制
        max_concurrent = self.config.max_concurrent_requests

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]推理实体属性...", total=len(self.entities_df)
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
        console.print(f"[green]✓ 实体属性推理完成，结果已保存至: {output_path}[/]")

    async def _infer_relationships(
            self, source, target, description, progress: Progress, task: TaskID
    ):
        """使用 Chat Model 生成关系"""
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
        """并发推理所有关系"""
        console.print("[blue]开始推理实体间关系...[/]")

        # 从配置中获取并发请求限制
        max_concurrent = self.config.max_concurrent_requests

        with Progress() as progress:
            task = progress.add_task(
                "[magenta]推理关系...", total=len(self.relationships_df)
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
        console.print(f"[green]✓ 关系推理完成，结果已保存至: {output_path}[/]")

    async def get_embeddings(self, text_list, batch_size=20):
        """获取文本嵌入，支持批量请求"""
        embeddings = []

        with Progress() as progress:
            task = progress.add_task("[green]获取嵌入...", total=len(text_list))

            for i in range(0, len(text_list), batch_size):
                batch = text_list[i: i + batch_size]

                response = await self.embedding_client.embeddings.create(
                    model=self.config.embedding_model_name, input=batch, dimensions=512
                )
                data = response.data

                if data:
                    batch_embeddings = [item.embedding for item in data]
                    embeddings.extend(batch_embeddings)
                else:
                    console.print("[red]错误: 嵌入 API 调用失败[/]")
                    return None

                progress.update(task, advance=len(batch))

        return embeddings

    async def compute_all_embeddings(self):
        """计算所有实体的嵌入"""
        console.print("[blue]开始计算实体嵌入...[/]")

        # 获取实体名称的嵌入
        entity_texts = self.entities_df["title"].tolist()
        entity_embeddings = await self.get_embeddings(entity_texts)

        if entity_embeddings is None:
            console.print("[red]错误: 无法获取实体嵌入，请检查 API[/]")
            raise RuntimeError("无法获取实体嵌入，请检查 API")

        self.entities_df["embedding"] = entity_embeddings
        output_path = self.parquet_input_path / "entity_embeddings.npy"
        np.save(output_path, np.array(entity_embeddings))
        console.print(f"[green]✓ 实体嵌入计算完成，结果已保存至: {output_path}[/]")

    def _optimal_kmeans(self, X, max_k=10):
        """使用肘部法和轮廓系数选择最优 K"""
        console.print("[blue]正在计算最优聚类数量...[/]")
        distortions = []
        silhouette_scores = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        # 绘制肘部法和轮廓系数图表
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
        console.print(f"[green]✓ 最优聚类数量: {best_k}[/]")
        return best_k

    async def cluster_entities(self):
        """使用 KMeans 进行聚类"""
        console.print("[blue]开始聚类实体...[/]")

        # 读取已计算的嵌入
        embeddings_path = self.parquet_input_path / "entity_embeddings.npy"
        if not embeddings_path.exists():
            console.print("[red]错误: 找不到嵌入文件，请先计算嵌入[/]")
            return

        embeddings = np.load(embeddings_path)

        # 选择最优 K 值
        optimal_k = self._optimal_kmeans(embeddings)

        # 运行 KMeans 聚类
        console.print(f"[blue]使用 K={optimal_k} 进行聚类...[/]")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        # 绑定聚类结果到 DataFrame
        self.entities_df["cluster"] = clusters

        selected_entities = await self.select_random_entities_per_cluster()

        selected_entities_records = selected_entities[
            ["title", "description", "cluster"]
        ].to_dict(orient="records")

        console.print("[blue]正在生成聚类名称...[/]")
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
            task = progress.add_task("[green]生成聚类名称...", total=1)
            # 调用 LLM API 生成聚类名称
            content = await self.chat_client.chat.completions.create(
                model=self.config.chat_model_name,
                messages=[{"role": "user", "content": clustered_entities_prompt}],
            )
            # 处理聚类名称
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

            # 保存聚类结果
            output_path = self.parquet_input_path / "clustered_entities.json"
            self.entities_df.to_json(
                output_path, orient="records", force_ascii=False, indent=4
            )

            progress.update(task, advance=1)

        console.print(f"[green]✓ 聚类完成，结果已保存至: {output_path}[/]")

    async def select_random_entities_per_cluster(
            self, max_entities_per_cluster=10
    ) -> DataFrame:
        """从每个 cluster 中随机选择最多 10 个实体"""

        # 按照 cluster 分组
        clustered_entities = self.entities_df.groupby("cluster")

        selected_entities = []

        for cluster_id, group in clustered_entities:
            # 如果当前 cluster 的实体少于 max_entities_per_cluster，则取所有
            num_entities_to_select = min(len(group), max_entities_per_cluster)

            selected_entities_for_cluster = group.sample(
                n=num_entities_to_select, random_state=42
            )

            selected_entities.append(selected_entities_for_cluster)

        # 合并所有选中的实体
        selected_entities_df = pd.concat(selected_entities, ignore_index=True)

        return selected_entities_df

    @staticmethod
    def _save_to_json(data, filename):
        """保存数据到 JSON 文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
