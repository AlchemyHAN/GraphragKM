import logging
import os
import numpy as np
import torch
import owlready2
import networkx as nx
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict, Any
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress
import re
import json
import asyncio

try:
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    HAS_PYKEEN = True
except ImportError:
    HAS_PYKEEN = False

from .config.config import Config

console = Console()


class KnowledgeEnricher:
    """Knowledge graph completion and enrichment module"""

    def __init__(
        self,
        config: Config,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize the knowledge enrichment module

        Args:
            config: Configuration object
            input_path: Input OWL file path (defaults to output/ontology.owl)
            output_path: Output OWL file path (defaults to output/enriched_ontology.owl)
        """
        self.config = config

        # Set default paths
        if input_path is None:
            input_path = str(
                Path(__file__).resolve().parents[2] / "output" / "ontology.owl"
            )

        if output_path is None:
            output_path = str(
                Path(__file__).resolve().parents[2] / "output" / "enriched_ontology.owl"
            )

        self.input_path = input_path
        self.output_path = output_path

        # Knowledge enrichment parameters
        self.num_new_relations = 10  # Number of new relations to generate
        self.device = "cpu"  # Training device, "cuda" or "cpu"
        self.seed = 42  # Random seed

        # Ensure PyKEEN is available
        if not HAS_PYKEEN:
            console.print(
                "[yellow]Warning: PyKEEN not installed. Knowledge enrichment will be limited.[/]"
            )

        # Initialize OpenAI client
        self.chat_client = AsyncOpenAI(
            api_key=self.config.chat_model_api_key,
            base_url=self.config.chat_model_api_base,
            max_retries=5,
        )

        # Logging and random seed settings
        np.random.seed(self.seed)
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

    def _to_camel_case(self, s: str) -> str:
        # Remove all illegal characters, including double quotes, angle brackets, spaces, punctuation, etc.
        s = re.sub(r"[^a-zA-Z0-9_]", "", s)
        parts = re.split(r"[_\-\s]+", s)
        if not parts:
            return s
        head, *tail = parts
        return head.lower() + "".join(w.capitalize() for w in tail)

    async def run(self):
        """Execute the knowledge enrichment process"""
        console.print("[blue]Starting knowledge enrichment...[/]")

        if not os.path.exists(self.input_path):
            console.print(f"[red]Error: Input OWL file not found: {self.input_path}[/]")
            return False

        # 1. Load ontology
        try:
            onto = owlready2.get_ontology(self.input_path).load()
            console.print("[green]✓ Loaded ontology[/]")
        except Exception as e:
            console.print(f"[red]Error loading ontology: {str(e)}[/]")
            return False

        # 2. Extract triples
        triples = self._extract_triples(onto)
        if not triples:
            console.print("[yellow]Warning: No triples extracted from ontology[/]")
            # Even if no triples, we might still want to proceed if there are classes for enrichment.
            # However, candidate generation might fail.

        # Get class label mapping
        class_labels = self._get_class_labels(onto)

        # 3. If PyKEEN is available, train embedding model and generate candidates
        candidates = []
        if HAS_PYKEEN and triples:  # Ensure triples exist for PyKEEN
            candidates = self._train_and_generate_candidates(triples)
        elif triples:  # Fallback if PyKEEN not available but triples exist
            console.print(
                "[yellow]PyKEEN not available or no triples for training, using simple heuristics for candidates[/]"
            )
            candidates = self._generate_simple_candidates(triples)
        else:
            console.print(
                "[yellow]No triples available to generate candidates from.[/]"
            )

        # 4. Validate candidates with LLM
        final_triples = []
        if candidates:
            final_triples = await self._validate_with_llm(
                candidates, class_labels
            )  # Use await here
        else:
            console.print("[yellow]No candidates generated to validate.[/]")

        # 5. Write new relations to ontology
        if final_triples:
            success = self._enrich_ontology(onto, final_triples)
            if success:
                console.print(
                    f"[green]✓ Enhanced ontology saved to: {self.output_path}[/]"
                )
                return True
        else:
            console.print("[yellow]No new relations were generated or validated[/]")

        return False

    def _extract_triples(self, onto) -> List[Tuple[str, str, str]]:
        """
        Extract triples from the ontology

        Args:
            onto: owlready2 ontology object

        Returns:
            List of triples [(head, relation, tail), ...]
        """
        console.print("[blue]Extracting triples from ontology...[/]")

        kg_graph = nx.MultiDiGraph()
        triples = []

        # Get all classes
        classes = list(onto.classes())

        # Extract object property relations
        for subj in onto.individuals():
            for prop, values in subj.get_properties():
                if isinstance(prop, owlready2.ObjectPropertyClass):
                    for obj in values:
                        kg_graph.add_edge(subj, obj, label=prop.name)
                        triples.append((subj.name, prop.name, obj.name))

        # Extract class inheritance relations
        for cls in classes:
            for parent in cls.is_a:
                if parent in classes:
                    kg_graph.add_edge(cls, parent, label="subClassOf")
                    triples.append((cls.name, "subClassOf", parent.name))

        console.print(f"[green]✓ Extracted {len(triples)} triples[/]")
        return triples

    def _get_class_labels(self, onto) -> Dict[str, str]:
        """Get class label mapping"""
        class_labels = {}

        # Class labels
        for c in onto.classes():
            class_labels[c.name] = getattr(c.label, "first", lambda: c.name)() or c.name

        # Individual labels
        for ind in onto.individuals():
            class_labels[ind.name] = (
                getattr(ind.label, "first", lambda: ind.name)() or ind.name
            )

        return class_labels

    def _train_and_generate_candidates(
        self, triples: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Train embedding model and generate candidate triples

        Args:
            triples: List of existing triples

        Returns:
            List of candidate triples
        """
        console.print("[blue]Training embedding model...[/]")

        cleaned_triples = []
        for h, r, t in triples:
            if all([h, r, t]) and "None" not in (h, r, t):
                h = re.sub(r"[^\w\-]", "", h.strip())
                r = re.sub(r"[^\w\-]", "", r.strip())
                t = re.sub(r"[^\w\-]", "", t.strip())
                if h and r and t:
                    cleaned_triples.append((h, r, t))

        triples = list(set(cleaned_triples))  # 去重
        if not triples:
            console.print("[red]Error: No valid triples after cleaning[/]")
            return []
        max_triples = 1000
        if len(triples) > max_triples:
            console.print(
                f"[yellow]Too many triples ({len(triples)}), sampling {max_triples} for training[/]"
            )
            import random

            random.seed(self.seed)
            triples = random.sample(triples, max_triples)

        # Generate candidate triples
        try:
            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Training knowledge graph embeddings...", total=100
                )

                # Build TriplesFactory
                all_tf = TriplesFactory.from_labeled_triples(
                    np.array(triples, dtype=str)
                )
                progress.update(task, advance=10)

                # Map triples to IDs
                mapped = all_tf.mapped_triples

                # Shuffle randomly
                g = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(mapped.shape[0], generator=g)
                mapped = mapped[perm]

                # Split into training/validation/test sets (80/10/10)
                n_total = mapped.shape[0]
                n_train = int(0.8 * n_total)
                n_valid = int(0.9 * n_total)

                progress.update(task, advance=10)

                # Create training/validation/test TriplesFactory
                training_tf = TriplesFactory(
                    mapped_triples=mapped[:n_train],
                    entity_to_id=all_tf.entity_to_id,
                    relation_to_id=all_tf.relation_to_id,
                )

                valid_tf = TriplesFactory(
                    mapped_triples=mapped[n_train:n_valid],
                    entity_to_id=all_tf.entity_to_id,
                    relation_to_id=all_tf.relation_to_id,
                )

                test_tf = TriplesFactory(
                    mapped_triples=mapped[n_valid:],
                    entity_to_id=all_tf.entity_to_id,
                    relation_to_id=all_tf.relation_to_id,
                )

                progress.update(task, advance=10)

                # Reduce training complexity, decrease training time
                num_epochs = min(50, 200)  # Max 50 epochs, not 200
                batch_size = (
                    min(512, n_train // 2) if n_train > 2 else 1
                )  # Reasonable batch size

                console.print(
                    f"[blue]Training with {num_epochs} epochs, batch size {batch_size}[/]"
                )

                # Train model - use TransE (simplest and most efficient model)
                result = pipeline(
                    training=training_tf,
                    validation=valid_tf,
                    testing=test_tf,
                    model="TransE",
                    model_kwargs=dict(
                        embedding_dim=32
                    ),  # Reduce dimension from 64 to 32
                    random_seed=self.seed,
                    device=self.device,
                    training_kwargs=dict(
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        checkpoint_name=None,  # Do not save checkpoint files
                        checkpoint_frequency=0,
                        checkpoint_directory=None,
                    ),
                )

                model = result.model
                progress.update(task, advance=40)

                # Generate candidate triples - limit entity count to avoid excessive computation
                entity_to_id = training_tf.entity_to_id
                relation_to_id = training_tf.relation_to_id

                # If there are too many entities, randomly sample to reduce computation
                max_entities = 50  # Limit max entity count
                nodes = list(entity_to_id.keys())
                if len(nodes) > max_entities:
                    import random

                    random.seed(self.seed)
                    nodes = random.sample(nodes, max_entities)
                    console.print(
                        f"[yellow]Too many entities, sampling {max_entities} for candidate generation[/]"
                    )

                # Limit relation count
                max_relations = 10  # Max 10 relation types
                relations_list = list(relation_to_id.keys())
                if len(relations_list) > max_relations:
                    import random

                    random.seed(self.seed)
                    relations_list = random.sample(relations_list, max_relations)

                # Record existing triples
                existing_triples = {(h, r, t) for h, r, t in triples}

                # Score and sort candidates
                candidates = []
                progress.update(task, advance=10)

                # Set scoring counter, update progress periodically
                total_pairs = len(nodes) * (len(nodes) - 1)
                scored = 0

                for i, h in enumerate(nodes):
                    h_id = entity_to_id[h]
                    for j, t in enumerate(nodes):
                        if h == t:
                            continue

                        # Calculate progress
                        scored += 1
                        if scored % 100 == 0 and total_pairs > 0:
                            progress_pct = min(99, int(scored / total_pairs * 100))
                            # Safely update progress
                            progress.update(
                                task,
                                completed=progress_pct,
                            )

                        t_id = entity_to_id[t]

                        # Batch predict all relations
                        batch_indices = []
                        batch_rel_indices = []

                        for r_idx, rel in enumerate(relations_list):
                            if (h, rel, t) in existing_triples:
                                continue
                            batch_indices.append([h_id, relation_to_id[rel], t_id])
                            batch_rel_indices.append(r_idx)

                        if not batch_indices:
                            continue

                        # Batch score, handle gradient issues
                        try:
                            batch = torch.tensor(
                                batch_indices, dtype=torch.long, device=self.device
                            )
                            # Use detach() to separate gradients, avoid numpy() error
                            scores = model.score_hrt(batch).detach().cpu().numpy()

                            # Add scoring results
                            for score_idx, r_idx in enumerate(batch_rel_indices):
                                candidates.append(
                                    (scores[score_idx], h, relations_list[r_idx], t)
                                )
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning during scoring: {str(e)}[/]"
                            )
                            # If scoring fails, try using a default score
                            for r_idx in batch_rel_indices:
                                candidates.append((0.0, h, relations_list[r_idx], t))

                # Take top N by score
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = [
                    (h, rel, t)
                    for score, h, rel, t in candidates[: self.num_new_relations]
                ]

                progress.update(task, completed=100)
                console.print(
                    f"[green]✓ Generated {len(top_candidates)} candidate relations[/]"
                )
                return top_candidates

        except Exception as e:
            console.print(f"[red]Error during training: {str(e)}[/]")
            # Fallback to simple heuristic method on error
            console.print("[yellow]Falling back to simple heuristic method[/]")
            return self._generate_simple_candidates(triples)

    def _generate_simple_candidates(
        self, triples: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        When PyKEEN is unavailable or training fails, use simple heuristics to generate candidates

        Args:
            triples: List of existing triples

        Returns:
            List of candidate triples
        """
        console.print("[blue]Generating candidates using simple heuristics...[/]")

        # Get all entities and relations
        entities = set()
        relations = set()

        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        # Record existing triples
        existing_triples = {(h, r, t) for h, r, t in triples}

        # Try each relation for each entity pair
        candidates = []
        entity_list = list(entities)

        # Limit candidate count for efficiency - max 20 head entities and 20 tail entities
        max_head_entities = min(20, len(entity_list))
        max_tail_entities = min(20, len(entity_list))

        # Limit the number of relations used
        max_relations = min(10, len(relations))
        relation_list = list(relations)
        if len(relation_list) > max_relations:
            import random

            random.seed(self.seed)
            relation_list = random.sample(relation_list, max_relations)

        np.random.shuffle(entity_list)
        head_entities = entity_list[:max_head_entities]

        # Different set of tail entities (avoid self-loops)
        tail_entities = []
        for i in range(min(max_tail_entities, len(entity_list))):
            if entity_list[i] not in head_entities:
                tail_entities.append(entity_list[i])

        # If tail entities are insufficient, supplement from head entities
        if len(tail_entities) < max_tail_entities and len(head_entities) > 0:
            needed = max_tail_entities - len(tail_entities)
            tail_entities.extend(head_entities[:needed])

        # Generate candidate triples
        console.print("[cyan]Generating candidate relations...[/]")
        total_attempts = 0
        max_attempts = max_head_entities * max_tail_entities

        for h in head_entities:
            for t in tail_entities:
                if h == t:
                    continue

                for r in relation_list:
                    if (h, r, t) not in existing_triples and (
                        t,
                        r,
                        h,
                    ) not in existing_triples:
                        candidates.append((h, r, t))

                # Periodically display progress
                total_attempts += 1
                if total_attempts % 10 == 0:
                    percent = min(100, int(total_attempts * 100 / max_attempts))
                    console.print(f"[blue]Generated candidates: {percent}% complete[/]")

                # Stop after generating enough candidates
                if len(candidates) >= self.num_new_relations * 3:
                    break

            if len(candidates) >= self.num_new_relations * 3:
                break

        # Randomly sample
        if len(candidates) > self.num_new_relations:
            np.random.shuffle(candidates)
            candidates = candidates[: self.num_new_relations]

        console.print(f"[green]✓ Generated {len(candidates)} candidate relations[/]")
        return candidates

    async def _validate_candidate_with_llm(
        self,
        h: str,
        t: str,
        class_labels: Dict[str, str],
        system_content: str,
        progress: Progress,
        task_id: Any,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Validate a single candidate triple using LLM and update progress.
        """
        async with semaphore:
            user_content = (
                f'Class A: "{class_labels.get(h, h)}"\n'
                f'Class B: "{class_labels.get(t, t)}"\n'
                "Relation?"
            )
            try:
                from openai.types.chat import (
                    ChatCompletionSystemMessageParam,
                    ChatCompletionUserMessageParam,
                )

                system_message = ChatCompletionSystemMessageParam(
                    role="system", content=system_content
                )
                user_message = ChatCompletionUserMessageParam(
                    role="user", content=user_content
                )

                # Correct message structure
                reply = await self.chat_client.chat.completions.create(
                    model=self.config.chat_model_name,
                    messages=[system_message, user_message],
                )

                # Safely get content
                reply_content = reply.choices[0].message.content
                reply_text = reply_content.strip() if reply_content else ""

                # Process reply, take only the first line
                if "\n" in reply_text:
                    reply_text = reply_text.split("\n")[0].strip()

                # Remove possible code block markers
                if reply_text.startswith("```"):
                    reply_text = reply_text.strip("`").lstrip("json").strip()

                console.log(
                    f"[dim]LLM raw response for ({h}, ?, {t}): '{reply_text}'[/dim]"
                )

                # If reply is not "None", add to final triples
                if reply_text and reply_text.lower() != "none":
                    # Clean string
                    clean_rel = self._to_camel_case(reply_text)
                    clean_rel = re.sub(r'["\'\s<>]', "", clean_rel)

                    # Filter out cases where h or t is "None" (uppercase None)
                    if (
                        h.lower() == "none"
                        or t.lower() == "none"
                        or clean_rel.lower() == "none"
                        or not clean_rel  # Ensure clean_rel is not empty
                    ):
                        console.print(
                            f"[yellow]Filtered out invalid triple (None or empty): ({h}, {clean_rel}, {t})[/]"
                        )
                        progress.update(task_id, advance=1)
                        return None

                    progress.update(task_id, advance=1)
                    return h, clean_rel, t

            except Exception as e:
                console.print(f"[red]Error calling LLM for ({h}, ?, {t}): {str(e)}[/]")

            progress.update(task_id, advance=1)
            return None

    async def _validate_with_llm(
        self, candidates: List[Tuple[str, str, str]], class_labels: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """
        Asynchronously validate candidate triples using LLM

        Args:
            candidates: List of candidate triples [(head, relation_guess, tail), ...]
            class_labels: Class label mapping

        Returns:
            List of validated triples
        """
        console.print("[blue]Validating candidate relations with LLM (async)...[/]")

        system_content = (
            "You are a knowledge-graph completion assistant. "
            "Given two OWL class names, reply with ONE plausible English noun-phrase "
            "relation connecting them, or exactly 'None' if none applies. "
            "Return plain text only. Ensure the relation is concise and meaningful."
        )

        final_triples = []

        # Get concurrent request limit
        max_concurrent = self.config.max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent)

        with Progress() as progress_bar:
            task_id = progress_bar.add_task(
                "[cyan]Validating relations...", total=len(candidates)
            )

            validation_tasks = []
            for (
                h,
                rel_guess,
                t,
            ) in candidates:  # rel_guess is not used here as LLM generates new one
                validation_tasks.append(
                    self._validate_candidate_with_llm(
                        h,
                        t,
                        class_labels,
                        system_content,
                        progress_bar,
                        task_id,
                        semaphore,
                    )
                )

            results = await asyncio.gather(*validation_tasks)

            for result in results:
                if result:
                    final_triples.append(result)
                    console.print(
                        f"[green]Validated: ({result[0]}, {result[1]}, {result[2]})[/]"
                    )

        console.print(f"[green]✓ Validated {len(final_triples)} new relations[/]")

        # Log results
        log_path = Path(self.output_path).parent / "new_triples.json"
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "new_triples": [
                        {"head": h, "relation": r, "tail": t}
                        for h, r, t in final_triples
                    ]
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        console.print(f"[blue]New triples log saved to: {log_path}[/]")
        return final_triples

    def _enrich_ontology(self, onto, final_triples: List[Tuple[str, str, str]]) -> bool:
        """
        Write new relations to the ontology

        Args:
            onto: owlready2 ontology object
            final_triples: List of triples to add

        Returns:
            Whether successful
        """
        console.print("[blue]Enriching ontology with new relations...[/]")

        try:
            # Get ontology base URI
            base_uri = self.config.owl_namespace
            if not base_uri.endswith("/"):
                base_uri = base_uri + "/"

            with onto:
                for h, rel, t in final_triples:
                    # Find head and tail entities
                    cls_h = onto.search_one(iri=f"*{h}")
                    cls_t = onto.search_one(iri=f"*{t}")

                    if not (cls_h and cls_t):
                        console.print(
                            f"[yellow]Warning: Classes not found for ({h}, {t})"
                        )
                        continue

                    # Create a valid property name - remove characters that might cause XML issues
                    clean_rel = re.sub(r'["\'\s<>]', "", rel)

                    # Construct full URI, ensure no extra quotes
                    prop_iri = f"{base_uri}{clean_rel}"

                    # Check if property already exists
                    prop = onto.search_one(iri=prop_iri)

                    if not prop:
                        import types

                        # Create new property using a clean name, ensure no special characters or quotes
                        try:
                            prop = types.new_class(
                                clean_rel, (owlready2.ObjectProperty,)
                            )
                            # Set label, without quotes or special characters
                            prop.label = [clean_rel]
                            prop.namespace = onto
                        except Exception as e:
                            console.print(
                                f"[red]Error creating property {clean_rel}: {str(e)}[/]"
                            )
                            continue

                    # Establish someValuesFrom relation
                    try:
                        # Check if entity classes are valid before adding relation
                        if cls_h is None or cls_t is None or prop is None:
                            console.print(
                                f"[yellow]Warning: Invalid entities or property for relation: {h} {rel} {t}[/]"
                            )
                            continue

                        cls_h.is_a.append(prop.some(cls_t))
                    except Exception as e:
                        console.print(
                            f"[red]Error adding relation {h} {rel} {t}: {str(e)}[/]"
                        )

            # Check and fix quote issues in OWL file before saving
            self._fix_owl_quotes(onto)

            # Save to file
            onto.save(file=self.output_path)
            return True

        except Exception as e:
            console.print(f"[red]Error enriching ontology: {str(e)}[/]")
            return False

    def _fix_owl_quotes(self, onto):
        for entity in list(onto.classes()) + list(onto.properties()):
            if hasattr(entity, "name"):
                entity.name = entity.name.replace('"', "").replace("'", "")
        console.print("[green]✓ Removed extra quotes from entity names[/]")
