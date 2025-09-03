"""VLM concept interpretability analysis with PCA sensitivity plots."""

from __future__ import annotations

import io
import os
import re
import sqlite3
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors using numpy with robust error handling.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    # Check for NaN or infinite values
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        print('Warning: NaN or infinite values detected in tensors')
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors or invalid norms
    if norm_a == 0 or norm_b == 0 or not (np.isfinite(norm_a) and np.isfinite(norm_b)):
        return 0.0

    dot_product = np.dot(a, b)

    # Check if dot product is valid
    if not np.isfinite(dot_product):
        print('Warning: Invalid dot product')
        return 0.0

    return dot_product / (norm_a * norm_b)


def extract_tensor_from_object(tensor_obj: Any) -> Optional[torch.Tensor]:
    """Return a single 1D embedding vector from a deserialized object.

    Prefer pooled outputs; if we get sequence/token grids, mean-pool.

    Args:
        tensor_obj: Deserialized tensor object from model output

    Returns:
        Single 1D torch tensor or None if extraction fails
    """
    def _to_1d(t: Any) -> Optional[torch.Tensor]:
        if not torch.is_tensor(t):
            return None
        if t.dim() == 3:
            t = t[0]  # assume batch size 1
            t = t.mean(dim=0)  # mean over seq
        elif t.dim() == 2:
            t = t.mean(dim=0)  # mean over seq
        elif t.dim() == 1:
            pass
        else:
            t = t.flatten()
        return t

    if hasattr(tensor_obj, 'pooler_output'):
        t = _to_1d(tensor_obj.pooler_output)
        if t is not None:
            return t
    if hasattr(tensor_obj, 'last_hidden_state'):
        t = _to_1d(tensor_obj.last_hidden_state)
        if t is not None:
            return t
    if hasattr(tensor_obj, 'hidden_states'):
        hs = tensor_obj.hidden_states
        if isinstance(hs, (list, tuple)) and len(hs) > 0:
            t = _to_1d(hs[-1])  # last layer
            if t is not None:
                return t
        else:
            t = _to_1d(hs)
            if t is not None:
                return t
    if torch.is_tensor(tensor_obj):
        return _to_1d(tensor_obj)

    for attr_name in dir(tensor_obj):
        if attr_name.startswith('_'):
            continue
        try:
            attr_value = getattr(tensor_obj, attr_name)
            if torch.is_tensor(attr_value):
                t = _to_1d(attr_value)
                if t is not None:
                    print(f"Using attribute \'{attr_name}\' from {type(tensor_obj).__name__}")
                    return t
        except Exception:
            continue

    print(f'Could not find tensor data in {type(tensor_obj).__name__}')
    return None


def load_tensors_by_layer(db_path: str, device: str = 'cpu') -> dict[str, list[tuple[np.ndarray, Any, int, str]]]:
    """Load all tensors from a database, grouped by layer.

    Args:
        db_path: Path to the SQLite database
        device: PyTorch device for tensor loading

    Returns:
        Dictionary mapping layer names to lists of (tensor_np, label, row_id, image_filename) tuples
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # First check what columns are available
    cursor.execute('PRAGMA table_info(tensors)')
    columns = [column[1] for column in cursor.fetchall()]
    print(f'Available columns in {db_path}: {columns}')

    query = 'SELECT rowid, layer, tensor, label, image_path FROM tensors'
    cursor.execute(query)
    results = cursor.fetchall()
    connection.close()

    layers_dict = defaultdict(list)

    for result in results:
        row_id, layer, tensor_bytes, label, image_filename = result

        try:
            # Load tensor object
            tensor_obj = torch.load(io.BytesIO(tensor_bytes), map_location=device, weights_only=False)

            # Extract actual tensor from object
            tensor = extract_tensor_from_object(tensor_obj)
            if tensor is None:
                print(f'Warning: Could not extract tensor from row {row_id} in layer {layer}')
                continue

            # Convert to numpy for analysis
            if tensor.requires_grad:
                tensor_np = tensor.detach().cpu().numpy().flatten()
            else:
                tensor_np = tensor.cpu().numpy().flatten()

            layers_dict[layer].append((tensor_np, label, row_id, image_filename))

        except Exception as e:
            print(f'Warning: Could not deserialize tensor at row {row_id}, layer {layer}: {e}')
            continue

    return dict(layers_dict)


def extract_concept_from_filename(image_filename: str) -> Optional[str]:
    """Extract concept name from image filename.

    Args:
        image_filename: e.g., './data/concepts/images/blue_01.jpg'

    Returns:
        concept name, e.g., 'blue'
    """
    if not image_filename:
        return None

    # Get the base filename without path and extension
    base_name = os.path.splitext(os.path.basename(image_filename))[0]

    # Extract concept name (everything before the last underscore and number)
    # e.g., 'blue_01' -> 'blue'
    match = re.match(r'^(.+)_\d+$', base_name)
    if match:
        return match.group(1)
    else:
        # If no underscore pattern, use the whole base name
        return base_name


def group_tensors_by_concept(layer_tensors: list[tuple[np.ndarray, Any, int, str]]) -> dict[str, list[tuple[np.ndarray, Any, int, str]]]:
    """Group tensors by concept based on their image filenames.

    Args:
        layer_tensors: List of (tensor_np, label, row_id, image_filename) tuples

    Returns:
        Dictionary mapping concept names to lists of tensor data
    """
    concept_groups = defaultdict(list)

    for tensor_data in layer_tensors:
        tensor_np, label, row_id, image_filename = tensor_data
        concept = extract_concept_from_filename(image_filename)

        if concept:
            concept_groups[concept].append(tensor_data)
        else:
            print(f'Warning: Could not extract concept from filename: {image_filename}')

    return dict(concept_groups)


def apply_pca_to_layer(
    target_tensors: list[tuple[np.ndarray, Any, int, str]],
    concept_tensors: list[tuple[np.ndarray, Any, int, str]],
    n_components: Optional[int] = None
) -> tuple[list[tuple[np.ndarray, Any, int, str]], list[tuple[np.ndarray, Any, int, str]], Optional[PCA]]:
    """Apply PCA dimensionality reduction to tensors from the same layer.

    PCA is fit on CONCEPT TENSORS ONLY to avoid target leakage.

    Args:
        target_tensors: List of target tensor data
        concept_tensors: List of concept tensor data
        n_components: Number of PCA components (None to skip PCA)

    Returns:
        Tuple of (transformed_target_tensors, transformed_concept_tensors, pca_model)
    """
    if n_components is None:
        return target_tensors, concept_tensors, None

    print(f'Applying PCA with {n_components} components...')

    concept_arrays = [data[0] for data in concept_tensors]

    if len(concept_arrays) == 0:
        print('Warning: no concept tensors to fit PCA; skipping PCA.')
        return target_tensors, concept_tensors, None

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(np.vstack(concept_arrays))

    print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
    print(f'Total explained variance: {pca.explained_variance_ratio_.sum():.4f}')

    transformed_target_tensors = []
    for tensor_np, label, row_id, image_filename in target_tensors:
        transformed = pca.transform(tensor_np.reshape(1, -1)).flatten()
        transformed_target_tensors.append((transformed, label, row_id, image_filename))

    transformed_concept_tensors = []
    for tensor_np, label, row_id, image_filename in concept_tensors:
        transformed = pca.transform(tensor_np.reshape(1, -1)).flatten()
        transformed_concept_tensors.append((transformed, label, row_id, image_filename))

    return transformed_target_tensors, transformed_concept_tensors, pca


def analyze_target_vs_concepts(
    target_tensors: list[tuple[np.ndarray, Any, int, str]],
    concept_tensors: list[tuple[np.ndarray, Any, int, str]],
    layer_name: str
) -> list[dict[str, Any]]:
    """Analyze similarity between target images and concept groups.

    Adds centroid-based metrics while preserving existing stats.

    Args:
        target_tensors: List of target tensor data
        concept_tensors: List of concept tensor data
        layer_name: Name of the current layer

    Returns:
        List of analysis results for each target image
    """
    concept_groups = group_tensors_by_concept(concept_tensors)
    print(f'Found {len(concept_groups)} concepts: {list(concept_groups.keys())}')
    for concept, tensors in concept_groups.items():
        print(f'  {concept}: {len(tensors)} images')

    # Precompute concept centroids
    concept_centroids = {}
    for concept_name, tensor_list in concept_groups.items():
        vecs = [t[0] for t in tensor_list]
        if len(vecs) > 0:
            concept_centroids[concept_name] = np.mean(np.vstack(vecs), axis=0)
        else:
            concept_centroids[concept_name] = None

    results = []

    for target_data in target_tensors:
        target_tensor, target_label, target_row_id, target_image_filename = target_data

        target_result = {
            'layer': layer_name,
            'target_row_id': target_row_id,
            'target_label': target_label,
            'target_image_filename': target_image_filename,
            'concept_analysis': {}
        }

        for concept_name, concept_tensor_list in concept_groups.items():
            similarities = []

            # Original per-prototype pairwise similarities
            for concept_data in concept_tensor_list:
                concept_tensor, concept_label, concept_row_id, concept_image_filename = concept_data
                if target_tensor.shape != concept_tensor.shape:
                    print(f'Warning: Shape mismatch between target {target_row_id} and concept {concept_row_id}')
                    continue
                sim = cosine_similarity_numpy(target_tensor, concept_tensor)
                similarities.append(sim)

            concept_stats = {}
            if similarities:
                similarities = np.array(similarities)
                distances = 1.0 - similarities

                concept_stats.update({
                    'min_similarity': float(np.min(similarities)),
                    'max_similarity': float(np.max(similarities)),
                    'mean_similarity': float(np.mean(similarities)),
                    'min_distance': float(np.min(distances)),
                    'mean_distance': float(np.mean(distances)),
                    'num_comparisons': int(len(similarities)),
                })

            # New: centroid-based similarity
            centroid = concept_centroids.get(concept_name, None)
            if centroid is not None and centroid.shape == target_tensor.shape:
                cen_sim = cosine_similarity_numpy(target_tensor, centroid)
                cen_ang = float(np.degrees(np.arccos(np.clip(cen_sim, -1.0, 1.0))))
                concept_stats.update({
                    'centroid_similarity': float(cen_sim),
                    'centroid_angular_deg': cen_ang
                })

            if concept_stats:
                target_result['concept_analysis'][concept_name] = concept_stats

        results.append(target_result)

        target_display = target_image_filename if target_image_filename else f'Target_{target_row_id}'
        print(f'Analyzed {target_display} against {len(concept_groups)} concepts')

    return results


def concept_similarity_analysis(
    target_db_path: str,
    concept_db_path: str,
    layer_names: Optional[list[str]] = None,
    n_pca_components: Optional[int] = None,
    device: str = 'cpu'
) -> dict[str, dict[str, Any]]:
    """Main function for concept-based similarity analysis.

    Args:
        target_db_path: Path to target images database
        concept_db_path: Path to concept images database
        layer_names: List of layer names to analyze (None for all common layers)
        n_pca_components: Number of PCA components (None to skip PCA)
        device: PyTorch device

    Returns:
        Dictionary of analysis results by layer
    """
    print('Starting concept-based similarity analysis...')
    print(f'Target DB: {target_db_path}')
    print(f'Concept DB: {concept_db_path}')
    print(f'PCA components: {n_pca_components}')

    # Load tensors from both databases
    print(f'\nLoading tensors from {target_db_path}...')
    target_tensors = load_tensors_by_layer(target_db_path, device)

    print(f'Loading tensors from {concept_db_path}...')
    concept_tensors = load_tensors_by_layer(concept_db_path, device)

    # Find common layers
    common_layers = set(target_tensors.keys()) & set(concept_tensors.keys())
    print(f'\nFound {len(common_layers)} common layers: {sorted(common_layers)}')

    if not common_layers:
        print('No common layers found between databases!')
        return {}

    # Determine which layers to analyze
    if layer_names is None:
        layers_to_analyze = sorted(common_layers)
        print('Analyzing all common layers')
    else:
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        layers_to_analyze = [layer for layer in layer_names if layer in common_layers]
        print(f'Analyzing specified layers: {layers_to_analyze}')

        # Warn about missing layers
        missing_layers = set(layer_names) - common_layers
        if missing_layers:
            print(f'Warning: Requested layers not found: {missing_layers}')

    if not layers_to_analyze:
        print('No valid layers to analyze!')
        return {}

    all_results = {}

    # Process each layer
    for layer in layers_to_analyze:
        print(f'\n{"=" * 50}')
        print(f'Processing Layer: {layer}')
        print(f'{"=" * 50}')

        target_layer_tensors = target_tensors[layer]
        concept_layer_tensors = concept_tensors[layer]

        print(f'Target tensors: {len(target_layer_tensors)}')
        print(f'Concept tensors: {len(concept_layer_tensors)}')

        # Apply PCA if requested
        if n_pca_components is not None:
            target_layer_tensors, concept_layer_tensors, pca_model = apply_pca_to_layer(
                target_layer_tensors, concept_layer_tensors, n_pca_components
            )
        else:
            pca_model = None

        # Analyze similarities
        layer_results = analyze_target_vs_concepts(
            target_layer_tensors, concept_layer_tensors, layer
        )

        all_results[layer] = {
            'results': layer_results,
            'pca_model': pca_model,
            'n_pca_components': n_pca_components
        }

        # Print layer summary
        if layer_results:
            print(f"\nLayer \'{layer}\' Summary:")
            print(f'  Analyzed {len(layer_results)} target images')

            # Get all concept names from first result
            if layer_results[0]['concept_analysis']:
                concept_names = list(layer_results[0]['concept_analysis'].keys())
                print(f'  Against {len(concept_names)} concepts: {concept_names}')

    return all_results


def save_concept_analysis_results(results: dict[str, dict[str, Any]], output_file: str = 'output/concept_similarity_analysis.txt') -> None:
    """Save concept analysis results to a text file.

    Args:
        results: Dictionary of analysis results by layer
        output_file: Output filename
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('Concept-Based VLM Embedding Similarity Analysis\n')
        f.write('=' * 60 + '\n\n')

        for layer, layer_data in results.items():
            layer_results = layer_data['results']
            n_pca_components = layer_data['n_pca_components']

            f.write(f'Layer: {layer}\n')
            if n_pca_components:
                f.write(f'PCA Components: {n_pca_components}\n')
            f.write('-' * 40 + '\n\n')

            for result in layer_results:
                target_display = result['target_image_filename'] or f'Target_{result["target_row_id"]}'
                f.write(f'Target: {target_display}\n')

                for concept_name, stats in result['concept_analysis'].items():
                    f.write(f'  {concept_name}:\n')
                    if 'min_similarity' in stats:
                        f.write(f'    Min Similarity: {stats["min_similarity"]:.4f}\n')
                        f.write(f'    Max Similarity: {stats["max_similarity"]:.4f}\n')
                        f.write(f'    Mean Similarity: {stats["mean_similarity"]:.4f}\n')
                        f.write(f'    Min Distance: {stats["min_distance"]:.4f}\n')
                        f.write(f'    Mean Distance: {stats["mean_distance"]:.4f}\n')
                        f.write(f'    Comparisons: {stats["num_comparisons"]}\n')
                    if 'centroid_similarity' in stats:
                        f.write(f'    Centroid Similarity: {stats["centroid_similarity"]:.4f}\n')
                        f.write(f'    Centroid Angular (deg): {stats["centroid_angular_deg"]:.2f}\n')
                f.write('\n')

            f.write('\n')

    print(f'Results saved to {output_file}')


def analyze_concept_trends(results: dict[str, dict[str, Any]]) -> None:
    """Analyze trends across all targets and concepts.

    Args:
        results: Dictionary of analysis results by layer
    """
    print(f'\n{"=" * 50}')
    print('CONCEPT ANALYSIS TRENDS')
    print(f'{"=" * 50}')

    for layer, layer_data in results.items():
        layer_results = layer_data['results']
        n_pca_components = layer_data['n_pca_components']

        print(f'\nLayer: {layer}')
        if n_pca_components:
            print(f'PCA Components: {n_pca_components}')
        print('-' * 30)

        if not layer_results:
            print('No results for this layer')
            continue

        concept_stats = defaultdict(list)
        for result in layer_results:
            for concept_name, stats in result['concept_analysis'].items():
                concept_stats[concept_name].append(stats)

        for concept_name in sorted(concept_stats.keys()):
            stats_list = concept_stats[concept_name]
            all_min_sim = [s['min_similarity'] for s in stats_list if 'min_similarity' in s]
            all_max_sim = [s['max_similarity'] for s in stats_list if 'max_similarity' in s]
            all_mean_sim = [s['mean_similarity'] for s in stats_list if 'mean_similarity' in s]
            all_min_dist = [s['min_distance'] for s in stats_list if 'min_distance' in s]
            all_cen_sim = [s['centroid_similarity'] for s in stats_list if 'centroid_similarity' in s]

            print(f'  {concept_name}:')
            if all_min_sim:
                print(f'    Avg Min Similarity:   {np.mean(all_min_sim):.4f}')
            if all_max_sim:
                print(f'    Avg Max Similarity:   {np.mean(all_max_sim):.4f}')
            if all_mean_sim:
                print(f'    Avg Mean Similarity:  {np.mean(all_mean_sim):.4f}')
            if all_min_dist:
                print(f'    Avg Min Distance:     {np.mean(all_min_dist):.4f}')
            if all_cen_sim:
                print(f'    Avg Centroid Cosine:  {np.mean(all_cen_sim):.4f}')
            print(f'    Targets analyzed:     {len(stats_list)}')


def plot_pca_sensitivity_analysis(
    target_db_path: str,
    concept_db_path: str,
    layer_names: Optional[list[str]] = None,
    max_components: int = 50,
    device: str = 'cpu',
    output_dir: str = 'output'
) -> None:
    """Plot centroid similarity vs number of PCA components for interpretability analysis.

    Args:
        target_db_path: Path to target images database
        concept_db_path: Path to concept images database
        layer_names: List of layer names to analyze (None for all common layers)
        max_components: Maximum number of PCA components to test
        device: PyTorch device
        output_dir: Directory to save plots
    """
    print(f'\n{"=" * 50}')
    print('PCA SENSITIVITY ANALYSIS')
    print(f'{"=" * 50}')

    # Load tensors from both databases
    print(f'Loading tensors from {target_db_path}...')
    target_tensors = load_tensors_by_layer(target_db_path, device)

    print(f'Loading tensors from {concept_db_path}...')
    concept_tensors = load_tensors_by_layer(concept_db_path, device)

    # Find common layers
    common_layers = set(target_tensors.keys()) & set(concept_tensors.keys())

    # Determine which layers to analyze
    if layer_names is None:
        layers_to_analyze = sorted(common_layers)
    else:
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        layers_to_analyze = [layer for layer in layer_names if layer in common_layers]

    os.makedirs(output_dir, exist_ok=True)

    # Process each layer
    for layer in layers_to_analyze:
        print(f'\nProcessing layer: {layer}')

        target_layer_tensors = target_tensors[layer]
        concept_layer_tensors = concept_tensors[layer]

        if not target_layer_tensors or not concept_layer_tensors:
            print(f'Skipping layer {layer} - insufficient data')
            continue

        # Determine actual max components based on data
        concept_arrays = [data[0] for data in concept_layer_tensors]
        if not concept_arrays:
            continue

        n_features = concept_arrays[0].shape[0]
        n_samples = len(concept_arrays)
        actual_max_components = min(max_components, n_features, n_samples)

        print(f'  Features: {n_features}, Samples: {n_samples}')
        print(f'  Testing PCA components: 1 to {actual_max_components}')

        # Component range to test
        component_range = range(1, actual_max_components + 1)

        # Store results for each target image
        target_results: dict[str, dict[str, Any]] = {}

        # Test each number of components
        for n_comp in component_range:
            print(f'  Testing {n_comp} components...', end='', flush=True)

            # Apply PCA with n_comp components
            transformed_targets, transformed_concepts, _ = apply_pca_to_layer(
                target_layer_tensors, concept_layer_tensors, n_comp
            )

            # Analyze similarities
            layer_results = analyze_target_vs_concepts(
                transformed_targets, transformed_concepts, layer
            )

            # Store results for each target
            for result in layer_results:
                target_id = result['target_row_id']
                target_name = result['target_image_filename'] or f'Target_{target_id}'

                if target_name not in target_results:
                    target_results[target_name] = {
                        'n_components': [],
                        'concept_similarities': defaultdict(list)
                    }

                target_results[target_name]['n_components'].append(n_comp)

                # Store centroid similarities for each concept
                for concept_name, stats in result['concept_analysis'].items():
                    if 'centroid_similarity' in stats:
                        similarity = stats['centroid_similarity']
                        target_results[target_name]['concept_similarities'][concept_name].append(similarity)

            print(' done')

        # Create plots for this layer
        if target_results:
            # Get all concepts from the first target
            first_target = next(iter(target_results.values()))
            all_concepts = list(first_target['concept_similarities'].keys())

            # Create subplots - one for each concept
            n_concepts = len(all_concepts)
            n_cols = min(3, n_concepts)
            n_rows = (n_concepts + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            fig.suptitle(f'Centroid Similarity vs PCA Components - Layer: {layer}', fontsize=16)

            if n_concepts == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_concepts > 1 else [axes]
            else:
                axes = axes.flatten()

            # Plot each concept
            for concept_idx, concept_name in enumerate(all_concepts):
                ax = axes[concept_idx] if concept_idx < len(axes) else None
                if ax is None:
                    continue

                # Plot lines for each target image
                for target_name, target_data in target_results.items():
                    n_components = target_data['n_components']
                    similarities = target_data['concept_similarities'][concept_name]

                    if len(similarities) == len(n_components):
                        # Clean target name for legend
                        clean_target_name = os.path.splitext(os.path.basename(target_name))[0]
                        ax.plot(n_components, similarities,
                                marker='o', markersize=3, linewidth=1.5,
                                label=clean_target_name, alpha=0.8)

                ax.set_xlabel('Number of PCA Components')
                ax.set_ylabel('Centroid Similarity')
                ax.set_title(f'Concept: {concept_name}')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

            # Hide unused subplots
            for idx in range(n_concepts, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            # Save plot
            plot_filename = f'{output_dir}/pca_sensitivity_layer_{layer.replace("/", "_")}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f'  Plot saved: {plot_filename}')

        else:
            print(f'  No results to plot for layer {layer}')

    print(f'\nPCA sensitivity analysis complete. Plots saved in {output_dir}/')


if __name__ == '__main__':
    # Configuration
    target_db_path = 'output/llava.db'
    concept_db_path = 'output/llava-7b-concepts-colors.db'

    # Analysis parameters
    layer_names = None  # None for all layers, or specify: ['layer_name1', 'layer_name2']
    n_pca_components = None  # None for raw embeddings, or specify: 5, 10, etc. (production: use None)

    print('=' * 60)
    print('CONCEPT-BASED VLM EMBEDDING ANALYSIS')
    print('=' * 60)

    try:
        # Run main analysis
        results = concept_similarity_analysis(
            target_db_path=target_db_path,
            concept_db_path=concept_db_path,
            layer_names=layer_names,
            n_pca_components=n_pca_components,
            device='cpu'
        )

        if results:
            # Save detailed results
            output_file = 'output/concept_similarity_analysis.txt'
            save_concept_analysis_results(results, output_file)

            # Show aggregate trends
            analyze_concept_trends(results)

            print(f'\n{"=" * 50}')
            print('ANALYSIS COMPLETE')
            print(f'{"=" * 50}')
            print(f'Processed {len(results)} layers')
            print(f'Results saved to: {output_file}')

        else:
            print('No results generated. Check database compatibility and parameters.')

        # Run PCA sensitivity analysis (separate from main analysis)
        print(f'\n{"=" * 60}')
        print('STARTING PCA SENSITIVITY ANALYSIS')
        print(f'{"=" * 60}')

        plot_pca_sensitivity_analysis(
            target_db_path=target_db_path,
            concept_db_path=concept_db_path,
            layer_names=layer_names,  # Same layers as main analysis
            max_components=50,        # Adjust based on your data size
            device='cpu',
            output_dir='output'
        )

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
