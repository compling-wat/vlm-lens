"""Instance-based k-NN extension for VLM concept analysis.

This module extends the existing VLM concept analysis with nearest-neighbor
prototype-based classification. It reuses the existing functions and adds
instance-based readout capabilities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
# Import from the existing analysis module
from pca import (analyze_concept_trends, cosine_similarity_numpy,
                 extract_concept_from_filename, group_tensors_by_concept,
                 load_tensors_by_layer)


def _build_normalized_prototype_bank(
    concept_tensors: list[tuple[np.ndarray, Any, int, str]]
) -> tuple[Optional[np.ndarray], list[dict[str, Any]]]:
    """Build an (N,d) bank of L2-normalized prototype vectors and metadata.

    Args:
        concept_tensors: List of tuples (vec, label, row_id, image_path)

    Returns:
        Tuple of (X matrix (N,d), meta list of dicts with concept/row_id/image_path)
    """
    X_list, meta = [], []
    for vec, label, row_id, image_path in concept_tensors:
        if vec is None:
            continue
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm == 0:
            continue
        X_list.append(vec / norm)
        meta.append({
            'concept': extract_concept_from_filename(image_path),
            'row_id': row_id,
            'image_path': image_path,
            'label': label
        })
    if not X_list:
        return None, []
    X = np.vstack(X_list)
    return X, meta


def _nearest_prototypes(
    target_vec: np.ndarray,
    X_bank: Optional[np.ndarray],
    meta: list[dict[str, Any]],
    topk: int = 5
) -> list[dict[str, Any]]:
    """Compute cosine similarities target vs all prototypes (already normalized).

    Args:
        target_vec: Target vector (d,), will be L2-normalized here
        X_bank: Prototype bank matrix (N, d), already normalized
        meta: List of metadata dicts for each prototype
        topk: Number of top neighbors to return

    Returns:
        Top list of dicts sorted by similarity with keys:
        ['concept', 'row_id', 'image_path', 'label', 'sim']
    """
    if X_bank is None or len(meta) == 0:
        return []

    # L2-normalize target
    t = target_vec
    t_norm = np.linalg.norm(t)
    if not np.isfinite(t_norm) or t_norm == 0:
        return []

    t = t / t_norm
    sims = X_bank @ t  # cosine since both normalized

    k = min(topk, sims.shape[0])
    # argpartition is O(N); then sort the small top-k slice
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    out = []
    for i in idx:
        m = meta[i]
        out.append({
            'concept': m['concept'],
            'row_id': m['row_id'],
            'image_path': m['image_path'],
            'label': m['label'],
            'sim': float(sims[i]),
        })
    return out


def _knn_weighted_vote(
    neighbors: list[dict[str, Any]],
    p: float = 1.0
) -> tuple[Optional[str], dict[str, float]]:
    """Weighted majority vote over top-k neighbors.

    Args:
        neighbors: List of neighbor dicts with 'concept' and 'sim' keys
        p: Power for weighting (weight = sim^p, negatives clipped to 0)

    Returns:
        Tuple of (winner_concept, score_dict)
    """
    wsum = defaultdict(float)
    for nb in neighbors:
        w = max(0.0, nb['sim']) ** p
        wsum[nb['concept']] += w
    if not wsum:
        return None, {}
    winner = max(wsum.items(), key=lambda kv: kv[1])[0]
    return winner, dict(wsum)


def analyze_target_vs_concepts_with_knn(
    target_tensors: list[tuple[np.ndarray, Any, int, str]],
    concept_tensors: list[tuple[np.ndarray, Any, int, str]],
    layer_name: str,
    knn_topk: int = 5,
    knn_power: float = 1.0
) -> list[dict[str, Any]]:
    """Analyze similarity between targets and concepts with k-NN instance-based prediction.

    Keeps existing per-prototype stats and centroid metrics.
    Adds instance-based nearest-neighbor prediction (1-NN + k-NN vote).

    Args:
        target_tensors: List of target tensor data
        concept_tensors: List of concept tensor data
        layer_name: Name of the current layer
        knn_topk: Number of nearest neighbors to consider
        knn_power: Power for weighted voting (weight = sim^p)

    Returns:
        List of analysis results with added 'instance_knn' section
    """
    # Group by concept (existing behavior)
    concept_groups = group_tensors_by_concept(concept_tensors)
    print(f'Found {len(concept_groups)} concepts: {list(concept_groups.keys())}')
    for concept, tensors in concept_groups.items():
        print(f'  {concept}: {len(tensors)} images')

    # Precompute centroids (as before)
    concept_centroids = {}
    for concept_name, tensor_list in concept_groups.items():
        vecs = [t[0] for t in tensor_list]
        if len(vecs) > 0:
            concept_centroids[concept_name] = np.mean(np.vstack(vecs), axis=0)
        else:
            concept_centroids[concept_name] = None

    # NEW: build prototype bank once for this layer
    X_bank, bank_meta = _build_normalized_prototype_bank(concept_tensors)
    if X_bank is None:
        print('Warning: prototype bank is empty for this layer; skipping instance-NN.')

    results = []

    for target_data in target_tensors:
        target_vec, target_label, target_row_id, target_image_filename = target_data

        target_result = {
            'layer': layer_name,
            'target_row_id': target_row_id,
            'target_label': target_label,
            'target_image_filename': target_image_filename,
            'concept_analysis': {},   # existing per-concept stats live here
            'instance_knn': {}        # NEW: instance-based readout lives here
        }

        # --- Existing per-concept stats (unchanged) ---
        for concept_name, concept_tensor_list in concept_groups.items():
            similarities = []
            for concept_data in concept_tensor_list:
                concept_vec, concept_label, concept_row_id, concept_image_filename = concept_data
                if target_vec.shape != concept_vec.shape:
                    print(f'Warning: Shape mismatch between target {target_row_id} and concept {concept_row_id}')
                    continue
                sim = cosine_similarity_numpy(target_vec, concept_vec)
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

            centroid = concept_centroids.get(concept_name, None)
            if centroid is not None and centroid.shape == target_vec.shape:
                cen_sim = cosine_similarity_numpy(target_vec, centroid)
                cen_ang = float(np.degrees(np.arccos(np.clip(cen_sim, -1.0, 1.0))))
                concept_stats.update({
                    'centroid_similarity': float(cen_sim),
                    'centroid_angular_deg': cen_ang
                })

            if concept_stats:
                target_result['concept_analysis'][concept_name] = concept_stats

        # --- NEW: instance-based nearest neighbor prediction ---
        if X_bank is not None:
            nbs = _nearest_prototypes(target_vec, X_bank, bank_meta, topk=knn_topk)
            winner_1nn = nbs[0]['concept'] if nbs else None
            voted, vote_scores = _knn_weighted_vote(nbs, p=knn_power) if nbs else (None, {})

            target_result['instance_knn'] = {
                'top1_concept': winner_1nn,
                'top1_similarity': nbs[0]['sim'] if nbs else None,
                'topk_neighbors': nbs,            # list with concept,row_id,image_path,sim
                'topk_voted_concept': voted,      # weighted by sim^p over topk (non-negative)
                'vote_scores': vote_scores,       # dict concept->weight
                'topk': knn_topk,
                'vote_power': knn_power
            }

        results.append(target_result)

        target_display = target_image_filename if target_image_filename else f'Target_{target_row_id}'
        print(f'Analyzed {target_display} against {len(concept_groups)} concepts')

    return results


def concept_similarity_analysis_with_knn(
    target_db_path: str,
    concept_db_path: str,
    layer_names: Optional[list[str]] = None,
    n_pca_components: Optional[int] = None,
    knn_topk: int = 5,
    knn_power: float = 1.0,
    device: str = 'cpu'
) -> dict[str, dict[str, Any]]:
    """Main function for concept-based similarity analysis with k-NN prediction.

    Args:
        target_db_path: Path to target images database
        concept_db_path: Path to concept images database
        layer_names: List of layer names to analyze (None for all common layers)
        n_pca_components: Number of PCA components (None to skip PCA)
        knn_topk: Number of nearest neighbors for k-NN prediction
        knn_power: Power for weighted voting in k-NN
        device: PyTorch device

    Returns:
        Dictionary of analysis results by layer with k-NN predictions
    """
    print('Starting concept-based similarity analysis with k-NN...')
    print(f'Target DB: {target_db_path}')
    print(f'Concept DB: {concept_db_path}')
    print(f'PCA components: {n_pca_components}')
    print(f'k-NN parameters: topk={knn_topk}, power={knn_power}')

    # Load tensors from both databases (reuse existing function)
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

        # Apply PCA if requested (reuse existing function)
        if n_pca_components is not None:
            # Import the PCA function
            from pca import apply_pca_to_layer
            target_layer_tensors, concept_layer_tensors, pca_model = apply_pca_to_layer(
                target_layer_tensors, concept_layer_tensors, n_pca_components
            )
        else:
            pca_model = None

        # Analyze similarities with k-NN
        layer_results = analyze_target_vs_concepts_with_knn(
            target_layer_tensors, concept_layer_tensors, layer,
            knn_topk=knn_topk, knn_power=knn_power
        )

        all_results[layer] = {
            'results': layer_results,
            'pca_model': pca_model,
            'n_pca_components': n_pca_components,
            'knn_topk': knn_topk,
            'knn_power': knn_power
        }

        # Print layer summary
        if layer_results:
            print(f"\nLayer \'{layer}\' Summary:")
            print(f'  Analyzed {len(layer_results)} target images')

            # Get all concept names from first result
            if layer_results[0]['concept_analysis']:
                concept_names = list(layer_results[0]['concept_analysis'].keys())
                print(f'  Against {len(concept_names)} concepts: {concept_names}')

            # Print k-NN summary
            knn_predictions = []
            for result in layer_results:
                ik = result.get('instance_knn', {})
                if ik.get('top1_concept'):
                    knn_predictions.append(ik['top1_concept'])

            if knn_predictions:
                from collections import Counter
                pred_counts = Counter(knn_predictions)
                print(f'  k-NN Predictions: {dict(pred_counts)}')

    return all_results


def save_knn_analysis_results(
    results: dict[str, dict[str, Any]],
    output_file: str = 'output/knn_similarity_analysis.txt'
) -> None:
    """Save k-NN analysis results to a text file.

    Args:
        results: Dictionary of analysis results by layer
        output_file: Output filename
    """
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('VLM Concept Analysis with Instance-based k-NN Prediction\n')
        f.write('=' * 60 + '\n\n')

        for layer, layer_data in results.items():
            layer_results = layer_data['results']
            n_pca_components = layer_data['n_pca_components']
            knn_topk = layer_data.get('knn_topk', 5)
            knn_power = layer_data.get('knn_power', 1.0)

            f.write(f'Layer: {layer}\n')
            if n_pca_components:
                f.write(f'PCA Components: {n_pca_components}\n')
            f.write(f'k-NN Parameters: topk={knn_topk}, power={knn_power}\n')
            f.write('-' * 40 + '\n\n')

            for result in layer_results:
                target_display = result['target_image_filename'] or f'Target_{result["target_row_id"]}'
                f.write(f'Target: {target_display}\n')

                # k-NN predictions
                ik = result.get('instance_knn', {})
                if ik:
                    f.write(f'  1-NN Concept: {ik.get("top1_concept")}  (sim={ik.get("top1_similarity", 0):.4f})\n')
                    if ik.get('topk_voted_concept') is not None and ik.get('topk', 1) > 1:
                        f.write(f'  k-NN Vote (k={ik["topk"]}, p={ik["vote_power"]}): {ik["topk_voted_concept"]}\n')

                    # Show top neighbors
                    neighbors = ik.get('topk_neighbors', [])
                    if neighbors:
                        f.write('  Top Neighbors:\n')
                        for i, nb in enumerate(neighbors[:3], 1):  # Show top 3
                            f.write(f'    {i}. {nb["concept"]} (sim={nb["sim"]:.4f})\n')

                # Original concept analysis
                for concept_name, stats in result['concept_analysis'].items():
                    f.write(f'  Concept {concept_name}:\n')
                    if 'centroid_similarity' in stats:
                        f.write(f'    Centroid Similarity: {stats["centroid_similarity"]:.4f}\n')
                    if 'mean_similarity' in stats:
                        f.write(f'    Mean Similarity: {stats["mean_similarity"]:.4f}\n')
                f.write('\n')

            f.write('\n')

    print(f'k-NN results saved to {output_file}')


def analyze_knn_accuracy(
    results: dict[str, dict[str, Any]],
    ground_truth_concept_extractor: Optional[callable] = None
) -> None:
    """Analyze k-NN prediction accuracy if ground truth is available.

    Args:
        results: Dictionary of analysis results by layer
        ground_truth_concept_extractor: Function to extract true concept from target filename
    """
    if ground_truth_concept_extractor is None:
        ground_truth_concept_extractor = extract_concept_from_filename

    print(f'\n{"=" * 50}')
    print('k-NN PREDICTION ACCURACY ANALYSIS')
    print(f'{"=" * 50}')

    for layer, layer_data in results.items():
        layer_results = layer_data['results']
        knn_topk = layer_data.get('knn_topk', 5)

        print(f'\nLayer: {layer}')
        print('-' * 30)

        if not layer_results:
            print('No results for this layer')
            continue

        correct_1nn = 0
        correct_knn = 0
        total = 0

        for result in layer_results:
            # Extract ground truth
            true_concept = ground_truth_concept_extractor(result['target_image_filename'])
            if not true_concept:
                continue

            ik = result.get('instance_knn', {})
            if not ik:
                continue

            total += 1

            # Check 1-NN accuracy
            pred_1nn = ik.get('top1_concept')
            if pred_1nn == true_concept:
                correct_1nn += 1

            # Check k-NN vote accuracy
            pred_knn = ik.get('topk_voted_concept')
            if pred_knn == true_concept:
                correct_knn += 1

        if total > 0:
            acc_1nn = correct_1nn / total
            acc_knn = correct_knn / total
            print(f'  1-NN Accuracy: {correct_1nn}/{total} = {acc_1nn:.3f}')
            print(f'  k-NN Accuracy (k={knn_topk}): {correct_knn}/{total} = {acc_knn:.3f}')
        else:
            print('  No valid predictions to evaluate')


if __name__ == '__main__':
    # Configuration
    target_db_path = 'output/llava.db'
    concept_db_path = 'output/llava-colors.db'

    # Analysis parameters
    layer_names = None  # None for all layers
    n_pca_components = 5  # None for raw embeddings
    knn_topk = 5
    knn_power = 1.0

    print('=' * 60)
    print('VLM CONCEPT ANALYSIS WITH INSTANCE-BASED k-NN')
    print('=' * 60)

    try:
        # Run k-NN analysis
        results = concept_similarity_analysis_with_knn(
            target_db_path=target_db_path,
            concept_db_path=concept_db_path,
            layer_names=layer_names,
            n_pca_components=n_pca_components,
            knn_topk=knn_topk,
            knn_power=knn_power,
            device='cpu'
        )

        if results:
            # Save detailed results
            output_file = 'output/knn_similarity_analysis.txt'
            save_knn_analysis_results(results, output_file)

            # Analyze k-NN accuracy
            analyze_knn_accuracy(results)

            # Show aggregate trends (reuse existing function)
            analyze_concept_trends(results)

            print(f'\n{"=" * 50}')
            print('k-NN ANALYSIS COMPLETE')
            print(f'{"=" * 50}')
            print(f'Processed {len(results)} layers')
            print(f'Results saved to: {output_file}')

        else:
            print('No results generated. Check database compatibility and parameters.')

    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
