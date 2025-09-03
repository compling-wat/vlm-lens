"""PCA scatter plot visualization for VLM concept analysis.

Creates 2D scatter plots of concepts and targets in PCA space for interpretability.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pca import (apply_pca_to_layer, extract_concept_from_filename,
                 group_tensors_by_concept, load_tensors_by_layer)


def create_pca_scatter_plots(
    target_db_path: str,
    concept_db_path: str,
    layer_names: Optional[list[str]] = None,
    output_dir: str = 'output',
    figsize: tuple[int, int] = (12, 8),
    alpha: float = 0.7,
    target_marker_size: int = 100,
    concept_marker_size: int = 50
) -> None:
    """Create 2D PCA scatter plots for concepts and targets.

    Args:
        target_db_path: Path to target images database
        concept_db_path: Path to concept images database
        layer_names: List of layer names to visualize (None for all layers)
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
        alpha: Transparency for concept points
        target_marker_size: Size of target markers
        concept_marker_size: Size of concept markers
    """
    print('Creating PCA scatter plots...')

    # Load tensors from both databases
    print(f'Loading tensors from {target_db_path}...')
    target_tensors = load_tensors_by_layer(target_db_path, 'cpu')

    print(f'Loading tensors from {concept_db_path}...')
    concept_tensors = load_tensors_by_layer(concept_db_path, 'cpu')

    # Find common layers
    common_layers = set(target_tensors.keys()) & set(concept_tensors.keys())
    print(f'Found {len(common_layers)} common layers: {sorted(common_layers)}')

    if not common_layers:
        print('No common layers found between databases!')
        return

    # Determine which layers to visualize
    if layer_names is None:
        layers_to_analyze = sorted(common_layers)
    else:
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        layers_to_analyze = [layer for layer in layer_names if layer in common_layers]

    os.makedirs(output_dir, exist_ok=True)

    # Create plots for each layer
    for layer in layers_to_analyze:
        print(f'\nProcessing layer: {layer}')

        target_layer_tensors = target_tensors[layer]
        concept_layer_tensors = concept_tensors[layer]

        if not target_layer_tensors or not concept_layer_tensors:
            print(f'Skipping layer {layer} - insufficient data')
            continue

        # Apply PCA with 2 components
        print('  Applying PCA with 2 components...')
        transformed_targets, transformed_concepts, pca_model = apply_pca_to_layer(
            target_layer_tensors, concept_layer_tensors, n_components=2
        )

        if pca_model is None:
            print(f'  Failed to apply PCA for layer {layer}')
            continue

        # Group concepts for coloring
        concept_groups = group_tensors_by_concept(transformed_concepts)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Define colors for concepts (use a colormap)
        concept_names = sorted(concept_groups.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(concept_names)))
        color_map = dict(zip(concept_names, colors))

        # Plot concept prototypes
        for concept_name, concept_data in concept_groups.items():
            concept_coords = np.array([data[0] for data in concept_data])

            ax.scatter(
                concept_coords[:, 0],
                concept_coords[:, 1],
                c=[color_map[concept_name]],
                s=concept_marker_size,
                alpha=alpha,
                label=f'{concept_name} (prototypes)',
                marker='o',
                edgecolors='white',
                linewidth=0.5
            )

        # Plot targets
        target_coords = np.array([data[0] for data in transformed_targets])
        target_concepts = []

        # Extract target concepts for coloring
        for data in transformed_targets:
            target_concept = extract_concept_from_filename(data[3])  # data[3] is image_filename
            target_concepts.append(target_concept)

        # Plot targets with concept-based coloring
        for i, (coord, target_concept) in enumerate(zip(target_coords, target_concepts)):
            if target_concept in color_map:
                color = color_map[target_concept]
                label = f'{target_concept} (target)' if i == 0 or target_concept != target_concepts[i-1] else None
            else:
                color = 'black'
                label = 'Unknown (target)' if i == 0 else None

            ax.scatter(
                coord[0],
                coord[1],
                c=[color],
                s=target_marker_size,
                alpha=0.9,
                marker='^',  # Triangle for targets
                edgecolors='black',
                linewidth=1.0,
                label=label
            )

        # Customize the plot
        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.3f} variance explained)')
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.3f} variance explained)')
        ax.set_title(f'PCA Visualization: Concepts vs Targets\nLayer: {layer}')
        ax.grid(True, alpha=0.3)

        # Create legend with better organization
        handles, labels = ax.get_legend_handles_labels()

        # Separate prototype and target entries
        prototype_handles, prototype_labels = [], []
        target_handles, target_labels = [], []

        for handle, label in zip(handles, labels):
            if '(prototypes)' in label:
                prototype_handles.append(handle)
                prototype_labels.append(label.replace(' (prototypes)', ''))
            elif '(target)' in label:
                target_handles.append(handle)
                target_labels.append(label.replace(' (target)', ''))

        # Create two-column legend
        if prototype_handles and target_handles:
            legend1 = ax.legend(
                prototype_handles,
                [f'{label} (○)' for label in prototype_labels],
                title='Concept Prototypes',
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                fontsize=9
            )
            ax.add_artist(legend1)

            ax.legend(
                target_handles,
                [f'{label} (△)' for label in target_labels],
                title='Target Images',
                loc='upper left',
                bbox_to_anchor=(1.02, 0.6),
                fontsize=9
            )
        else:
            ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9)

        # Add statistics text
        stats_text = (
            f'Total variance explained: {pca_model.explained_variance_ratio_.sum():.3f}\n'
            f'Concepts: {len(concept_groups)}\n'
            f'Prototypes: {len(transformed_concepts)}\n'
            f'Targets: {len(transformed_targets)}'
        )

        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9
        )

        plt.tight_layout()

        # Save plot
        plot_filename = f'{output_dir}/pca_scatter_layer_{layer.replace("/", "_")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'  Plot saved: {plot_filename}')

        # Print summary statistics
        print(f'  Variance explained: PC1={pca_model.explained_variance_ratio_[0]:.3f}, '
              f'PC2={pca_model.explained_variance_ratio_[1]:.3f}, '
              f'Total={pca_model.explained_variance_ratio_.sum():.3f}')
        print(f'  Plotted {len(concept_groups)} concept groups with {len(transformed_concepts)} prototypes')
        print(f'  Plotted {len(transformed_targets)} target images')

    print(f'\nPCA scatter plots complete. Plots saved in {output_dir}/')


def create_concept_separation_analysis(
    target_db_path: str,
    concept_db_path: str,
    layer_names: Optional[list[str]] = None,
    output_dir: str = 'output'
) -> None:
    """Analyze concept separation in PCA space.

    Args:
        target_db_path: Path to target images database
        concept_db_path: Path to concept images database
        layer_names: List of layer names to analyze (None for all layers)
        output_dir: Directory to save analysis
    """
    print('\nAnalyzing concept separation in PCA space...')

    # Load tensors
    target_tensors = load_tensors_by_layer(target_db_path, 'cpu')
    concept_tensors = load_tensors_by_layer(concept_db_path, 'cpu')

    common_layers = set(target_tensors.keys()) & set(concept_tensors.keys())

    if layer_names is None:
        layers_to_analyze = sorted(common_layers)
    else:
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        layers_to_analyze = [layer for layer in layer_names if layer in common_layers]

    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/pca_separation_analysis.txt', 'w') as f:
        f.write('PCA Concept Separation Analysis\n')
        f.write('=' * 40 + '\n\n')

        for layer in layers_to_analyze:
            target_layer_tensors = target_tensors[layer]
            concept_layer_tensors = concept_tensors[layer]

            if not concept_layer_tensors:
                continue

            # Apply PCA
            _, transformed_concepts, pca_model = apply_pca_to_layer(
                target_layer_tensors, concept_layer_tensors, n_components=2
            )

            if pca_model is None:
                continue

            f.write(f'Layer: {layer}\n')
            f.write('-' * 20 + '\n')

            # Group concepts
            concept_groups = group_tensors_by_concept(transformed_concepts)

            # Calculate concept centroids in PCA space
            concept_centroids = {}
            for concept_name, concept_data in concept_groups.items():
                coords = np.array([data[0] for data in concept_data])
                concept_centroids[concept_name] = np.mean(coords, axis=0)

            # Calculate pairwise distances between concept centroids
            concept_names = list(concept_centroids.keys())
            f.write('Concept centroid distances in PC1-PC2 space:\n')

            for i, concept1 in enumerate(concept_names):
                for j, concept2 in enumerate(concept_names[i+1:], i+1):
                    centroid1 = concept_centroids[concept1]
                    centroid2 = concept_centroids[concept2]
                    distance = np.linalg.norm(centroid1 - centroid2)
                    f.write(f'  {concept1} - {concept2}: {distance:.3f}\n')

            # Calculate within-concept scatter
            f.write('\nWithin-concept scatter (std dev):\n')
            for concept_name, concept_data in concept_groups.items():
                coords = np.array([data[0] for data in concept_data])
                if len(coords) > 1:
                    std_pc1 = np.std(coords[:, 0])
                    std_pc2 = np.std(coords[:, 1])
                    f.write(f'  {concept_name}: PC1={std_pc1:.3f}, PC2={std_pc2:.3f}\n')

            f.write('\nPCA Statistics:\n')
            f.write(f'  PC1 variance explained: {pca_model.explained_variance_ratio_[0]:.3f}\n')
            f.write(f'  PC2 variance explained: {pca_model.explained_variance_ratio_[1]:.3f}\n')
            f.write(f'  Total variance explained: {pca_model.explained_variance_ratio_.sum():.3f}\n')
            f.write('\n\n')

    print(f'Separation analysis saved to {output_dir}/pca_separation_analysis.txt')


if __name__ == '__main__':
    # Configuration
    target_db_path = 'output/llava.db'
    concept_db_path = 'output/llava-7b-concepts-colors.db'

    # Visualization parameters
    layer_names = None  # None for all layers, or specify: ['layer_name1', 'layer_name2']

    print('=' * 60)
    print('VLM PCA VISUALIZATION')
    print('=' * 60)

    try:
        # Create scatter plots
        create_pca_scatter_plots(
            target_db_path=target_db_path,
            concept_db_path=concept_db_path,
            layer_names=layer_names,
            output_dir='output',
            figsize=(12, 8),
            alpha=0.7,
            target_marker_size=100,
            concept_marker_size=50
        )

        # Analyze concept separation
        create_concept_separation_analysis(
            target_db_path=target_db_path,
            concept_db_path=concept_db_path,
            layer_names=layer_names,
            output_dir='output'
        )

        print('\nVisualization complete!')

    except Exception as e:
        print(f'Error during visualization: {e}')
        import traceback
        traceback.print_exc()
