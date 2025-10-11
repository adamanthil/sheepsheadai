#!/usr/bin/env python3
"""
Analyze trained card embeddings from a PPOAgent's StateEncoder.

Compares learned embeddings against initialized values and produces
detailed visualizations of how the model has refined its card representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Tuple
import csv

from state_encoder import StateEncoder, CardEmbeddingConfig, PAD_CARD_ID, UNDER_CARD_ID
from sheepshead import DECK_IDS, TRUMP


def get_initial_embeddings(d_card: int = 12) -> torch.Tensor:
    """Get initialized card embeddings (before any training)."""
    config = CardEmbeddingConfig(use_informed_init=True, d_card=d_card)
    encoder = StateEncoder(card_config=config)
    return encoder.card.weight.data.clone()


def load_trained_embeddings(checkpoint_path: str) -> Tuple[torch.Tensor, int]:
    """Load card embeddings from a trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_encoder_state_dict']

    # Extract card embedding weights
    card_weights = state_dict['card.weight']
    d_card = card_weights.shape[1]

    return card_weights, d_card


def compute_embedding_delta(initial: torch.Tensor, trained: torch.Tensor) -> torch.Tensor:
    """Compute change in embeddings from initialization to trained state."""
    return trained - initial


def get_card_names_ordered(include_special: bool = True) -> list:
    """Get card names in a logical order.

    If include_special is True, include PAD and UNDER at the top.
    Then list trumps (in strength order) followed by fail suits.
    """
    names: list[str] = []
    if include_special:
        # By convention in this project: PAD index 0, UNDER index 33
        names.extend(["PAD", "UNDER"])  # symbolic names; not in DECK_IDS

    # Trump cards in strength order
    trump_cards = TRUMP

    # Fail cards by suit
    FAIL_ORDER = ["A", "10", "K", "9", "8", "7"]
    fail_cards = []
    for suit in ['C', 'S', 'H']:
        for rank in FAIL_ORDER:
            card = rank + suit
            if card in DECK_IDS:
                fail_cards.append(card)

    names.extend(trump_cards + fail_cards)
    return names


def name_to_id(name: str) -> int:
    """Map a human-readable card name (or special token) to its embedding row index."""
    if name == 'PAD':
        return PAD_CARD_ID
    if name == 'UNDER':
        return UNDER_CARD_ID
    return DECK_IDS[name]


def write_embeddings_csv(path: Path, embeddings: torch.Tensor, card_names: list, card_ids: list):
    """Write embeddings to CSV with header: name,id,D0..D{d-1}."""
    d = embeddings.size(1)
    header = ['name', 'id'] + [f'D{i}' for i in range(d)]
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, cid in zip(card_names, card_ids):
            row = [name, cid] + [float(x) for x in embeddings[cid].tolist()]
            writer.writerow(row)


def analyze_dimension_importance(delta: torch.Tensor, card_ids: list) -> Dict[str, np.ndarray]:
    """Analyze which embedding dimensions changed the most during training."""
    # Exclude PAD (0) and UNDER (33)
    real_card_delta = delta[card_ids, :]  # (32, d_card)

    # Compute various metrics per dimension
    mean_abs_change = torch.abs(real_card_delta).mean(dim=0).numpy()
    std_change = real_card_delta.std(dim=0).numpy()
    max_abs_change = torch.abs(real_card_delta).max(dim=0)[0].numpy()

    return {
        'mean_abs_change': mean_abs_change,
        'std_change': std_change,
        'max_abs_change': max_abs_change,
    }


def analyze_card_movement(delta: torch.Tensor, card_ids: list, card_names: list) -> Dict[str, float]:
    """Compute L2 norm of change for each card."""
    card_movements = {}
    for cid, name in zip(card_ids, card_names):
        l2_norm = torch.norm(delta[cid]).item()
        card_movements[name] = l2_norm
    return card_movements


def plot_embedding_heatmap(embeddings: torch.Tensor, card_names: list, card_ids: list,
                           title: str, filename: str, vmin=None, vmax=None):
    """Create heatmap visualization of card embeddings."""
    # Extract embeddings for selected cards (may include PAD/UNDER)
    card_embeds = embeddings[card_ids, :].numpy()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    im = ax.imshow(card_embeds, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_yticks(range(len(card_names)))
    ax.set_yticklabels(card_names, fontsize=9)
    ax.set_xlabel('Embedding Dimension', fontsize=11)
    ax.set_ylabel('Card', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=10)

    # Add grid for readability
    ax.set_xticks(np.arange(card_embeds.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(card_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_dimension_importance(importance_dict: Dict[str, np.ndarray], filename: str):
    """Plot which embedding dimensions changed most during training."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    dim_labels = [f"D{i}" for i in range(len(importance_dict['mean_abs_change']))]

    # Known dimension names for first 7
    known_dims = ['Trump', 'Clubs', 'Spades', 'Hearts', 'Rank', 'Points', 'Under']
    for i, label in enumerate(known_dims):
        if i < len(dim_labels):
            dim_labels[i] = f"D{i}\n{label}"

    # Mean absolute change
    axes[0].bar(range(len(dim_labels)), importance_dict['mean_abs_change'], color='steelblue')
    axes[0].set_ylabel('Mean |Δ|', fontsize=11)
    axes[0].set_title('Mean Absolute Change per Dimension', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(dim_labels)))
    axes[0].set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Standard deviation of change
    axes[1].bar(range(len(dim_labels)), importance_dict['std_change'], color='coral')
    axes[1].set_ylabel('Std Dev', fontsize=11)
    axes[1].set_title('Standard Deviation of Change per Dimension', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(dim_labels)))
    axes[1].set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    # Max absolute change
    axes[2].bar(range(len(dim_labels)), importance_dict['max_abs_change'], color='mediumseagreen')
    axes[2].set_ylabel('Max |Δ|', fontsize=11)
    axes[2].set_title('Maximum Absolute Change per Dimension', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(dim_labels)))
    axes[2].set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_card_movement(movements: Dict[str, float], filename: str):
    """Plot L2 norm of embedding change for each card."""
    # Sort by movement magnitude
    sorted_cards = sorted(movements.items(), key=lambda x: x[1], reverse=True)
    card_names, movement_vals = zip(*sorted_cards)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Color code: special tokens, trump, fail
    def color_for(card: str) -> str:
        if card == 'PAD':
            return 'gray'
        if card == 'UNDER':
            return 'black'
        return 'darkred' if card in TRUMP else 'steelblue'

    colors = [color_for(card) for card in card_names]

    ax.bar(range(len(card_names)), movement_vals, color=colors, alpha=0.8)
    ax.set_ylabel('L2 Norm of Change', fontsize=12)
    ax.set_xlabel('Card', fontsize=12)
    ax.set_title('Embedding Movement per Card (Sorted by Magnitude)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(card_names)))
    ax.set_xticklabels(card_names, rotation=90, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, label='PAD'),
        Patch(facecolor='black', alpha=0.8, label='UNDER'),
        Patch(facecolor='darkred', alpha=0.8, label='Trump'),
        Patch(facecolor='steelblue', alpha=0.8, label='Fail'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_trump_vs_fail_analysis(initial: torch.Tensor, trained: torch.Tensor,
                                 card_names: list, card_ids: list, filename: str):
    """Compare trump and fail card embeddings."""
    # Identify groups by name
    trump_ids = [card_ids[i] for i, name in enumerate(card_names) if name in TRUMP]
    fail_ids = [card_ids[i] for i, name in enumerate(card_names) if (name not in TRUMP and name not in ('PAD', 'UNDER'))]
    special_ids = [card_ids[i] for i, name in enumerate(card_names) if name in ('PAD', 'UNDER')]

    # Compute average embeddings
    trump_init = initial[trump_ids, :].mean(dim=0).numpy()
    trump_trained = trained[trump_ids, :].mean(dim=0).numpy()
    fail_init = initial[fail_ids, :].mean(dim=0).numpy()
    fail_trained = trained[fail_ids, :].mean(dim=0).numpy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    dim_labels = [f"D{i}" for i in range(len(trump_init))]
    known_dims = ['Trump', 'Clubs', 'Spades', 'Hearts', 'Rank', 'Points', 'Under']
    for i, label in enumerate(known_dims):
        if i < len(dim_labels):
            dim_labels[i] = f"{label}"

    # Trump initial
    axes[0, 0].bar(range(len(trump_init)), trump_init, color='darkred', alpha=0.7)
    axes[0, 0].set_title('Trump Cards - Initial Embedding', fontweight='bold')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].set_xticks(range(len(dim_labels)))
    axes[0, 0].set_xticklabels(dim_labels, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Trump trained
    axes[0, 1].bar(range(len(trump_trained)), trump_trained, color='darkred', alpha=0.7)
    axes[0, 1].set_title('Trump Cards - Trained Embedding', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].set_xticks(range(len(dim_labels)))
    axes[0, 1].set_xticklabels(dim_labels, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Fail initial
    axes[1, 0].bar(range(len(fail_init)), fail_init, color='steelblue', alpha=0.7)
    axes[1, 0].set_title('Fail Cards - Initial Embedding', fontweight='bold')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_xticks(range(len(dim_labels)))
    axes[1, 0].set_xticklabels(dim_labels, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Fail trained
    axes[1, 1].bar(range(len(fail_trained)), fail_trained, color='steelblue', alpha=0.7)
    axes[1, 1].set_title('Fail Cards - Trained Embedding', fontweight='bold')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].set_xticks(range(len(dim_labels)))
    axes[1, 1].set_xticklabels(dim_labels, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Special tokens (PAD/UNDER) - averages over the 2 tokens
    if special_ids:
        special_init = initial[special_ids, :].mean(dim=0).numpy()
        special_trained = trained[special_ids, :].mean(dim=0).numpy()

        axes[2, 0].bar(range(len(special_init)), special_init, color='gray', alpha=0.7)
        axes[2, 0].set_title('Special Tokens - Initial Embedding', fontweight='bold')
        axes[2, 0].set_ylabel('Mean Value')
        axes[2, 0].set_xticks(range(len(dim_labels)))
        axes[2, 0].set_xticklabels(dim_labels, rotation=45, ha='right')
        axes[2, 0].grid(axis='y', alpha=0.3)

        axes[2, 1].bar(range(len(special_trained)), special_trained, color='black', alpha=0.7)
        axes[2, 1].set_title('Special Tokens - Trained Embedding', fontweight='bold')
        axes[2, 1].set_ylabel('Mean Value')
        axes[2, 1].set_xticks(range(len(dim_labels)))
        axes[2, 1].set_xticklabels(dim_labels, rotation=45, ha='right')
        axes[2, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_title('Fail Cards - Trained Embedding', fontweight='bold')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].set_xticks(range(len(dim_labels)))
    axes[1, 1].set_xticklabels(dim_labels, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_special_cards(initial: torch.Tensor, trained: torch.Tensor, delta: torch.Tensor,
                       card_names: list, card_ids: list, filename: str):
    """Focus on special cards: queens, jacks, and aces."""
    special_groups = {
        'Queens': ['QD', 'QH', 'QS', 'QC'],
        'Jacks': ['JD', 'JH', 'JS', 'JC'],
        'Aces': ['AD', 'AH', 'AS', 'AC'],
        'Special Tokens': ['PAD', 'UNDER'],
    }

    fig, axes = plt.subplots(len(special_groups), 1, figsize=(12, 4 * len(special_groups)))

    for idx, (group_name, group_cards) in enumerate(special_groups.items()):
        group_ids = [card_ids[card_names.index(card)] for card in group_cards if card in card_names]
        group_delta = delta[group_ids, :].numpy()

        ax = axes[idx]
        im = ax.imshow(group_delta, aspect='auto', cmap='RdBu_r',
                      vmin=-abs(group_delta).max(), vmax=abs(group_delta).max())

        ax.set_yticks(range(len(group_cards)))
        ax.set_yticklabels([c for c in group_cards if c in card_names])
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Card')
        ax.set_title(f'{group_name} - Embedding Change', fontweight='bold')

        plt.colorbar(im, ax=ax, label='Δ Value')

        # Add grid
        ax.set_xticks(np.arange(group_delta.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(group_cards)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def print_summary_statistics(initial: torch.Tensor, trained: torch.Tensor,
                             delta: torch.Tensor, card_ids: list):
    """Print detailed summary statistics."""
    print("\n" + "="*80)
    print("CARD EMBEDDING ANALYSIS SUMMARY")
    print("="*80)

    # Overall statistics
    real_card_delta = delta[card_ids, :]
    print(f"\nOverall Statistics ({len(card_ids)} cards):")
    print(f"  Mean absolute change:     {torch.abs(real_card_delta).mean():.6f}")
    print(f"  Std dev of change:        {real_card_delta.std():.6f}")
    print(f"  Max absolute change:      {torch.abs(real_card_delta).max():.6f}")
    print(f"  Min change:               {real_card_delta.min():.6f}")
    print(f"  Max change:               {real_card_delta.max():.6f}")

    # Per-card L2 norms
    card_norms = torch.norm(real_card_delta, dim=1)
    print("\n  Mean L2 norm per card:    {:.6f}".format(card_norms.mean()))
    print("  Max L2 norm per card:     {:.6f}".format(card_norms.max()))
    print("  Min L2 norm per card:     {:.6f}".format(card_norms.min()))

    # Frobenius norm (overall embedding matrix change)
    frob_norm = torch.norm(real_card_delta, p='fro')
    print("\n  Frobenius norm (total):   {:.6f}".format(frob_norm))

    # Dimension-wise analysis
    dim_mean_abs = torch.abs(real_card_delta).mean(dim=0)
    most_changed_dim = torch.argmax(dim_mean_abs).item()
    least_changed_dim = torch.argmin(dim_mean_abs).item()

    print("\nDimension Analysis:")
    print(f"  Most changed dimension:   D{most_changed_dim} (mean |Δ| = {dim_mean_abs[most_changed_dim]:.6f})")
    print(f"  Least changed dimension:  D{least_changed_dim} (mean |Δ| = {dim_mean_abs[least_changed_dim]:.6f})")

    # Special card analysis
    print("\nSpecial Cards:")
    for special_name, special_ids_str in [('PAD', 0), ('UNDER', 33)]:
        if special_ids_str < len(delta):
            norm = torch.norm(delta[special_ids_str]).item()
            print(f"  {special_name:6s} L2 change:     {norm:.6f}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze card embeddings from trained PPO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint (.pt or .pth file)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='card_embedding_analysis',
        help='Directory to save analysis outputs (default: card_embedding_analysis)',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Load embeddings
    print(f"\nLoading checkpoint: {args.checkpoint}")
    trained_embeddings, d_card = load_trained_embeddings(args.checkpoint)
    print(f"Embedding dimension: {d_card}")

    print("Generating initial embeddings...")
    initial_embeddings = get_initial_embeddings(d_card=d_card)

    # Compute delta
    delta_embeddings = compute_embedding_delta(initial_embeddings, trained_embeddings)

    # Get card ordering
    card_names = get_card_names_ordered(include_special=True)
    card_ids = [name_to_id(name) for name in card_names]

    # Print summary statistics
    print_summary_statistics(initial_embeddings, trained_embeddings, delta_embeddings, card_ids)

    # Generate visualizations
    print("Generating visualizations...")

    # Determine shared color scale for comparison
    vmax = max(
        torch.abs(initial_embeddings[card_ids, :]).max().item(),
        torch.abs(trained_embeddings[card_ids, :]).max().item(),
    )

    plot_embedding_heatmap(
        initial_embeddings, card_names, card_ids,
        'Initial Card Embeddings (Informed Initialization)',
        output_dir / 'initial_embeddings.png',
        vmin=-vmax, vmax=vmax,
    )

    plot_embedding_heatmap(
        trained_embeddings, card_names, card_ids,
        'Trained Card Embeddings',
        output_dir / 'trained_embeddings.png',
        vmin=-vmax, vmax=vmax,
    )

    # Delta visualization with its own scale
    delta_max = torch.abs(delta_embeddings[card_ids, :]).max().item()
    plot_embedding_heatmap(
        delta_embeddings, card_names, card_ids,
        'Embedding Changes (Trained - Initial)',
        output_dir / 'delta_embeddings.png',
        vmin=-delta_max, vmax=delta_max,
    )

    # Dimension importance analysis
    importance = analyze_dimension_importance(delta_embeddings, card_ids)
    plot_dimension_importance(importance, output_dir / 'dimension_importance.png')

    # Card movement analysis
    movements = analyze_card_movement(delta_embeddings, card_ids, card_names)
    plot_card_movement(movements, output_dir / 'card_movement.png')

    # Trump vs fail analysis
    plot_trump_vs_fail_analysis(
        initial_embeddings, trained_embeddings, card_names, card_ids,
        output_dir / 'trump_vs_fail.png',
    )

    # Special cards (Queens, Jacks, Aces)
    plot_special_cards(
        initial_embeddings, trained_embeddings, delta_embeddings,
        card_names, card_ids,
        output_dir / 'special_cards.png',
    )

    # CSV exports for further analysis
    print("Writing CSV exports...")
    # All tokens (including PAD/UNDER)
    write_embeddings_csv(output_dir / 'initial_embeddings_all.csv', initial_embeddings, card_names, card_ids)
    write_embeddings_csv(output_dir / 'final_embeddings_all.csv', trained_embeddings, card_names, card_ids)
    # Real cards only (exclude PAD/UNDER)
    real_mask = [name not in ('PAD', 'UNDER') for name in card_names]
    real_names = [n for n, keep in zip(card_names, real_mask) if keep]
    real_ids = [i for i, keep in zip(card_ids, real_mask) if keep]
    write_embeddings_csv(output_dir / 'initial_embeddings_real_cards.csv', initial_embeddings, real_names, real_ids)

    print(f"\nAnalysis complete! All outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - initial_embeddings.png     : Heatmap of initialized embeddings")
    print("  - trained_embeddings.png     : Heatmap of trained embeddings")
    print("  - delta_embeddings.png       : Heatmap of changes (trained - initial)")
    print("  - dimension_importance.png   : Which dimensions changed most")
    print("  - card_movement.png          : L2 norm of change per card")
    print("  - trump_vs_fail.png          : Comparison of trump vs fail card groups")
    print("  - special_cards.png          : Focus on Queens, Jacks, and Aces")
    print()


if __name__ == '__main__':
    main()

