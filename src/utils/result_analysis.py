"""
Result Analysis Module
======================
Visualizzazione e analisi confronto GNN vs LKH-3.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import torch


class ResultAnalyzer:
    """Analizza e visualizza risultati GNN vs LKH"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_solution_comparison(self, 
                                 instance,
                                 lkh_routes: List[List[int]],
                                 edge_probs: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 threshold: float = 0.5,
                                 save_path: Optional[Path] = None):
        """
        Confronta visivamente soluzione LKH vs predizioni GNN.
        
        Args:
            instance: CVRPInstance
            lkh_routes: Route da LKH-3 [[2,3,4], [5,6,7], ...]
            edge_probs: Probabilità edge da GNN (tensor)
            edge_index: Indici edge (tensor 2xE)
            threshold: Soglia per considerare edge come "predetto"
            save_path: Dove salvare il plot
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        coords = instance.coordinates
        
        # --- PLOT 1: Soluzione LKH ---
        ax = axes[0]
        self._plot_routes(ax, coords, lkh_routes, "Soluzione LKH-3")
        
        # --- PLOT 2: Predizioni GNN ---
        ax = axes[1]
        # Converti edge_probs in route "predette"
        gnn_edges = self._extract_predicted_edges(edge_index, edge_probs, threshold)
        self._plot_edges(ax, coords, gnn_edges, "Predizioni GNN (threshold={:.2f})".format(threshold))
        
        # --- PLOT 3: Overlay (LKH in verde, GNN in rosso) ---
        ax = axes[2]
        self._plot_overlay(ax, coords, lkh_routes, gnn_edges, "Overlay: LKH (verde) vs GNN (rosso)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Salvato: {save_path}")
        
        return fig
    
    def _plot_routes(self, ax, coords, routes, title):
        """Plot route colorate"""
        # Deposito
        ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s', 
                  edgecolors='black', linewidth=2, label='Depot', zorder=5)
        
        # Clienti
        ax.scatter(coords[1:, 0], coords[1:, 1], c='lightgray', s=100,
                  edgecolors='black', linewidth=1, alpha=0.7, zorder=4)
        
        # Route
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        for route_idx, (route, color) in enumerate(zip(routes, colors)):
            # Converti in 0-based se necessario
            route_0based = [max(0, n-1) for n in route]
            
            # Aggiungi depot
            full_route = [0] + route_0based + [0]
            route_coords = coords[full_route]
            
            ax.plot(route_coords[:, 0], route_coords[:, 1],
                   'o-', color=color, linewidth=2, markersize=6,
                   label=f'Route {route_idx+1}', alpha=0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_edges(self, ax, coords, edges, title):
        """Plot singoli edge predetti"""
        # Deposito
        ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s',
                  edgecolors='black', linewidth=2, label='Depot', zorder=5)
        
        # Clienti
        ax.scatter(coords[1:, 0], coords[1:, 1], c='lightgray', s=100,
                  edgecolors='black', linewidth=1, alpha=0.7, zorder=4)
        
        # Edge predetti
        for src, dst in edges:
            ax.plot([coords[src, 0], coords[dst, 0]],
                   [coords[src, 1], coords[dst, 1]],
                   'b-', linewidth=1.5, alpha=0.5)
        
        ax.set_title(f"{title}\n({len(edges)} archi predetti)", fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    def _plot_overlay(self, ax, coords, lkh_routes, gnn_edges, title):
        """Overlay LKH (verde) e GNN (rosso)"""
        # Deposito
        ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s',
                  edgecolors='black', linewidth=2, label='Depot', zorder=5)
        
        # Clienti
        ax.scatter(coords[1:, 0], coords[1:, 1], c='lightgray', s=100,
                  edgecolors='black', linewidth=1, alpha=0.7, zorder=4)
        
        # LKH route (verde)
        lkh_edges = self._routes_to_edges(lkh_routes)
        for src, dst in lkh_edges:
            ax.plot([coords[src, 0], coords[dst, 0]],
                   [coords[src, 1], coords[dst, 1]],
                   'g-', linewidth=3, alpha=0.6, label='LKH' if src == 0 and dst == lkh_routes[0][0]-1 else '')
        
        # GNN edges (rosso)
        for src, dst in gnn_edges:
            ax.plot([coords[src, 0], coords[dst, 0]],
                   [coords[src, 1], coords[dst, 1]],
                   'r-', linewidth=1.5, alpha=0.8, label='GNN' if src == 0 else '')
        
        # Rimuovi label duplicate
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    def _extract_predicted_edges(self, edge_index, edge_probs, threshold):
        """Estrae edge con prob > threshold"""
        mask = edge_probs > threshold
        predicted_edges = edge_index[:, mask].cpu().numpy().T
        return [(int(src), int(dst)) for src, dst in predicted_edges]
    
    def _routes_to_edges(self, routes):
        """Converte route in lista di edge"""
        edges = []
        for route in routes:
            route_0based = [max(0, n-1) for n in route]
            # Depot -> primo nodo
            edges.append((0, route_0based[0]))
            # Nodi consecutivi
            for i in range(len(route_0based)-1):
                edges.append((route_0based[i], route_0based[i+1]))
            # Ultimo -> depot
            edges.append((route_0based[-1], 0))
        return edges
    
    def plot_edge_probability_distribution(self,
                                          edge_probs: torch.Tensor,
                                          true_edges: torch.Tensor,
                                          save_path: Optional[Path] = None):
        """
        Istogramma distribuzione probabilità edge: true vs false.
        
        Args:
            edge_probs: Probabilità predette (tensor)
            true_edges: Label vere (tensor 0/1)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Converti in numpy
        probs = edge_probs.cpu().numpy()
        labels = true_edges.cpu().numpy()
        
        # Separa true/false edges
        true_probs = probs[labels == 1]
        false_probs = probs[labels == 0]
        
        # Plot istogrammi
        ax.hist(false_probs, bins=50, alpha=0.6, label=f'False Edges (n={len(false_probs)})', 
                color='red', edgecolor='black')
        ax.hist(true_probs, bins=50, alpha=0.6, label=f'True Edges (n={len(true_probs)})', 
                color='green', edgecolor='black')
        
        # Soglia 0.5
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold 0.5')
        
        ax.set_xlabel('Probabilità Predetta', fontsize=12)
        ax.set_ylabel('Frequenza', fontsize=12)
        ax.set_title('Distribuzione Probabilità: True vs False Edges', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Aggiungi statistiche
        stats_text = f"""
        True Edges - Mean: {true_probs.mean():.3f}, Std: {true_probs.std():.3f}
        False Edges - Mean: {false_probs.mean():.3f}, Std: {false_probs.std():.3f}
        """
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Salvato: {save_path}")
        
        return fig
    
    def compute_edge_metrics(self, 
                            edge_probs: torch.Tensor,
                            true_edges: torch.Tensor,
                            thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> Dict:
        """
        Calcola metriche per diverse soglie.
        
        Returns:
            Dict con precision, recall, f1 per ogni soglia
        """
        results = {}
        
        for thresh in thresholds:
            preds = (edge_probs > thresh).float()
            
            tp = ((preds == 1) & (true_edges == 1)).sum().item()
            fp = ((preds == 1) & (true_edges == 0)).sum().item()
            fn = ((preds == 0) & (true_edges == 1)).sum().item()
            tn = ((preds == 0) & (true_edges == 0)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            results[f'thresh_{thresh}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'num_predicted': tp + fp
            }
        
        return results
    
    def plot_metrics_by_threshold(self,
                                  metrics_dict: Dict,
                                  save_path: Optional[Path] = None):
        """Plot precision/recall/f1 per diverse soglie"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        thresholds = sorted([float(k.split('_')[1]) for k in metrics_dict.keys()])
        
        precisions = [metrics_dict[f'thresh_{t}']['precision'] for t in thresholds]
        recalls = [metrics_dict[f'thresh_{t}']['recall'] for t in thresholds]
        f1s = [metrics_dict[f'thresh_{t}']['f1'] for t in thresholds]
        
        ax.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
        ax.plot(thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=8)
        ax.plot(thresholds, f1s, '^-', label='F1-Score', linewidth=2, markersize=8)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metriche per Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Salvato: {save_path}")
        
        return fig


def analyze_single_instance(instance, lkh_solution, gnn_predictions, output_dir):
    """
    Funzione utility per analizzare una singola istanza.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = ResultAnalyzer()
    
    # 1. Confronto soluzioni (SEMPRE)
    print("\n📊 Generazione confronto soluzioni...")
    fig1 = analyzer.plot_solution_comparison(
        instance=instance,
        lkh_routes=lkh_solution['routes'],
        edge_probs=torch.sigmoid(gnn_predictions['edge_probs']),
        edge_index=gnn_predictions['edge_index'],
        threshold=0.6,
        save_path=output_dir / f"comparison_{instance.id}.png"
    )
    plt.close(fig1)
    
    # 2-3. Distribuzione e metriche (SOLO SE ABBIAMO y_edges)
    if gnn_predictions.get('y_edges') is not None:  # ← FIX: controlla anche None
        print("📊 Generazione distribuzione probabilità...")
        fig2 = analyzer.plot_edge_probability_distribution(
            edge_probs=torch.sigmoid(gnn_predictions['edge_probs']),
            true_edges=gnn_predictions['y_edges'],
            save_path=output_dir / f"prob_dist_{instance.id}.png"
        )
        plt.close(fig2)
        
        # 3. Metriche per soglia
        print("📊 Calcolo metriche per threshold...")
        metrics = analyzer.compute_edge_metrics(
            edge_probs=torch.sigmoid(gnn_predictions['edge_probs']),
            true_edges=gnn_predictions['y_edges'],
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        
        fig3 = analyzer.plot_metrics_by_threshold(
            metrics,
            save_path=output_dir / f"metrics_threshold_{instance.id}.png"
        )
        plt.close(fig3)
        
        # Stampa best threshold
        best_f1_thresh = max(metrics.keys(), key=lambda k: metrics[k]['f1'])
        print(f"\n🎯 Best F1 threshold: {best_f1_thresh}")
        print(f"   Precision: {metrics[best_f1_thresh]['precision']:.3f}")
        print(f"   Recall: {metrics[best_f1_thresh]['recall']:.3f}")
        print(f"   F1: {metrics[best_f1_thresh]['f1']:.3f}")
    else:
        print("\n⚠️  y_edges non disponibile - skip distribuzione e metriche")
    
    print(f"\n✅ Analisi completata! File salvati in: {output_dir}")
