# scripts/train_gnn_model.py
"""
Script di Training GNN per CVRP
================================
Training completo con visualizzazione risultati.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse
import json
import time

from models.architectures.gnn_base import CVRPGNNBase, CVRPDecoder
from src.training.trainer import CVRPTrainer, CVRPLoss
from src.data_generation.graph_builder import GraphDataset


class CVRPEvaluator:
    """Valuta performance del modello CVRP"""
    
    def __init__(self, model: nn.Module, decoder: CVRPDecoder, device: str = 'cpu'):
        self.model = model.to(device)
        self.decoder = decoder
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(self, test_loader: GeometricDataLoader, 
                     save_predictions: bool = False,  # ← AGGIUNGI
                     output_dir: Optional[Path] = None) -> Dict:  # ← AGGIUNGI
        """Valuta il modello su un dataset completo"""
        print("\n🔍 Valutazione modello su test set...")
        
        all_results = []
        all_predictions = []  # ← AGGIUNGI
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                print(f"Batch {batch_idx + 1}/{len(test_loader)}")
                batch = batch.to(self.device)
                
                # Predizioni
                predictions = self.model(batch)
                
                # Separa il batch in grafi individuali
                data_list = batch.to_data_list()
                
                # Indici di inizio/fine per ogni grafo nel batch
                if hasattr(batch, 'ptr'):
                    ptr = batch.ptr.cpu().numpy()
                else:
                    # Fallback se ptr non esiste
                    ptr = [0]
                    cumsum = 0
                    for g in data_list:
                        cumsum += g.num_nodes
                        ptr.append(cumsum)
                    ptr = np.array(ptr)
                
                # Processa ogni grafo separatamente
                for graph_idx, single_graph in enumerate(data_list):
                    try:
                        start_node = ptr[graph_idx]
                        end_node = ptr[graph_idx + 1]
                        
                        # Trova gli edge di questo grafo
                        edge_mask = (batch.edge_index[0] >= start_node) & (batch.edge_index[0] < end_node)
                        graph_edge_index = batch.edge_index[:, edge_mask] - start_node
                        graph_edge_preds = predictions['edge_predictions'][edge_mask]
                        

                        # Calcola metriche SENZA decodificare (per evitare il loop infinito)
                        result = self._compute_test_metrics_simple(
                            graph_edge_preds,
                            single_graph
                        )
                        all_results.append(result)
                        
                        if save_predictions:
                            all_predictions.append({
                                'instance_id': single_graph.instance_id if hasattr(single_graph, 'instance_id') else f'instance_{batch_idx}_{graph_idx}',
                                'edge_probs': torch.sigmoid(graph_edge_preds).cpu(),
                                'edge_index': graph_edge_index.cpu(),
                                'y_edges': single_graph.y_edges.cpu() if hasattr(single_graph, 'y_edges') else None
                            })
                    except Exception as e:
                        print(f"   ⚠️ Errore nel valutare grafo {graph_idx}: {e}")
                        continue
        
        # Aggrega risultati
        if not all_results:
            print("   ❌ Nessun risultato valido!")
            return {}
        
        aggregated = self._aggregate_results(all_results)
        # ← AGGIUNGI: Salva predizioni
        if save_predictions and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_file = output_dir / 'test_predictions.pkl'
            torch.save(all_predictions, pred_file)
            print(f"\n💾 Predizioni salvate: {pred_file}")
        
        return aggregated
    
    def _compute_test_metrics_simple(self, edge_preds: torch.Tensor, graph) -> Dict:
        """
        Calcola metriche semplici senza decodificare le route.
        Questo evita il problema del loop infinito nel decoder.
        """
        metrics = {}
        
        # Edge accuracy
        if hasattr(graph, 'y_edges'):
            edge_probs = torch.sigmoid(edge_preds)
            edge_binary = (edge_probs > 0.15).float()
            correct = (edge_binary == graph.y_edges).float()
            metrics['edge_accuracy'] = correct.mean().item()
            
            # Precision e Recall
            tp = ((edge_binary == 1) & (graph.y_edges == 1)).sum().item()
            fp = ((edge_binary == 1) & (graph.y_edges == 0)).sum().item()
            fn = ((edge_binary == 0) & (graph.y_edges == 1)).sum().item()
            
            metrics['edge_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['edge_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 score
            if metrics['edge_precision'] + metrics['edge_recall'] > 0:
                metrics['edge_f1'] = 2 * (metrics['edge_precision'] * metrics['edge_recall']) / \
                                    (metrics['edge_precision'] + metrics['edge_recall'])
            else:
                metrics['edge_f1'] = 0
        
        # Numero di edge predetti come parte della soluzione
        edge_probs = torch.sigmoid(edge_preds)
        metrics['num_edges_predicted'] = (edge_probs > 0.5).sum().item()
        
        return metrics
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggrega risultati di valutazione"""
        aggregated = {}
        
        # Raccogli tutte le metriche
        all_metrics = {}
        for result in results:
            for key, value in result.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Calcola statistiche
        for key, values in all_metrics.items():
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated


def visualize_training_results(history: Dict, save_dir: Path):
    """Visualizza e salva i risultati del training"""
    print("\n📊 Generazione grafici...")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup style
    # plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", 4)
    
    # Crea figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', color=colors[0], 
            label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 's-', color=colors[1], 
            label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoca', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Edge Accuracy
    ax = axes[0, 1]
    train_acc = [m.get('edge_acc', 0) for m in history['train_metrics']]
    val_acc = [m.get('edge_acc', 0) for m in history['val_metrics']]
    
    ax.plot(epochs, train_acc, 'o-', color=colors[2], 
            label='Train Accuracy', linewidth=2, markersize=4)
    ax.plot(epochs, val_acc, 's-', color=colors[3], 
            label='Val Accuracy', linewidth=2, markersize=4)
    ax.set_xlabel('Epoca', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Edge Classification Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. Learning curve detail (ultimi 50% epoche)
    ax = axes[1, 0]
    mid_point = len(epochs) // 2
    ax.plot(epochs[mid_point:], history['train_loss'][mid_point:], 'o-', 
            color=colors[0], label='Train Loss', linewidth=2)
    ax.plot(epochs[mid_point:], history['val_loss'][mid_point:], 's-', 
            color=colors[1], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoca', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Detail (Ultime Epoche)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calcola statistiche finali
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    best_val_loss = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val_loss) + 1
    
    final_train_acc = train_acc[-1] if train_acc else 0
    final_val_acc = val_acc[-1] if val_acc else 0
    
    summary_text = f"""
    RIEPILOGO TRAINING
    {'='*40}
    
    Loss Finale:
    • Train: {final_train_loss:.4f}
    • Validation: {final_val_loss:.4f}
    
    Best Validation Loss:
    • Valore: {best_val_loss:.4f}
    • Epoca: {best_epoch}
    
    Accuracy Finale:
    • Train: {final_train_acc:.2%}
    • Validation: {final_val_acc:.2%}
    
    Overfitting Gap:
    • Loss: {abs(final_train_loss - final_val_loss):.4f}
    • Accuracy: {abs(final_train_acc - final_val_acc):.2%}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(' CVRP GNN Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Salva figura
    plt.savefig(save_dir / 'training_results.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Grafici salvati in: {save_dir}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Train GNN per CVRP")
    
    # Dataset
    parser.add_argument('--data-dir', type=str, default=r'C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data',
                       help='Directory con i dati')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    
    # Modello
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Numero di GAT layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Numero di attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Numero di epoche')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=r'C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\models\checkpoints',
                       help='Directory per checkpoints')
    parser.add_argument('--results-dir', type=str, default=r'C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\experiments\results',
                       help='Directory per risultati')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 CVRP GNN TRAINING - STEP 3")
    print("="*60)
    
    # Carica dataset
    print("\n📂 Caricamento dataset...")
    data_dir = Path(args.data_dir)
    
    train_graphs = GraphDataset.load(data_dir / 'processed' / 'train_graphs.pkl')
    val_graphs = GraphDataset.load(data_dir / 'processed' / 'val_graphs.pkl')
    test_graphs = GraphDataset.load(data_dir / 'processed' / 'test_graphs.pkl')
    
    print(f"   ✅ Train: {len(train_graphs)} grafi")
    print(f"   ✅ Val: {len(val_graphs)} grafi")
    print(f"   ✅ Test: {len(test_graphs)} grafi")
    
    # Crea DataLoaders
    train_loader = GeometricDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = GeometricDataLoader(test_graphs, batch_size=args.batch_size)
    
    # Determina dimensioni input
    sample_data = train_graphs[0]
    node_features = sample_data.x.shape[1]
    edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 1
    
    print(f"\n🔧 Configurazione modello:")
    print(f"   - Node features: {node_features}")
    print(f"   - Edge features: {edge_features}")
    print(f"   - Hidden dim: {args.hidden_dim}")
    print(f"   - Layers: {args.num_layers}")
    
    # Crea modello
    model = CVRPGNNBase(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Loss con Focal Loss e penalità VRP (pesi già ottimizzati)
    loss_fn = CVRPLoss(
        edge_weight=1.0,
        node_weight=0.3,
        # Penalità VRP ridotte per non sopprimere predizioni
        self_loop_penalty=0.5,
        node_revisit_penalty=0.3,
        capacity_penalty=0.8,
        route_validity_penalty=0.2,
        # Focal Loss per sbilanciamento
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        # Warmup graduale delle penalità
        penalty_warmup_epochs=20
    )
    
    # Crea trainer
    trainer = CVRPTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir)
    )
    
    # Training
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        scheduler=scheduler
    )
    
    # Valutazione finale
    print("\n" + "="*60)
    print("📈 VALUTAZIONE FINALE")
    print("="*60)
    
    decoder = CVRPDecoder(temperature=1.0)
    evaluator = CVRPEvaluator(model, decoder, device)
    
    # Valutazione con salvataggio predizioni
    results_dir = Path(args.results_dir)
    test_results = evaluator.evaluate_dataset(
        test_loader, 
        save_predictions=True,  # ← ABILITA salvataggio
        output_dir=results_dir   # ← Directory output
    )
    
    print("\n📊 Performance su Test Set:")
    print(f"   • Edge Accuracy: {test_results.get('edge_accuracy_mean', 0):.2%} "
          f"(±{test_results.get('edge_accuracy_std', 0):.2%})")
    print(f"   • Edge Precision: {test_results.get('edge_precision_mean', 0):.2%}")
    print(f"   • Edge Recall: {test_results.get('edge_recall_mean', 0):.2%}")
    print(f"   • Edge F1 Score: {test_results.get('edge_f1_mean', 0):.2%}")
    
    # Visualizza risultati
    results_dir = Path(args.results_dir)
    fig = visualize_training_results(history, results_dir)
    
    # Converti numpy types in Python natives per JSON
    def convert_to_native(obj):
        """Converte numpy types in Python natives"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    report = {
        'training_args': vars(args),
        'model_config': {
            'class': model.__class__.__name__,
            'parameters': int(sum(p.numel() for p in model.parameters())),
            'node_features': int(node_features),
            'edge_features': int(edge_features)
        },
        'training_history': convert_to_native(history),
        'test_results': convert_to_native(test_results),
        'training_time': time.time()
    }
    
    report_path = results_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report completo salvato in: {report_path}")
    
    # --- VISUALIZZAZIONE RISULTATI ---
    print("\n" + "="*60)
    print("🎨 GENERAZIONE VISUALIZZAZIONI")
    print("="*60)
    
    # Carica predizioni salvate
    predictions_file = results_dir / 'test_predictions.pkl'
    if predictions_file.exists():
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from src.utils.result_analysis import analyze_single_instance
        from src.data_generation.cvrp_generator import CVRPInstance
        
        predictions = torch.load(predictions_file)
        
        # Carica istanze test e soluzioni LKH
        test_instances_dir = Path(args.data_dir) / 'raw' / 'test'
        test_data = np.load(test_instances_dir / 'data.npz', allow_pickle=True)
        
        # Visualizza prime 5 istanze
        num_to_visualize = min(5, len(predictions))
        print(f"\n📊 Visualizzazione {num_to_visualize} istanze di esempio...")
        
        for i in range(num_to_visualize):
            pred = predictions[i]
            # print(f"GNN prediction = {pred}")
            instance_id = pred['instance_id']
            
            # Ricostruisci istanza
            instance = CVRPInstance(
                instance_id=instance_id,
                coordinates=test_data['coordinates'][i],
                demands=test_data['demands'][i],
                capacity=int(test_data['capacities'][i])
            )
            
            # Carica soluzione LKH
            lkh_solution = test_data['solutions'][i]
            if isinstance(lkh_solution, np.ndarray):
                lkh_solution = lkh_solution.item()
            # Analizza
            viz_dir = results_dir / 'visualizations' / f'instance_{i}'
            analyze_single_instance(
                instance=instance,
                lkh_solution=lkh_solution,
                gnn_predictions=pred,
                output_dir=viz_dir
            )
            
            print(f"   ✅ Istanza {i+1}/{num_to_visualize} completata")
        
        print(f"\n📁 Visualizzazioni salvate in: {results_dir}/visualizations/")
        
        print("\n" + "="*60)
        print("✅ STEP 3 COMPLETATO CON SUCCESSO!")
        print("="*60)


if __name__ == "__main__":
    main()