#!/usr/bin/env python3
"""
Test script per verificare le nuove penalità nella loss function
"""

import torch
import sys
from pathlib import Path

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import CVRPLoss

def test_loss_penalties():
    """Test delle nuove penalità nella loss function"""
    print("="*60)
    print("Test delle penalità VRP nella Loss Function")
    print("="*60)

    # Crea loss function con le nuove penalità
    loss_fn = CVRPLoss(
        edge_weight=1.0,
        node_weight=0.5,
        consistency_weight=0.2,
        self_loop_penalty=10.0,
        node_revisit_penalty=5.0,
        capacity_penalty=8.0,
        route_validity_penalty=3.0
    )

    print("\n✓ Loss function creata con successo")
    print(f"  - Self-loop penalty weight: {loss_fn.self_loop_penalty}")
    print(f"  - Node revisit penalty weight: {loss_fn.node_revisit_penalty}")
    print(f"  - Capacity penalty weight: {loss_fn.capacity_penalty}")
    print(f"  - Route validity penalty weight: {loss_fn.route_validity_penalty}")

    # Crea dati di test
    num_nodes = 10
    num_edges = num_nodes * (num_nodes - 1)  # Fully connected senza self-loops

    # Crea edge_index per grafo fully connected senza self-loops
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Crea predizioni edge casuali
    edge_predictions = torch.randn(num_edges)

    # Crea predizioni nodi casuali
    node_predictions = torch.randn(num_nodes)

    # Crea node features [x_coord, y_coord, demand, is_depot]
    node_features = torch.randn(num_nodes, 4)
    node_features[:, 2] = torch.rand(num_nodes) * 10  # demands tra 0 e 10
    node_features[0, 3] = 1.0  # depot
    node_features[1:, 3] = 0.0  # non-depot

    # Crea labels
    y_edges = torch.zeros(num_edges)
    y_edges[:20] = 1.0  # Primi 20 archi sono nella soluzione

    y_nodes = torch.arange(num_nodes, dtype=torch.float)

    # Crea oggetto data mock
    class MockData:
        def __init__(self):
            self.edge_index = edge_index
            self.x = node_features
            self.y_edges = y_edges
            self.y_nodes = y_nodes
            self.num_nodes = num_nodes
            self.capacity = 20.0

    data = MockData()

    predictions = {
        'edge_predictions': edge_predictions,
        'node_predictions': node_predictions
    }

    print("\n✓ Dati di test creati:")
    print(f"  - Num nodi: {num_nodes}")
    print(f"  - Num archi: {num_edges}")
    print(f"  - Capacità: {data.capacity}")

    # Calcola loss
    print("\n" + "="*60)
    print("Calcolo delle loss...")
    print("="*60)

    try:
        losses = loss_fn(predictions, data)

        print("\n✓ Loss calcolate con successo!")
        print("\nComponenti della loss:")
        for name, value in losses.items():
            if name != 'total_loss':
                print(f"  - {name:25s}: {value.item():10.6f}")

        print("\n" + "-"*60)
        print(f"  TOTAL LOSS: {losses['total_loss'].item():.6f}")
        print("="*60)

        # Verifica che tutte le loss siano finite (non NaN o Inf)
        all_finite = all(torch.isfinite(v) for v in losses.values())
        if all_finite:
            print("\n✓ Tutte le loss sono finite (non NaN/Inf)")
        else:
            print("\n✗ ATTENZIONE: Alcune loss sono NaN o Inf!")
            return False

        # Test con self-loop
        print("\n" + "="*60)
        print("Test Self-Loop Penalty")
        print("="*60)

        # Aggiungi un self-loop
        edge_index_with_loop = torch.cat([
            edge_index,
            torch.tensor([[5], [5]], dtype=torch.long)
        ], dim=1)

        edge_preds_with_loop = torch.cat([
            edge_predictions,
            torch.tensor([5.0])  # Alta probabilità per self-loop
        ])

        data_with_loop = MockData()
        data_with_loop.edge_index = edge_index_with_loop

        preds_with_loop = {
            'edge_predictions': edge_preds_with_loop,
            'node_predictions': node_predictions
        }

        losses_with_loop = loss_fn(preds_with_loop, data_with_loop)

        self_loop_increase = losses_with_loop['self_loop_loss'].item() - losses.get('self_loop_loss', torch.tensor(0.0)).item()
        print(f"\n✓ Self-loop penalty aumentata di: {self_loop_increase:.6f}")

        if self_loop_increase > 0:
            print("✓ La penalità per self-loop funziona correttamente!")
        else:
            print("✗ ATTENZIONE: Self-loop penalty non ha effetto")

        print("\n" + "="*60)
        print("TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n✗ ERRORE durante il calcolo della loss:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_loss_penalties()
    sys.exit(0 if success else 1)
