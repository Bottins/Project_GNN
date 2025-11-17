#!/usr/bin/env python3
"""
Test Visualizzazione Risultati
================================
Script per testare l'analisi visiva GNN vs LKH su un'istanza.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np

# Import dei moduli del progetto
from src.data_generation.cvrp_generator import CVRPInstance
from src.data_generation.graph_builder import CVRPGraphBuilder
from src.utils.result_analysis import analyze_single_instance
from models.architectures.gnn_base import CVRPGNNBase


def load_vrp_instance(vrp_file: Path) -> CVRPInstance:
    """Carica un'istanza VRP da file"""
    coordinates = []
    demands = []
    capacity = 0
    
    with open(vrp_file, 'r') as f:
        lines = f.readlines()
    
    section = None
    for line in lines:
        line = line.strip()
        
        if line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line == 'NODE_COORD_SECTION':
            section = 'coords'
        elif line == 'DEMAND_SECTION':
            section = 'demands'
        elif line == 'DEPOT_SECTION' or line == 'EOF':
            section = None
        elif section == 'coords' and line:
            parts = line.split()
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                coordinates.append([x, y])
        elif section == 'demands' and line:
            parts = line.split()
            if len(parts) == 2:
                demands.append(int(parts[1]))
    
    # Normalizza coordinate (assumo siano scalate a 1000)
    coordinates = np.array(coordinates) / 1000.0
    demands = np.array(demands)
    
    instance_id = vrp_file.stem
    return CVRPInstance(instance_id, coordinates, demands, capacity)


def load_lkh_solution(sol_file: Path, num_nodes: int) -> dict:
    """
    Carica soluzione LKH da file .sol.
    
    Args:
        sol_file: Path al file .sol
        num_nodes: Numero di nodi REALI nell'istanza (per filtrare depot virtuali)
    """
    with open(sol_file, 'r') as f:
        lines = f.readlines()
    
    tour = []
    in_tour = False
    cost = None
    
    for line in lines:
        line = line.strip()
        
        if 'Length' in line:
            cost = float(line.split('=')[1].strip())
        elif line == 'TOUR_SECTION':
            in_tour = True
        elif line == '-1' or line == 'EOF':
            in_tour = False
        elif in_tour and line:
            try:
                tour.append(int(line))
            except:
                pass
    
    # Filtra depot virtuali: nodi > num_nodes+1 sono depot virtuali
    # (num_nodes+1 perché il file VRP è 1-based e include il depot)
    max_valid_node = num_nodes + 1
    
    # Converti tour in routes (split al depot reale=1 o depot virtuali)
    routes = []
    current_route = []
    
    for node in tour:
        if node == 1:  # Depot reale
            if current_route:
                routes.append(current_route)
                current_route = []
        elif node > max_valid_node:  # Depot virtuale
            if current_route:
                routes.append(current_route)
                current_route = []
        else:  # Cliente normale
            current_route.append(node)
    
    if current_route:
        routes.append(current_route)
    
    return {'routes': routes, 'cost': cost}


def get_gnn_predictions(model, instance, graph_builder, device='cpu'):
    """Ottiene predizioni GNN per un'istanza"""
    # Costruisci grafo
    graph = graph_builder.build_graph(instance, solution=None)
    graph = graph.to(device)
    
    # Predizione
    model.eval()
    with torch.no_grad():
        predictions = model(graph)
    
    # Ritorna predizioni con edge_index e y_edges
    return {
        'edge_predictions': predictions['edge_predictions'],
        'edge_index': graph.edge_index,
        'y_edges': graph.y_edges if hasattr(graph, 'y_edges') else None
    }


def main():
    print("\n" + "="*60)
    print("🔍 TEST VISUALIZZAZIONE RISULTATI")
    print("="*60)
    
    # --- 1. CARICA ISTANZA ---
    print("\n📂 Caricamento istanza CVRP...")
    vrp_file = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data\instances\vrp_files\train_0.vrp")
    instance = load_vrp_instance(vrp_file)
    print(f"   ✅ Istanza '{instance.id}' caricata")
    print(f"      - Nodi: {instance.num_nodes}")
    print(f"      - Capacità: {instance.capacity}")
    
    # --- 2. CARICA SOLUZIONE LKH ---
    print("\n📂 Caricamento soluzione LKH...")
    sol_file = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data\instances\solutions\train_0.sol")
    lkh_solution = load_lkh_solution(sol_file, instance.num_nodes)
    print(f"   ✅ Soluzione caricata")
    print(f"      - Costo: {lkh_solution['cost']}")
    print(f"      - Route: {len(lkh_solution['routes'])}")
    
    # --- 3. CARICA MODELLO GNN ---
    print("\n📂 Caricamento modello GNN...")
    checkpoint_path = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\models\checkpoints\best_model.pth")
    
    if not checkpoint_path.exists():
        print(f"   ⚠️  Checkpoint non trovato: {checkpoint_path}")
        print("   ℹ️  Userò predizioni casuali per demo")
        
        # Crea predizioni fake per demo
        graph_builder = CVRPGraphBuilder(
            node_features=['x_coord', 'y_coord', 'demand', 'is_depot', 
                          'distance_to_depot', 'polar_angle'],
            edge_features=['distance', 'demand_sum', 'capacity_feasible'],
            normalize=False
        )
        
        graph = graph_builder.build_graph(instance)
        num_edges = graph.edge_index.shape[1]
        
        # Predizioni casuali (ma con bias verso pochi edge)
        gnn_predictions = {
            'edge_predictions': torch.randn(num_edges) - 2.0,  # Bias negativo
            'edge_index': graph.edge_index,
            'y_edges': None
        }
    else:
        # Carica modello vero
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        
        # Ricrea modello
        model = CVRPGNNBase(
            node_features=6,
            edge_features=3,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Graph builder
        graph_builder = CVRPGraphBuilder(
            node_features=['x_coord', 'y_coord', 'demand', 'is_depot',
                          'distance_to_depot', 'polar_angle'],
            edge_features=['distance', 'demand_sum', 'capacity_feasible'],
            normalize=True
        )
        
        # Carica stats normalizzazione se esistono
        stats_path = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data\processed\normalization_stats.pkl")
        if stats_path.exists():
            graph_builder.load_stats(stats_path)
        
        print(f"   ✅ Modello caricato")
        
        # Ottieni predizioni
        gnn_predictions = get_gnn_predictions(model, instance, graph_builder)
    
    # --- 4. ANALISI VISIVA ---
    print("\n📊 Generazione visualizzazioni...")
    output_dir = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\outputs\analysis_results")
    
    analyze_single_instance(
        instance=instance,
        lkh_solution=lkh_solution,
        gnn_predictions=gnn_predictions,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETATO!")
    print("="*60)
    print(f"\n📁 Risultati salvati in: {output_dir}")
    print("\nFile generati:")
    print(f"   1. comparison_{instance.id}.png - Confronto visivo LKH vs GNN")
    if gnn_predictions['y_edges'] is not None:
        print(f"   2. prob_dist_{instance.id}.png - Distribuzione probabilità")
        print(f"   3. metrics_threshold_{instance.id}.png - Metriche per threshold")


if __name__ == "__main__":
    main()
