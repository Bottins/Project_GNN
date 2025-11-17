#!/usr/bin/env python3
"""
Script di Verifica Dataset
==========================
Verifica che il dataset sia caricabile e ben formato.
"""

import sys
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT")
sys.path.append(str(project_root))

from src.data_generation.graph_builder import GraphDataset

def verify_dataset(data_dir: str = r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data"):
    """Verifica l'integrità del dataset"""
    
    print("\n" + "="*60)
    print("🔍 VERIFICA DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    
    # Verifica che esistano i file
    required_files = [
        processed_dir / "train_graphs.pkl",
        processed_dir / "val_graphs.pkl",
        processed_dir / "test_graphs.pkl",
        processed_dir / "normalization_stats.pkl"
    ]
    
    print("\n📁 Controllo file necessari...")
    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path.name}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ Alcuni file mancano! Esegui prima prepare_dataset.py")
        return False
    
    # Carica i dataset
    print("\n📊 Caricamento grafi...")
    try:
        train_graphs = GraphDataset.load(processed_dir / "train_graphs.pkl")
        print(f"   ✅ Train: {len(train_graphs)} grafi")
        
        val_graphs = GraphDataset.load(processed_dir / "val_graphs.pkl")
        print(f"   ✅ Val: {len(val_graphs)} grafi")
        
        test_graphs = GraphDataset.load(processed_dir / "test_graphs.pkl")
        print(f"   ✅ Test: {len(test_graphs)} grafi")
        
    except Exception as e:
        print(f"\n❌ Errore nel caricamento: {e}")
        return False
    
    # Verifica struttura dei grafi
    print("\n🔬 Analisi struttura grafi...")
    sample = train_graphs[0]
    
    print(f"   • Numero nodi: {sample.num_nodes}")
    print(f"   • Node features shape: {sample.x.shape}")
    print(f"   • Edge index shape: {sample.edge_index.shape}")
    
    if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
        print(f"   • Edge features shape: {sample.edge_attr.shape}")
    
    if hasattr(sample, 'y_edges'):
        print(f"   • Edge labels shape: {sample.y_edges.shape}")
    
    if hasattr(sample, 'y_nodes'):
        print(f"   • Node labels shape: {sample.y_nodes.shape}")
    
    # Statistiche del dataset
    print("\n📈 Statistiche dataset:")
    
    total_nodes = sum(g.num_nodes for g in train_graphs)
    avg_nodes = total_nodes / len(train_graphs)
    print(f"   • Nodi medi per grafo (train): {avg_nodes:.1f}")
    
    total_edges = sum(g.edge_index.shape[1] for g in train_graphs)
    avg_edges = total_edges / len(train_graphs)
    print(f"   • Edge medi per grafo (train): {avg_edges:.1f}")
    
    if hasattr(sample, 'capacity'):
        capacities = [g.capacity for g in train_graphs]
        print(f"   • Capacità veicoli: {capacities[0]}")
    
    print("\n" + "="*60)
    print("✅ DATASET VERIFICATO CORRETTAMENTE!")
    print("="*60)
    print("\n💡 Puoi ora procedere con il training usando:")
    print("   python scripts/train_gnn_model.py")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verifica integrità dataset")
    parser.add_argument('--data-dir', type=str, 
                       default=r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data",
                       help='Directory contenente il dataset')
    
    args = parser.parse_args()
    
    success = verify_dataset(args.data_dir)
    sys.exit(0 if success else 1)
