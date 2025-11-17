#!/usr/bin/env python3
"""
Demo Veloce STEP 1 - Generazione Istanze CVRP
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_generation.cvrp_generator import CVRPConfig, CVRPGenerator
from src.utils.visualization import CVRPVisualizer
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("\n" + "="*60)
    print("🚀 DEMO STEP 1: GENERAZIONE ISTANZE CVRP")
    print("="*60)
    
    # 1. Configurazione
    print("\n1️⃣ Configurazione parametri:")
    config = CVRPConfig(
        num_nodes=20,
        capacity=25,
        demand_range=(1, 8),
        distribution="uniform",
        depot_position="center",
        seed=42
    )
    print(f"   - Nodi: {config.num_nodes}")
    print(f"   - Capacità: {config.capacity}")
    print(f"   - Distribuzione: {config.distribution}")
    
    # 2. Generazione istanza
    print("\n2️⃣ Generazione istanza:")
    generator = CVRPGenerator(config)
    instance = generator.generate_instance("demo_001")
    
    info = instance.get_info()
    print(f"   - ID: {info['id']}")
    print(f"   - Domanda totale: {info['total_demand']}")
    print(f"   - Veicoli minimi: {info['min_vehicles']}")
    print(f"   - Domanda media: {info['avg_demand']:.2f}")
    
    # 3. Salvataggio
    print("\n3️⃣ Salvataggio file VRP:")
    output_dir = Path("outputs/demo_vrp")
    output_dir.mkdir(parents=True, exist_ok=True)
    vrp_file = output_dir / "demo.vrp"
    instance.save_vrp(vrp_file)
    print(f"   ✓ File salvato: {vrp_file}")
    
    # 4. Visualizzazione
    print("\n4️⃣ Generazione visualizzazione:")
    visualizer = CVRPVisualizer(figsize=(10, 8))
    
    # Crea figura con 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot istanza
    visualizer.plot_instance(
        instance,
        show_demands=True,
        title="Istanza CVRP Generata",
        ax=ax1
    )
    
    # Plot matrice distanze
    visualizer.plot_distance_matrix(
        instance,
        annotate=True,
        ax=ax2
    )
    
    plt.tight_layout()
    plot_file = output_dir / "visualizzazione.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualizzazione salvata: {plot_file}")
    
    # 5. Mini dataset
    print("\n5️⃣ Generazione mini-dataset:")
    instances = generator.generate_dataset(5, verbose=False)
    stats = generator.save_dataset(
        instances,
        output_dir / "mini_dataset",
        save_vrp=True
    )
    print(f"   ✓ Dataset salvato con {stats['num_instances']} istanze")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETATA CON SUCCESSO!")
    print("="*60)
    print("\n📁 File generati in: outputs/demo_vrp/")
    print("   - demo.vrp: Istanza in formato VRP")
    print("   - visualizzazione.png: Grafici dell'istanza")
    print("   - mini_dataset/: Dataset di esempio")
    
    return instance

if __name__ == "__main__":
    instance = main()
