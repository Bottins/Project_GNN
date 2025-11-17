# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:00:57 2025

@author: alexq
"""

#!/usr/bin/env python3
"""
Dataset Preparation Script
==========================
Genera dataset completo per training GNN su CVRP.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import time
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from src.data_generation.cvrp_generator import CVRPConfig, CVRPGenerator
from src.data_generation.graph_builder import CVRPGraphBuilder, GraphDataset
from src.solvers.lkh_solver import LKHSolver, LKHSolutionValidator
from src.utils.visualization import CVRPVisualizer


class DatasetCreator:
    """Orchestratore per la creazione del dataset completo"""
    
    def __init__(self,
                 config: CVRPConfig,
                 lkh_solver: LKHSolver,
                 graph_builder: CVRPGraphBuilder,
                 output_dir: Path):
        """
        Args:
            config: Configurazione CVRP
            lkh_solver: Solver LKH-3
            graph_builder: Builder per grafi
            output_dir: Directory di output
        """
        self.config = config
        self.generator = CVRPGenerator(config)
        self.lkh_solver = lkh_solver
        self.graph_builder = graph_builder
        self.output_dir = Path(output_dir)
        
        # Crea struttura directory
        self._setup_directories()
    
    def _setup_directories(self):
        """Crea la struttura delle directory"""
        dirs = [
            self.output_dir / "raw" / "train",
            self.output_dir / "raw" / "val",
            self.output_dir / "raw" / "test",
            self.output_dir / "processed",
            self.output_dir / "instances" / "vrp_files",
            self.output_dir / "instances" / "solutions",
            self.output_dir / "instances" / "parameters",
            self.output_dir / "visualizations"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_split(self,
                     split_name: str,
                     num_instances: int,
                     start_id: int = 0) -> Dict:
        """
        Crea un singolo split del dataset.
        
        Args:
            split_name: Nome dello split (train/val/test)
            num_instances: Numero di istanze da generare
            start_id: ID iniziale
            
        Returns:
            Statistiche dello split
        """
        print(f"\n{'='*60}")
        print(f"Creazione split: {split_name.upper()}")
        print(f"{'='*60}")
        
        # Directory per questo split
        split_dir = self.output_dir / "raw" / split_name
        vrp_dir = self.output_dir / "instances" / "vrp_files"
        sol_dir = self.output_dir / "instances" / "solutions"
        par_dir = self.output_dir / "instances" / "parameters"
        
        # Genera istanze
        print(f"\n📊 Generazione {num_instances} istanze...")
        instances = self.generator.generate_dataset(
            num_instances, 
            start_id=start_id,
            verbose=True
        )
        
        # Risolvi istanze con LKH-3
        print(f"\n🔧 Risoluzione con LKH-3...")
        solutions = []
        invalid_count = 0
        
        with tqdm(total=num_instances) as pbar:
            for instance in instances:
                # Salva file VRP
                vrp_file = vrp_dir / f"{split_name}_{instance.id}.vrp"
                instance.save_vrp(vrp_file)
                
                # Risolvi con LKH
                try:
                    solution = self.lkh_solver.solve(
                        vrp_file,
                        output_dir=sol_dir,
                        verbose=False
                    )
                    
                    # Valida soluzione
                    validator = LKHSolutionValidator()
                    validation = validator.validate_solution(instance, solution)
                    
                    if validation['is_valid']:
                        solutions.append(solution)
                    else:
                        print(f"\n⚠️ Soluzione invalida per {instance.id}: {validation['errors']}")
                        solutions.append(None)
                        invalid_count += 1
                        
                except Exception as e:
                    print(f"\n❌ Errore nel risolvere {instance.id}: {e}")
                    solutions.append(None)
                    invalid_count += 1
                
                pbar.update(1)
        
        # Filtra istanze con soluzioni valide
        valid_instances = []
        valid_solutions = []
        for inst, sol in zip(instances, solutions):
            if sol is not None:
                valid_instances.append(inst)
                valid_solutions.append(sol)
        
        print(f"\n✅ Istanze valide: {len(valid_instances)}/{num_instances}")
        
        # Costruisci grafi
        print(f"\n🔨 Costruzione grafi...")
        if split_name == 'train':
            # Fit normalization sul training set
            self.graph_builder.fit_normalization(valid_instances)
            
            # Salva statistiche
            stats_file = self.output_dir / "processed" / "normalization_stats.pkl"
            self.graph_builder.save_stats(stats_file)
            print(f"   📈 Statistiche normalizzazione salvate")
        
        # Crea dataset di grafi
        graph_dataset = GraphDataset(
            valid_instances,
            valid_solutions,
            self.graph_builder
        )
        
        # Salva grafi processati
        graph_file = self.output_dir / "processed" / f"{split_name}_graphs.pkl"
        graph_dataset.save(graph_file)
        
        # Salva anche istanze e soluzioni raw
        raw_file = split_dir / "data.npz"
        self._save_raw_data(valid_instances, valid_solutions, raw_file)
        
        # Calcola statistiche
        stats = self._compute_statistics(valid_instances, valid_solutions)
        stats['split_name'] = split_name
        stats['num_instances'] = len(valid_instances)
        stats['invalid_count'] = invalid_count
        
        # Visualizza alcune istanze di esempio
        if len(valid_instances) > 0:
            self._visualize_examples(
                valid_instances[:3],
                valid_solutions[:3],
                split_name
            )
        
        return stats
    
    def _save_raw_data(self, 
                      instances: List,
                      solutions: List[Dict],
                      output_file: Path):
        """Salva dati raw in formato NPZ"""
        data = {
            'instance_ids': [inst.id for inst in instances],
            'coordinates': [inst.coordinates for inst in instances],
            'demands': [inst.demands for inst in instances],
            'capacities': [inst.capacity for inst in instances],
            'solutions': solutions
        }
        
        np.savez_compressed(output_file, **data)
        print(f"   💾 Dati raw salvati: {output_file}")
    
    def _compute_statistics(self, 
                          instances: List,
                          solutions: List[Dict]) -> Dict:
        """Calcola statistiche sul dataset"""
        stats = {
            'num_nodes': [inst.num_nodes for inst in instances],
            'total_demands': [inst.get_total_demand() for inst in instances],
            'min_vehicles': [inst.get_min_vehicles_lb() for inst in instances],
            'solution_costs': [sol.get('cost', 0) for sol in solutions],
            'num_routes': [len(sol.get('routes', [])) for sol in solutions]
        }
        
        # Calcola medie e deviazioni standard
        summary = {}
        for key, values in stats.items():
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))
        
        return summary
    
    def _visualize_examples(self,
                          instances: List,
                          solutions: List[Dict],
                          split_name: str):
        """Visualizza alcune istanze di esempio"""
        print(f"\n🎨 Generazione visualizzazioni di esempio...")
        
        viz_dir = self.output_dir / "visualizations" / split_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = CVRPVisualizer(figsize=(12, 10))
        
        for i, (instance, solution) in enumerate(zip(instances, solutions)):
            if i >= 3:  # Visualizza solo le prime 3
                break
            
            # Crea visualizzazione con soluzione
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # 1. Istanza
            visualizer.plot_instance(
                instance,
                show_demands=True,
                title=f"Instance {instance.id}",
                ax=axes[0, 0]
            )
            
            # 2. Soluzione
            visualizer.plot_solution(
                instance,
                solution['routes'],
                title=f"Solution (Cost: {solution['cost']:.1f})",
                ax=axes[0, 1]
            )
            
            # 3. Matrice distanze
            visualizer.plot_distance_matrix(
                instance,
                annotate=False,
                ax=axes[1, 0]
            )
            
            # 4. Info text
            axes[1, 1].axis('off')
            info_text = f"""
            Instance Info:
            - Nodes: {instance.num_nodes}
            - Capacity: {instance.capacity}
            - Total Demand: {instance.get_total_demand()}
            
            Solution Info:
            - Total Cost: {solution['cost']:.1f}
            - Num Routes: {len(solution['routes'])}
            - Execution Time: {solution.get('execution_time', 0):.2f}s
            """
            axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, va='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"example_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✅ Visualizzazioni salvate in: {viz_dir}")
    
    def create_full_dataset(self,
                          train_size: int = 1000,
                          val_size: int = 200,
                          test_size: int = 200) -> Dict:
        """
        Crea il dataset completo con train/val/test splits.
        
        Args:
            train_size: Numero istanze training
            val_size: Numero istanze validation
            test_size: Numero istanze test
            
        Returns:
            Statistiche complete del dataset
        """
        print("\n" + "="*60)
        print("🚀 CREAZIONE DATASET COMPLETO")
        print("="*60)
        print(f"📊 Dimensioni pianificate:")
        print(f"   - Training: {train_size} istanze")
        print(f"   - Validation: {val_size} istanze")
        print(f"   - Test: {test_size} istanze")
        
        all_stats = {}
        start_time = time.time()
        
        # Crea ogni split
        current_id = 0
        for split_name, split_size in [
            ('train', train_size),
            ('val', val_size),
            ('test', test_size)
        ]:
            split_stats = self.create_split(split_name, split_size, current_id)
            all_stats[split_name] = split_stats
            current_id += split_size
        
        # Salva configurazione e statistiche
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'cvrp_config': self.config.to_dict(),
                'dataset_stats': all_stats,
                'creation_time': time.time() - start_time
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ DATASET CREATO CON SUCCESSO!")
        print("="*60)
        print(f"⏱️ Tempo totale: {time.time() - start_time:.1f} secondi")
        print(f"📁 Output directory: {self.output_dir}")
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(description="Prepara dataset CVRP per GNN")
    
    # Parametri dataset
    parser.add_argument('--train-size', type=int, default=80,
                       help='Numero istanze training')
    parser.add_argument('--val-size', type=int, default=10,
                       help='Numero istanze validation')
    parser.add_argument('--test-size', type=int, default=10,
                       help='Numero istanze test')
    
    # Parametri CVRP
    parser.add_argument('--num-nodes', type=int, default=20,
                       help='Numero di nodi per istanza')
    parser.add_argument('--capacity', type=int, default=30,
                       help='Capacità veicolo')
    parser.add_argument('--distribution', type=str, default='uniform',
                       choices=['uniform', 'gaussian', 'clustered'],
                       help='Distribuzione coordinate')
    
    # Parametri LKH
    parser.add_argument('--lkh-path', type=str, default=r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\external\LKH-3\LKH-3.exe",
                       help='Path a LKH-3 executable')
    parser.add_argument('--time-limit', type=int, default=10,
                       help='Time limit per LKH-3 (secondi)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=r"C:\Users\alexq\Desktop\GNN_CVRP_PROJECT\data",
                       help='Directory di output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup configurazione CVRP
    cvrp_config = CVRPConfig(
        num_nodes=args.num_nodes,
        capacity=args.capacity,
        distribution=args.distribution,
        seed=args.seed
    )
    
    # Setup LKH solver
    lkh_solver = LKHSolver(
        lkh_path=args.lkh_path,
        time_limit=args.time_limit,
        runs=5,
        seed=args.seed
    )
    
    # Setup graph builder
    graph_builder = CVRPGraphBuilder(
        node_features=['x_coord', 'y_coord', 'demand', 'is_depot',
                      'distance_to_depot', 'polar_angle'],
        edge_features=['distance', 'demand_sum', 'capacity_feasible'],
        normalize=True,
        fully_connected=True
    )
    
    # Crea dataset
    creator = DatasetCreator(
        config=cvrp_config,
        lkh_solver=lkh_solver,
        graph_builder=graph_builder,
        output_dir=Path(args.output_dir)
    )
    
    stats = creator.create_full_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size
    )
    
    # Stampa riassunto finale
    print("\n📊 RIASSUNTO STATISTICHE DATASET:")
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  - Istanze: {split_stats.get('num_instances', 0)}")
        print(f"  - Costo medio: {split_stats.get('solution_costs_mean', 0):.1f}")
        print(f"  - Routes medie: {split_stats.get('num_routes_mean', 0):.1f}")


if __name__ == "__main__":
    main()