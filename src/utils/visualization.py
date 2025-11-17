"""
CVRP Visualization Module
=========================
Modulo per la visualizzazione di istanze e soluzioni CVRP.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path


class CVRPVisualizer:
    """Classe per visualizzare istanze e soluzioni CVRP"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'seaborn-v0_8'):
        """
        Args:
            figsize: Dimensione della figura
            style: Stile matplotlib da usare
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_instance(self,
                      instance,
                      show_demands: bool = True,
                      show_node_ids: bool = True,
                      title: Optional[str] = None,
                      save_path: Optional[Union[str, Path]] = None,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualizza un'istanza CVRP.
        
        Args:
            instance: Istanza CVRPInstance da visualizzare
            show_demands: Se True, mostra le domande dei nodi
            show_node_ids: Se True, mostra gli ID dei nodi
            title: Titolo del plot
            save_path: Path dove salvare la figura
            ax: Axes matplotlib esistente (se None, ne crea uno nuovo)
            
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        coords = instance.coordinates
        demands = instance.demands
        
        # Separa deposito e clienti
        depot = coords[0]
        customers = coords[1:]
        
        # Plot deposito
        ax.scatter(depot[0], depot[1], 
                  c='red', s=200, marker='s', 
                  edgecolors='black', linewidth=2,
                  label='Deposito', zorder=5)
        
        # Plot clienti con dimensione proporzionale alla domanda
        if show_demands:
            sizes = 50 + (demands[1:] * 20)  # Scala le dimensioni
        else:
            sizes = 100
            
        scatter = ax.scatter(customers[:, 0], customers[:, 1],
                           c=demands[1:], s=sizes,
                           cmap='YlOrRd', edgecolors='black',
                           linewidth=1, alpha=0.7, zorder=4)
        
        # Aggiungi colorbar per le domande
        if show_demands:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Domanda', rotation=270, labelpad=15)
        
        # Aggiungi etichette ai nodi
        if show_node_ids:
            ax.text(depot[0], depot[1], '0', 
                   fontsize=10, fontweight='bold',
                   ha='center', va='center', color='white')
            
            for i, coord in enumerate(customers):
                ax.text(coord[0], coord[1], str(i+1),
                       fontsize=8, ha='center', va='center')
        
        # Aggiungi annotazioni per domande alte
        if show_demands:
            high_demand_threshold = instance.capacity * 0.4
            for i, (coord, demand) in enumerate(zip(customers, demands[1:])):
                if demand > high_demand_threshold:
                    ax.annotate(f'd={int(demand)}',
                              xy=(coord[0], coord[1]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=7, color='red', fontweight='bold')
        
        # Imposta limiti e labels
        margin = 0.05
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        ax.set_xlim(coords[:, 0].min() - margin * x_range,
                   coords[:, 0].max() + margin * x_range)
        ax.set_ylim(coords[:, 1].min() - margin * y_range,
                   coords[:, 1].max() + margin * y_range)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # Titolo
        if title is None:
            title = f'Istanza CVRP - {instance.num_nodes} nodi, Capacità: {instance.capacity}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Info aggiuntive
        info_text = (f'Domanda totale: {instance.get_total_demand()}\n'
                    f'Min veicoli: {instance.get_min_vehicles_lb()}')
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return ax
    
    def plot_solution(self,
                     instance,
                     tours: List[List[int]],
                     tour_cost: Optional[float] = None,
                     title: Optional[str] = None,
                     save_path: Optional[Union[str, Path]] = None,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualizza una soluzione CVRP con i tour dei veicoli.
        
        Args:
            instance: Istanza CVRPInstance
            tours: Lista di tour (ogni tour è una lista di indici di nodi)
            tour_cost: Costo totale della soluzione
            title: Titolo del plot
            save_path: Path dove salvare la figura
            ax: Axes matplotlib esistente
            
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        coords = instance.coordinates
        demands = instance.demands
        
        # Colori per i diversi tour
        colors = plt.cm.tab20(np.linspace(0, 1, len(tours)))
        
        # Plot nodi
        depot = coords[0]
        customers = coords[1:]
        
        ax.scatter(depot[0], depot[1], 
                  c='red', s=300, marker='s', 
                  edgecolors='black', linewidth=2,
                  label='Deposito', zorder=5)
        
        ax.scatter(customers[:, 0], customers[:, 1],
                  c='lightgray', s=100,
                  edgecolors='black', linewidth=1,
                  alpha=0.7, zorder=4)
        # visualization.py, dentro plot_solution, prima del loop che plottà i tour
        # Se sembra 1-based (max indice >= len(coords)), converti on-the-fly
        if len(tours) > 0 and max((max(t) if t else 0) for t in tours) >= len(coords):
            tours = [[n-1 for n in t] for t in tours]

        # Plot tour
        for tour_idx, (tour, color) in enumerate(zip(tours, colors)):
            if len(tour) == 0:
                continue
                
            # Aggiungi deposito all'inizio e alla fine se non presente
            if tour[0] != 0:
                tour = [0] + tour
            if tour[-1] != 0:
                tour = tour + [0]
            
            # print(tour)
            # print(coords)
            # Coordinate del tour
            tour_coords = coords[tour]
            
            # Disegna il percorso
            ax.plot(tour_coords[:, 0], tour_coords[:, 1],
                   'o-', color=color, linewidth=2,
                   markersize=8, alpha=0.7,
                   label=f'Tour {tour_idx+1}')
            
            # Aggiungi frecce direzionali
            for i in range(len(tour_coords) - 1):
                dx = tour_coords[i+1, 0] - tour_coords[i, 0]
                dy = tour_coords[i+1, 1] - tour_coords[i, 1]
                ax.arrow(tour_coords[i, 0], tour_coords[i, 1],
                        dx*0.8, dy*0.8,
                        head_width=0.015, head_length=0.02,
                        fc=color, ec=color, alpha=0.5)
            
            # Calcola e mostra il carico del veicolo
            tour_demand = sum(demands[node] for node in tour if node != 0)
            mid_point = len(tour_coords) // 2
            ax.text(tour_coords[mid_point, 0], tour_coords[mid_point, 1],
                   f'{tour_demand}/{instance.capacity}',
                   fontsize=8, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Aggiungi ID dei nodi
        for i, coord in enumerate(coords):
            ax.text(coord[0], coord[1], str(i),
                   fontsize=8, ha='center', va='center')
        
        # Titolo e info
        if title is None:
            title = f'Soluzione CVRP - {len(tours)} veicoli'
            if tour_cost:
                title += f', Costo: {tour_cost:.2f}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Statistiche della soluzione
        total_demand_served = sum(sum(demands[node] for node in tour if node != 0) 
                                 for tour in tours)
        vehicle_utilization = [sum(demands[node] for node in tour if node != 0) / instance.capacity 
                              for tour in tours]
        
        stats_text = (f'Veicoli utilizzati: {len(tours)}\n'
                     f'Domanda servita: {total_demand_served}/{instance.get_total_demand()}\n'
                     f'Utilizzo medio: {np.mean(vehicle_utilization)*100:.1f}%')
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return ax
    
    def plot_distance_matrix(self,
                            instance,
                            annotate: bool = False,
                            save_path: Optional[Union[str, Path]] = None,
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualizza la matrice delle distanze come heatmap.
        
        Args:
            instance: Istanza CVRPInstance
            annotate: Se True, mostra i valori nella heatmap
            save_path: Path dove salvare la figura
            ax: Axes matplotlib esistente
            
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        dist_matrix = instance.distance_matrix
        
        # Crea heatmap
        im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        
        # Aggiungi colorbar
        plt.colorbar(im, ax=ax, label='Distanza')
        
        # Etichette
        n_nodes = len(dist_matrix)
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels(range(n_nodes))
        ax.set_yticklabels(range(n_nodes))
        
        # Evidenzia il deposito
        ax.axhline(y=0.5, color='blue', linewidth=2, alpha=0.5)
        ax.axvline(x=0.5, color='blue', linewidth=2, alpha=0.5)
        
        # Annotazioni
        if annotate and n_nodes <= 20:
            for i in range(n_nodes):
                for j in range(n_nodes):
                    ax.text(j, i, f'{dist_matrix[i, j]:.1f}',
                           ha='center', va='center',
                           color='white' if dist_matrix[i, j] > dist_matrix.max()/2 else 'black',
                           fontsize=6)
        
        ax.set_title(f'Matrice delle Distanze - {instance.num_nodes} nodi', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Nodo destinazione', fontsize=12)
        ax.set_ylabel('Nodo origine', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return ax
    
    def plot_demand_distribution(self,
                                instances: List,
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualizza la distribuzione delle domande per un set di istanze.
        
        Args:
            instances: Lista di istanze CVRPInstance
            save_path: Path dove salvare la figura
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Raccogli tutte le domande
        all_demands = []
        all_capacities = []
        all_utilizations = []
        
        for instance in instances:
            demands = instance.demands[1:]  # Escludi deposito
            all_demands.extend(demands)
            all_capacities.append(instance.capacity)
            all_utilizations.append(instance.get_total_demand() / 
                                   (instance.capacity * instance.get_min_vehicles_lb()))
        
        # 1. Istogramma delle domande
        ax = axes[0, 0]
        ax.hist(all_demands, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_demands), color='red', linestyle='--', 
                  label=f'Media: {np.mean(all_demands):.2f}')
        ax.set_xlabel('Domanda', fontsize=12)
        ax.set_ylabel('Frequenza', fontsize=12)
        ax.set_title('Distribuzione delle Domande', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Boxplot domande per istanza
        ax = axes[0, 1]
        demands_per_instance = [instance.demands[1:] for instance in instances[:20]]  # Max 20 istanze
        ax.boxplot(demands_per_instance)
        ax.set_xlabel('Istanza', fontsize=12)
        ax.set_ylabel('Domanda', fontsize=12)
        ax.set_title('Domande per Istanza (prime 20)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Rapporto domanda/capacità
        ax = axes[1, 0]
        demand_capacity_ratios = [instance.get_total_demand() / instance.capacity 
                                 for instance in instances]
        ax.hist(demand_capacity_ratios, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(demand_capacity_ratios), color='red', linestyle='--',
                  label=f'Media: {np.mean(demand_capacity_ratios):.2f}')
        ax.set_xlabel('Rapporto Domanda Totale/Capacità', fontsize=12)
        ax.set_ylabel('Frequenza', fontsize=12)
        ax.set_title('Utilizzo della Flotta', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Numero minimo di veicoli
        ax = axes[1, 1]
        min_vehicles = [instance.get_min_vehicles_lb() for instance in instances]
        unique, counts = np.unique(min_vehicles, return_counts=True)
        ax.bar(unique, counts, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Numero Minimo di Veicoli', fontsize=12)
        ax.set_ylabel('Frequenza', fontsize=12)
        ax.set_title('Distribuzione Veicoli Minimi', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Analisi Dataset - {len(instances)} istanze', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_dataset_comparison(datasets: Dict[str, List],
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Confronta le statistiche di diversi dataset.
    
    Args:
        datasets: Dizionario {nome_dataset: lista_istanze}
        save_path: Path dove salvare la figura
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    
    stats = {}
    for name, instances in datasets.items():
        stats[name] = {
            'num_nodes': [inst.num_nodes for inst in instances],
            'total_demand': [inst.get_total_demand() for inst in instances],
            'min_vehicles': [inst.get_min_vehicles_lb() for inst in instances],
            'avg_demand': [np.mean(inst.demands[1:]) for inst in instances],
            'capacity': [inst.capacity for inst in instances],
            'utilization': [inst.get_total_demand() / (inst.capacity * inst.get_min_vehicles_lb()) 
                          for inst in instances]
        }
    
    # Plot comparisons
    metrics = [
        ('num_nodes', 'Numero di Nodi'),
        ('total_demand', 'Domanda Totale'),
        ('min_vehicles', 'Veicoli Minimi'),
        ('avg_demand', 'Domanda Media'),
        ('capacity', 'Capacità'),
        ('utilization', 'Utilizzo Flotta')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        data = [stats[name][metric] for name in datasets.keys()]
        bp = ax.boxplot(data, labels=list(datasets.keys()), patch_artist=True)
        
        # Colora i box
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylabel('Valore', fontsize=10)
        
        # Aggiungi media
        for i, name in enumerate(datasets.keys()):
            mean_val = np.mean(stats[name][metric])
            ax.plot(i+1, mean_val, 'r*', markersize=10)
    
    plt.suptitle('Confronto Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
