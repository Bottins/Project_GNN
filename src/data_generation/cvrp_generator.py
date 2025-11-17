"""
CVRP Instance Generator
=======================
Modulo per la generazione parametrizzabile di istanze CVRP.
Supporta diverse distribuzioni per coordinate e domande.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from scipy.spatial.distance import pdist, squareform


@dataclass
class CVRPConfig:
    """Configurazione per la generazione di istanze CVRP"""
    num_nodes: int = 25
    capacity: int = 20
    coord_range: Tuple[float, float] = (0, 1)
    demand_range: Tuple[int, int] = (1, 10)
    scale: float = 1000.0
    distribution: str = "uniform"  # uniform, gaussian, clustered
    depot_position: str = "random"  # random, center, corner
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Converte la configurazione in dizionario"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Crea configurazione da dizionario"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]):
        """Carica configurazione da file JSON"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class CVRPInstance:
    """Rappresenta una singola istanza del problema CVRP"""
    
    def __init__(self, 
                 instance_id: Union[int, str],
                 coordinates: np.ndarray, 
                 demands: np.ndarray, 
                 capacity: int,
                 config: Optional[CVRPConfig] = None):
        """
        Args:
            instance_id: Identificatore univoco dell'istanza
            coordinates: Array di coordinate (N+1, 2) dove il primo nodo è il deposito
            demands: Array di domande (N+1,) dove demands[0] = 0 (deposito)
            capacity: Capacità del veicolo
            config: Configurazione usata per generare l'istanza
        """
        self.id = instance_id
        self.coordinates = coordinates
        self.demands = demands
        self.capacity = capacity
        self.config = config or CVRPConfig()
        self.num_nodes = len(coordinates) - 1  # Escluso il deposito
        
        # Pre-calcola la matrice delle distanze
        self._distance_matrix = None
        
    @property
    def distance_matrix(self) -> np.ndarray:
        """Calcola e cachea la matrice delle distanze euclidee"""
        if self._distance_matrix is None:
            self._distance_matrix = squareform(pdist(self.coordinates, metric='euclidean'))
        return self._distance_matrix
    
    def get_total_demand(self) -> int:
        """Ritorna la domanda totale (escluso deposito)"""
        return int(np.sum(self.demands[1:]))
    
    def get_min_vehicles_lb(self) -> int:
        """Calcola il lower bound sul numero minimo di veicoli"""
        return int(np.ceil(self.get_total_demand() / self.capacity))
    
    def validate(self) -> bool:
        """Valida l'istanza CVRP"""
        # Verifica che il deposito abbia domanda 0
        if self.demands[0] != 0:
            return False
        
        # Verifica che nessuna domanda singola superi la capacità
        if np.any(self.demands > self.capacity):
            return False
        
        # Verifica dimensioni consistenti
        if len(self.coordinates) != len(self.demands):
            return False
        
        return True
    
    def to_vrp_format(self, scale: Optional[float] = None) -> str:
        """Converte l'istanza nel formato .vrp standard"""
        scale = scale or self.config.scale
        
        vrp_content = []
        vrp_content.append(f"NAME : {self.id}")
        vrp_content.append(f"COMMENT : Generated instance No. {self.id}")
        vrp_content.append("TYPE : CVRP")
        vrp_content.append(f"DIMENSION : {len(self.coordinates)}")
        vrp_content.append("EDGE_WEIGHT_TYPE : EUC_2D")
        vrp_content.append(f"CAPACITY : {self.capacity}")
        vrp_content.append("NODE_COORD_SECTION")
        
        for i, coord in enumerate(self.coordinates):
            x = int(coord[0] * scale)
            y = int(coord[1] * scale)
            vrp_content.append(f" {i+1} {x} {y}")
        
        vrp_content.append("DEMAND_SECTION")
        for i, demand in enumerate(self.demands):
            vrp_content.append(f"{i+1} {int(demand)}")
        
        vrp_content.append("DEPOT_SECTION")
        vrp_content.append(" 1")
        vrp_content.append(" -1")
        vrp_content.append("EOF")
        
        return "\n".join(vrp_content)
    
    def save_vrp(self, output_path: Union[str, Path], scale: Optional[float] = None):
        """Salva l'istanza in formato .vrp"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(self.to_vrp_format(scale))
    
    def get_info(self) -> Dict:
        """Ritorna informazioni sull'istanza"""
        return {
            'id': self.id,
            'num_nodes': self.num_nodes,
            'capacity': self.capacity,
            'total_demand': self.get_total_demand(),
            'min_vehicles': self.get_min_vehicles_lb(),
            'avg_demand': float(np.mean(self.demands[1:])),
            'std_demand': float(np.std(self.demands[1:])),
            'demand_range': (int(np.min(self.demands[1:])), int(np.max(self.demands[1:])))
        }


class CVRPGenerator:
    """Generatore di istanze CVRP con diverse distribuzioni"""
    
    def __init__(self, config: Optional[CVRPConfig] = None):
        """
        Args:
            config: Configurazione per la generazione
        """
        self.config = config or CVRPConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def generate_coordinates(self, num_nodes: int) -> np.ndarray:
        """
        Genera coordinate per i nodi secondo la distribuzione specificata.
        
        Args:
            num_nodes: Numero di nodi (escluso deposito)
            
        Returns:
            Array di coordinate (num_nodes+1, 2) con deposito in prima posizione
        """
        distribution = self.config.distribution
        coord_range = self.config.coord_range
        
        if distribution == "uniform":
            # Distribuzione uniforme nel range specificato
            coords = np.random.uniform(
                coord_range[0], coord_range[1], 
                size=(num_nodes, 2)
            )
            
        elif distribution == "gaussian":
            # Distribuzione gaussiana centrata
            center = (coord_range[0] + coord_range[1]) / 2
            scale = (coord_range[1] - coord_range[0]) / 6  # 99.7% entro il range
            coords = np.random.normal(center, scale, size=(num_nodes, 2))
            # Clipping per rimanere nel range
            coords = np.clip(coords, coord_range[0], coord_range[1])
            
        elif distribution == "clustered":
            # Genera cluster di nodi
            n_clusters = max(2, num_nodes // 10)
            coords = []
            
            # Genera centri dei cluster
            centers = np.random.uniform(
                coord_range[0] + 0.1, coord_range[1] - 0.1,
                size=(n_clusters, 2)
            )
            
            # Assegna nodi ai cluster
            nodes_per_cluster = num_nodes // n_clusters
            remaining = num_nodes % n_clusters
            
            for i, center in enumerate(centers):
                n_nodes = nodes_per_cluster + (1 if i < remaining else 0)
                cluster_std = 0.05 * (coord_range[1] - coord_range[0])
                cluster_coords = np.random.normal(center, cluster_std, size=(n_nodes, 2))
                coords.append(cluster_coords)
            
            coords = np.vstack(coords)
            coords = np.clip(coords, coord_range[0], coord_range[1])
        else:
            raise ValueError(f"Distribuzione non supportata: {distribution}")
        
        # Aggiungi deposito
        depot = self._generate_depot_position(coord_range)
        coordinates = np.vstack([depot, coords])
        
        return coordinates
    
    def _generate_depot_position(self, coord_range: Tuple[float, float]) -> np.ndarray:
        """Genera la posizione del deposito"""
        position = self.config.depot_position
        
        if position == "random":
            depot = np.random.uniform(coord_range[0], coord_range[1], size=(2,))
        elif position == "center":
            center = (coord_range[0] + coord_range[1]) / 2
            depot = np.array([center, center])
        elif position == "corner":
            # Sceglie un angolo a caso
            corners = [
                [coord_range[0], coord_range[0]],
                [coord_range[0], coord_range[1]],
                [coord_range[1], coord_range[0]],
                [coord_range[1], coord_range[1]]
            ]
            depot = np.array(corners[np.random.randint(0, 4)])
        else:
            raise ValueError(f"Posizione deposito non supportata: {position}")
        
        return depot
    
    def generate_demands(self, num_nodes: int) -> np.ndarray:
        """
        Genera domande per i nodi.
        
        Args:
            num_nodes: Numero di nodi (escluso deposito)
            
        Returns:
            Array di domande (num_nodes+1,) con demands[0] = 0 per il deposito
        """
        demand_range = self.config.demand_range
        
        # Genera domande per i clienti
        demands = np.random.randint(
            demand_range[0], 
            demand_range[1] + 1, 
            size=num_nodes
        )
        
        # Aggiungi domanda 0 per il deposito
        demands = np.concatenate([[0], demands])
        
        return demands
    
    def generate_instance(self, instance_id: Optional[Union[int, str]] = None) -> CVRPInstance:
        """
        Genera una singola istanza CVRP.
        
        Args:
            instance_id: ID univoco per l'istanza
            
        Returns:
            Istanza CVRP generata
        """
        if instance_id is None:
            instance_id = np.random.randint(0, 1000000)
        
        # Genera coordinate e domande
        coordinates = self.generate_coordinates(self.config.num_nodes)
        demands = self.generate_demands(self.config.num_nodes)
        
        # Crea l'istanza
        instance = CVRPInstance(
            instance_id=instance_id,
            coordinates=coordinates,
            demands=demands,
            capacity=self.config.capacity,
            config=self.config
        )
        
        # Valida l'istanza
        if not instance.validate():
            raise ValueError(f"Istanza {instance_id} non valida!")
        
        return instance
    
    def generate_dataset(self, 
                        num_instances: int,
                        start_id: int = 0,
                        verbose: bool = True) -> List[CVRPInstance]:
        """
        Genera un dataset di istanze CVRP.
        
        Args:
            num_instances: Numero di istanze da generare
            start_id: ID iniziale per le istanze
            verbose: Se True, mostra progresso
            
        Returns:
            Lista di istanze CVRP
        """
        instances = []
        
        for i in range(num_instances):
            instance_id = start_id + i
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Generata istanza {i + 1}/{num_instances}")
            
            instance = self.generate_instance(instance_id)
            instances.append(instance)
        
        return instances
    
    def save_dataset(self, 
                    instances: List[CVRPInstance],
                    output_dir: Union[str, Path],
                    save_vrp: bool = True) -> Dict:
        """
        Salva un dataset di istanze su disco.
        
        Args:
            instances: Lista di istanze da salvare
            output_dir: Directory di output
            save_vrp: Se True, salva anche i file .vrp
            
        Returns:
            Dizionario con statistiche del dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepara array per salvare in formato NPZ
        all_coords = []
        all_demands = []
        all_capacities = []
        all_ids = []
        
        for instance in instances:
            all_coords.append(instance.coordinates)
            all_demands.append(instance.demands)
            all_capacities.append(instance.capacity)
            all_ids.append(str(instance.id))
            
            if save_vrp:
                vrp_path = output_dir / "vrp_files" / f"{instance.id}.vrp"
                instance.save_vrp(vrp_path)
        
        # Salva in formato NPZ
        np.savez_compressed(
            output_dir / "dataset.npz",
            coords=np.stack(all_coords),
            demands=np.stack(all_demands),
            capacities=np.array(all_capacities),
            ids=np.array(all_ids)
        )
        
        # Salva configurazione
        with open(output_dir / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Calcola statistiche
        stats = {
            'num_instances': len(instances),
            'num_nodes': self.config.num_nodes,
            'capacity': self.config.capacity,
            'total_demands': [inst.get_total_demand() for inst in instances],
            'min_vehicles': [inst.get_min_vehicles_lb() for inst in instances]
        }
        
        # Salva statistiche
        with open(output_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
