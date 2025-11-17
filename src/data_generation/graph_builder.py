# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:58:37 2025

@author: alexq
"""

"""
Graph Builder for CVRP
======================
Converte istanze CVRP in grafi per GNN.
Supporta diverse rappresentazioni e feature engineering.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle


class CVRPGraphBuilder:
    """Costruttore di grafi per istanze CVRP"""
    
    def __init__(self,
                 node_features: List[str] = None,
                 edge_features: List[str] = None,
                 normalize: bool = True,
                 add_self_loops: bool = False,
                 fully_connected: bool = True):
        """
        Args:
            node_features: Lista di feature da includere nei nodi
            edge_features: Lista di feature da includere negli edge
            normalize: Se True, normalizza le feature
            add_self_loops: Se True, aggiunge self-loops al grafo
            fully_connected: Se True, crea grafo fully connected
        """
        # Node features migliorate con info di capacità e distanza
        self.node_features = node_features or [
            'x_coord', 'y_coord', 'demand', 'is_depot',
            'demand_capacity_ratio', 'distance_to_depot'
        ]
        # Edge features migliorate con info di capacità
        self.edge_features = edge_features or ['distance', 'capacity_feasible']
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.fully_connected = fully_connected
        
        # Statistiche per normalizzazione
        self.normalization_stats = {}
    
    def build_graph(self, 
                   instance,
                   solution: Optional[Dict] = None) -> Data:
        """
        Costruisce un grafo PyTorch Geometric da un'istanza CVRP.
        
        Args:
            instance: Istanza CVRP
            solution: Soluzione opzionale da includere come label
            
        Returns:
            Oggetto Data di PyTorch Geometric
        """
        # Estrai feature dei nodi
        node_features = self._extract_node_features(instance)
        
        # Costruisci edge index e features
        edge_index, edge_features = self._build_edges(instance)
        
        # Normalizza se richiesto
        if self.normalize:
            node_features = self._normalize_features(
                node_features, 'node'
            )
            edge_features = self._normalize_features(
                edge_features, 'edge'
            )
        
        # Crea oggetto Data
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_features)
        )
        
        # Aggiungi metadati
        data.num_nodes = instance.num_nodes + 1  # Include depot
        data.capacity = instance.capacity
        data.instance_id = str(instance.id)
        data.depot_idx = 0  # Il deposito è sempre il nodo 0
        
        # Aggiungi solution come label se disponibile
        if solution is not None:
            data = self._add_solution_labels(data, instance, solution)
        
        return data
    
    def _extract_node_features(self, instance) -> np.ndarray:
        """Estrae le feature dei nodi"""
        num_nodes = len(instance.coordinates)
        features = []
        
        for feat_name in self.node_features:
            if feat_name == 'x_coord':
                feat = instance.coordinates[:, 0]
            elif feat_name == 'y_coord':
                feat = instance.coordinates[:, 1]
            elif feat_name == 'demand':
                feat = instance.demands
            elif feat_name == 'is_depot':
                feat = np.zeros(num_nodes)
                feat[0] = 1
            elif feat_name == 'demand_capacity_ratio':
                feat = instance.demands / instance.capacity
            elif feat_name == 'polar_angle':
                # Angolo polare rispetto al deposito
                depot = instance.coordinates[0]
                angles = np.arctan2(
                    instance.coordinates[:, 1] - depot[1],
                    instance.coordinates[:, 0] - depot[0]
                )
                feat = angles
            elif feat_name == 'distance_to_depot':
                depot = instance.coordinates[0]
                distances = np.linalg.norm(
                    instance.coordinates - depot, axis=1
                )
                feat = distances
            else:
                raise ValueError(f"Feature nodo non supportata: {feat_name}")
            
            features.append(feat.reshape(-1, 1))
        
        return np.hstack(features)
    
    def _build_edges(self, instance) -> Tuple[np.ndarray, np.ndarray]:
        """Costruisce edge index e features"""
        num_nodes = len(instance.coordinates)
        
        if self.fully_connected:
            # Grafo completamente connesso
            src, dst = np.meshgrid(
                range(num_nodes), range(num_nodes)
            )
            edge_index = np.stack([src.ravel(), dst.ravel()])
            
            # Rimuovi self-loops se non richiesti
            if not self.add_self_loops:
                mask = edge_index[0] != edge_index[1]
                edge_index = edge_index[:, mask]
        else:
            # TODO: Implementare altre strategie (KNN, Delaunay, etc.)
            raise NotImplementedError("Solo fully connected supportato per ora")
        
        # Calcola edge features
        edge_features = self._extract_edge_features(instance, edge_index)
        
        return edge_index, edge_features
    
    def _extract_edge_features(self, 
                              instance,
                              edge_index: np.ndarray) -> np.ndarray:
        """Estrae le feature degli edge"""
        features = []
        
        for feat_name in self.edge_features:
            if feat_name == 'distance':
                # Distanza euclidea tra nodi
                src_coords = instance.coordinates[edge_index[0]]
                dst_coords = instance.coordinates[edge_index[1]]
                distances = np.linalg.norm(
                    src_coords - dst_coords, axis=1
                )
                feat = distances
            elif feat_name == 'demand_sum':
                # Somma delle domande dei nodi connessi
                demands = instance.demands[edge_index[0]] + \
                         instance.demands[edge_index[1]]
                feat = demands
            elif feat_name == 'capacity_feasible':
                # 1 se i due nodi possono stare nella stessa route
                demands = instance.demands[edge_index[0]] + \
                         instance.demands[edge_index[1]]
                feat = (demands <= instance.capacity).astype(float)
            else:
                raise ValueError(f"Feature edge non supportata: {feat_name}")
            
            features.append(feat.reshape(-1, 1))
        
        return np.hstack(features)
    
    def _add_solution_labels(self, 
                            data: Data,
                            instance,
                            solution: Dict) -> Data:
        """Aggiunge le label della soluzione al grafo"""
        routes = solution.get('routes', [])
        
        # Edge labels: 1 se l'edge è nella soluzione
        num_nodes = instance.num_nodes + 1
        solution_edges = set()
        
        for route in routes:
            # Aggiungi edges dal deposito al primo nodo
            if route:
                solution_edges.add((0, route[0]-1))  # -1 per indice 0-based
                solution_edges.add((route[-1]-1, 0))
                
                # Aggiungi edges nella route
                for i in range(len(route)-1):
                    src = route[i] - 1
                    dst = route[i+1] - 1
                    solution_edges.add((src, dst))
        
        # Crea edge labels
        edge_labels = torch.zeros(data.edge_index.shape[1])
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if (src, dst) in solution_edges:
                edge_labels[i] = 1.0
        
        data.y_edges = edge_labels
        
        # Node labels: ordine di visita
        node_order = torch.full((num_nodes,), -1, dtype=torch.float)
        order_idx = 1
        for route in routes:
            for node_id in route:
                node_order[node_id-1] = order_idx
                order_idx += 1
        
        data.y_nodes = node_order
        
        # Aggiungi costo totale
        data.total_cost = solution.get('cost', 0)
        
        return data
    
    def _normalize_features(self, 
                          features: np.ndarray,
                          feature_type: str) -> np.ndarray:
        """Normalizza le feature usando statistiche salvate o calcolate"""
        key = f"{feature_type}_features"
        
        if key not in self.normalization_stats:
            # Calcola statistiche
            self.normalization_stats[key] = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0) + 1e-8
            }
        
        stats = self.normalization_stats[key]
        normalized = (features - stats['mean']) / stats['std']
        
        return normalized
    
    def fit_normalization(self, instances: List) -> 'CVRPGraphBuilder':
        """
        Calcola statistiche di normalizzazione su un dataset.
        
        Args:
            instances: Lista di istanze CVRP
            
        Returns:
            Self per chaining
        """
        all_node_features = []
        all_edge_features = []
        
        for instance in instances:
            node_feat = self._extract_node_features(instance)
            all_node_features.append(node_feat)
            
            edge_index, edge_feat = self._build_edges(instance)
            all_edge_features.append(edge_feat)
        
        # Calcola statistiche globali
        all_node_features = np.vstack(all_node_features)
        all_edge_features = np.vstack(all_edge_features)
        
        self.normalization_stats['node_features'] = {
            'mean': np.mean(all_node_features, axis=0),
            'std': np.std(all_node_features, axis=0) + 1e-8
        }
        
        self.normalization_stats['edge_features'] = {
            'mean': np.mean(all_edge_features, axis=0),
            'std': np.std(all_edge_features, axis=0) + 1e-8
        }
        
        return self
    
    def save_stats(self, path: Union[str, Path]):
        """Salva statistiche di normalizzazione"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.normalization_stats, f)
    
    def load_stats(self, path: Union[str, Path]):
        """Carica statistiche di normalizzazione"""
        with open(path, 'rb') as f:
            self.normalization_stats = pickle.load(f)
        
        return self


class GraphDataset:
    """Dataset di grafi per training GNN"""
    
    def __init__(self, 
                 instances: List,
                 solutions: List[Dict],
                 graph_builder: CVRPGraphBuilder):
        """
        Args:
            instances: Lista di istanze CVRP
            solutions: Lista di soluzioni
            graph_builder: Builder per creare grafi
        """
        self.instances = instances
        self.solutions = solutions
        self.graph_builder = graph_builder
        self.graphs = []
        
        # Costruisci tutti i grafi
        self._build_graphs()
    
    def _build_graphs(self):
        """Costruisce tutti i grafi del dataset"""
        for instance, solution in zip(self.instances, self.solutions):
            if solution is not None:
                graph = self.graph_builder.build_graph(instance, solution)
                self.graphs.append(graph)
    
    def save(self, path: Union[str, Path]):
        """Salva il dataset di grafi"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.graphs, path)
        print(f"Dataset salvato: {len(self.graphs)} grafi in {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> List[Data]:
        """Carica un dataset di grafi salvato"""
        return torch.load(path,weights_only=False)