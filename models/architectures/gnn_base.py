# models/architectures/gnn_base.py
"""
Base GNN Architecture for CVRP
================================
Architettura GNN con meccanismo di attention per CVRP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional, List


class CVRPGNNBase(nn.Module):
    """Graph Neural Network base per CVRP con GAT layers"""
    
    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            node_features: Dimensione feature nodi
            edge_features: Dimensione feature edges  
            hidden_dim: Dimensione hidden layers
            num_layers: Numero di GAT layers
            num_heads: Numero di attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer = GATConv(
                    hidden_dim, 
                    hidden_dim, 
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            else:
                layer = GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=i < num_layers - 1
                )
            self.gat_layers.append(layer)
        
        # Output layers
        final_dim = hidden_dim if num_layers > 1 else hidden_dim * num_heads
        
        # Edge classifier (quale edge fa parte della soluzione)
        self.edge_classifier = nn.Sequential(
            nn.Linear(final_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Node sequence predictor (ordine di visita)
        self.node_sequence = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"🔨 Inizializzato CVRPGNNBase:")
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - Layers: {num_layers}")
        print(f"   - Heads: {num_heads}")
        print(f"   - Parametri totali: {self._count_parameters():,}")
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass della rete.
        
        Returns:
            Dict con predizioni per edges e nodes
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Embedding
        x = self.node_embedding(x)
        edge_feat = self.edge_embedding(edge_attr) if edge_attr is not None else None
        
        # GAT layers con residual connections
        x_residual = x
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Residual connection ogni 2 layers
                if i % 2 == 1 and x.shape == x_residual.shape:
                    x = x + x_residual
                    x_residual = x
        
        # Edge predictions
        edge_predictions = self._predict_edges(x, edge_index, edge_feat)
        
        # Node sequence predictions
        node_predictions = self.node_sequence(x).squeeze(-1)
        
        # Global graph features (per informazioni aggiuntive)
        if batch is not None:
            graph_feat_mean = global_mean_pool(x, batch)
            graph_feat_max = global_max_pool(x, batch)
            graph_features = torch.cat([graph_feat_mean, graph_feat_max], dim=-1)
        else:
            graph_features = torch.cat([x.mean(0, keepdim=True), 
                                       x.max(0)[0].unsqueeze(0)], dim=-1)
        
        return {
            'edge_predictions': edge_predictions,
            'node_predictions': node_predictions,
            'graph_features': graph_features,
            'node_embeddings': x
        }
    
    def _predict_edges(self, 
                      node_features: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Predice quali edges fanno parte della soluzione"""
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        
        if edge_features is not None:
            combined = torch.cat([src_features, dst_features, edge_features], dim=-1)
        else:
            combined = torch.cat([src_features, dst_features], dim=-1)
        
        return self.edge_classifier(combined).squeeze(-1)
    
    def _count_parameters(self) -> int:
        """Conta i parametri del modello"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CVRPDecoder(nn.Module):
    """Decoder per convertire predizioni GNN in routes CVRP"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                edge_predictions: torch.Tensor,
                edge_index: torch.Tensor,
                num_nodes: int,
                capacity_constraint: Optional[torch.Tensor] = None) -> Dict:
        """
        Decodifica le predizioni in routes.
        
        Returns:
            Dict con routes e probabilità
        """
        # Applica sigmoid per ottenere probabilità
        edge_probs = torch.sigmoid(edge_predictions / self.temperature)
        
        # Costruisci matrice di adiacenza
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=edge_predictions.device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_probs
        
        # Greedy decoding: costruisci routes partendo dal depot
        routes = self._greedy_decode(adj_matrix, capacity_constraint)
        
        return {
            'routes': routes,
            'edge_probs': edge_probs,
            'adj_matrix': adj_matrix
        }
    
    def _greedy_decode(self,
                      adj_matrix: torch.Tensor,
                      capacity: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Decodifica greedy delle routes"""
        num_nodes = adj_matrix.shape[0]
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
        visited[0] = True  # Depot sempre visitato
        
        routes = []
        while not visited[1:].all():
            route = []
            current = 0  # Parti dal depot

            while True:
                # Trova prossimo nodo non visitato con probabilità più alta

                probs = adj_matrix[current].clone()
                probs[visited] = 0  # Maschera nodi visitati
                if probs.sum() == 0:
                    break

                next_node = probs.argmax().item()

                if next_node == 0:  # Tornato al depot
                    break

                route.append(next_node)
                visited[next_node] = True
                current = next_node

                # Check capacità se fornita
                if capacity is not None and len(route) >= capacity:
                    break
            
            if route:
                routes.append(route)
        
        return routes