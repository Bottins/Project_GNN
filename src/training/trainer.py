# src/training/trainer.py
"""
Training Pipeline for CVRP GNN
===============================
Sistema completo di training con validazione e logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import json
import torch.nn.functional as F

class CVRPLoss(nn.Module):
    """
    Loss completamente riprogettata per CVRP basata su TOUR FORMATION.

    Problema fondamentale: VRP richiede PERCORSI validi, non archi indipendenti.

    Nuova strategia (in ordine di importanza):
    1. COVERAGE: Ogni nodo deve essere visitato esattamente una volta
    2. TOUR FORMATION: Le probabilità devono formare cicli (flow conservation)
    3. DEPOT BALANCE: k routes dal depot (k stimato da capacità)
    4. CAPACITY TOURS: Numero tours ≈ ceil(total_demand / capacity)
    5. SIMILARITY: Guida verso soluzione ottima (peso MOLTO ridotto)
    """

    def __init__(self,
                 # Pesi principali: TOUR-BASED LOSSES (alte)
                 coverage_weight: float = 5.0,           # CRITICA: ogni nodo raggiunto
                 tour_formation_weight: float = 3.0,     # ALTA: flow conservation
                 depot_balance_weight: float = 2.0,      # Bilancia in/out depot
                 capacity_tours_weight: float = 1.5,     # Numero tours corretto

                 # Peso similarity: MOLTO RIDOTTO (solo guida)
                 similarity_weight: float = 0.3,         # BASSA: non dominare

                 # Node sequence loss (opzionale)
                 node_weight: float = 0.1,

                 # Focal loss per similarity
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()

        # Pesi tour-based (PRINCIPALI)
        self.coverage_weight = coverage_weight
        self.tour_formation_weight = tour_formation_weight
        self.depot_balance_weight = depot_balance_weight
        self.capacity_tours_weight = capacity_tours_weight

        # Peso similarity (GUIDA DEBOLE)
        self.similarity_weight = similarity_weight

        # Node weight
        self.node_weight = node_weight

        # Focal loss
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict, data) -> Dict[str, torch.Tensor]:
        """
        Calcola le loss TOUR-BASED per CVRP.

        Focus: Formare PERCORSI VALIDI, non classificare archi indipendenti.

        Returns:
            Dict con loss totale e componenti
        """
        losses = {}

        # Calcola probabilità da logits
        edge_probs = torch.sigmoid(predictions['edge_predictions'])

        # ===== LOSS TOUR-BASED (PRINCIPALI) =====

        # 1. COVERAGE LOSS - CRITICA
        #    Ogni nodo (non depot) DEVE essere raggiunto esattamente 1 volta
        #    Forte penalità se sum(in_probs) != 1 o sum(out_probs) != 1
        coverage_loss = self._coverage_loss(
            edge_probs,
            data.edge_index,
            data.num_nodes
        )
        losses['coverage_loss'] = coverage_loss * self.coverage_weight

        # 2. TOUR FORMATION LOSS - ALTA
        #    Flow conservation: per ogni nodo, sum(in) = sum(out)
        #    Questo forza la creazione di CICLI
        tour_formation_loss = self._tour_formation_loss(
            edge_probs,
            data.edge_index,
            data.num_nodes
        )
        losses['tour_formation_loss'] = tour_formation_loss * self.tour_formation_weight

        # 3. DEPOT BALANCE LOSS
        #    Depot: sum(out) = sum(in) = numero di routes
        depot_balance_loss = self._depot_balance_loss(
            edge_probs,
            data.edge_index
        )
        losses['depot_balance_loss'] = depot_balance_loss * self.depot_balance_weight

        # 4. CAPACITY TOURS LOSS
        #    Numero di tours dal depot ≈ ceil(total_demand / capacity)
        if hasattr(data, 'x') and hasattr(data, 'capacity'):
            capacity_tours_loss = self._capacity_tours_loss(
                edge_probs,
                data.edge_index,
                data.x,
                data.capacity
            )
            losses['capacity_tours_loss'] = capacity_tours_loss * self.capacity_tours_weight

        # ===== SIMILARITY LOSS (GUIDA DEBOLE) =====

        # 5. SIMILARITY TO GROUND TRUTH
        #    Focal Loss con peso MOLTO ridotto, solo per guidare l'apprendimento
        #    NON deve dominare le loss tour-based
        if hasattr(data, 'y_edges'):
            if self.use_focal_loss:
                similarity_loss = self._focal_loss(
                    predictions['edge_predictions'],
                    data.y_edges,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                )
            else:
                similarity_loss = F.binary_cross_entropy_with_logits(
                    predictions['edge_predictions'],
                    data.y_edges
                )
            losses['similarity_loss'] = similarity_loss * self.similarity_weight

        # 6. NODE SEQUENCE LOSS (opzionale, peso molto basso)
        if hasattr(data, 'y_nodes') and 'node_predictions' in predictions:
            mask = data.y_nodes >= 0
            if mask.any():
                node_loss = self.mse_loss(
                    predictions['node_predictions'][mask],
                    data.y_nodes[mask]
                )
                losses['node_loss'] = node_loss * self.node_weight

        # Loss totale
        losses['total_loss'] = sum(losses.values())

        return losses

    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        Focal Loss per gestire lo sbilanciamento delle classi.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            inputs: Logits predetti
            targets: Target binari (0 o 1)
            alpha: Peso per classe positiva
            gamma: Focusing parameter (gamma > 0 riduce peso agli esempi facili)
        """
        # Calcola BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calcola probabilità
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calcola alpha_t
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        # Focal term
        focal_weight = (1 - p_t) ** gamma

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()
    
    # ===== TOUR-BASED LOSS FUNCTIONS =====

    def _coverage_loss(self, edge_probs, edge_index, num_nodes):
        """
        COVERAGE LOSS - CRITICA per VRP

        Ogni nodo (eccetto depot) DEVE essere raggiunto esattamente UNA volta.

        Per ogni nodo cliente:
        - sum(incoming edge probs) dovrebbe essere = 1.0
        - sum(outgoing edge probs) dovrebbe essere = 1.0

        Forte penalità quadratica se != 1.0

        Args:
            edge_probs: Probabilità degli archi [num_edges]
            edge_index: Indici degli archi [2, num_edges]
            num_nodes: Numero totale di nodi (incluso depot)

        Returns:
            Penalità media per violazione di coverage
        """
        penalty = torch.tensor(0.0, device=edge_probs.device)

        # Per ogni nodo cliente (skip depot=0)
        for node in range(1, num_nodes):
            # Trova archi in ingresso
            in_mask = edge_index[1] == node
            in_sum = edge_probs[in_mask].sum() if in_mask.any() else torch.tensor(0.0, device=edge_probs.device)

            # Trova archi in uscita
            out_mask = edge_index[0] == node
            out_sum = edge_probs[out_mask].sum() if out_mask.any() else torch.tensor(0.0, device=edge_probs.device)

            # Penalizza se != 1.0 (forte penalità quadratica)
            penalty += (in_sum - 1.0) ** 2
            penalty += (out_sum - 1.0) ** 2

        # Media su tutti i nodi clienti (2 termini per nodo: in e out)
        return penalty / (2 * max(num_nodes - 1, 1))

    def _tour_formation_loss(self, edge_probs, edge_index, num_nodes):
        """
        TOUR FORMATION LOSS - Flow Conservation

        Forza le probabilità a formare CICLI (tours) validi.

        Principio: Per ogni nodo, sum(incoming) = sum(outgoing)
        Questo garantisce flow conservation = cicli chiusi

        Applicato a TUTTI i nodi (incluso depot).

        Args:
            edge_probs: Probabilità degli archi [num_edges]
            edge_index: Indici degli archi [2, num_edges]
            num_nodes: Numero totale di nodi

        Returns:
            Penalità media per violazione di flow conservation
        """
        penalty = torch.tensor(0.0, device=edge_probs.device)

        # Per ogni nodo (incluso depot)
        for node in range(num_nodes):
            # Somma incoming
            in_mask = edge_index[1] == node
            in_sum = edge_probs[in_mask].sum() if in_mask.any() else torch.tensor(0.0, device=edge_probs.device)

            # Somma outgoing
            out_mask = edge_index[0] == node
            out_sum = edge_probs[out_mask].sum() if out_mask.any() else torch.tensor(0.0, device=edge_probs.device)

            # Flow conservation: in = out
            penalty += (in_sum - out_sum) ** 2

        return penalty / num_nodes

    def _depot_balance_loss(self, edge_probs, edge_index):
        """
        DEPOT BALANCE LOSS

        Il depot deve avere:
        - sum(outgoing) = k (numero di routes)
        - sum(incoming) = k (numero di routes)
        - Quindi: sum(out) = sum(in)

        Args:
            edge_probs: Probabilità degli archi
            edge_index: Indici degli archi

        Returns:
            Penalità per sbilanciamento depot
        """
        depot_in = edge_probs[edge_index[1] == 0].sum()
        depot_out = edge_probs[edge_index[0] == 0].sum()

        # Dovrebbero essere uguali
        return (depot_in - depot_out) ** 2

    def _capacity_tours_loss(self, edge_probs, edge_index, node_features, capacity):
        """
        CAPACITY TOURS LOSS

        Il numero di tours dal depot dovrebbe essere ≈ ceil(total_demand / capacity).

        Questo guida il modello a creare il numero corretto di routes in base
        alla capacità del veicolo.

        Args:
            edge_probs: Probabilità degli archi
            edge_index: Indici degli archi
            node_features: Features dei nodi (contiene demands)
            capacity: Capacità del veicolo

        Returns:
            Penalità se numero tours != expected
        """
        # Estrai demands (skip depot)
        demands = node_features[:, 2]
        total_demand = demands[1:].sum()  # Escludi depot

        # Calcola numero atteso di tours
        if isinstance(capacity, torch.Tensor):
            cap_value = capacity.float().mean().item() if capacity.numel() > 1 else capacity.float().item()
        else:
            cap_value = float(capacity)

        expected_tours = torch.ceil(total_demand / cap_value)

        # Conta tours: sum(outgoing dal depot)
        depot_out_sum = edge_probs[edge_index[0] == 0].sum()

        # Penalizza se != expected
        return (depot_out_sum - expected_tours) ** 2

    # Fine delle loss functions tour-based


class CVRPTrainer:
    """Trainer per modelli CVRP GNN"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: Optional[Path] = None):
        """
        Args:
            model: Modello GNN
            optimizer: Ottimizzatore
            loss_fn: Funzione di loss
            device: Device per training
            checkpoint_dir: Directory per salvare checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"🚀 Trainer inizializzato su {device}")
    
    def train_epoch(self, train_loader: GeometricDataLoader) -> Dict[str, float]:
        """Esegue un'epoca di training"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'edge_acc': [],
            'node_mae': [],
            'pos_rate@0.5': [],
            'mean_logit': [],
            'mean_prob': []
        }
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Calcola loss
            losses = self.loss_fn(predictions, batch)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Tracking
            epoch_losses.append(total_loss.item())
            
            # Calcola metriche
            with torch.no_grad():
                metrics = self._compute_metrics(predictions, batch)
                for key, value in metrics.items():
                    if value is not None:
                        epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'edge_acc': f"{metrics.get('edge_acc', 0):.2%}"
            })
        
        # Calcola medie
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader: GeometricDataLoader) -> Dict[str, float]:
        """Valida il modello"""
        self.model.eval()
        val_losses = []
        val_metrics = {
            'edge_acc': [],
            'node_mae': [],
            'route_quality': [],
            'pos_rate@0.5': [],
            'mean_logit': [],
            'mean_prob': []
        }

        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Loss
                losses = self.loss_fn(predictions, batch)
                val_losses.append(losses['total_loss'].item())
                
                # Metriche
                metrics = self._compute_metrics(predictions, batch)
                for key, value in metrics.items():
                    if value is not None:
                        val_metrics[key].append(value)
        
        # Calcola medie
        avg_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in val_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def _compute_metrics(self, predictions: Dict, batch) -> Dict[str, float]:
        """Calcola metriche di valutazione"""
        metrics = {}
        
        # Edge accuracy Claude
        # if 'edge_predictions' in predictions and hasattr(batch, 'y_edges'):
        #     edge_probs = torch.sigmoid(predictions['edge_predictions'])
        #     edge_preds = (edge_probs > 0.5).float()
        #     correct = (edge_preds == batch.y_edges).float()
        #     metrics['edge_acc'] = correct.mean().item()
        
        # Node MAE
        if 'node_predictions' in predictions and hasattr(batch, 'y_nodes'):
            mask = batch.y_nodes >= 0
            if mask.any():
                mae = torch.abs(
                    predictions['node_predictions'][mask] - batch.y_nodes[mask]
                ).mean()
                metrics['node_mae'] = mae.item()
                
        if 'edge_predictions' in predictions and hasattr(batch, 'y_edges'):
            edge_logits = predictions['edge_predictions']
            edge_probs  = torch.sigmoid(edge_logits)
            edge_preds  = (edge_probs > 0.5).float()
        
            correct = (edge_preds == batch.y_edges).float()
            metrics['edge_acc'] = correct.mean().item()
        
            # logging extra
            metrics['pos_rate@0.5'] = (edge_preds.mean()).item()  # % archi predetti positivi
            with torch.no_grad():
                metrics['mean_logit'] = edge_logits.mean().item()
                metrics['mean_prob']  = edge_probs.mean().item()

        
        return metrics
    
    def train(self,
             train_loader: GeometricDataLoader,
             val_loader: GeometricDataLoader,
             epochs: int,
             patience: int = 10,
             scheduler: Optional[object] = None) -> Dict:
        """
        Training loop completo.
        
        Args:
            train_loader: DataLoader per training
            val_loader: DataLoader per validation
            epochs: Numero di epoche
            patience: Early stopping patience
            scheduler: Learning rate scheduler opzionale
            
        Returns:
            Storia del training
        """
        print("\n" + "="*60)
        print("🎯 INIZIO TRAINING")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training
            train_stats = self.train_epoch(train_loader)
            
            # Validation
            val_stats = self.validate(val_loader)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_stats['loss'])
            
            # Logging
            epoch_time = time.time() - start_time
            self._log_epoch(epoch, train_stats, val_stats, epoch_time)
            
            # Salva history
            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_loss'].append(val_stats['loss'])
            self.history['train_metrics'].append(train_stats)
            self.history['val_metrics'].append(val_stats)
            
            # Checkpoint se miglioramento
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                self.patience_counter = 0
                
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_stats['loss'])
                    print(f"   💾 Checkpoint salvato (best loss: {val_stats['loss']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⚠️ Early stopping dopo {epoch} epoche")
                    break
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETATO")
        print("="*60)
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def _log_epoch(self, epoch: int, train_stats: Dict, val_stats: Dict, epoch_time: float):
        """Log delle statistiche per epoca"""
        print(f"\n📈 Epoca {epoch}")
        print(f"   ⏱️  Tempo: {epoch_time:.1f}s")
        print(f"   📉 Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
        print(f"   🎯 Train Edge Acc: {train_stats.get('edge_acc', 0):.2%} | "
              f"Val Edge Acc: {val_stats.get('edge_acc', 0):.2%}")
        
        if 'node_mae' in train_stats:
            print(f"   📏 Train Node MAE: {train_stats['node_mae']:.2f} | "
                  f"Val Node MAE: {val_stats.get('node_mae', 0):.2f}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Salva checkpoint del modello"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': {
                'model_class': self.model.__class__.__name__,
                'device': str(self.device)
            }
        }
        
        path = self.checkpoint_dir / f"best_model.pth"
        torch.save(checkpoint, path)
        
        # Salva anche history come JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)