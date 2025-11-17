# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:00:18 2025

@author: alexq
"""

"""
LKH-3 Solver Wrapper
====================
Interfaccia Python per il solver LKH-3.
Gestisce la creazione dei file di parametro e l'esecuzione del solver.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time
import platform
import shutil


class LKHSolver:
    """Wrapper per il solver LKH-3"""
    
    def __init__(self, 
                 lkh_path: Optional[Union[str, Path]] = None,
                 time_limit: int = 60,
                 runs: int = 5,
                 seed: Optional[int] = None):
        """
        Args:
            lkh_path: Percorso all'eseguibile LKH-3
            time_limit: Limite di tempo in secondi
            runs: Numero di run del solver
            seed: Seed per la riproducibilità
        """
        # Trova LKH-3 executable
        if lkh_path is None:
            lkh_path = self._find_lkh_executable()
        
        self.lkh_path = Path(lkh_path)
        if not self.lkh_path.exists():
            raise FileNotFoundError(f"LKH-3 non trovato in: {self.lkh_path}")
        
        self.time_limit = time_limit
        self.runs = runs
        self.seed = seed or int(time.time())
        
    def _find_lkh_executable(self) -> Path:
        """Trova automaticamente l'eseguibile LKH-3"""
        # Percorsi comuni dove cercare LKH-3
        possible_paths = [
            Path("external/LKH-3/LKH-3"),
            Path("external/LKH-3/LKH-3.exe"),
            Path("external/LKH-3/LKH"),
            Path("external/LKH-3/LKH.exe"),
        ]
        
        # Aggiungi estensione per Windows
        if platform.system() == "Windows":
            possible_paths = [p.with_suffix('.exe') if not p.suffix else p 
                             for p in possible_paths]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Prova a cercare nel PATH di sistema
        lkh_cmd = "LKH-3.exe" if platform.system() == "Windows" else "LKH-3"
        if shutil.which(lkh_cmd):
            return Path(shutil.which(lkh_cmd))
        
        raise FileNotFoundError(
            "LKH-3 non trovato. Scaricalo da: "
            "http://webhotel4.ruc.dk/~keld/research/LKH-3/"
        )
    
    def solve(self, 
              vrp_file: Union[str, Path],
              output_dir: Optional[Union[str, Path]] = None,
              verbose: bool = False) -> Dict:
        """
        Risolve un'istanza CVRP usando LKH-3.
        
        Args:
            vrp_file: Path al file .vrp
            output_dir: Directory per i file di output (default: temp)
            verbose: Se True, mostra output del solver
            
        Returns:
            Dizionario con soluzione e statistiche
        """
        vrp_file = Path(vrp_file)
        num_nodes = self._get_vrp_dimension(vrp_file)
        
        # Crea directory temporanea se necessario
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="lkh_")
            output_dir = Path(temp_dir)
            cleanup = True
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        
        try:
            # Crea file dei parametri
            par_file = output_dir / f"{vrp_file.stem}.par"
            sol_file = output_dir / f"{vrp_file.stem}.sol"
            
            self._create_parameter_file(vrp_file, par_file, sol_file)
            
            # Esegui LKH-3
            start_time = time.time()
            result = self._run_lkh(par_file, verbose)
            execution_time = time.time() - start_time
            
            # Parsing della soluzione
            solution_data = self._parse_solution(sol_file, num_nodes)
            
            # Aggiungi statistiche
            solution_data['execution_time'] = execution_time
            solution_data['solver'] = 'LKH-3'
            solution_data['parameters'] = {
                'runs': self.runs,
                'time_limit': self.time_limit,
                'seed': self.seed
            }
            
            return solution_data
            
        finally:
            # Pulizia file temporanei se necessario
            if cleanup and output_dir.exists():
                shutil.rmtree(output_dir)
    
    def _create_parameter_file(self, 
                               vrp_file: Path, 
                               par_file: Path, 
                               sol_file: Path):
        """Crea il file dei parametri per LKH-3"""
        parameters = [
            f"PROBLEM_FILE = {vrp_file.absolute()}",
            f"RUNS = {self.runs}",
            f"TIME_LIMIT = {self.time_limit}",
            f"TOUR_FILE = {sol_file.absolute()}",
            f"SEED = {self.seed}",
            "TRACE_LEVEL = 0"  # Riduci output
        ]
        
        with open(par_file, 'w') as f:
            f.write('\n'.join(parameters))
    
    def _run_lkh(self, par_file: Path, verbose: bool = False) -> subprocess.CompletedProcess:
        """Esegue LKH-3 con il file dei parametri"""
        cmd = [str(self.lkh_path), str(par_file)]
        
        if verbose:
            result = subprocess.run(cmd, capture_output=False, text=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode != 0:
            error_msg = result.stderr if hasattr(result, 'stderr') else "Unknown error"
            raise RuntimeError(f"LKH-3 fallito: {error_msg}")
        
        return result
    
    def _get_vrp_dimension(self, vrp_file: Path) -> int:
        """Estrae DIMENSION dal file VRP"""
        with open(vrp_file, 'r') as f:
            for line in f:
                if line.startswith("DIMENSION"):
                    return int(line.split(':')[1].strip())
        raise ValueError(f"DIMENSION non trovata in {vrp_file}")
    
    def _parse_solution(self, sol_file: Path, num_nodes: int) -> Dict:
        """Parse del file .sol generato da LKH-3"""
        if not sol_file.exists():
            raise FileNotFoundError(f"File soluzione non trovato: {sol_file}")
        
        solution_data = {
            'tour': [],
            'cost': None,
            'routes': [],
            'name': None,
            'comment': None
        }
        
        with open(sol_file, 'r') as f:
            lines = f.readlines()
        
        in_tour_section = False
        current_tour = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("NAME"):
                solution_data['name'] = line.split(':', 1)[1].strip()
            elif line.startswith("COMMENT"):
                if "Length" in line:
                    # Estrai il costo della soluzione
                    try:
                        cost_str = line.split('=')[1].strip()
                        solution_data['cost'] = float(cost_str)
                    except:
                        pass
                solution_data['comment'] = line.split(':', 1)[1].strip()

            elif line == "TOUR_SECTION":
                in_tour_section = True
            elif in_tour_section:
                if line == "-1" or line == "EOF":
                    in_tour_section = False
                else:
                    try:
                        node = int(line)
                        current_tour.append(node)
                    except ValueError:
                        pass
        
        # Converti tour in routes (split al deposito)
        solution_data['tour'] = current_tour
        
        solution_data['routes'] = self._tour_to_routes_with_virtual_depots(current_tour,num_nodes)
        
        return solution_data
    
    def _tour_to_routes_with_virtual_depots(self, tour: List[int], num_nodes: int) -> List[List[int]]:
        """
        Converte un tour con depot virtuali in route separate.
        
        Args:
            tour: Tour completo da LKH-3
            num_nodes: Numero di nodi reali nel problema (include depot)
        """
        if not tour:
            return []
        
        routes = []
        current_route = []
        
        for node in tour:
            if node == 1:  # Depot reale
                if current_route:
                    routes.append(current_route)
                    current_route = []
            elif node > num_nodes:  # Depot virtuale
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:  # Cliente normale (2 <= node <= num_nodes)
                current_route.append(node)
        
        # Aggiungi l'ultima route se non vuota
        if current_route:
            routes.append(current_route)
        
        return routes

    def _tour_to_routes(self, tour: List[int]) -> List[List[int]]:
        """Converte un tour completo in route separate"""
        if not tour:
            return []
        
        routes = []
        current_route = []
        
        for node in tour:
            if node == 1:  # Deposito
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        
        # Aggiungi l'ultima route se non vuota
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def batch_solve(self, 
                   vrp_files: List[Union[str, Path]],
                   output_dir: Optional[Union[str, Path]] = None,
                   n_jobs: int = 1,
                   verbose: bool = False) -> List[Dict]:
        """
        Risolve multiple istanze CVRP.
        
        Args:
            vrp_files: Lista di file .vrp
            output_dir: Directory per output
            n_jobs: Numero di processi paralleli (TODO: implementare parallelizzazione)
            verbose: Mostra progresso
            
        Returns:
            Lista di soluzioni
        """
        solutions = []
        
        for i, vrp_file in enumerate(vrp_files):
            if verbose:
                print(f"Solving instance {i+1}/{len(vrp_files)}: {Path(vrp_file).name}")
            
            try:
                solution = self.solve(vrp_file, output_dir, verbose=True)
                solutions.append(solution)
            except Exception as e:
                print(f"Errore nel risolvere {vrp_file}: {e}")
                solutions.append(None)
        
        return solutions


class LKHSolutionValidator:
    """Validatore per le soluzioni generate da LKH-3"""
    
    @staticmethod
    def validate_solution(instance, solution: Dict) -> Dict:
        """
        Valida una soluzione rispetto all'istanza CVRP.
        
        Returns:
            Dizionario con risultati della validazione
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        routes = solution.get('routes', [])
        
        # Filtra eventuali nodi invalidi nelle route
        max_node = instance.num_nodes + 1  # +1 perché i nodi sono 1-based
        cleaned_routes = []
        for route in routes:
            cleaned_route = [n for n in route if 1 <= n <= max_node]
            if cleaned_route != route:
                validation['warnings'].append(
                    f"Route conteneva nodi invalidi: {set(route) - set(cleaned_route)}"
                )
            cleaned_routes.append(cleaned_route)
        
        routes = cleaned_routes
        # Verifica che tutti i nodi siano visitati
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route)
        
        expected_nodes = set(range(2, instance.num_nodes + 2))  # Nodi 2 a N+1
        missing_nodes = expected_nodes - visited_nodes
        extra_nodes = visited_nodes - expected_nodes
        
        if missing_nodes:
            validation['is_valid'] = False
            validation['errors'].append(f"Nodi mancanti: {missing_nodes}")
        
        if extra_nodes:
            # Solo un warning se sono nodi > max_node (depot virtuali)
            if any(n > max_node for n in extra_nodes):
                validation['warnings'].append(f"Depot virtuali trovati: {extra_nodes}")
            else:
                validation['is_valid'] = False
                validation['errors'].append(f"Nodi extra: {extra_nodes}")
    
        
        # Verifica vincoli di capacità
        for i, route in enumerate(routes):
            try:
                # Usa min() per evitare index out of bounds
                route_demand = sum(
                    instance.demands[min(node-1, len(instance.demands)-1)] 
                    for node in route if node <= len(instance.demands)
                )
                if route_demand > instance.capacity:
                    validation['is_valid'] = False
                    validation['errors'].append(
                        f"Route {i+1} supera la capacità: {route_demand} > {instance.capacity}"
                    )
            except Exception as e:
                validation['errors'].append(f"Errore nel calcolare domanda route {i+1}: {e}")
    
        
        # Calcola statistiche
        validation['stats'] = {
            'num_routes': len(routes),
            'total_cost': solution.get('cost', 0),
            'avg_route_length': np.mean([len(r) for r in routes]) if routes else 0,
            'max_route_length': max([len(r) for r in routes]) if routes else 0,
            'min_route_length': min([len(r) for r in routes]) if routes else 0
        }
        
        return validation