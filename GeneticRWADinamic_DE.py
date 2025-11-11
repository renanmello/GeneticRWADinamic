import heapq
import random
import os
from itertools import islice
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import json
import datetime

from pymoo.core.problem import ElementwiseProblem
class MyProblem(ElementwiseProblem):
    def __init__(self, gene_size, calc_fitness, manual_pair):

        self.calc_fitness = calc_fitness
        self.manual_pair = manual_pair

        super().__init__(n_var=gene_size,
                         n_obj=1,
                         xl=np.array([0]*gene_size),
                         xu=np.array([gene_size - 1]*gene_size),
                         vtype=int)
    
    def _evaluate(self, x, out, *args, **kwargs):

        x_int = x.astype(int)

        fitness = -self.calc_fitness(x_int.tolist(), self.manual_pair)

        out["F"] = [fitness]


class WDMSimulator:
    """
    Simulador de rede WDM (Wavelength Division Multiplexing) que usa
    algoritmo genético para resolver o problema RWA (Routing and Wavelength Assignment).
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 4,
                 gene_size: int = 5,
                 manual_selection: bool = False,
                 gene_variation_mode: str = "fixed",
                 k: int = 150, 
                 population_size: int = 120,
                 num_generations: int = 30,
                 crossover_rate: float = 0.6,
                 mutation_rate: float = 0.02, 
                 hops_weight: float = 0.7,
                 wavelength_weight: float = 0.3,
                 tournament_size: float = 3,
                 # --- novos parâmetros do PSO ---
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 n_gen: int = 40,
                 # --- novos parâmetros do DE ---
                 CR: float = 0.9,
                 F: float = 0.8
                 ):
        """
        Inicializa o simulador WDM.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda disponíveis
            gene_size: Tamanho do gene (número de pares origem-destino)
            manual_selection: Se True, usa pares manuais predefinidos
            gene_variation_mode: "fixed" ou "custom"
            k: Número máximo de caminhos mais curtos a considerar
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.gene_size = gene_size
        self.k = k
        self.manual_pairs = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
        self.manual_selection = manual_selection
        self.gene_variation_mode = gene_variation_mode
        self.hops_weight = hops_weight
        self.wavelength_weight = wavelength_weight
        self.tournament_size = tournament_size
        # --- novos parâmetros do PSO ---
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_gen = n_gen

        # --- novos parâmetros do DE ---
        self.CR = CR
        self.F = F

        # Calcula todos os k-shortest paths uma vez
        self.k_shortest_paths = self._get_all_k_shortest_paths(k=self.k)
        self.reset_network()

    def reset_network(self) -> None:
        """Reseta os canais de comprimento de onda na rede."""
        for u, v in self.graph.edges:
            self.graph[u][v]['wavelengths'] = np.ones(self.num_wavelengths, dtype=bool)
            self.graph[u][v]['current_wavelength'] = -1

    def release_wavelength(self, route: List[int], wavelength: int) -> None:
        """
        Libera um comprimento de onda de uma rota.

        Args:
            route: Lista de nós representando a rota
            wavelength: Índice do comprimento de onda a ser liberado
        """
        if not self._is_valid_wavelength(wavelength):
            raise ValueError(f"Comprimento de onda inválido: {wavelength}")

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                self.graph[u][v]['wavelengths'][wavelength] = True
                # Reset apenas se este era o wavelength em uso
                if self.graph[u][v]['current_wavelength'] == wavelength:
                    self.graph[u][v]['current_wavelength'] = -1

    def allocate_wavelength(self, route: List[int], wavelength: int) -> bool:
        """
        Aloca um comprimento de onda para uma rota.

        Args:
            route: Lista de nós representando a rota
            wavelength: Índice do comprimento de onda a ser alocado

        Returns:
            True se a alocação foi bem-sucedida, False caso contrário
        """
        if not self._is_valid_wavelength(wavelength):
            return False

        # Verifica se o wavelength está disponível em toda a rota
        if not self._is_wavelength_available(route, wavelength):
            return False

        # Aloca o wavelength
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self.graph[u][v]['wavelengths'][wavelength] = False
            self.graph[u][v]['current_wavelength'] = wavelength

        return True

    def _is_valid_wavelength(self, wavelength: int) -> bool:
        """Verifica se o wavelength é válido."""
        return 0 <= wavelength < self.num_wavelengths

    def _is_wavelength_available(self, route: List[int], wavelength: int) -> bool:
        """Verifica se um wavelength está disponível em toda a rota."""
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if not self.graph.has_edge(u, v):
                return False
            if not self.graph[u][v]['wavelengths'][wavelength]:
                return False
        return True

    def find_available_wavelength(self, route: List[int]) -> Optional[int]:
        """
        Encontra o primeiro wavelength disponível para uma rota.

        Args:
            route: Lista de nós representando a rota

        Returns:
            Índice do wavelength disponível ou None se nenhum estiver disponível
        """
        for wavelength in range(self.num_wavelengths):
            if self._is_wavelength_available(route, wavelength):
                return wavelength
        return None

    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """Calcula os k menores caminhos entre dois nós."""
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []

    def _get_all_k_shortest_paths(self, k: int) -> Dict[Tuple[int, int], List[List[int]]]:
        """Calcula os k menores caminhos para todos os pares origem-destino."""
        paths = {}
        pairs_to_process = self.manual_pairs if self.manual_selection else []

        # Se não está usando seleção manual, gera pares aleatórios
        if not self.manual_selection:
            nodes = list(self.graph.nodes)
            for _ in range(self.gene_size):
                source, target = random.sample(nodes, 2)
                pairs_to_process.append((source, target))

        for source, target in pairs_to_process:
            paths[(source, target)] = self._get_k_shortest_paths(source, target, k)

        return paths

    def _initialize_population(self, source_targets: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Inicializa a população do algoritmo genético.

        Args:
            source_targets: Lista de pares (origem, destino)

        Returns:
            População inicial
        """
        population = []

        for _ in range(self.population_size):
            individual = []
            for source, target in source_targets:
                if (source, target) not in self.k_shortest_paths:
                    raise ValueError(f"Nenhum caminho encontrado para ({source}, {target})")

                num_paths = len(self.k_shortest_paths[(source, target)])
                if num_paths == 0:
                    raise ValueError(f"Nenhum caminho válido para ({source}, {target})")

                # Escolhe um índice válido baseado no modo de variação do gene
                if self.gene_variation_mode == "fixed":
                    max_index = min(self.gene_size - 1, num_paths - 1)
                else:  # custom
                    max_index = num_paths - 1

                individual.append(random.randint(0, max_index))

            population.append(individual)

        return population

    def _fitness_route(self, route: List[int]) -> float:
        """
        Calcula a aptidão de uma rota específica.

        Args:
            route: Lista de nós representando a rota

        Returns:
            Valor de fitness da rota
        """
        if len(route) < 2:
            return 0.0

        # Número de saltos
        hops = len(route) - 1

        # Número de trocas de comprimentos de onda necessárias
        wavelength_changes = 0
        for i in range(len(route) - 2):
            u1, v1 = route[i], route[i + 1]
            u2, v2 = route[i + 1], route[i + 2]

            if (self.graph.has_edge(u1, v1) and self.graph.has_edge(u2, v2) and
                    self.graph[u1][v1].get('current_wavelength', -1) != -1 and
                    self.graph[u2][v2].get('current_wavelength', -1) != -1 and
                    self.graph[u1][v1]['current_wavelength'] != self.graph[u2][v2]['current_wavelength']):
                wavelength_changes += 1

        # Função de fitness combinada (normalizada)
        fitness = (self.hops_weight * (1 / (hops + 1)) +
                   self.wavelength_weight * (1 / (wavelength_changes + 1)))

        return fitness

    def _fitness(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> float:
        """
        Calcula a aptidão de um indivíduo.

        Args:
            individual: Cromossomo (lista de índices de rotas)
            source_targets: Lista de pares origem-destino

        Returns:
            Aptidão total do cromossomo
        """
        total_fitness = 0.0

        for i, (source, target) in enumerate(source_targets):
            if i >= len(individual):
                continue

            route_idx = individual[i]
            available_routes = self.k_shortest_paths.get((source, target), [])

            if not available_routes or route_idx >= len(available_routes):
                continue  # Penaliza rotas inválidas

            route = available_routes[route_idx]
            total_fitness += self._fitness_route(route)

        return total_fitness

    def _tournament_selection(self, population: List[List[int]],
                              source_targets: List[Tuple[int, int]],
                              tournament_size: int = 3) -> List[int]:
        """
        Realiza seleção por torneio.

        Args:
            population: População atual
            source_targets: Lista de pares origem-destino
            tournament_size: Tamanho do torneio

        Returns:
            Indivíduo selecionado
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: self._fitness(ind, source_targets))

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Realiza crossover entre dois pais.

        Args:
            parent1: Primeiro pai
            parent2: Segundo pai

        Returns:
            Tupla com dois filhos
        """
        if len(parent1) <= 1:
            return parent1[:], parent2[:]

        cut_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]

        return child1, child2

    def _mutate(self, individual: List[int], source_targets: List[Tuple[int, int]]) -> None:
        """
        Aplica mutação em um indivíduo.

        Args:
            individual: Indivíduo a ser mutado
            source_targets: Lista de pares origem-destino
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                source, target = source_targets[i]
                available_routes = self.k_shortest_paths.get((source, target), [])

                if available_routes:
                    if self.gene_variation_mode == "fixed":
                        max_index = min(self.gene_size - 1, len(available_routes) - 1)
                    else:  # custom
                        max_index = len(available_routes) - 1

                    individual[i] = random.randint(0, max_index)

    def genetic_algorithm(self, source_targets: List[Tuple[int, int]]) -> List[int]:
        """
        Executa o algoritmo genético.

        Args:
            source_targets: Lista de pares origem-destino

        Returns:
            Melhor indivíduo encontrado
        """
        # Inicializa população
        population = self._initialize_population(source_targets)

        best_fitness_history = []
        stagnation_count = 0
        max_stagnation = 10

        for generation in range(self.num_generations):
            print(f"Processando geração {generation + 1}/{self.num_generations}...")

            # Avalia e ordena população
            population.sort(key=lambda ind: self._fitness(ind, source_targets), reverse=True)
            population = population[:self.population_size]  # Mantém apenas os melhores

            best_individual = population[0]
            best_fitness = self._fitness(best_individual, source_targets)
            best_fitness_history.append(best_fitness)

            # Verifica estagnação
            if (len(best_fitness_history) > 1 and
                    abs(best_fitness - best_fitness_history[-2]) < 1e-6):
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= max_stagnation:
                print(f"Estagnação detectada após {generation + 1} gerações.")
                break

            # Cria nova geração
            next_generation = []

            # Elitismo: mantém os melhores
            elite_size = max(1, self.population_size // 10)
            next_generation.extend(population[:elite_size])

            # Gera resto da população
            while len(next_generation) < self.population_size:
                parent1 = self._tournament_selection(population, source_targets, self.tournament_size)
                parent2 = self._tournament_selection(population, source_targets, self.tournament_size)

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                    next_generation.extend([child1, child2])
                else:
                    next_generation.extend([parent1[:], parent2[:]])

            # Aplica mutação
            for individual in next_generation[elite_size:]:  # Não muta a elite
                self._mutate(individual, source_targets)

            population = next_generation[:self.population_size]

        best_individual = max(population, key=lambda ind: self._fitness(ind, source_targets))
        best_fitness_value = self._fitness(best_individual, source_targets)
    
        return best_individual, best_fitness_value
    

    def pso_algorithm(self):
        """
        Executa o algoritmo PSO.

        Args:
            source_targets: Lista de pares origem-destino

        Returns:
            Melhor indivíduo encontrado
            Melhor fitness encontrada
        """
    
        # Definição do problema para pymoo
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs)

        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.operators.sampling.rnd import IntegerRandomSampling

        algorithm = PSO(
            pop_size=self.population_size,
            w=self.w,
            c1=self.c1,
            c2=self.c2,
            sampling=IntegerRandomSampling(),
        )

        from pymoo.termination import get_termination
        termination = get_termination("n_gen", self.n_gen)

        from pymoo.optimize import minimize
        res_pso = minimize(problem, 
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=True)

        X_PSO = res_pso.X.astype(int)
        F_PSO = -res_pso.F

        print("-" * 30)
        print("PSO")
        print("Melhor Indivíduo da População (PSO):")
        print(X_PSO)
        print("Valor de Fitness do Melhor Indivíduo (PSO):")
        print(F_PSO)

        return X_PSO, F_PSO
    
    def de_algorithm(self):

        # Definição do problema para pymoo
        problem = MyProblem(self.gene_size, self._fitness, self.manual_pairs)
        
        from pymoo.operators.sampling.rnd import IntegerRandomSampling

        from pymoo.algorithms.soo.nonconvex.de import DE

        algorithm = DE(
            pop_size=self.population_size, # Tamanho da População
            # CR: Crossover Rate (Taxa de Crossover), define a chance de usar o vetor mutante.
            CR=self.CR, 
            # F: Scale Factor (Fator de Escala), define o quanto o vetor mutante é escalado.
            F=self.F,
            # O 'variant' define a estratégia de mutação e crossover (DE/rand/1/bin é comum).
            variant="DE/rand/1/bin",
            # Define a estratégia de amostragem
            sampling=IntegerRandomSampling(),
        )


        from pymoo.termination import get_termination
        termination = get_termination("n_gen", self.n_gen)

        from pymoo.optimize import minimize
        res_DE = minimize(problem, 
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=True)
        
        X_DE = res_DE.X.astype(int)
        F_DE = res_DE.F

        return X_DE, F_DE


    def _calculate_blocking_probability(self, route: List[int], load: int) -> float:
        """
        Calcula a probabilidade de bloqueio para uma rota e carga específicas.

        Args:
            route: Rota a ser avaliada
            load: Carga de tráfego

        Returns:
            Probabilidade de bloqueio
        """
        # Modelo de probabilidade de bloqueio baseado na teoria de tráfego
        route_load = 0.05 * load
        utilization = route_load * len(route) / (self.num_wavelengths + 5)

        # Aplica limite para manter probabilidade entre 0 e 1
        return max(0.0, min(1.0, utilization))

    def simulate_network(self, num_simulations: int = 1,
                         output_file: str = "blocking_results.txt") -> Dict[str, List[float]]:
        """
        Simula a rede WDM e calcula probabilidades de bloqueio.

        Args:
            num_simulations: Número de simulações
            output_file: Arquivo de saída para resultados

        Returns:
            Dicionário com resultados por rota
        """
        results = {}

        with open(output_file, "w") as f:
            f.write("# Resultados de Probabilidade de Bloqueio\n")
            f.write("# Formato: Load Probabilidade\n")

            for sim in range(num_simulations):
                print(f"Executando simulação {sim + 1}/{num_simulations}")

                # Executa algoritmo genético para todos os pares
                # best_individual, best_fitness_value = self.genetic_algorithm(self.manual_pairs)
                best_individual, best_fitness_value = self.de_algorithm()

                for gene_idx, (source, target) in enumerate(self.manual_pairs):
                    if gene_idx >= len(best_individual):
                        continue

                    routes = self.k_shortest_paths.get((source, target), [])
                    if not routes or best_individual[gene_idx] >= len(routes):
                        continue

                    best_route = routes[best_individual[gene_idx]]

                    # Calcula probabilidades de bloqueio
                    blocking_probs = []
                    for load in range(1, 31):
                        blocked_calls = 0
                        total_calls = 1000  # Reduzido para melhor performance

                        for _ in range(total_calls):
                            block_prob = self._calculate_blocking_probability(best_route, load)
                            if random.random() < block_prob:
                                blocked_calls += 1

                        prob = blocked_calls / total_calls
                        blocking_probs.append(prob)
                        f.write(f"{load} {prob}\n")

                    # Armazena resultados
                    label = f'[{source},{target}]'
                    if label not in results:
                        results[label] = []
                    results[label].extend(blocking_probs)

                    # Salva arquivo individual
                    self._save_gene_results(gene_idx, source, target, blocking_probs)

        print(f"Resultados salvos em {output_file}")
        return results, best_individual, best_fitness_value

    def _save_gene_results(self, gene_idx: int, source: int, target: int,
                           blocking_probs: List[float]) -> None:
        """Salva resultados de um gene específico."""
        filename = f"gene_{gene_idx + 1}.txt"
        with open(filename, "w") as f:
            f.write(f"# [{source} -> {target}]\n")
            for load, prob in enumerate(blocking_probs, 1):
                f.write(f"{load} {prob}\n")

    def plot_individual_genes(self, save_path: str = "grafico_wdm_simulation.png") -> None:
        """
        Gera gráfico com resultados de todos os genes.

        Args:
            save_path: Caminho para salvar o gráfico
        """
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for gene_idx in range(self.gene_size):
            filename = f"gene_{gene_idx + 1}.txt"

            if not os.path.exists(filename):
                print(f"Arquivo {filename} não encontrado.")
                continue

            try:
                data = np.loadtxt(filename, comments='#')
                if data.size == 0:
                    continue

                if data.ndim == 1:
                    # Apenas uma linha de dados
                    loads = [1]
                    probs = [data]
                else:
                    loads = data[:, 0]
                    probs = data[:, 1]

                # Lê o cabeçalho
                with open(filename, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        label = first_line[2:]
                    else:
                        label = f"Gene {gene_idx + 1}"

                plt.plot(loads, probs,
                         label=label,
                         color=colors[gene_idx % len(colors)],
                         linewidth=2,
                         marker='o',
                         markersize=4)

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        plt.xlabel('Carga de Tráfego', fontsize=12)
        plt.ylabel('Probabilidade de Bloqueio', fontsize=12)
        plt.title('Probabilidade de Bloqueio por Rota', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 31, 5))
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Gráfico salvo em {save_path}")

    def generate_comparison_table(self, output_file: str = "comparacao_rotas.csv") -> pd.DataFrame:
        """
        Gera tabela de comparação das probabilidades de bloqueio.

        Args:
            output_file: Arquivo de saída CSV

        Returns:
            DataFrame com comparação
        """
        data = {}
        loads = None

        for gene_idx in range(self.gene_size):
            filename = f"gene_{gene_idx + 1}.txt"

            if not os.path.exists(filename):
                continue

            try:
                file_data = np.loadtxt(filename, comments='#')
                if file_data.size == 0:
                    continue

                if file_data.ndim == 1:
                    loads = [1]
                    probs = [file_data[1]] if len(file_data) > 1 else [file_data]
                else:
                    loads = file_data[:, 0].astype(int)
                    probs = file_data[:, 1]

                data[f"Rota {gene_idx + 1}"] = probs

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        if not data or loads is None:
            print("Nenhum dado válido encontrado para gerar tabela.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.insert(0, "Carga (Load)", loads)

        print("Tabela de Comparação:")
        print(df.to_string(index=False))

        df.to_csv(output_file, index=False)
        print(f"Tabela salva em {output_file}")

        return df


def objective(trial: optuna.Trial) -> float:
    """Função objetivo para otimização com Optuna."""
    
    # Parâmetros de pesos
    hops_weight = trial.suggest_float('hops_weight', 0.1, 0.9)
    wavelength_weight = trial.suggest_float('wavelength_weight', 0.1, 0.9)
    
    # --- novos parâmetros DE ---
    CR = trial.suggest_float('CR', 0.1, 1.0)  # Crossover Rate (literatura: 0.3–0.9)
    F = trial.suggest_float('F', 0.3, 1.0)    # Scale Factor (literatura: 0.4–0.9)
    n_gen = trial.suggest_int('n_gen', 10, 200)  # Número de gerações do PSO

    # Garantir que os pesos somem 1
    total_weight = hops_weight + wavelength_weight
    if total_weight != 1.0:
        # Normalizar os pesos
        hops_weight /= total_weight
        wavelength_weight /= total_weight
    
    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)
    
    # Configuração do simulador com os hiperparâmetros sugeridos
    wdm_simulator = WDMSimulator(
        graph=graph,
        num_wavelengths=4,
        gene_size=5,
        manual_selection=True,
        gene_variation_mode="fixed",
        k=5,
        hops_weight=hops_weight,
        wavelength_weight=wavelength_weight,
        n_gen=n_gen,
        CR=CR,
        F=F
    )
    
    # Executa a simulação
    results, best_individual, best_fitness_value = wdm_simulator.simulate_network(num_simulations=5)
    
    # Calcula a métrica de desempenho (média das probabilidades de bloqueio)
    performance_metric = calculate_performance_metric(results)
    
    return performance_metric

def calculate_performance_metric(results: Dict[str, List[float]]) -> float:
    """
    Calcula a métrica de desempenho a partir dos resultados.
    
    Args:
        results: Dicionário com os resultados da simulação
        
    Returns:
        Média das probabilidades de bloqueio de todas as requisições
    """
    all_means = []
    
    for key, values in results.items():
        if values:  # Verifica se a lista não está vazia
            mean_value = np.mean(values)
            all_means.append(mean_value)
    
    if all_means:
        return np.mean(all_means)
    else:
        return float('inf')  # Retorna um valor ruim se não houver resultados

def optimize_hyperparameters(n_trials: int = 100) -> Dict[str, any]:
    """
    Otimiza os hiperparâmetros usando Optuna.
    
    Args:
        n_trials: Número de tentativas de otimização
        
    Returns:
        Dicionário com os melhores hiperparâmetros encontrados
    """
    # Criar estudo de otimização
    study = optuna.create_study(
        direction='minimize',  # Diminuir a probabilidade de bloqueio
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Executar otimização
    study.optimize(objective, n_trials=n_trials)

    # Salva apenas os resultados principais em JSON
    best_results = {
        "best_params_DE_algorithm": study.best_params,
        "best_value_DE_algorithm": study.best_value,
        "datetime": str(datetime.datetime.now())
    }

    output_path = "best_params_DE_algorithm.json"
    with open(output_path, "w") as f:
        json.dump(best_results, f, indent=4)
    
    print(f"Resultados DE algorithm salvos em {output_path}")
    
    # Retornar os melhores hiperparâmetros
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


def main():
    """Função principal para executar a simulação otimizada."""
    
    # Otimizar hiperparâmetros
    print("Otimizando hiperparâmetros com Optuna...")
    optimization_results = optimize_hyperparameters(n_trials=2)
    
    best_params = optimization_results['best_params']

    # Normaliza os pesos assim como foi feito no código de otimização
    if 'hops_weight' in best_params and 'wavelength_weight' in best_params:
        total_weight = best_params['hops_weight'] + best_params['wavelength_weight']
        best_params['hops_weight'] = best_params['hops_weight']/total_weight
        best_params['wavelength_weight'] = best_params['wavelength_weight']/total_weight

    print(f"Melhores hiperparâmetros encontrados: {best_params}")
    print(f"Melhor valor da média de probabilidade de bloqueio das requisições: {optimization_results['best_value']}")
    
    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)

    # Configuração e execução da simulação com os melhores hiperparâmetros
    print("Iniciando simulação WDM com melhores hiperparâmetros...")
    wdm_simulator = WDMSimulator(
        graph=graph,
        num_wavelengths=4,
        gene_size=5,
        manual_selection=True,
        gene_variation_mode="fixed",
        k=5,
        **{k: v for k, v in best_params.items() if k in [
            'population_size', 'num_generations', 'crossover_rate',
            'mutation_rate', 'hops_weight', 'wavelength_weight',
            'tournament_size', 'w', 'c1', 'c2', 'n_gen' 
        ]}
    )

    # Executa simulação
    results, best_individual, best_fitness_value = wdm_simulator.simulate_network(num_simulations=5)  # numero de simulações

    # Gera visualizações e relatórios
    wdm_simulator.plot_individual_genes()
    wdm_simulator.generate_comparison_table()

    print("-" * 30)
    print("PSO + OPTUNA")
    print("Melhor Indivíduo da População (PSO) - Após otimização com Optuna:")
    print(best_individual)
    print("Valor de Fitness do Melhor Indivíduo (PSO) - Após otimização com Optuna:")
    print(best_fitness_value)


    print("Simulação concluída com sucesso!")



if __name__ == "__main__":
    main()