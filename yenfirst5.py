def _release_expired_wavelengths(self) -> None:
    """Libera wavelengths cuja duração expirou."""
    for edge in self.wavelength_allocation:
        # Encontra wavelengths expirados
        expired_wavelengths = [
            wl for wl, end_time in self.wavelength_allocation[edge].items()
            if end_time <= self.current_time
        ]

        # Remove wavelengths expirados
        for wl in expired_wavelengths:
            del self.wavelength_allocation[edge][wl]
            import os


from itertools import islice
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class WDMYenFirstFit:
    """
    Simulador WDM usando abordagem clássica:
    - Roteamento: YEN (sempre pega o menor caminho k[0])
    - Alocação de Wavelength: First Fit
    """

    def __init__(self,
                 graph: nx.Graph,
                 num_wavelengths: int = 4,
                 k: int = 5,
                 requests: List[Tuple[int, int]] = None):
        """
        Inicializa o simulador com abordagem clássica.

        Args:
            graph: Grafo da rede
            num_wavelengths: Número de comprimentos de onda
            k: Número de k-shortest paths para YEN
            requests: Lista de requisições (origem, destino)
        """
        self.graph = graph
        self.num_wavelengths = num_wavelengths
        self.k = k

        # Requisições padrão (pode ser customizado)
        self.requests = requests if requests else [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]

        # Estrutura para armazenar alocações com tempo
        self.wavelength_allocation = {}  # (u, v): {wavelength: end_time}
        self.call_records = []  # Histórico de chamadas
        self.current_time = 0  # Tempo atual da simulação

        self.reset_network()

    def reset_network(self) -> None:
        """Reseta o estado da rede."""
        for u, v in self.graph.edges:
            # Normaliza para manter (u, v) com u < v
            edge = (min(u, v), max(u, v))
            # Estrutura: {wavelength: end_time}
            self.wavelength_allocation[edge] = {}
        self.current_time = 0

    def _get_k_shortest_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        """
        Calcula os k menores caminhos usando YEN (Dijkstra + desvios).

        Args:
            source: Nó de origem
            target: Nó de destino
            k: Número de caminhos

        Returns:
            Lista com k menores caminhos
        """
        if not nx.has_path(self.graph, source, target):
            return []
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target), k))
        except nx.NetworkXNoPath:
            return []

    def first_fit_allocation_with_conversion(self, route: List[int],
                                             call_duration: float) -> Optional[Dict[int, int]]:
        """
        Aloca wavelengths usando First Fit com conversão permitida.
        Cada enlace pode usar um wavelength diferente (permite conversão).

        Args:
            route: Rota (lista de nós)
            call_duration: Duração da chamada (unidades de tempo)

        Returns:
            Dicionário {índice_enlace: wavelength} ou None se rota não puder ser alocada
        """
        wavelength_allocation_route = {}
        end_time = self.current_time + call_duration

        # Para cada enlace da rota
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))

            # Encontra o primeiro wavelength disponível neste enlace
            wavelength_found = None
            for wavelength in range(self.num_wavelengths):
                if wavelength not in self.wavelength_allocation[edge]:
                    wavelength_found = wavelength
                    break

            if wavelength_found is None:
                # Nenhum wavelength disponível neste enlace
                return None

            # Aloca este wavelength no enlace com tempo de expiração
            self.wavelength_allocation[edge][wavelength_found] = end_time
            wavelength_allocation_route[i] = wavelength_found

        return wavelength_allocation_route

    def _is_wavelength_available(self, route: List[int], wavelength: int) -> bool:
        """Verifica se um wavelength está livre em toda a rota."""
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))

            if wavelength in self.wavelength_allocation[edge]:
                return False

        return True

    def _allocate_wavelength_on_route(self, route: List[int], wavelength: int) -> None:
        """Aloca um wavelength em todos os enlaces da rota."""
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))

            if wavelength not in self.wavelength_allocation[edge]:
                self.wavelength_allocation[edge].append(wavelength)

    def _deallocate_wavelength_on_route(self, route: List[int], wavelength: int) -> None:
        """Libera um wavelength de todos os enlaces da rota."""
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge = (min(u, v), max(u, v))

            if wavelength in self.wavelength_allocation[edge]:
                self.wavelength_allocation[edge].remove(wavelength)

    def _release_expired_wavelengths(self) -> None:
        """Libera wavelengths cuja duração expirou."""
        for edge in self.wavelength_allocation:
            # Encontra wavelengths expirados
            expired_wavelengths = [
                wl for wl, end_time in self.wavelength_allocation[edge].items()
                if end_time <= self.current_time
            ]

            # Remove wavelengths expirados
            for wl in expired_wavelengths:
                del self.wavelength_allocation[edge][wl]

    def route_with_yen_shortest(self, source: int, target: int) -> Optional[List[int]]:
        """
        Roteamento usando YEN: sempre pega o menor caminho (k[0]).

        Args:
            source: Nó origem
            target: Nó destino

        Returns:
            Rota selecionada (menor caminho) ou None se não disponível
        """
        paths = self._get_k_shortest_paths(source, target, self.k)

        if not paths:
            return None

        # YEN clássico: sempre pega o primeiro (menor) caminho
        shortest_path = paths[0]

        # Verifica se existe algum wavelength disponível para este caminho
        for wavelength in range(self.num_wavelengths):
            if self._is_wavelength_available(shortest_path, wavelength):
                return shortest_path

        return None  # Nenhum wavelength disponível neste caminho

    def process_call(self, source: int, target: int, call_id: int,
                     call_duration: float) -> Tuple[bool, Optional[List[int]], Optional[Dict[int, int]]]:
        """
        Processa uma chamada usando YEN + First Fit com conversão.

        Args:
            source: Nó origem
            target: Nó destino
            call_id: ID da chamada
            call_duration: Duração da chamada

        Returns:
            (bloqueado, rota, wavelength_allocation)
        """
        # Libera wavelengths expirados
        self._release_expired_wavelengths()

        # Roteamento com YEN (sempre o menor caminho)
        route = self.route_with_yen_shortest(source, target)

        if route is None:
            return (True, None, None)  # Chamada bloqueada

        # Alocação com First Fit (com conversão de wavelength permitida)
        wavelength_allocation = self.first_fit_allocation_with_conversion(route, call_duration)

        if wavelength_allocation is None:
            return (True, None, None)  # Chamada bloqueada

        # Sucesso
        self.call_records.append({
            'call_id': call_id,
            'source': source,
            'target': target,
            'route': route,
            'wavelength_allocation': wavelength_allocation,
            'blocked': False,
            'hops': len(route) - 1,
            'start_time': self.current_time,
            'end_time': self.current_time + call_duration,
            'duration': call_duration
        })

        return (False, route, wavelength_allocation)

    def simulate_traffic_multiple_runs(self, num_runs: int = 10,
                                       total_calls_per_load: int = 1000,
                                       call_duration_range: Tuple[float, float] = (1.0, 5.0)) -> Dict[str, List[float]]:
        """
        Simula tráfego com múltiplas execuções para cada carga (1 a 30).
        Incluindo duração de chamadas para alocação/desalocação temporal.

        Args:
            num_runs: Número de rodadas por carga (padrão 10)
            total_calls_per_load: Chamadas por rodada por carga
            call_duration_range: Tupla (min_duration, max_duration) para duração das chamadas

        Returns:
            Dicionário com probabilidades de bloqueio médias por requisição
        """
        results = {}

        # Inicializa estrutura de resultados
        for req_idx, (source, target) in enumerate(self.requests):
            results[f'[{source},{target}]'] = [0.0]  # Começa com ponto (0,0)

        # Para cada carga de 1 a 30
        for load in range(1, 31):
            print(f"\n=== CARGA {load}/30 ===")

            # Múltiplas rodadas para esta carga
            load_results = {}
            for req_idx, (source, target) in enumerate(self.requests):
                load_results[f'[{source},{target}]'] = []

            for run in range(num_runs):
                print(f"Rodada {run + 1}/{num_runs} da carga {load}...")
                self.reset_network()
                self.current_time = 0  # Reseta tempo

                for req_idx, (source, target) in enumerate(self.requests):
                    blocked_count = 0

                    for call_idx in range(total_calls_per_load):
                        # Duração aleatória da chamada
                        call_duration = np.random.uniform(call_duration_range[0], call_duration_range[1])

                        # Incrementa tempo simulado com a duração da chamada
                        # Assim wavelengths podem expirar durante a simulação
                        self.current_time += call_duration * 0.5  # Chamadas podem se sobrepor

                        # 80% para requisição específica, 20% aleatório
                        if np.random.random() < 0.8:
                            s, t = source, target
                        else:
                            nodes = list(self.graph.nodes)
                            s, t = np.random.choice(nodes, 2, replace=False)

                        blocked, route, wavelength_alloc = self.process_call(s, t, call_idx, call_duration)

                        if blocked:
                            blocked_count += 1

                    # Probabilidade de bloqueio desta rodada
                    blocking_prob = blocked_count / total_calls_per_load
                    load_results[f'[{source},{target}]'].append(blocking_prob)

            # Média das rodadas para cada requisição
            for req_idx, (source, target) in enumerate(self.requests):
                req_key = f'[{source},{target}]'
                avg_blocking = np.mean(load_results[req_key])
                results[req_key].append(avg_blocking)
                print(f"  {req_key}: {avg_blocking:.6f}")

        return results

    def save_results(self, results: Dict[str, List[float]],
                     output_file: str = "yen_firstfit_results.txt") -> None:
        """
        Salva resultados em arquivo de texto.

        Args:
            results: Resultados da simulação
            output_file: Arquivo de saída
        """
        with open(output_file, "w") as f:
            f.write("=== YEN ROUTING + FIRST FIT WAVELENGTH ALLOCATION ===\n\n")
            f.write(f"Network Parameters:\n")
            f.write(f"  Wavelengths: {self.num_wavelengths}\n")
            f.write(f"  YEN Strategy: Always shortest path (k[0])\n")
            f.write(f"  Wavelength Strategy: First Fit (WITH WAVELENGTH CONVERSION)\n")
            f.write(f"  Temporal Model: Calls have duration and release wavelengths when expire\n")
            f.write(f"  Requests tested: {self.requests}\n\n")

            f.write("Blocking Probability Results:\n")
            f.write("Load\t" + "\t".join([f"Req{i + 1}" for i in range(len(self.requests))]) + "\n")

            # Assume que todos têm o mesmo tamanho (0 a 30)
            num_loads = len(next(iter(results.values())))

            for load_idx in range(num_loads):
                if load_idx == 0:
                    f.write(f"0\t")  # Carga 0
                else:
                    f.write(f"{load_idx}\t")  # Cargas 1-30

                values = []
                for req_key in sorted(results.keys()):
                    if load_idx < len(results[req_key]):
                        values.append(f"{results[req_key][load_idx]:.6f}")
                    else:
                        values.append("N/A")
                f.write("\t".join(values) + "\n")

            # Estatísticas (ignorando ponto 0,0)
            f.write(f"\n=== STATISTICS (Load 1-30) ===\n")
            for req_key in sorted(results.keys()):
                blocking_probs = results[req_key][1:]  # Ignora ponto (0,0)
                if blocking_probs:
                    f.write(f"\n{req_key}:\n")
                    f.write(f"  Average: {np.mean(blocking_probs):.6f}\n")
                    f.write(f"  Max: {np.max(blocking_probs):.6f}\n")
                    f.write(f"  Min: {np.min(blocking_probs):.6f}\n")
                    f.write(f"  Std Dev: {np.std(blocking_probs):.6f}\n")

        print(f"\nResultados salvos em: {output_file}")

    def plot_results(self, results: Dict[str, List[float]],
                     save_path: str = "yen_firstfit_results.png") -> None:
        """
        Gera gráfico com resultados (de carga 0 a 30).

        Args:
            results: Resultados da simulação
            save_path: Caminho para salvar o gráfico
        """
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for idx, (req_key, blocking_probs) in enumerate(sorted(results.items())):
            # Carga vai de 0 a 30 (31 pontos)
            loads = np.arange(0, len(blocking_probs))
            plt.plot(loads, blocking_probs,
                     label=req_key,
                     color=colors[idx % len(colors)],
                     linewidth=2,
                     marker='o',
                     markersize=5)

        plt.xlabel('Traffic Load', fontsize=12)
        plt.ylabel('Blocking Probability', fontsize=12)
        plt.title('YEN (Shortest Path) + First Fit Wavelength Allocation\nBlocking Probability vs Traffic Load',
                  fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 31, 5))
        plt.ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Gráfico salvo em: {save_path}")


def main():
    """Função principal para executar simulação com YEN + First Fit."""

    # Criação do grafo NSFNet
    graph = nx.Graph()
    nsfnet_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
        (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
        (9, 13), (10, 11), (10, 13), (11, 12)
    ]
    graph.add_edges_from(nsfnet_edges)

    # ========== CAMPO CUSTOMIZÁVEL ==========
    # Defina aqui as requisições que quer testar
    custom_requests = [(0, 12), (2, 6), (5, 10), (4, 11), (3, 8)]
    # =======================================

    # Inicializa simulador
    print("=== YEN ROUTING (SHORTEST PATH) + FIRST FIT WAVELENGTH ALLOCATION ===\n")
    simulator = WDMYenFirstFit(
        graph=graph,
        num_wavelengths=4,
        k=5,
        requests=custom_requests
    )

    print(f"Requisições testadas: {custom_requests}")
    print(f"Cargas testadas: 0 a 30")
    print(f"Rodadas por carga: 10")
    print(f"Chamadas por rodada: 1.000\n")

    # Executa simulação com 10 rodadas por carga
    print("Iniciando simulação de tráfego...\n")
    # call_duration_range: (min, max) em unidades de tempo
    # Reduzido para 0.5-2.0 para permitir que wavelengths sejam liberados mais rapidamente
    results = simulator.simulate_traffic_multiple_runs(
        num_runs=10,
        total_calls_per_load=1000,
        call_duration_range=(0.5, 2.0)  # Duração das chamadas entre 0.5 e 2.0
    )

    # Salva resultados
    simulator.save_results(results, "yen_firstfit_results.txt")

    # Gera gráfico
    simulator.plot_results(results, "yen_firstfit_results.png")

    print("\nSimulação concluída com sucesso!")


if __name__ == "__main__":
    main()