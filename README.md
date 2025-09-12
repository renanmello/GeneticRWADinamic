# WDM Network Simulator with Genetic Algorithm

## 📋 Overview

This project implements a WDM (Wavelength Division Multiplexing) network simulator with dynamic traffic that uses a Genetic Algorithm to solve the RWA (Routing and Wavelength Assignment) problem. The simulator finds optimal lightpaths (route + wavelength combinations) to minimize blocking probability in optical networks.

- Genetic Algorithm for RWA optimization
- Dynamic traffic simulation in WDM networks
- K-shortest paths calculation for routing
- Blocking probability analysis
- Interactive result visualizations
- NSFNet topology support (14 nodes, 21 links)

##  🚀 Quick Start
Prerequisites

- Python 3.8 or higher

- pip package manager

## Installation

1.Clone the repository:
```
git clone https://github.com/renanmello/GeneticRWADinamic.git
cd GeneticRWADinamic
```
2.Install dependencies:
```
pip install -r requirements.txt
```
3.Run the simulator:
```
python GeneticRWADinamic.py
```

## Dependencies
```
networkx>=3.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 📚 Academic References

- Ramaswami, R., & Sivarajan, K. N. (2001). Optical Networks: A Practical Perspective
- Zang, H., Jue, J. P., & Mukherjee, B. (2000). "A review of routing and wavelength assignment approaches for wavelength-routed optical WDM networks"
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
