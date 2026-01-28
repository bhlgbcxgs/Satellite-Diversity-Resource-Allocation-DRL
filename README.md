# Satellite Power Allocation and MCS decision with Deep Reinforcement Learning

This repository contains the implementation of a power allocation algorithm for satellite communications based on Deep Reinforcement Learning (DRL). The project simulates a Low Earth Orbit (LEO) satellite network environment, where an intelligent agent automatically optimizes channel power allocation to maximize spectral efficiency and mitigate jamming attacks.

## File Structure

- **`main.py`**: The main entry point of the program. It handles configuration loading, environment and agent initialization, the training loop, and data logging.
- **`agent.py`**: The implementation of the DRL agent. It includes the neural network architectures for the Actor, DQN, and Multiplier.
- **`environment.py`**: The satellite communication simulation environment. It calculates throughput of the system handles the dynamics of satellite and jammer positions.
- **`config.py`**: This is the template for the configuration file, containing placeholders for environmental physical constants and training hyperparameters.
- **`satellite_param.py`**: A utility library for calculating satellite orbital positions and antenna gains.
- **`rician_data_collect.py`**: Utilities for calculating BER and rewards under Rician fading channels.

## Requirements

This project is developed using Python 3. Please ensure the following dependencies are installed:

```bash
pip install numpy torch matplotlib
```

## Configuration

All hyperparameters are managed within **`config.py`**, allowing you to adjust the experiment without modifying the source code:

- **Environment**: Controls physical parameters.
- **Training**: Controls RL training hyperparameters.
- **Network**: Controls the weight initialization distributions for the neural networks.


## Results

- **Data**: Training logs will be saved in the **`./results/data/`** directory.
- **Models**: Models will be saved in the **`./results/model/`** directory.
