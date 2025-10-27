import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class DQNetwork(nn.Module):
    """Réseau de neurones pour approximer la Q-function"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    """
    Experience Replay Memory pour stocker les transitions (s, a, r, s', done).
    Permet de rompre la corrélation temporelle entre les expériences.
    """
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition à la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch aléatoire de transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNDeepMindAgent:
    """
    Agent DQN avec les techniques de DeepMind:
    1. Experience Replay: stocke et réutilise les expériences passées
    2. Target Network: réseau cible fixe pour stabiliser l'apprentissage
    3. Batch Learning: apprend sur des mini-batches aléatoires
    """
    def __init__(self, env, alpha=0.0005, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, hidden_size=128,
                 memory_capacity=10000, batch_size=64, target_update_freq=10):
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        
        # Hyperparamètres
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseaux de neurones: Policy Network (en ligne) et Target Network (cible)
        self.policy_net = DQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        
        # Initialiser le target network avec les poids du policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Mode évaluation uniquement
        
        # Optimizer et loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Experience Replay Memory
        self.memory = ReplayMemory(capacity=memory_capacity)
        
        # Compteur pour mise à jour du target network
        self.update_counter = 0
        
        print(f"DQN DeepMind Agent initialisé sur {self.device}")
        print(f"Architecture: {self.state_size} -> {hidden_size} -> {hidden_size} -> {self.action_size}")
        print(f"Memory capacity: {memory_capacity}, Batch size: {batch_size}")
        print(f"Target network update frequency: {target_update_freq} episodes")
    
    def state_to_tensor(self, state):
        """Convertit un état (entier) en vecteur one-hot tensor"""
        state_vector = np.zeros(self.state_size)
        state_vector[state] = 1.0
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def select_action(self, state):
        """Sélection d'action avec epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans la replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=False):
        """
        Entraîne le réseau sur un batch de la replay memory.
        Cette fonction peut être appelée:
        - Avec des paramètres (compatibilité): stocke la transition puis apprend
        - Sans paramètres: apprend uniquement si assez de mémoire
        """
        # Si des paramètres sont fournis, stocker la transition
        if state is not None:
            self.store_transition(state, action, reward, next_state, done)
        
        # Vérifier si assez d'expériences pour un batch
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Échantillonner un batch
        transitions = self.memory.sample(self.batch_size)
        
        # Préparer les tensors
        states = torch.cat([self.state_to_tensor(t[0]) for t in transitions])
        actions = torch.LongTensor([t[1] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        next_states = torch.cat([self.state_to_tensor(t[3]) for t in transitions])
        dones = torch.FloatTensor([t[4] for t in transitions]).to(self.device)
        
        # Calcul des Q-values actuelles: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calcul des Q-values cibles avec le target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcul de la loss et backpropagation
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copie les poids du policy network vers le target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Décroissance de epsilon après chaque épisode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        
        # Mise à jour périodique du target network
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
            print(f"Target network updated (episode {self.update_counter})")
    
    def get_q_values(self, state):
        """Retourne les Q-values pour un état donné (via policy network)"""
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            return self.policy_net(state_tensor).cpu().numpy()[0]
    
    def save(self, filepath):
        """Sauvegarde les deux réseaux et l'état de l'optimiseur"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load(self, filepath):
        """Charge les deux réseaux et l'état de l'optimiseur"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']
        print(f"Modèle chargé: {filepath}")