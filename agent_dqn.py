import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


class DQNAgent:
    """
    Agent DQN simple sans Experience Replay ni Target Network.
    Apprentissage direct en ligne avec mise à jour après chaque step.
    """
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, hidden_size=128):
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        
        # Hyperparamètres
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        
        # Réseau de neurones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        print(f"DQN Agent initialisé sur {self.device}")
        print(f"Architecture: {self.state_size} -> {hidden_size} -> {hidden_size} -> {self.action_size}")
    
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
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done=False):
        """
        Mise à jour du réseau après chaque step (apprentissage en ligne).
        
        Target: r + γ * max_a' Q(s', a') si non terminal
                r si terminal
        """
        # État actuel
        state_tensor = self.state_to_tensor(state)
        
        # Prédiction Q(s, a)
        q_values = self.model(state_tensor)
        q_value = q_values[0, action]
        
        # Calcul de la target
        with torch.no_grad():
            if done:
                target = reward
            else:
                next_state_tensor = self.state_to_tensor(next_state)
                next_q_values = self.model(next_state_tensor)
                max_next_q = next_q_values.max().item()
                target = reward + self.gamma * max_next_q
        
        target_tensor = torch.FloatTensor([target]).to(self.device)
        
        # Backpropagation
        loss = self.criterion(q_value.unsqueeze(0), target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Décroissance de epsilon après chaque épisode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state):
        """Retourne les Q-values pour un état donné"""
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            return self.model(state_tensor).cpu().numpy()[0]
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load(self, filepath):
        """Charge le modèle"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Modèle chargé: {filepath}")