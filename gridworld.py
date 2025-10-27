import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    Environnement GridWorld configurable avec obstacles et goals mobiles.
    
    Paramètres:
        grid_width (int): Largeur de la grille
        grid_height (int): Hauteur de la grille
        goal_states (list): Liste des positions des goals
        obstacles (list): Liste des positions des obstacles
        moving_goals (bool): Si True, les goals se déplacent aléatoirement
        moving_obstacles (bool): Si True, les obstacles se déplacent aléatoirement
        move_probability (float): Probabilité de mouvement (0.0 à 1.0)
        render_mode (str): Mode de rendu
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_width=5, grid_height=5, goal_states=None, obstacles=None, 
                 moving_goals=False, moving_obstacles=False, move_probability=0.3,
                 render_mode=None):
        super(GridWorldEnv, self).__init__()
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_states = self.grid_width * self.grid_height
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # 0: haut, 1: droite, 2: bas, 3: gauche
        
        # Paramètres de mouvement
        self.moving_goals = moving_goals
        self.moving_obstacles = moving_obstacles
        self.move_probability = move_probability
        
        # Start toujours en haut à gauche (case 0)
        self.start_state = 0
        
        # Initialisation des goals
        if goal_states is None:
            self.initial_goal_states = [self.n_states - 1]  # Coin bas-droite par défaut
        else:
            self.initial_goal_states = goal_states if isinstance(goal_states, list) else [goal_states]
        
        # Initialisation des obstacles
        if obstacles is None:
            self.initial_obstacles = []
        else:
            self.initial_obstacles = obstacles if isinstance(obstacles, list) else [obstacles]
        
        # Vérification de la validité de la configuration initiale
        self._validate_initial_configuration()
        
        # États actuels (seront modifiés si mouvement activé)
        self.goal_states = self.initial_goal_states.copy()
        self.obstacles = self.initial_obstacles.copy()
        
        # État de l'agent
        self.state = self.start_state
        
        # Compteur de steps pour statistiques
        self.step_count = 0
    
    def _validate_initial_configuration(self):
        """Vérifie que la configuration initiale est valide"""
        all_positions = set([self.start_state] + self.initial_goal_states + self.initial_obstacles)
        
        # Vérifier qu'il n'y a pas de chevauchements
        total_elements = 1 + len(self.initial_goal_states) + len(self.initial_obstacles)
        if len(all_positions) != total_elements:
            raise ValueError("Chevauchement détecté entre start, goals et obstacles!")
        
        # Vérifier que toutes les positions sont valides
        for pos in all_positions:
            if pos < 0 or pos >= self.n_states:
                raise ValueError(f"Position {pos} hors limites (0-{self.n_states-1})")
        
        # Vérifier qu'il reste au moins une case libre
        if len(all_positions) >= self.n_states:
            raise ValueError("Pas assez de cases libres dans la grille!")
    
    def _get_valid_neighbors(self, state):
        """Retourne les voisins valides d'une case (pour le mouvement)"""
        row, col = divmod(state, self.grid_width)
        neighbors = []
        
        # Haut
        if row > 0:
            neighbors.append((row - 1) * self.grid_width + col)
        # Droite
        if col < self.grid_width - 1:
            neighbors.append(row * self.grid_width + (col + 1))
        # Bas
        if row < self.grid_height - 1:
            neighbors.append((row + 1) * self.grid_width + col)
        # Gauche
        if col > 0:
            neighbors.append(row * self.grid_width + (col - 1))
        
        return neighbors
    
    def _move_element(self, current_pos, forbidden_positions):
        """
        Déplace un élément (goal ou obstacle) vers une position voisine valide.
        
        Args:
            current_pos: Position actuelle
            forbidden_positions: Set de positions interdites (agent, autres goals/obstacles)
        
        Returns:
            Nouvelle position (ou position actuelle si aucun mouvement possible)
        """
        if np.random.random() > self.move_probability:
            return current_pos  # Pas de mouvement
        
        neighbors = self._get_valid_neighbors(current_pos)
        # Filtrer les positions interdites
        valid_moves = [n for n in neighbors if n not in forbidden_positions]
        
        if valid_moves:
            return np.random.choice(valid_moves)
        return current_pos  # Aucun mouvement valide
    
    def _update_moving_elements(self):
        """Met à jour les positions des goals et obstacles mobiles"""
        new_goals = []
        new_obstacles = []
        
        # Ensemble des positions interdites (pour éviter les collisions)
        forbidden = {self.state}  # Position de l'agent
        
        # Déplacer les goals
        if self.moving_goals:
            for goal in self.goal_states:
                # Ajouter les obstacles actuels aux positions interdites
                temp_forbidden = forbidden.union(set(self.obstacles))
                new_pos = self._move_element(goal, temp_forbidden)
                new_goals.append(new_pos)
                forbidden.add(new_pos)
            self.goal_states = new_goals
        
        # Déplacer les obstacles
        if self.moving_obstacles:
            for obs in self.obstacles:
                # Ajouter les goals actuels aux positions interdites
                temp_forbidden = forbidden.union(set(self.goal_states))
                new_pos = self._move_element(obs, temp_forbidden)
                new_obstacles.append(new_pos)
                forbidden.add(new_pos)
            self.obstacles = new_obstacles
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement"""
        super().reset(seed=seed)
        self.state = self.start_state
        self.step_count = 0
        
        # Réinitialiser les positions des goals et obstacles
        self.goal_states = self.initial_goal_states.copy()
        self.obstacles = self.initial_obstacles.copy()
        
        return self.state, {}
    
    def step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: 0=haut, 1=droite, 2=bas, 3=gauche
        
        Returns:
            next_state, reward, terminated, truncated, info
        """
        self.step_count += 1
        row, col = divmod(self.state, self.grid_width)
        new_row, new_col = row, col
        
        # Calculer la nouvelle position selon l'action
        if action == 0 and row > 0:  # haut
            new_row = row - 1
        elif action == 1 and col < self.grid_width - 1:  # droite
            new_col = col + 1
        elif action == 2 and row < self.grid_height - 1:  # bas
            new_row = row + 1
        elif action == 3 and col > 0:  # gauche
            new_col = col - 1
        
        new_state = new_row * self.grid_width + new_col
        
        # Gestion des collisions avec obstacles
        if new_state in self.obstacles:
            # Collision avec obstacle: pénalité et reste sur place
            reward = -2.0
            terminated = False
            # L'agent reste à sa position actuelle
        else:
            # Mouvement valide
            self.state = new_state
            
            # Vérifier si l'agent atteint un goal
            if self.state in self.goal_states:
                reward = 10.0
                terminated = True
            else:
                # Pénalité pour encourager l'efficacité
                reward = -0.01
                terminated = False
        
        # Déplacer les éléments mobiles (après le mouvement de l'agent)
        if not terminated:
            self._update_moving_elements()
            
            # Re-vérifier si l'agent est maintenant sur un goal (goal mobile arrivé sur l'agent)
            if self.state in self.goal_states:
                reward = 10.0
                terminated = True
            # Re-vérifier si un obstacle a bougé sur l'agent
            elif self.state in self.obstacles:
                reward = -2.0
                terminated = True  # Game over si écrasé par obstacle mobile
        
        truncated = False
        info = {
            'step_count': self.step_count,
            'goal_states': self.goal_states.copy(),
            'obstacles': self.obstacles.copy()
        }
        
        return self.state, reward, terminated, truncated, info
    
    def get_next_state(self, state, action):
        """
        Retourne le prochain état sans modifier l'environnement.
        Utile pour la planification (VI, PI).
        
        Note: Ne prend pas en compte le mouvement des éléments mobiles.
        """
        row, col = divmod(state, self.grid_width)
        new_row, new_col = row, col
        
        if action == 0 and row > 0:
            new_row = row - 1
        elif action == 1 and col < self.grid_width - 1:
            new_col = col + 1
        elif action == 2 and row < self.grid_height - 1:
            new_row = row + 1
        elif action == 3 and col > 0:
            new_col = col - 1
        
        next_state = new_row * self.grid_width + new_col
        
        # Si la nouvelle position est un obstacle, reste sur place
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def render(self):
        """Affiche la grille en mode console"""
        grid = np.array([["."] * self.grid_width for _ in range(self.grid_height)])
        
        # Placer les obstacles
        for obs in self.obstacles:
            r, c = divmod(obs, self.grid_width)
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                grid[r, c] = "X"
        
        # Placer les goals
        for goal in self.goal_states:
            gr, gc = divmod(goal, self.grid_width)
            if 0 <= gr < self.grid_height and 0 <= gc < self.grid_width:
                grid[gr, gc] = "G"
        
        # Placer l'agent (prioritaire sur tout)
        row, col = divmod(self.state, self.grid_width)
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            grid[row, col] = "A"
        
        # Affichage
        print("\n" + "="*50)
        print(f"Step: {self.step_count} | Agent: {self.state} | Goals: {self.goal_states}")
        if self.moving_goals or self.moving_obstacles:
            status = []
            if self.moving_goals:
                status.append("Goals mobiles")
            if self.moving_obstacles:
                status.append("Obstacles mobiles")
            print(f"Mode: {' + '.join(status)}")
        print("="*50)
        print("\n".join(" ".join(row) for row in grid))
        print()