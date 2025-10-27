from gridworld import GridWorldEnv
from agent_dqn import DQNAgent
from agent_dqn_deepmind import DQNDeepMindAgent
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import sys
import io
import torch
import time # Ajout pour le timing

# --- AJOUT COMMENCE ---
# Définir le device (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- AJOUT TERMINE ---

# Configuration de l'encodage pour Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not set stdout encoding to utf-8. {e}")

# ===========================
# CONFIGURATION GLOBALE
# ===========================
RANDOM_SEED = 42
# MAX_EPISODES = 400 # Version originale
MAX_EPISODES = 100 # MODIFIÉ: Réduit pour un test rapide
NUM_SCENARIOS = 3 # MODIFIÉ: Réduit de 10 à 3 pour un test rapide

# Hyperparamètres DQN
ALPHA = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995 # Decay plus rapide pour moins d'épisodes
EPSILON_MIN = 0.01
HIDDEN_SIZE = 128

# Hyperparamètres DeepMind
MEMORY_CAPACITY = 10000 # Augmenté pour être plus réaliste
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10

# Fixer les seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ===========================
# GÉNÉRATION DES SCÉNARIOS
# ===========================

def generate_scenarios(num_scenarios=3): # Val par défaut changée à 3
    """
    Génère N scénarios différents.
    MODIFIÉ: S'assure que le goal est à au moins 3 cases de distance (Manhattan) du start.
    """
    print(f"Génération de {num_scenarios} scénarios...")
    scenarios = []
    
    # Tailles de grille alternées
    grid_sizes = [(5, 5), (6, 6), (5, 5), (6, 6), (5, 5), 
                  (6, 6), (5, 5), (6, 6), (5, 5), (6, 6)]
    
    for i in range(num_scenarios):
        width, height = grid_sizes[i]
        total_cells = width * height
        
        num_obstacles = max(2, int(total_cells * 0.1))
        num_goals = 1 # Gardons 1 seul but pour simplifier
        
        # Positions available (excluant start 0)
        all_positions = list(range(1, total_cells))

        # 1. Filtrer pour les goals (loin du start)
        min_distance = 3 # MODIFIÉ: Manhattan distance >= 3 (row + col)
        potential_goals = []
        for pos in all_positions:
            r, c = divmod(pos, width)
            if (r + c) >= min_distance: # Manhattan distance from (0,0)
                potential_goals.append(pos)
        
        if not potential_goals:
            print(f"Warning: Scénario {i+1} - N'a pas pu trouver de goal à {min_distance} cases. Utilisation de la méthode standard.")
            potential_goals = all_positions # Fallback

        np.random.shuffle(potential_goals)
        
        # S'assurer qu'on a un goal
        if not potential_goals:
            # Fallback si vraiment aucun goal n'est possible (grille 2x2 p.ex.)
            goals = [all_positions[-1]]
        else:
            goals = [potential_goals.pop(0)] # Prend le premier goal
        
        # 2. Placer les obstacles
        # Les obstacles peuvent être n'importe où (sauf start et goal)
        available_for_obstacles = list(set(all_positions) - set(goals))
        np.random.shuffle(available_for_obstacles)
        
        obstacles = available_for_obstacles[:num_obstacles]
        
        move_prob = np.random.uniform(0.2, 0.5)
        
        scenario = {
            'id': i + 1,
            'width': width,
            'height': height,
            'goals': goals,
            'obstacles': obstacles,
            'moving_goals': True, # MODIFIÉ: Mettre True pour tester le cas dynamique
            'moving_obstacles': False,
            'move_probability': move_prob
        }
        
        scenarios.append(scenario)
    
    return scenarios

# ===========================
# FONCTIONS DE VISUALISATION
# ===========================

def render_scenario_grid(scenario, ax, title="Scénario"):
    """Affiche la configuration initiale d'un scénario"""
    width = scenario['width']
    height = scenario['height']
    grid = np.zeros((height, width, 3))
    
    background_color = np.array([173, 216, 230])/255
    start_color = np.array([0, 255, 0])/255
    goal_color = np.array([255, 0, 0])/255
    obstacle_color = np.array([0, 0, 0])/255
    
    grid[:, :] = background_color
    grid[0, 0] = start_color
    
    for obs in scenario['obstacles']:
        r, c = divmod(obs, width)
        if 0 <= r < height and 0 <= c < width:
            grid[r, c] = obstacle_color
    
    for goal in scenario['goals']:
        gr, gc = divmod(goal, width)
        if 0 <= gr < height and 0 <= gc < width:
            grid[gr, gc] = goal_color
    
    ax.imshow(grid)
    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linewidth=1)
    ax.set_title(f"{title}\n{width}x{height} - {len(scenario['goals'])} goals - {len(scenario['obstacles'])} obs\nMove prob: {scenario['move_probability']:.2f}", 
                fontsize=9, fontweight='bold')

# --- NOUVELLE FONCTION ---
def plot_path_on_grid(ax, scenario, path, title):
    """Dessine une grille statique avec le chemin de l'agent."""
    width = scenario['width']
    height = scenario['height']
    
    # 1. Dessiner la grille de base
    grid = np.zeros((height, width, 3))
    background_color = np.array([255, 255, 255])/255 # Blanc pour le chemin
    start_color = np.array([0, 255, 0])/255
    goal_color = np.array([255, 0, 0])/255
    obstacle_color = np.array([0, 0, 0])/255
    path_color = np.array([173, 216, 230])/255 # Bleu clair
    
    grid[:, :] = background_color
    
    for obs in scenario['obstacles']:
        r, c = divmod(obs, width)
        if 0 <= r < height and 0 <= c < width: grid[r, c] = obstacle_color
    
    for goal in scenario['goals']:
        gr, gc = divmod(goal, width)
        if 0 <= gr < height and 0 <= gc < width: grid[gr, gc] = goal_color

    # 2. Dessiner le chemin
    if path:
        for pos in path[1:-1]: # Le chemin (sans start/end)
            r, c = divmod(pos, width)
            if 0 <= r < height and 0 <= c < width: grid[r, c] = path_color
    
    # 3. Dessiner Start et End (par-dessus le chemin)
    grid[0, 0] = start_color # Start
    if path and path[-1] in scenario['goals']:
        gr, gc = divmod(path[-1], width)
        if 0 <= gr < height and 0 <= gc < width: grid[gr, gc] = goal_color # Goal atteint
    elif path: # Si l'agent a échoué
        fr, fc = divmod(path[-1], width)
        fail_color = np.array([255, 165, 0])/255 # Orange
        if 0 <= fr < height and 0 <= fc < width: grid[fr, fc] = fail_color

    
    ax.imshow(grid)
    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linewidth=1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # 4. Dessiner les flèches du chemin
    if path:
        for i in range(len(path) - 1):
            r1, c1 = divmod(path[i], width)
            r2, c2 = divmod(path[i+1], width)
            
            # Coordonnées du centre des cases
            c1_center = c1
            r1_center = r1
            c2_center = c2
            r2_center = r2
            
            # Calculer delta
            dc = (c2_center - c1_center) * 0.4 # Flèche courte
            dr = (r2_center - r1_center) * 0.4
            
            if dc != 0 or dr != 0: # Éviter flèche sur place
                ax.arrow(c1_center, r1_center, dc, dr, 
                         head_width=0.2, head_length=0.15, 
                         fc='black', ec='black', length_includes_head=True)

def plot_scenario_comparison(results, scenario):
    """
    Trace la comparaison pour un scénario donné
    MODIFIÉ: Affiche le trajet final au lieu des barres de stats.
    """
    print(f"Génération du graphique pour le scénario {scenario['id']}...")
    try:
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle(f"SCÉNARIO {scenario['id']} - {scenario['width']}x{scenario['height']} "
                     f"({len(scenario['goals'])} goals, {len(scenario['obstacles'])} obstacles)", 
                     fontsize=14, fontweight='bold')
        
        window = 20
        colors = {'simple': 'blue', 'deepmind': 'red'}
        
        # 1: Configuration
        ax_config = fig.add_subplot(gs[0, 0])
        render_scenario_grid(scenario, ax_config, "Configuration")
        
        # Data
        rewards_simple = results['simple']['rewards']
        rewards_deepmind = results['deepmind']['rewards']
        steps_simple = results['simple']['steps']
        steps_deepmind = results['deepmind']['steps']
        losses_simple = results['simple']['losses']
        losses_deepmind = results['deepmind']['losses']

        # 2: Rewards DQN Simple
        ax1 = fig.add_subplot(gs[0, 1])
        if len(rewards_simple) >= window:
            rewards_smooth = np.convolve(rewards_simple, np.ones(window)/window, mode='valid')
            ax1.plot(rewards_smooth, color=colors['simple'], linewidth=2, label='DQN Simple')
        else:
            ax1.plot(rewards_simple, color=colors['simple'], linewidth=2, label='DQN Simple')
        ax1.set_title('DQN Simple - Rewards', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 3: Rewards DQN DeepMind
        ax2 = fig.add_subplot(gs[0, 2])
        if len(rewards_deepmind) >= window:
            rewards_smooth = np.convolve(rewards_deepmind, np.ones(window)/window, mode='valid')
            ax2.plot(rewards_smooth, color=colors['deepmind'], linewidth=2, label='DQN DeepMind')
        else:
            ax2.plot(rewards_deepmind, color=colors['deepmind'], linewidth=2, label='DQN DeepMind')
        ax2.set_title('DQN DeepMind - Rewards', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 4: Steps comparés
        ax3 = fig.add_subplot(gs[1, 0])
        if len(steps_simple) >= window:
            steps_smooth_s = np.convolve(steps_simple, np.ones(window)/window, mode='valid')
            ax3.plot(steps_smooth_s, color=colors['simple'], label='DQN Simple')
        if len(steps_deepmind) >= window:
            steps_smooth_d = np.convolve(steps_deepmind, np.ones(window)/window, mode='valid')
            ax3.plot(steps_smooth_d, color=colors['deepmind'], label='DQN DeepMind')
        ax3.set_title('Comparaison Steps', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 5: Loss comparées
        ax4 = fig.add_subplot(gs[1, 1])
        if len(losses_simple) >= window:
            losses_smooth_s = np.convolve(losses_simple, np.ones(window)/window, mode='valid')
            ax4.plot(losses_smooth_s, color=colors['simple'], label='DQN Simple')
        if len(losses_deepmind) >= window:
            losses_smooth_d = np.convolve(losses_deepmind, np.ones(window)/window, mode='valid')
            ax4.plot(losses_smooth_d, color=colors['deepmind'], label='DQN DeepMind')
        ax4.set_title('Comparaison Loss', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 6: Epsilon
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(results['simple']['epsilons'], color=colors['simple'], label='DQN Simple')
        ax5.plot(results['deepmind']['epsilons'], color=colors['deepmind'], label='DQN DeepMind')
        ax5.set_title('Décroissance Epsilon', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # --- MODIFICATION COMMENCE ---
        
        # Subplot 7: Trajet DQN Simple
        ax6 = fig.add_subplot(gs[2, 0])
        path_simple = results['simple'].get('final_path', [])
        plot_path_on_grid(ax6, scenario, path_simple, "Trajet Final (DQN Simple)")
        
        # Subplot 8: Trajet DQN DeepMind
        ax7 = fig.add_subplot(gs[2, 1])
        path_deepmind = results['deepmind'].get('final_path', [])
        plot_path_on_grid(ax7, scenario, path_deepmind, "Trajet Final (DQN DeepMind)")

        # Subplot 9: Statistiques textuelles (reste pareil)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Recalculer les moyennes pour le texte
        mean_rewards_simple = np.mean(rewards_simple[-50:])
        mean_rewards_deepmind = np.mean(rewards_deepmind[-50:])
        mean_steps_simple = np.mean(steps_simple[-50:])
        mean_steps_deepmind = np.mean(steps_deepmind[-50:])
        
        stats_text = f"""
    STATS FINALES (50 derniers épisodes)

    DQN Simple:
      Reward: {mean_rewards_simple:.2f}
      Steps: {mean_steps_simple:.2f}
      Succès: {results['simple']['success_rate']:.1f}%
      
    DQN DeepMind:
      Reward: {mean_rewards_deepmind:.2f}
      Steps: {mean_steps_deepmind:.2f}
      Succès: {results['deepmind']['success_rate']:.1f}%
      Memory: {results['deepmind']['memory_size']}
      
    GAGNANT: {results.get('winner', 'N/A')}
        """
        ax8.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(f'scenario_{scenario["id"]}_comparison.png', dpi=100, bbox_inches='tight')
        plt.close(fig) # Fermer la figure pour libérer la mémoire
        print(f"✓ Graphique 'scenario_{scenario['id']}_comparison.png' sauvegardé.")
    except Exception as e:
        print(f"ERREUR lors de la génération du graphique pour scénario {scenario['id']}: {e}")


# ===========================
# ENTRAÎNEMENT
# ===========================

# --- NOUVELLE FONCTION ---
def run_test_episode(agent, env):
    """Exécute un épisode en mode test (exploitation) et retourne le chemin."""
    path = []
    agent.epsilon = 0 # Mode exploitation pure
    
    state, _ = env.reset()
    path.append(state)
    done = False
    steps = 0
    # Limite de steps (pour éviter boucle infinie si la politique est mauvaise)
    max_steps = env.grid_width * env.grid_height * 3 # Augmenté un peu
    
    while not done and steps < max_steps:
        action = agent.select_action(state) # action sera la meilleure
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        path.append(state)
        steps += 1
    
    success = state in env.goal_states
    return path, success

def train_agent_on_scenario(agent, env, max_episodes, agent_name="Agent"):
    """Entraîne un agent sur un scénario"""
    rewards_per_episode = []
    steps_per_episode = []
    losses_per_episode = []
    epsilon_per_episode = []
    successes = 0
    
    start_time_agent = time.time()
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0
        loss_count = 0
        
        # Limite de steps par épisode
        # MODIFIÉ: Utilisation de 'grid_width' et 'grid_height'
        max_steps_per_episode = env.grid_width * env.grid_height * 2 
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            loss = agent.update(state, action, reward, next_state, terminated)
            
            if loss > 0:
                total_loss += loss
                loss_count += 1
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if terminated or truncated:
                done = True
                if state in env.goal_states:
                    successes += 1
        
        agent.decay_epsilon()
        
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        losses_per_episode.append(avg_loss)
        epsilon_per_episode.append(agent.epsilon)

        # --- AJOUT: Affichage de la progression ---
        if (episode + 1) % (max_episodes // 5) == 0 or episode == max_episodes - 1:
            # Affichage tous les 20%
            if max_episodes >= 5:
                print(f"    [{agent_name} - Ep. {episode + 1}/{max_episodes}] "
                      f"Reward: {total_reward:.2f}, Steps: {steps}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
            elif episode == max_episodes - 1:
                # Gérer le cas de très peu d'épisodes (affiche au moins la fin)
                print(f"    [{agent_name} - Ep. {episode + 1}/{max_episodes}] "
                      f"Reward: {total_reward:.2f}, Steps: {steps}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")

    end_time_agent = time.time()
    print(f"    -> Temps d'entraînement ({agent_name}): {end_time_agent - start_time_agent:.2f}s")
    
    success_rate = (successes / max_episodes) * 100 if max_episodes > 0 else 0
    
    return {
        'rewards': rewards_per_episode,
        'steps': steps_per_episode,
        'losses': losses_per_episode,
        'epsilons': epsilon_per_episode,
        'success_rate': success_rate
    }

def run_scenario(scenario, max_episodes):
    """
    Exécute un scénario complet avec les deux agents
    MODIFIÉ: Exécute un test final pour obtenir le chemin.
    """
    print(f"\n{'='*70}")
    print(f"SCÉNARIO {scenario['id']}/{NUM_SCENARIOS}: {scenario['width']}x{scenario['height']}")
    print(f"Goals: {len(scenario['goals'])} (Mobiles: {scenario['moving_goals']})")
    print(f"Obstacles: {len(scenario['obstacles'])} (Fixes)")
    print(f"Move probability: {scenario['move_probability']:.2f}")
    print(f"{'='*70}")
    
    # Créer l'environnement
    env = GridWorldEnv(
        grid_width=scenario['width'],
        grid_height=scenario['height'],
        goal_states=scenario['goals'],
        obstacles=scenario['obstacles'],
        moving_goals=scenario['moving_goals'],
        moving_obstacles=scenario['moving_obstacles'],
        move_probability=scenario['move_probability']
    )
    
    # --- AJOUT DE COMPATIBILITÉ (IMPORTANT) ---
    # Les agents s'attendent à 'state_space_size' et 'action_space_size'.
    # L'environnement Gym fournit 'observation_space.n' et 'action_space.n'.
    # Nous "patchons" l'objet 'env' pour le rendre compatible.
    try:
        env.state_space_size = env.observation_space.n
        env.action_space_size = env.action_space.n
    except AttributeError:
        print("ERREUR: L'environnement 'env' n'a pas les attributs 'observation_space.n' ou 'action_space.n'.")
        print("Veuillez vérifier que 'gridworld.py' est bien basé sur Gymnasium.")
        return {} # Quitter ce scénario
    # --- FIN DE L'AJOUT ---
    
    # DQN Simple
    print("\n[1/2] Entraînement DQN Simple...")
    agent_simple = DQNAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                           epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                           hidden_size=HIDDEN_SIZE)
    
    results_simple = train_agent_on_scenario(agent_simple, env, max_episodes, "DQN Simple")
    
    print(f"    -> Reward moyen final (50 derniers): {np.mean(results_simple['rewards'][-50:]):.2f}")
    print(f"    -> Taux de succès: {results_simple['success_rate']:.1f}%")
    
    # DQN DeepMind
    print("\n[2/2] Entraînement DQN DeepMind...")
    # Reset l'env pour le 2e agent (même si train_agent le fait, c'est plus propre)
    env.reset()
    agent_deepmind = DQNDeepMindAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                                     epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                                     hidden_size=HIDDEN_SIZE, memory_capacity=MEMORY_CAPACITY,
                                     batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ)
    
    results_deepmind = train_agent_on_scenario(agent_deepmind, env, max_episodes, "DQN DeepMind")
    results_deepmind['memory_size'] = len(agent_deepmind.memory)
    
    print(f"    -> Reward moyen final (50 derniers): {np.mean(results_deepmind['rewards'][-50:]):.2f}")
    print(f"    -> Taux de succès: {results_deepmind['success_rate']:.1f}%")
    print(f"    -> Taille mémoire: {results_deepmind['memory_size']}")
    
    # --- AJOUT: Exécuter un test final pour voir le chemin ---
    print("\n[3/3] Exécution des tests finaux (exploitation)...")
    env.reset()
    path_simple, success_simple_test = run_test_episode(agent_simple, env)
    print(f"    -> Chemin Simple (longueur {len(path_simple)}): {'Succès' if success_simple_test else 'Echec'}")
    
    env.reset()
    path_deepmind, success_deepmind_test = run_test_episode(agent_deepmind, env)
    print(f"    -> Chemin DeepMind (longueur {len(path_deepmind)}): {'Succès' if success_deepmind_test else 'Echec'}")
    
    # Ajouter les chemins aux résultats
    results_simple['final_path'] = path_simple
    results_deepmind['final_path'] = path_deepmind
    # --- FIN AJOUT ---

    # Déterminer le gagnant
    mean_reward_simple = np.mean(results_simple['rewards'][-50:])
    mean_reward_deepmind = np.mean(results_deepmind['rewards'][-50:])
    
    if mean_reward_deepmind > mean_reward_simple:
        winner = 'DQN DeepMind'
    elif mean_reward_simple > mean_reward_deepmind:
        winner = 'DQN Simple'
    else:
        winner = 'Égalité'
        
    print(f"\n✓ GAGNANT (Scénario {scenario['id']}): {winner}")
    
    return {
        'simple': results_simple,
        'deepmind': results_deepmind,
        'winner': winner
    }

# ===========================
# ANALYSE GLOBALE
# ===========================

def plot_global_summary(all_results, scenarios):
    """Trace un résumé global de tous les scénarios"""
    print("Génération du graphique de résumé global...")
    try:
        num_scenarios = len(scenarios)
        if num_scenarios == 0:
            print("Aucun résultat à afficher.")
            return

        fig = plt.figure(figsize=(16, 10)) # Taille ajustée pour 3 scénarios
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'RÉSUMÉ GLOBAL - COMPARAISON ({num_scenarios} SCÉNARIOS)', 
                     fontsize=16, fontweight='bold')
        
        scenario_ids = [s['id'] for s in scenarios]
        
        # Données
        rewards_simple = [np.mean(all_results[i]['simple']['rewards'][-50:]) for i in range(num_scenarios)]
        rewards_deepmind = [np.mean(all_results[i]['deepmind']['rewards'][-50:]) for i in range(num_scenarios)]
        success_simple = [all_results[i]['simple']['success_rate'] for i in range(num_scenarios)]
        success_deepmind = [all_results[i]['deepmind']['success_rate'] for i in range(num_scenarios)]
        winners = [all_results[i]['winner'] for i in range(num_scenarios)]
        
        x = np.arange(num_scenarios)
        width = 0.35
        
        # 1. Rewards moyens
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(x - width/2, rewards_simple, width, label='DQN Simple', color='blue', alpha=0.7)
        ax1.bar(x + width/2, rewards_deepmind, width, label='DQN DeepMind', color='red', alpha=0.7)
        ax1.set_ylabel('Reward moyen final')
        ax1.set_title('Rewards finaux par scénario', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Taux de succès
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x - width/2, success_simple, width, label='DQN Simple', color='blue', alpha=0.7)
        ax2.bar(x + width/2, success_deepmind, width, label='DQN DeepMind', color='red', alpha=0.7)
        ax2.set_ylabel('Taux de succès (%)')
        ax2.set_title('Taux de succès par scénario', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenario_ids)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Nombre de victoires
        ax3 = fig.add_subplot(gs[0, 2])
        win_counts = {'DQN Simple': winners.count('DQN Simple'), 
                      'DQN DeepMind': winners.count('DQN DeepMind'),
                      'Égalité': winners.count('Égalité')}
        ax3.bar(win_counts.keys(), win_counts.values(), color=['blue', 'red', 'grey'], alpha=0.7)
        ax3.set_ylabel('Nombre de victoires')
        ax3.set_title('Nombre de scénarios gagnés', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(win_counts.values()):
            ax3.text(i, v + 0.05, str(v), ha='center', fontweight='bold', fontsize=12)
        
        # 4. Statistiques globales
        ax4 = fig.add_subplot(gs[1, :]) # Prend toute la largeur du bas
        ax4.axis('off')
        
        stats_text = f"""
    STATISTIQUES GLOBALES ({num_scenarios} scénarios)

    DQN Simple:
      Reward moyen: {np.mean(rewards_simple):.2f}
      Succès moyen: {np.mean(success_simple):.1f}%
      Victoires: {win_counts['DQN Simple']}

    DQN DeepMind:
      Reward moyen: {np.mean(rewards_deepmind):.2f}
      Succès moyen: {np.mean(success_deepmind):.1f}%
      Victoires: {win_counts['DQN DeepMind']}
      
    Égalités: {win_counts['Égalité']}
      
    CHAMPION GLOBAL: {max(win_counts, key=win_counts.get)}
        """
        
        ax4.text(0.5, 0.5, stats_text, fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le titre principal
        plt.savefig('global_summary_all_scenarios.png', dpi=100, bbox_inches='tight')
        print("✓ Graphique 'global_summary_all_scenarios.png' sauvegardé.")
        # plt.show() # Optionnel: afficher le graphique
        plt.close(fig)

    except Exception as e:
        print(f"ERREUR lors de la génération du résumé global: {e}")

# ===========================
# MAIN
# ===========================

def main():
    """Fonction principale - Exécute N scénarios"""
    start_time_total = time.time()
    
    print("\n" + "="*70)
    print("COMPARAISON AUTOMATIQUE: DQN SIMPLE vs DQN DEEPMIND")
    print(f"{NUM_SCENARIOS} SCÉNARIOS ALÉATOIRES - {MAX_EPISODES} EPISODES/SCÉNARIO")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Device: {device}")
    print(f"  - Seed: {RANDOM_SEED}")
    print(f"  - Alpha: {ALPHA}, Gamma: {GAMMA}, Hidden: {HIDDEN_SIZE}")
    print(f"  - Epsilon: {EPSILON} -> {EPSILON_MIN} (Decay: {EPSILON_DECAY})")
    print(f"  - DeepMind Params: Mem={MEMORY_CAPACITY}, Batch={BATCH_SIZE}, Target Freq={TARGET_UPDATE_FREQ}")
    
    # 1. Générer les scénarios
    print("\n[Phase 1/3] Génération des scénarios...")
    scenarios = generate_scenarios(NUM_SCENARIOS)
    
    # 2. Visualiser les scénarios
    print("\n[Phase 2/3] Visualisation des scénarios...")
    try:
        fig, axes = plt.subplots(1, NUM_SCENARIOS, figsize=(5 * NUM_SCENARIOS, 5))
        if NUM_SCENARIOS == 1: axes = [axes] # Assurer que 'axes' est itérable
        fig.suptitle(f'CONFIGURATION DES {NUM_SCENARIOS} SCÉNARIOS', fontsize=16, fontweight='bold')
        
        for i, scenario in enumerate(scenarios):
            render_scenario_grid(scenario, axes[i], f"Scénario {scenario['id']}")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('all_scenarios_configuration.png', dpi=100, bbox_inches='tight')
        print("✓ Graphique 'all_scenarios_configuration.png' sauvegardé.")
        plt.close(fig)
    except Exception as e:
        print(f"ERREUR lors de la visualisation des scénarios: {e}")
    
    # 3. Exécuter tous les scénarios
    print("\n[Phase 3/3] Exécution des scénarios...")
    all_results = []
    
    for scenario in scenarios:
        results = run_scenario(scenario, MAX_EPISODES)
        if not results: # Si run_scenario a échoué (ex: patch env)
            continue
        all_results.append(results)
        
        # Générer le graphique de comparaison pour ce scénario
        plot_scenario_comparison(results, scenario)
    
    # 4. Résumé global
    print("\n" + "="*70)
    print("GÉNÉRATION DU RÉSUMÉ GLOBAL...")
    print("="*70)
    if all_results:
        plot_global_summary(all_results, scenarios)
    else:
        print("Aucun résultat n'a été collecté.")

    # 5. Statistiques finales
    print("\n" + "="*70)
    print("STATISTIQUES FINALES")
    print("="*70)
    
    if all_results:
        wins_simple = sum(1 for r in all_results if r['winner'] == 'DQN Simple')
        wins_deepmind = sum(1 for r in all_results if r['winner'] == 'DQN DeepMind')
        wins_tie = sum(1 for r in all_results if r['winner'] == 'Égalité')
        
        avg_rewards_simple = np.mean([np.mean(r['simple']['rewards'][-50:]) for r in all_results])
        avg_rewards_deepmind = np.mean([np.mean(r['deepmind']['rewards'][-50:]) for r in all_results])
        avg_success_simple = np.mean([r['simple']['success_rate'] for r in all_results])
        avg_success_deepmind = np.mean([r['deepmind']['success_rate'] for r in all_results])
        
        print("\nRÉSULTATS GLOBAUX:")
        print(f"\nDQN Simple:")
        print(f"  - Victoires: {wins_simple}/{NUM_SCENARIOS}")
        print(f"  - Reward moyen: {avg_rewards_simple:.2f}")
        print(f"  - Taux de succès moyen: {avg_success_simple:.1f}%")
        
        print(f"\nDQN DeepMind:")
        print(f"  - Victoires: {wins_deepmind}/{NUM_SCENARIOS}")
        print(f"  - Reward moyen: {avg_rewards_deepmind:.2f}")
        print(f"  - Taux de succès moyen: {avg_success_deepmind:.1f}%")
        
        print(f"\nÉgalités: {wins_tie}/{NUM_SCENARIOS}")
        
        print("\n" + "="*70)
        if wins_deepmind > wins_simple:
            print("🏆 CHAMPION GLOBAL: DQN DEEPMIND")
        elif wins_simple > wins_deepmind:
            print("🏆 CHAMPION GLOBAL: DQN SIMPLE")
        else:
            print("🤝 ÉGALITÉ PARFAITE")
        print("="*70)
    
    end_time_total = time.time()
    print(f"\nTemps d'exécution total: {end_time_total - start_time_total:.2f} secondes")
    print("EXPÉRIENCE TERMINÉE AVEC SUCCÈS!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

