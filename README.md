# 🧠 GridWorld RL Project V3 : Comparaison DQN vs DQN DeepMind

## 🎯 Description Générale
Ce projet explore l’apprentissage par renforcement profond (*Deep Reinforcement Learning*) à travers un environnement **GridWorld** personnalisé.  
Deux agents sont entraînés et comparés :

- **DQN Simple** : version de base du Deep Q-Learning.  
- **DQN DeepMind** : version avancée intégrant les principes de stabilisation développés par **DeepMind** (utilisation d’un *Replay Buffer* et de deux réseaux neuronaux distincts : *Online* et *Target*).

L’objectif est de mesurer l’impact de ces mécanismes sur la **vitesse d’apprentissage**, la **stabilité** et la **performance finale** des agents sur une variété de scénarios.

---

## 🗺️ Contexte et Environnement
L’environnement **GridWorld** simule une grille 2D où l’agent doit atteindre un objectif tout en évitant des obstacles.

- Chaque épisode commence à une position initiale et se termine lorsque l’agent atteint le but, tombe dans un piège ou dépasse le nombre maximal d’étapes.  
- L’agent reçoit des **récompenses positives ou négatives** selon ses actions.  
- Les grilles peuvent être **générées aléatoirement**, avec des **buts dynamiques** se déplaçant au fil du temps, forçant l’agent à apprendre une **politique adaptative** et robuste.  

---

## ⚙️ Philosophie et Méthodologie
Ce projet met en parallèle deux approches d’apprentissage par Q-Learning profond :

### 🧩 1. DQN Simple
- Apprentissage direct des valeurs d’action \( Q(s,a) \) via un seul réseau de neurones.  
- Mises à jour effectuées à chaque étape, sans mémoire d’expériences ni réseau secondaire.  
- Illustre les limites d’un DQN naïf : **instabilité**, **forte variance**, **apprentissage plus lent**.

### 🔬 2. DQN DeepMind (Version améliorée)
Inspiré du travail pionnier de **DeepMind (2015)** sur les jeux Atari :

- **Replay Buffer** : stocke des milliers de transitions `(état, action, récompense, état suivant)`, rééchantillonnées de manière aléatoire pour casser la corrélation temporelle et stabiliser l’apprentissage.  
- **Réseaux doubles (Online et Target)** :
  - Le réseau *Online* choisit les actions et s’entraîne.  
  - Le réseau *Target* est mis à jour moins fréquemment pour fournir une cible d’apprentissage stable.  
- Ces mécanismes réduisent l’instabilité et permettent une **meilleure convergence** vers la politique optimale.

---

## 🧪 Comparaison Expérimentale
Le projet entraîne ces deux agents sur une **série de scénarios GridWorld générés aléatoirement** :

- Chaque scénario présente une configuration unique : taille de la grille, nombre d’obstacles, position du but.  
- L’évaluation compare la **courbe d’apprentissage** (récompense cumulée, durée des épisodes, stabilité de la perte) et la **politique finale** (chemin appris vers le but).  
- Un **résumé global** agrège les performances de tous les scénarios pour visualiser les tendances générales.

---

## 📊 Analyse et Visualisation
Le système de visualisation génère automatiquement plusieurs graphiques :

- **Courbes d’apprentissage par scénario** : évolution des récompenses, du nombre d’étapes et de la fonction de perte.  
- **Trajectoires apprises** : représentation du chemin que chaque agent choisit en mode exploitation.  
- **Résumé global (`global_summary_all_scenarios.png`)** : comparaison synthétique entre les deux approches (taux de réussite, récompense moyenne, stabilité).  

Ces visualisations permettent d’évaluer concrètement **l’efficacité de la stabilisation DeepMind** face à un DQN simple.

---

## 📁 Structure du Projet
📁 gridworld-rl-project-V3/
├── main_dqn.py # Script principal de comparaison et d’entraînement
├── gridworld.py # Environnement GridWorld (héritant de gym.Env)
├── agent_dqn.py # Agent DQN simple
├── agent_dqn_deepmind.py # Agent DQN amélioré avec Replay Buffer et Target Network
├── results/ # Dossier généré contenant les graphiques et comparaisons
└── README.md # Documentation du projet


---

## 🚀 Fonctionnement Global
Lancer le script `main_dqn.py` :

1. Génère automatiquement plusieurs scénarios GridWorld.  
2. Entraîne successivement les deux agents sur chaque scénario.  
3. Sauvegarde les performances intermédiaires et finales.  
4. Produit un ensemble complet de visualisations comparatives.  

L’expérience met ainsi en évidence les **bénéfices des principes de DeepMind** :  
> une convergence plus stable, un apprentissage plus rapide et des politiques mieux généralisées.

---

## 🧭 Interprétation des Résultats (Optionnel)
- Le **DQN Simple** apprend plus vite au départ mais présente une forte instabilité et des performances finales variables.  
- Le **DQN DeepMind**, grâce à son *Replay Buffer* et à son *Réseau Cible*, converge plus lentement au début mais atteint une politique plus stable et efficace.  
- La variance de la récompense finale est significativement réduite, montrant l’effet stabilisateur de la méthodologie DeepMind.  

---

## 🛠️ Installation
1. **Cloner le dépôt :**
```bash
git clone https://github.com/votre-nom-utilisateur/gridworld-rl-project-V3.git
cd gridworld-rl-project-V3
