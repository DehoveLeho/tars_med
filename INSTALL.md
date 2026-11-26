# Guide d'installation — Simulation MuJoCo & ROS 2 (T.A.R.S.-Med)

Ce guide permet à toute l'équipe d'installer **exactement le même environnement** (ROS 2 Humble + MuJoCo + modèle Unitree G1 29-DOF) afin d'éviter les conflits de versions et de pouvoir tester les nœuds ROS 2, la cinématique inverse (IK) et l'orchestrateur chacun de son côté.

---

## Stack technique de référence (versions figées)

Pour garantir la compatibilité entre nos machines, tout le monde utilise :

- **OS cible :** Ubuntu 22.04 LTS (natif, via WSL2 sous Windows, ou via Distrobox sous Fedora/Arch)
- **ROS 2 :** Humble Hawksbill (`ros-humble-desktop`)
- **Python :** 3.10 (version système d'Ubuntu 22.04)
- **MuJoCo :** `3.2.6` (avec `numpy<2.0` pour éviter tout conflit avec les paquets ROS 2 Humble)
- **Modèle robot :** dépôt `unitree_mujoco` figé au commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` (`g1_29dof.xml`), complété par notre scène `tars_med_scene.xml` versionnée dans `tars_med_sim/models/`

---

## Étape 1 : Préparer Ubuntu 22.04 selon votre OS

### Option A — Vous êtes déjà sur Ubuntu 22.04 natif

Passez directement à l'**Étape 2**.

### Option B — Vous êtes sur Windows 10 / 11 (WSL2)

1. Ouvrez **PowerShell en mode Administrateur** et lancez :
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
2. Redémarrez le PC si demandé, puis ouvrez **Ubuntu 22.04** depuis le menu Démarrer.
3. Vérifiez que l'affichage graphique (WSLg) est actif :
   ```bash
   echo $DISPLAY
   ```
   *Résultat attendu :* `:0` (indique que la fenêtre 3D MuJoCo s'ouvrira nativement sous Windows).

### Option C — Vous êtes sur Fedora (ou autre Linux) via Distrobox

1. Installez Distrobox et Podman, puis créez le conteneur Ubuntu 22.04 *(retirez `--nvidia` si vous n'avez pas de GPU NVIDIA)* :
   ```bash
   sudo dnf install -y distrobox podman
   distrobox create --name ros2-humble --image ubuntu:22.04 --nvidia
   ```
2. Entrez dans le conteneur (à faire à chaque ouverture de terminal) :
   ```bash
   distrobox enter ros2-humble
   ```

---

## Étape 2 : Installer ROS 2 Humble et les dépendances système

Dans votre terminal **Ubuntu 22.04**, exécutez le bloc suivant :

```bash
sudo apt update && sudo apt install -y locales curl software-properties-common git python3-pip
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install -y \
  ros-humble-desktop \
  ros-dev-tools \
  ros-humble-joy \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  libxinerama1 \
  libxcursor1 \
  libxi6
```

Configurez ensuite le sourcing automatique de ROS 2 et un `ROS_DOMAIN_ID` personnel *(remplacez `<VOTRE_ID>` par un entier unique entre `1` et `100` pour éviter de piloter à distance la simulation d'un coéquipier sur le même réseau Wi-Fi)* :

```bash
cat << 'EOF' >> ~/.bashrc
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
if [ -f ~/tars_med_ws/install/setup.bash ]; then
  source ~/tars_med_ws/install/setup.bash
fi
export ROS_DOMAIN_ID=<VOTRE_ID>
EOF
source ~/.bashrc
```

**Vérification :**
```bash
ros2 --version
```

---

## Étape 3 : Installer MuJoCo (version figée)

Installez la version exacte de MuJoCo et NumPy compatibles avec ROS 2 Humble :

```bash
pip3 install "mujoco==3.2.6" "numpy<2.0"
```

**Vérification :**
```bash
python3 -c "import mujoco, numpy; print('MuJoCo:', mujoco.__version__, '| NumPy:', numpy.__version__)"
```

---

## Étape 4 : Récupérer le modèle Unitree G1 (commit figé) et le workspace

1. Créez le workspace ROS 2, clonez les modèles officiels Unitree et placez-vous sur le commit exact validé par l'équipe :
   ```bash
   mkdir -p ~/tars_med_ws/src
   cd ~/tars_med_ws/src
   git clone https://github.com/unitreerobotics/unitree_mujoco.git
   cd unitree_mujoco && git checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d && cd ..
   ```

2. **Étape indispensable :** empêchez `colcon` de compiler les exemples Go2 inclus dans le dépôt Unitree (sinon la compilation échoue sur un doublon `stand_go2`) :
   ```bash
   touch ~/tars_med_ws/src/unitree_mujoco/COLCON_IGNORE
   ```

3. Testez l'affichage du modèle Unitree G1 29-DOF en simulation brute *(on force X11 pour éviter un crash GLFW sous Wayland)* :
   ```bash
   env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 python3 -m mujoco.viewer --mjcf ~/tars_med_ws/src/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
   ```
   *Résultat attendu :* Une fenêtre 3D s'ouvre avec le robot G1 (qui tombe au sol car aucun moteur n'est encore piloté — c'est normal à ce stade).

4. Clonez ensuite les paquets de l'équipe dans `~/tars_med_ws/src/tars_med` :
   ```bash
   cd ~/tars_med_ws/src
   git clone https://github.com/DehoveLeho/tars_med.git tars_med
   ```

Votre dossier `~/tars_med_ws/src/` doit avoir exactement cette structure :

```
~/tars_med_ws/src/
├── unitree_mujoco/          # Dépôt Unitree figé (ignoré par colcon via COLCON_IGNORE)
└── tars_med/                # Dépôt Git de l'équipe
    ├── projet_interfaces/   # Messages et actions ROS 2 custom
    └── tars_med_sim/        # Nœuds MuJoCo, téléop manette et scène XML
```

---

## Étape 5 : Compiler et lancer la simulation complète

1. Compilez les paquets ROS 2 :
   ```bash
   cd ~/tars_med_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

2. Vérifiez que nos interfaces ROS 2 custom sont bien reconnues :
   ```bash
   ros2 interface show projet_interfaces/action/ExecuteGrasp
   ```

3. Lancez la simulation avec le G1 stabilisé, le plateau médical et le contrôle IK :
   ```bash
   env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ros2 launch tars_med_sim sim_teleop.launch.py
   ```

---

## Contrôles & Téléopération

### Commandes Manette (profil Xbox / xpad)
- **Stick gauche (X / Y) :** déplacer la main droite dans le plan horizontal
- **Stick droit (haut / bas) :** monter / descendre la main droite (axe Z)
- **Gâchette droite (RT / R2) :** publie le taux de fermeture sur `/teleop/gripper_cmd`
- **Bouton A / Croix :** ramener le bras à sa position initiale au-dessus du plateau

### Commandes Clavier (fenêtre MuJoCo 3D)
Cliquez sur la fenêtre 3D pour la rendre active :
- **D** : Démo autonome complète (approche, saisie physique, levage +20 cm, présentation avancée, relâchement par arrachement, retour home)
- **Flèches (↑ / ↓ / ← / →)** : Translation de la main droite dans le plan X/Y
- **W / S** (ou **PgUp / PgDn**) : Monter / descendre la main droite en Z
- **Espace** : Pince ouverte / fermée (saisie weld physique automatique si à portée d'un instrument)
- **R** : Déclencher l'arrachement de l'instrument tendu et retour home
- **H** : Retour à la posture initiale au-dessus du plateau

### Interfaces ROS 2 exposées
- `/joint_states` (`sensor_msgs/msg/JointState`) — état des articulations à 50 Hz
- `/perception/object_pose` (`projet_interfaces/msg/ObjectPose`) — pose 6D dans le repère `pelvis` pour `scalpel`, `tourniquet` et `bottle`
- `/teleop/ee_target` (`geometry_msgs/msg/Point`) — consigne cartésienne end-effector
- `/teleop/gripper_cmd` (`std_msgs/msg/Float32`) — commande de fermeture pince [0.0 - 1.0]
- `/manipulation/execute_grasp` (`projet_interfaces/action/ExecuteGrasp`) — serveur d'action de préhension autonome

Test de saisie autonome par ligne de commande :
```bash
ros2 action send_goal /manipulation/execute_grasp projet_interfaces/action/ExecuteGrasp '{object_id: "bottle", max_force_newton: 10.0}'
```

---

## Dépannage rapide

| Problème | Cause | Solution |
| --- | --- | --- |
| `Segmentation fault` ou `Wayland: Failed to connect` au lancement du viewer | Conflit GLFW avec Wayland / Distrobox | Toujours préfixer la commande avec `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11` (et exécuter `xhost +si:localuser:$USER` côté hôte Fedora si besoin) |
| `Duplicate package names not supported: stand_go2` lors du `colcon build` | `colcon` scanne les exemples du dépôt `unitree_mujoco` | Vérifier que `touch ~/tars_med_ws/src/unitree_mujoco/COLCON_IGNORE` a bien été exécuté |
| La manette ne répond pas sous Windows / WSL2 | WSL2 ne monte pas les périphériques USB par défaut et `joy` (SDL2) lit `/dev/input/event*` | Sous PowerShell Admin : `usbipd bind --busid <ID>` puis `usbipd attach --wsl --busid <ID>`. Dans Ubuntu : `sudo usermod -aG input $USER` puis redémarrer WSL |
| Deux coéquipiers interfèrent sur la même simulation | Même `ROS_DOMAIN_ID` sur le réseau local | Vérifier que `export ROS_DOMAIN_ID=<ID_UNIQUE>` est bien défini dans `~/.bashrc` |
