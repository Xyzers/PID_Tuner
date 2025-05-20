#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import configparser
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import winsound

# Importer PIDController si défini dans un autre module, sinon le copier ici.
# from pid_controller_module import PIDController

# --- Configuration du Logging ---
logger = logging.getLogger(__name__)

try:
    # Essayer d'obtenir le nom du fichier depuis __file__ (pour exécution en script)
    NOM_SCRIPT_SANS_EXTENSION = Path(__file__).stem
except NameError:
    # Fallback si __file__ n'est pas défini (ex: notebook Jupyter)
    # Vous pouvez définir ici manuellement le nom de base attendu pour le .ini
    NOM_SCRIPT_SANS_EXTENSION = "pid_tuner"
    logger.info(f"La variable __file__ n'est pas définie. Utilisation de '{NOM_SCRIPT_SANS_EXTENSION}' comme nom de base par défaut.")

# ... (suite de la classe PIDController et autres fonctions) ...

# --- Classe PIDController (copiée/adaptée de Pid_comparator.py) ---
class PIDController:
    def __init__(self, Kp, Ti, Td, Tsamp, mv_min, mv_max, direct_action=True, initial_mv=0.0):
        """
        Initialise le contrôleur PID.
        Kp: Gain proportionnel
        Ti: Temps intégral (secondes). Si <= 0 ou inf, l'action intégrale est désactivée.
        Td: Temps dérivé (secondes)
        Tsamp: Période d'échantillonnage (secondes)
        mv_min, mv_max: Limites de la variable manipulée (MV)
        direct_action: True si une erreur positive (SP > PV) doit augmenter la MV
        initial_mv: Valeur initiale de la MV pour un démarrage sans à-coup
        """
        self.Kp_param = Kp
        # Gérer Ti <= 0 ou inf comme pas d'action intégrale
        self.Ti_param = float('inf') if Ti <= 0 or Ti == float('inf') else Ti
        self.Td_param = Td
        self.Tsamp = Tsamp  # en secondes
        self.mv_min = mv_min
        self.mv_max = mv_max
        self.direct_action = direct_action

        self.proportional_term = 0.0
        self.integral_term = 0.0  # Ce sera la valeur cumulée du terme Intégral
        self.derivative_term = 0.0
        
        self.previous_pv = None
        self.previous_error = None # Pour le terme Dérivé sur l'erreur (non utilisé par défaut)
        self.mv = initial_mv
        self.last_active_mv = initial_mv

        self._update_internal_gains()

    def _update_internal_gains(self):
        """Met à jour les gains internes calculés à partir des paramètres Kp, Ti, Td, Tsamp."""
        self.kp_calc = self.Kp_param
        # Si Ti_param est infini (ou <=0), ki_calc est 0 (pas d'action intégrale)
        if self.Ti_param == float('inf') or self.Tsamp <= 0:
            self.ki_calc = 0.0
        else:
            self.ki_calc = (self.Kp_param * self.Tsamp) / self.Ti_param
        
        if self.Tsamp <= 0:
            self.kd_calc = 0.0
        else:
            self.kd_calc = (self.Kp_param * self.Td_param) / self.Tsamp

    def set_parameters(self, Kp, Ti, Td):
        """Met à jour les paramètres Kp, Ti, Td du PID et recalcule les gains internes."""
        parameter_changed = False
        if self.Kp_param != Kp:
            self.Kp_param = Kp
            parameter_changed = True
        
        new_ti_param = float('inf') if Ti <= 0 or Ti == float('inf') else Ti
        if self.Ti_param != new_ti_param:
            self.Ti_param = new_ti_param
            parameter_changed = True
            
        if self.Td_param != Td:
            self.Td_param = Td
            parameter_changed = True
            
        if parameter_changed:
            self._update_internal_gains()
            logger.debug(f"Paramètres PID mis à jour : Kp={self.Kp_param}, Ti={self.Ti_param}, Td={self.Td_param}")
            logger.debug(f"Gains internes : kp_calc={self.kp_calc}, ki_calc={self.ki_calc}, kd_calc={self.kd_calc}")

    def set_initial_state(self, pv_initial, sp_initial, mv_initial):
        """
        Initialise l'état interne du PID pour un démarrage sans à-coup,
        en ajustant le terme intégral pour que la sortie MV initiale soit mv_initial.
        """
        self.previous_pv = pv_initial
        self.mv = self._limit_mv(mv_initial)
        self.last_active_mv = self.mv
        
        error = sp_initial - pv_initial
        if not self.direct_action:
            error = -error
        self.previous_error = error # Initialiser previous_error

        self.proportional_term = self.kp_calc * error
        # Pour l'initialisation, on suppose que le terme dérivé est nul (pas de variation de PV passée connue)
        self.derivative_term = 0.0
        
        # Ajuster le terme intégral pour que P + I + D = mv_initial
        self.integral_term = self.mv - (self.proportional_term + self.derivative_term)
        
        logger.info(f"État initial du PID : PV={pv_initial:.2f}, SP={sp_initial:.2f}, MV={self.mv:.2f}")
        logger.debug(f"Termes initiaux : P={self.proportional_term:.2f}, I={self.integral_term:.2f}, D={self.derivative_term:.2f}")

    def _limit_mv(self, mv_candidate):
        """Limite la valeur de MV candidate entre mv_min et mv_max."""
        return max(self.mv_min, min(mv_candidate, self.mv_max))

    def update(self, sp, pv):
        """
        Calcule la sortie MV du PID pour la consigne (sp) et la mesure (pv) actuelles.
        Implémente l'anti-windup par back-calculation.
        Le terme dérivé est calculé sur la PV pour éviter les "coups de bélier" lors des changements de consigne.
        """
        if self.previous_pv is None: # Doit être initialisé par set_initial_state
            self.previous_pv = pv
            if self.previous_error is None: # Si c'est le tout premier appel
                 error_for_first_D = sp - pv
                 if not self.direct_action: error_for_first_D = -error_for_first_D
                 self.previous_error = error_for_first_D

        error = sp - pv
        if not self.direct_action:
            error = -error

        # Terme Proportionnel
        self.proportional_term = self.kp_calc * error
        
        # Terme Dérivé (sur la PV pour éviter le "derivative kick")
        self.derivative_term = 0.0
        if self.kd_calc > 0 and self.previous_pv is not None:
            delta_pv = pv - self.previous_pv
            self.derivative_term = -self.kd_calc * delta_pv # Négatif car D agit sur -PV

        # Calcul de la MV avant la mise à jour du terme intégral (pour anti-windup)
        # L'ancien terme intégral est utilisé ici.
        mv_sans_increment_integral = self.proportional_term + self.integral_term + self.derivative_term
        
        # Calcul de l'incrément pour le terme intégral actuel
        integral_increment = 0.0
        if self.ki_calc > 0: # ki_calc = Kp * Tsamp / Ti
            integral_increment = self.ki_calc * error 
            
        mv_candidate = mv_sans_increment_integral + integral_increment
        mv_limitee = self._limit_mv(mv_candidate)

        # Anti-Windup (Back-Calculation)
        # Si la sortie est saturée, recalculer le terme intégral pour qu'il corresponde à la sortie limitée.
        if self.ki_calc > 0 :
            if mv_candidate != mv_limitee:
                # I_new = MV_limitee - P - D (l'ancien I est remplacé)
                self.integral_term = mv_limitee - (self.proportional_term + self.derivative_term)
                # logger.debug(f"Anti-windup. MV_cand={mv_candidate:.2f}, MV_lim={mv_limitee:.2f}, I ajusté à {self.integral_term:.2f}")
            else:
                # Mise à jour normale du terme intégral si non saturé
                self.integral_term += integral_increment
        
        self.mv = mv_limitee
        self.previous_pv = pv
        self.previous_error = error

        return self.mv

# --- Fonctions de Configuration et Utilitaires ---
def load_config_and_setup_logging(config_file_path_str):
    """Charge la configuration depuis .ini et configure le logging."""
    config_path = Path(config_file_path_str)
    if not config_path.is_file():
        print(f"ERREUR CRITIQUE: Fichier de configuration '{config_path}' non trouvé.")
        sys.exit(1)

    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    config.optionxform = str  # Conserver la casse des clés
    config.read(config_path, encoding='utf-8')

    log_level_str = config.get('General', 'log_level', fallback='INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    log_file_base = config.get('General', 'log_file_base_name', fallback=NOM_SCRIPT_SANS_EXTENSION)
    log_file_name = f"{log_file_base}.txt"
    log_file_path = Path(log_file_name).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            try: handler.close()
            except Exception: pass
            logger.removeHandler(handler)
            
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info(f"Logging configuré. Niveau: {log_level_str}. Fichier log: {log_file_path}")
    logger.info(f"Configuration chargée depuis '{config_path}'.")
    return config

def load_process_model_and_scalers(config_modele):
    """Charge le modèle de procédé et les scalers depuis les chemins spécifiés."""
    model_path_str = config_modele.get('model_path')
    scalers_path_str = config_modele.get('scalers_path')

    if not model_path_str or not scalers_path_str:
        logger.error("Chemin du modèle ou des scalers non spécifié dans la config [ModeleProcede].")
        raise ValueError("Chemin modèle/scalers manquant dans la configuration.")

    model_path = Path(model_path_str)
    scalers_path = Path(scalers_path_str)

    if not model_path.is_file():
        logger.error(f"Fichier modèle de procédé non trouvé : {model_path}")
        raise FileNotFoundError(f"Fichier modèle non trouvé : {model_path}")
    if not scalers_path.is_file():
        logger.error(f"Fichier des scalers non trouvé : {scalers_path}")
        raise FileNotFoundError(f"Fichier scalers non trouvé : {scalers_path}")

    try:
        process_model = load(model_path)
        scalers_dict = load(scalers_path)
        scaler_X = scalers_dict['scaler_X']
        scaler_y = scalers_dict['scaler_y']
        logger.info(f"Modèle de procédé chargé depuis : {model_path}")
        logger.info(f"Scalers (X, Y) chargés depuis : {scalers_path}")
        return process_model, scaler_X, scaler_y
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle ou des scalers : {e}", exc_info=True)
        raise

# --- Préparation des Features pour l'Entrée du Modèle ---
def get_model_feature_names_from_config(config_modele_features):
    """
    (Fallback) Construit la liste ordonnée des noms de features attendus par le modèle
    en se basant sur la configuration [ModeleProcede] (pv_lags, mv_lags, etc.).
    L'ORDRE EST IMPORTANT et doit correspondre à celui utilisé lors de l'entraînement.
    Privilégier scaler_X.feature_names_in_ si disponible.
    """
    feature_names = []
    
    # Ordre supposé des groupes de features, basé sur une convention (ou l'analyse de pid_model_builder.py)
    # Votre pid_model_builder.py semble traiter : PV, MV, SP, Kp_hist, Ti_hist, Td_hist, puis Dist{i}
    lag_config_order = [
        ('pv_lags', 'PV'),
        ('mv_lags', 'MV'),
        ('sp_lags', 'SP'),
        ('kp_hist_lags', 'Kp_hist'),
        ('ti_hist_lags', 'Ti_hist'),
        ('td_hist_lags', 'Td_hist'),
    ]
    
    # Ajouter les perturbations dynamiquement
    i = 1
    while True:
        dist_lag_key = f'disturbance_{i}_lags'
        # Vérifier si l'option existe ET si elle a une valeur (pour éviter erreur si clé existe mais vide)
        if config_modele_features.has_option(dist_lag_key) and config_modele_features.get(dist_lag_key, '').strip():
            lag_config_order.append((dist_lag_key, f'Dist{i}'))
            i += 1
        elif not config_modele_features.has_option(dist_lag_key): 
            # Si l'option n'existe plus, on arrête (ex: disturbance_3_lags existe mais pas disturbance_4_lags)
            break
        else: 
            # L'option existe mais est vide (ex: disturbance_tag_2 = ), on l'ignore et on continue au cas où il y aurait un disturbance_3_lags renseigné.
            # Cependant, la logique de pid_model_builder semble s'arrêter si disturbance_tag_i est vide.
            # Pour être cohérent, on devrait s'arrêter si la clé de lag est vide ou si le tag associé était vide.
            # Pour l'instant, cette logique suppose que si disturbance_i_lags existe, il est pertinent.
             logger.debug(f"La clé {dist_lag_key} existe mais est peut-être vide. On continue à vérifier la suivante.")
             i += 1 # Continuer à vérifier au cas où il y aurait un trou (pas idéal)
             if i > 10: # Sécurité pour éviter boucle infinie si la logique de détection de fin est faillible
                logger.warning("Arrêt de la recherche de lags de perturbation après 10 tentatives.")
                break


    for config_lag_key, base_col_name in lag_config_order:
        num_lags = config_modele_features.getint(config_lag_key, 0)
        if num_lags > 0:
            for lag_idx in range(1, num_lags + 1):
                feature_names.append(f'{base_col_name}_lag_{lag_idx}')
                
    logger.info(f"Noms des features du modèle (générés depuis config, ordre important) : {feature_names}")
    if not feature_names:
        logger.warning("Aucun nom de feature généré depuis la config. Vérifiez [ModeleProcede] et les configurations de lags.")
    return feature_names

def prepare_model_input_frame(current_history_df, feature_names_ordered_list, config_modele_features):
    """
    Prépare une seule ligne (DataFrame) de features pour le modèle à partir de current_history_df.
    current_history_df : DataFrame contenant l'historique récent des variables de base (PV, MV, SP, etc.).
                        Les données les plus récentes sont en bas.
    feature_names_ordered_list : Liste ordonnée des noms de features exacts attendus par le modèle.
    config_modele_features : Section de configuration [ModeleProcede] pour les comptes de lags.
    """
    model_input_dict = {} # Dictionnaire pour construire la ligne de feature

    # Itérer sur chaque nom de feature attendu par le modèle
    for full_feature_name in feature_names_ordered_list:
        parts = full_feature_name.split('_lag_')
        if len(parts) == 2:
            base_col_name = parts[0] # Ex: 'PV', 'MV', 'Dist1'
            try:
                lag_idx = int(parts[1]) # Ex: 1, 2
            except ValueError:
                logger.error(f"Impossible d'extraire l'index de lag pour la feature '{full_feature_name}'. Elle sera NaN.")
                model_input_dict[full_feature_name] = np.nan
                continue

            if base_col_name not in current_history_df.columns:
                logger.warning(f"La colonne de base '{base_col_name}' pour la feature '{full_feature_name}' "
                               f"n'est pas dans history_df. La feature sera NaN.")
                model_input_dict[full_feature_name] = np.nan
                continue
            
            # current_history_df a les données les plus récentes à la fin.
            # Pour PV_lag_1, on a besoin de la valeur de PV à l'indice -1 (la dernière).
            # Pour PV_lag_2, on a besoin de la valeur de PV à l'indice -2 (l'avant-dernière).
            # Donc, on prend l'élément à l'index -(lag_idx) de la colonne history_df[base_col_name].
            if len(current_history_df) >= lag_idx:
                model_input_dict[full_feature_name] = current_history_df[base_col_name].iloc[-lag_idx]
            else:
                logger.error(f"Historique insuffisant pour créer '{full_feature_name}'. "
                               f"Requis: {lag_idx} points, Disponible: {len(current_history_df)}. La feature sera NaN.")
                model_input_dict[full_feature_name] = np.nan
        else:
            # Si la feature n'est pas une feature "lagged" (ex: une feature directe comme 'heure_du_jour' si le modèle l'utilisait)
            # Pour l'instant, on suppose que toutes les features sont "lagged" ou que leur nom est directement dans history_df.
            if full_feature_name in current_history_df.columns:
                 model_input_dict[full_feature_name] = current_history_df[full_feature_name].iloc[-1] # Prendre la dernière valeur
            else:
                logger.warning(f"Feature '{full_feature_name}' non reconnue comme 'lagged' et non trouvée "
                               f"directement dans history_df. Elle sera NaN.")
                model_input_dict[full_feature_name] = np.nan

    # Créer un DataFrame d'une seule ligne avec les colonnes dans l'ordre attendu
    # et s'assurer que toutes les features attendues sont présentes, même si avec NaN.
    single_row_data = {fn: [model_input_dict.get(fn, np.nan)] for fn in feature_names_ordered_list}
    final_input_df = pd.DataFrame(single_row_data, columns=feature_names_ordered_list)
    
    if final_input_df.isnull().values.any():
        cols_with_nan = final_input_df.columns[final_input_df.isnull().any()].tolist()
        logger.warning(f"NaNs présents dans l'entrée du modèle pour les features : {cols_with_nan}.")
        # Une stratégie de remplissage pourrait être nécessaire ici si le modèle ne gère pas les NaNs.
        # Ex: final_input_df = final_input_df.fillna(0) # Attention: peut fausser les prédictions.
        # Pour RandomForest, il peut parfois les gérer. Pour l'instant, on laisse les NaNs.
    return final_input_df


# --- Noyau de Simulation ---
def run_closed_loop_simulation(config, nom_jeu_reglage, Kp, Ti, Td,
                               modele_procede, scaler_X, scaler_y,
                               model_feature_names):
    """Exécute une simulation en boucle fermée pour un jeu de paramètres PID donné."""
    logger.info(f"--- Démarrage simulation pour jeu : {nom_jeu_reglage} (Kp={Kp}, Ti={Ti}, Td={Td}) ---")

    cfg_pid_base = config['ParametresPIDBase']
    cfg_sim_scenario = config['ScenarioSimulation']
    cfg_modele_features = config['ModeleProcede'] # Pour les comptes de lags

    tsamp_s = cfg_pid_base.getfloat('tsamp_pid_sim_ms') / 1000.0
    mv_min_sim = cfg_pid_base.getfloat('mv_min')
    mv_max_sim = cfg_pid_base.getfloat('mv_max')
    direct_action_sim = cfg_pid_base.getboolean('direct_action')

    sim_duration_s = cfg_sim_scenario.getfloat('simulation_duration_seconds')
    num_steps = int(sim_duration_s / tsamp_s)

    pv_initiale = cfg_sim_scenario.getfloat('initial_pv')
    mv_initiale = cfg_sim_scenario.getfloat('initial_mv')
    
    pid = PIDController(Kp, Ti, Td, tsamp_s, mv_min_sim, mv_max_sim, direct_action_sim, initial_mv=mv_initiale)

    time_points = np.arange(0, sim_duration_s, tsamp_s)[:num_steps] # Assurer la bonne longueur
    valeurs_sp = np.full_like(time_points, cfg_sim_scenario.getfloat('sp_constant_value')) # Défaut à constant

    type_sp = cfg_sim_scenario.get('setpoint_type', 'constant')
    if type_sp == 'step':
        steps_str = cfg_sim_scenario.get('sp_steps', '')
        sp_courante_pour_init = pv_initiale 
        if steps_str:
            parsed_steps = []
            for step_pair_str in steps_str.split(';'):
                time_val, sp_val = map(str.strip, step_pair_str.split(','))
                parsed_steps.append((float(time_val), float(sp_val)))
            parsed_steps.sort() 

            if parsed_steps and parsed_steps[0][0] == 0:
                 sp_courante_pour_init = parsed_steps[0][1]
            
            sp_idx_step = 0
            # Appliquer la valeur initiale avant le premier échelon (si pas à t=0)
            # ou la valeur du premier échelon (si à t=0)
            valeur_sp_actuelle_pour_boucle = sp_courante_pour_init 
            for i, t_sim_current in enumerate(time_points):
                while sp_idx_step < len(parsed_steps) and t_sim_current >= parsed_steps[sp_idx_step][0]:
                    valeur_sp_actuelle_pour_boucle = parsed_steps[sp_idx_step][1]
                    sp_idx_step += 1
                valeurs_sp[i] = valeur_sp_actuelle_pour_boucle
        sp_initiale_pour_pid = valeurs_sp[0]
    else: # constant
        sp_initiale_pour_pid = cfg_sim_scenario.getfloat('sp_constant_value')

    pid.set_initial_state(pv_initiale, sp_initiale_pour_pid, mv_initiale)

    pv_actuelle = pv_initiale
    
    max_lag = 0
    # Déterminer le max_lag à partir des noms de features du modèle si possible (plus robuste)
    # ou de la config en fallback.
    if model_feature_names:
        for fname in model_feature_names:
            if "_lag_" in fname:
                try: max_lag = max(max_lag, int(fname.split('_lag_')[-1]))
                except ValueError: pass
    else: # Fallback si model_feature_names est vide (ne devrait pas arriver si scaler_X.feature_names_in_ fonctionne)
        for key_cfg_lag in ['pv_lags', 'mv_lags', 'sp_lags', 'kp_hist_lags', 'ti_hist_lags', 'td_hist_lags']:
            max_lag = max(max_lag, cfg_modele_features.getint(key_cfg_lag, 0))
        idx_dist = 1
        while True:
            key_cfg_dist_lag = f'disturbance_{idx_dist}_lags'
            if cfg_modele_features.has_option(key_cfg_dist_lag):
                 max_lag = max(max_lag, cfg_modele_features.getint(key_cfg_dist_lag, 0))
                 idx_dist += 1
            else: break
    
    # Longueur du buffer d'historique : max_lag points passés.
    # Si max_lag = 0 (pas de features lag), on a quand même besoin d'1 point pour les valeurs actuelles (t-1)
    longueur_historique = max_lag if max_lag > 0 else 1


    logger.debug(f"Max lag requis par le modèle: {max_lag}. Longueur du buffer d'historique: {longueur_historique}")

    # Colonnes de base nécessaires pour construire les features.
    # Doit inclure toutes les variables de base qui ont des lags dans model_feature_names
    # Ex: 'PV', 'MV', 'SP', 'Dist1', etc.
    cols_base_historique = set()
    for fname in model_feature_names:
        if "_lag_" in fname:
            cols_base_historique.add(fname.split('_lag_')[0])
        else: # Feature directe
            cols_base_historique.add(fname)
    
    # S'assurer que PV, MV, SP sont là si le modèle les utilise implicitement ou explicitement.
    # (par exemple, si sp_lags=0, SP peut quand même être une feature directe "SP_lag_0" ou "SP")
    # Pour être sûr, on les inclut si des lags sont configurés pour eux, même si model_feature_names est vide.
    if cfg_modele_features.getint('pv_lags',0) > 0: cols_base_historique.add('PV')
    if cfg_modele_features.getint('mv_lags',0) > 0: cols_base_historique.add('MV')
    if cfg_modele_features.getint('sp_lags',0) > 0: cols_base_historique.add('SP')
    # TODO: Ajouter Kp_hist, Ti_hist, Td_hist, Disturbances si utilisés par le modèle

    colonnes_df_historique = sorted(list(cols_base_historique)) # Ordonner pour la consistance
    if not colonnes_df_historique: # Cas où model_feature_names est vide et aucun lag configuré pour PV/MV/SP
        if model_feature_names : # Si model_feature_names n'est pas vide, cela ne devrait pas arriver
             logger.warning("colonnes_df_historique est vide mais model_feature_names ne l'est pas. C'est incohérent.")
        # Si le modèle n'a aucune feature (très improbable), on a quand même besoin de PV, MV, SP pour la logique.
        colonnes_df_historique = ['PV', 'MV', 'SP']


    mode_init_lags = cfg_sim_scenario.get('lags_initialization_mode', 'constant')
    donnees_historique_init = {}
    if mode_init_lags == 'constant':
        if 'PV' in colonnes_df_historique: donnees_historique_init['PV'] = [pv_initiale] * longueur_historique
        if 'MV' in colonnes_df_historique: donnees_historique_init['MV'] = [mv_initiale] * longueur_historique
        if 'SP' in colonnes_df_historique: donnees_historique_init['SP'] = [sp_initiale_pour_pid] * longueur_historique
        # Initialiser les autres colonnes (Dist1, Kp_hist etc.) si elles sont dans colonnes_df_historique
        for col_hist in colonnes_df_historique:
            if col_hist not in donnees_historique_init:
                 logger.warning(f"Colonne '{col_hist}' dans colonnes_df_historique non explicitement initialisée "
                                f"en mode 'constant'. Valeur par défaut de 0.0 pour l'historique.")
                 donnees_historique_init[col_hist] = [0.0] * longueur_historique
    else: # TODO: 'historical_slice'
        logger.error(f"Mode d'initialisation des lags '{mode_init_lags}' non implémenté. "
                       "Utilisation de 'constant' par défaut.")
        # Fallback à constant
        if 'PV' in colonnes_df_historique: donnees_historique_init['PV'] = [pv_initiale] * longueur_historique
        if 'MV' in colonnes_df_historique: donnees_historique_init['MV'] = [mv_initiale] * longueur_historique
        if 'SP' in colonnes_df_historique: donnees_historique_init['SP'] = [sp_initiale_pour_pid] * longueur_historique


    df_historique = pd.DataFrame(donnees_historique_init, columns=colonnes_df_historique)

    resultats_sim = {'Time': [], 'SP': [], 'PV': [], 'MV': [], 'Error': []}
    logger.debug(f"df_historique initial pour entrées modèle :\n{df_historique.head()}")

    for i_step in range(num_steps):
        t_actuel = time_points[i_step]
        sp_actuelle = valeurs_sp[i_step]

        mv_actuelle = pid.update(sp_actuelle, pv_actuelle)

        # Prépare l'entrée pour le modèle de procédé en utilisant df_historique.
        # df_historique contient les données jusqu'à t-1 (ou l'équivalent) pour prédire la PV à t.
        input_X_df_modele = prepare_model_input_frame(df_historique, model_feature_names, cfg_modele_features)
        
        if input_X_df_modele.isnull().values.any():
            cols_nan_input = input_X_df_modele.columns[input_X_df_modele.isnull().any()].tolist()
            logger.error(f"Étape {i_step}, Temps {t_actuel:.2f}s : NaNs dans l'entrée du modèle pour {cols_nan_input}. "
                           "Arrêt de la simulation pour ce jeu de paramètres.")
            # Remplir le reste des résultats avec NaN et retourner un DataFrame partiel ou vide
            # pour indiquer un échec.
            remaining_steps = num_steps - i_step
            for key in resultats_sim.keys():
                resultats_sim[key].extend([np.nan] * remaining_steps)
            break 
            
        input_X_modele_scaled = scaler_X.transform(input_X_df_modele)
        
        pv_predite_scaled = modele_procede.predict(input_X_modele_scaled)
        pv_suivante = scaler_y.inverse_transform(pv_predite_scaled.reshape(-1, 1)).ravel()[0]
        # CORRECTION ICI :
        logger.debug(f"Étape {i_step}: input_X_df_modele (avant scale):\n{input_X_df_modele.head().to_string()}") # UTILISER input_X_df_modele

        # ET ICI AUSSI pour la variable des données scalées:
        logger.debug(f"Étape {i_step}: input_X_modele_scaled (1ere ligne):\n{input_X_modele_scaled[0]}")

        # ET ICI pour la variable pv prédite déscalée:
        logger.debug(f"Étape {i_step}: predicted_pv_scaled: {pv_predite_scaled[0]}, pv_suivante (déscalé): {pv_suivante}")
        
        resultats_sim['Time'].append(t_actuel)
        resultats_sim['SP'].append(sp_actuelle)
        resultats_sim['PV'].append(pv_actuelle) # PV au début de l'intervalle, utilisée par le PID
        resultats_sim['MV'].append(mv_actuelle) # MV calculée par le PID pour cet intervalle
        resultats_sim['Error'].append(sp_actuelle - pv_actuelle)

        # Mettre à jour pv_actuelle pour la prochaine itération (la PV à t devient la PV à t+1)
        pv_actuelle = pv_suivante 
        
        # Mettre à jour df_historique pour la prochaine prédiction du modèle
        # La nouvelle ligne contient les valeurs à la *fin* de l'intervalle t_actuel,
        # qui seront les entrées "t-1" pour prédire la PV de l'intervalle t_actuel+1.
        donnees_nouvelle_ligne_historique = {}
        if 'PV' in colonnes_df_historique: donnees_nouvelle_ligne_historique['PV'] = pv_actuelle # C'est pv_suivante
        if 'MV' in colonnes_df_historique: donnees_nouvelle_ligne_historique['MV'] = mv_actuelle
        if 'SP' in colonnes_df_historique: donnees_nouvelle_ligne_historique['SP'] = sp_actuelle # La SP peut changer
        # Mettre à jour les autres variables d'état (Disturbances, Kp_hist etc.) si elles sont dynamiques
        for col_hist_update in colonnes_df_historique:
             if col_hist_update not in donnees_nouvelle_ligne_historique:
                  # Si une colonne est nécessaire mais non mise à jour (ex: disturbance fixe ou Kp_hist),
                  # elle devrait être initialisée et conservée dans df_historique.
                  # Pour l'instant, on assume qu'elles sont soit constantes (déjà dans df_historique)
                  # soit mises à jour ici. Si une colonne de base manque, elle prendra la valeur de la ligne précédente.
                  # Ce n'est pas idéal, il faudrait une source pour chaque colonne de df_historique.
                  # Pour une simulation simple, on peut prendre la dernière valeur de cette colonne dans df_historique.
                  donnees_nouvelle_ligne_historique[col_hist_update] = df_historique[col_hist_update].iloc[-1]


        nouvelle_ligne_historique = pd.DataFrame([donnees_nouvelle_ligne_historique], columns=colonnes_df_historique)
        df_historique = pd.concat([df_historique.iloc[1:], nouvelle_ligne_historique], ignore_index=True)
        
        # Log occasionnel pour suivre la simulation
        if i_step < 5 or i_step == num_steps -1 or i_step % (max(1, num_steps // 10)) == 0:
             logger.debug(f"t={t_actuel:.2f}s, SP={sp_actuelle:.2f}, PV(entrée PID)={resultats_sim['PV'][-1]:.2f}, "
                          f"MV={mv_actuelle:.2f}, PV(prédite sortie modèle)={pv_actuelle:.2f}")

    logger.info(f"--- Simulation terminée pour le jeu : {nom_jeu_reglage} ---")
    return pd.DataFrame(resultats_sim)

# --- Calcul des Métriques de Performance ---
def calculate_performance_metrics(df_resultats, tsamp_s):
    """Calcule les métriques de performance à partir des résultats de simulation."""
    if df_resultats.empty or len(df_resultats) < 2 or df_resultats['PV'].isnull().all():
        logger.warning("DataFrame de résultats vide ou PV entièrement NaN. Métriques non calculables.")
        return {'IAE': np.nan, 'ISE': np.nan, 'ITAE': np.nan, 
                'Overshoot': np.nan, 'TempsStabilisation': np.nan, 'TempsMontee': np.nan}

    # Exclure les NaNs potentiels dus à un arrêt prématuré de la simulation
    df_resultats_valides = df_resultats.dropna(subset=['Error', 'PV', 'SP', 'Time'])
    if len(df_resultats_valides) < 2 :
        logger.warning("Pas assez de données valides après dropna pour calculer les métriques.")
        return {'IAE': np.nan, 'ISE': np.nan, 'ITAE': np.nan, 
                'Overshoot': np.nan, 'TempsStabilisation': np.nan, 'TempsMontee': np.nan}


    erreur = df_resultats_valides['Error']
    pv_vals = df_resultats_valides['PV']
    sp_vals = df_resultats_valides['SP']
    temps_vals = df_resultats_valides['Time']

    iae = np.sum(np.abs(erreur)) * tsamp_s
    ise = np.sum(erreur**2) * tsamp_s
    itae = np.sum(temps_vals * np.abs(erreur)) * tsamp_s
    
    metriques = {'IAE': iae, 'ISE': ise, 'ITAE': itae}

    # Métriques temporelles (simplifiées, supposent un échelon principal)
    # Identifier le dernier changement majeur de SP comme référence pour l'échelon
    sp_diff = sp_vals.diff().abs()
    major_step_time = 0 # Par défaut, échelon à t=0
    if sp_diff.max() > 1e-6: # S'il y a des changements de SP
        # Trouver l'index du dernier changement significatif de SP
        last_major_step_idx = sp_diff[sp_diff > 1e-6].index[-1] if not sp_diff[sp_diff > 1e-6].empty else 0
        # S'assurer que last_major_step_idx est un index numérique pour iloc
        iloc_idx = df_resultats_valides.index.get_loc(last_major_step_idx) if isinstance(last_major_step_idx, type(df_resultats_valides.index[0])) else last_major_step_idx

        if iloc_idx > 0 :
             sp_avant_echelon = sp_vals.iloc[iloc_idx-1]
             pv_avant_echelon = pv_vals.iloc[iloc_idx-1]
        else: # Echelon dès le début
             sp_avant_echelon = pv_vals.iloc[0] # On suppose que PV initiale = SP initiale avant l'échelon
             pv_avant_echelon = pv_vals.iloc[0]

        sp_apres_echelon = sp_vals.iloc[iloc_idx]
        
        # Analyser la réponse à partir de cet échelon
        pv_reponse = pv_vals.iloc[iloc_idx:]
        temps_reponse = temps_vals.iloc[iloc_idx:] - temps_vals.iloc[iloc_idx] # Temps relatif à l'échelon
        
        delta_sp = sp_apres_echelon - pv_avant_echelon # Amplitude de l'échelon par rapport à la PV d'origine
        
        if abs(delta_sp) > 1e-6 : # Si l'amplitude de l'échelon est significative
            if delta_sp > 0: # Échelon montant
                max_pv_apres_echelon = pv_reponse.max()
                overshoot_val = (max_pv_apres_echelon - sp_apres_echelon) / abs(delta_sp) * 100
                metriques['Overshoot'] = max(0, overshoot_val)
                
                try:
                    cible_10_pc = pv_avant_echelon + 0.1 * delta_sp
                    cible_90_pc = pv_avant_echelon + 0.9 * delta_sp
                    temps_10_pc = temps_reponse[pv_reponse >= cible_10_pc].iloc[0]
                    temps_90_pc = temps_reponse[pv_reponse >= cible_90_pc].iloc[0]
                    metriques['TempsMontee'] = temps_90_pc - temps_10_pc
                except IndexError:
                    metriques['TempsMontee'] = np.nan
            else: # Échelon descendant
                min_pv_apres_echelon = pv_reponse.min()
                undershoot_val = (sp_apres_echelon - min_pv_apres_echelon) / abs(delta_sp) * 100
                metriques['Overshoot'] = max(0, undershoot_val) # On stocke l'undershoot dans "Overshoot"
                # Temps de montée pour échelon descendant (temps de descente)
                try:
                    cible_10_pc_desc = pv_avant_echelon + 0.1 * delta_sp # Sera plus bas
                    cible_90_pc_desc = pv_avant_echelon + 0.9 * delta_sp # Sera plus haut que cible_10_pc_desc
                    # On cherche quand on passe de 90% à 10% de la valeur finale (par rapport à la descente)
                    temps_passage_90pc_valeur_init = temps_reponse[pv_reponse <= cible_90_pc_desc].iloc[0] # 10% de la descente effectuée
                    temps_passage_10pc_valeur_init = temps_reponse[pv_reponse <= cible_10_pc_desc].iloc[0] # 90% de la descente effectuée
                    metriques['TempsMontee'] = temps_passage_10pc_valeur_init - temps_passage_90pc_valeur_init

                except IndexError:
                     metriques['TempsMontee'] = np.nan


            # Temps de stabilisation : temps pour rester dans +/- 2% de sp_apres_echelon
            bande_sup_stab = sp_apres_echelon * 1.02
            bande_inf_stab = sp_apres_echelon * 0.98
            est_stabilise_mask = (pv_reponse >= bande_inf_stab) & (pv_reponse <= bande_sup_stab)
            
            temps_stabilisation_val = np.nan
            for k_stab in range(len(pv_reponse) -1, -1, -1): # Chercher à partir de la fin
                if not est_stabilise_mask.iloc[k_stab]:
                    if k_stab + 1 < len(temps_reponse):
                        temps_stabilisation_val = temps_reponse.iloc[k_stab+1]
                    else: # Instable jusqu'à la fin
                        temps_stabilisation_val = temps_reponse.iloc[-1] 
                    break
            if np.isnan(temps_stabilisation_val) and not est_stabilise_mask.empty and est_stabilise_mask.iloc[0]: # Si stable depuis le début de la réponse
                 temps_stabilisation_val = temps_reponse.iloc[0]
            metriques['TempsStabilisation'] = temps_stabilisation_val
        else: # Pas d'échelon significatif
            metriques['Overshoot'] = 0.0
            metriques['TempsMontee'] = 0.0
            metriques['TempsStabilisation'] = 0.0
            
    else: # Pas de changement de SP détecté après t=0
        metriques['Overshoot'] = np.nan
        metriques['TempsMontee'] = np.nan
        metriques['TempsStabilisation'] = np.nan

    for k, v in metriques.items(): logger.info(f"Métrique {k}: {v:.3f}" if isinstance(v, float) else f"Métrique {k}: {v}")
    return metriques

# --- Fonctions de Tracé des Résultats ---
def plot_simulation_results(tous_dfs_resultats, noms_jeux_reglage, config_sortie):
    """Trace les PV et MV pour toutes les simulations sur des graphiques combinés."""
    num_jeux = len(tous_dfs_resultats)
    if num_jeux == 0:
        logger.warning("Aucun résultat de simulation à tracer.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True) # Taille augmentée pour meilleure lisibilité
    
    # Utiliser un cycle de couleurs plus distinct si beaucoup de courbes
    # colors = plt.cm.get_cmap('tab10', num_jeux if num_jeux <=10 else 20) # 'viridis' est bien aussi
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']


    for i, df_resultats in enumerate(tous_dfs_resultats):
        if df_resultats.empty or df_resultats['PV'].isnull().all():
            logger.warning(f"Les résultats pour le jeu {noms_jeux_reglage[i]} sont vides ou PV est NaN, ignoré pour le tracé combiné.")
            continue
        
        nom_jeu = noms_jeux_reglage[i]
        # color = colors(i % colors.N if hasattr(colors, 'N') else i % len(colors) ) # Pour cmaps
        color = colors[i % len(colors)]


        axs[0].plot(df_resultats['Time'], df_resultats['PV'], label=f'PV ({nom_jeu})', color=color, linewidth=1.5)
        if i == 0: 
            axs[0].plot(df_resultats['Time'], df_resultats['SP'], 'k--', label='Consigne (SP)', alpha=0.8, linewidth=2)
        
        axs[1].plot(df_resultats['Time'], df_resultats['MV'], label=f'MV ({nom_jeu})', color=color, linewidth=1.5)

    axs[0].set_ylabel('Variable de Procédé (PV)')
    axs[0].legend(loc='best', fontsize='small')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_title('Comparaison des Réglages PID (Simulation en Boucle Fermée)')

    axs[1].set_ylabel('Variable Manipulée (MV)')
    axs[1].set_xlabel('Temps (secondes)')
    axs[1].legend(loc='best', fontsize='small')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Ajuster pour le titre global si besoin
    
    plot_save_path_str = config_sortie.get('results_plot_path')
    if plot_save_path_str:
        path_obj = Path(plot_save_path_str).resolve()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(path_obj, dpi=150) # Augmenter DPI pour meilleure qualité
            logger.info(f"Graphique combiné des résultats sauvegardé : {path_obj}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du graphique combiné : {e}", exc_info=True)
    
    try:
        plt.show()
    except Exception as e: # Peut échouer dans les environnements sans GUI (ex: scripts batch)
        logger.warning(f"Impossible d'afficher le graphique combiné (plt.show()): {e}")
    plt.close(fig)


def plot_individual_run(df_resultats, nom_jeu, config_sortie):
    """Trace PV, SP, MV pour une seule simulation."""
    if df_resultats.empty or df_resultats['PV'].isnull().all():
        logger.warning(f"Résultats vides ou PV NaN pour {nom_jeu}, graphique individuel non généré.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axs[0].plot(df_resultats['Time'], df_resultats['PV'], label='PV', linewidth=1.5)
    axs[0].plot(df_resultats['Time'], df_resultats['SP'], 'k--', label='SP', linewidth=1.5)
    axs[0].set_ylabel('Variable de Procédé (PV) / Consigne (SP)')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_title(f'Simulation : {nom_jeu}')

    axs[1].plot(df_resultats['Time'], df_resultats['MV'], label='MV', color='darkorange', linewidth=1.5)
    axs[1].set_ylabel('Variable Manipulée (MV)')
    axs[1].set_xlabel('Temps (secondes)')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_dir_str = config_sortie.get('individual_runs_plot_dir')
    if plot_dir_str:
        dir_path = Path(plot_dir_str).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        nom_fichier_safe = "".join(c if c.isalnum() or c in ['_','-'] else "_" for c in nom_jeu)
        plot_path = dir_path / f"run_{nom_fichier_safe}.png"
        try:
            plt.savefig(plot_path, dpi=120)
            logger.info(f"Graphique de la simulation individuelle sauvegardé : {plot_path}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde graphique individuel pour {nom_jeu}: {e}", exc_info=True)
    else: # Afficher si non sauvegardé dans un répertoire dédié
        try: plt.show()
        except Exception as e: logger.warning(f"Impossible d'afficher le graphique individuel {nom_jeu}: {e}")
    
    plt.close(fig) # Fermer pour libérer la mémoire, surtout si beaucoup de graphiques


# --- Exécution Principale ---
def main():
    config_file = f"{NOM_SCRIPT_SANS_EXTENSION}.ini" # Ex: pid_tuner.ini
    config = load_config_and_setup_logging(config_file)

    try:
        modele_procede, scaler_X, scaler_y = load_process_model_and_scalers(config['ModeleProcede'])
        
        noms_features_modele = []
        if hasattr(scaler_X, 'feature_names_in_'):
            noms_features_modele = scaler_X.feature_names_in_.tolist()
            logger.info(f"Noms des features du modèle récupérés depuis scaler_X : {noms_features_modele}")
        else:
            logger.warning("scaler_X n'a pas d'attribut 'feature_names_in_'. "
                           "Utilisation de la génération depuis la config (moins robuste). "
                           "Assurez-vous que l'ordre correspond à l'entraînement !")
            noms_features_modele = get_model_feature_names_from_config(config['ModeleProcede'])

        if not noms_features_modele:
             logger.error("Impossible de déterminer les noms des features du modèle. Arrêt.")
             return

        jeux_de_reglage = []
        if config.has_section('JeuxDeReglagePID'):
            for cle_jeu, str_val_jeu in config.items('JeuxDeReglagePID'):
                parts = [p.strip() for p in str_val_jeu.split(',')]
                try:
                    kp_val = float(parts[0])
                    # Gérer Ti = "inf" ou un nombre. float('inf') si non convertible en float directement.
                    try: ti_val = float(parts[1])
                    except ValueError:
                        if parts[1].lower() == 'inf': ti_val = float('inf')
                        else: raise
                    td_val = float(parts[2])
                    nom_jeu_val = parts[3] if len(parts) > 3 else cle_jeu
                    jeux_de_reglage.append({'nom': nom_jeu_val, 'Kp': kp_val, 'Ti': ti_val, 'Td': td_val})
                except (ValueError, IndexError) as e:
                    logger.error(f"Impossible de parser JeuxDeReglagePID '{cle_jeu}': '{str_val_jeu}'. Erreur: {e}. Ignoré.")
        
        if not jeux_de_reglage:
            logger.error("Aucun jeu de réglage PID valide trouvé dans la configuration. Arrêt.")
            return

        tous_dfs_resultats_simulation = []
        resume_toutes_metriques = []
        noms_jeux_pour_plot_global = []

        for jeu_pid in jeux_de_reglage:
            df_resultats = run_closed_loop_simulation(
                config, jeu_pid['nom'], jeu_pid['Kp'], jeu_pid['Ti'], jeu_pid['Td'],
                modele_procede, scaler_X, scaler_y, noms_features_modele
            )
            tous_dfs_resultats_simulation.append(df_resultats)
            noms_jeux_pour_plot_global.append(jeu_pid['nom'])

            if not df_resultats.empty and not df_resultats['PV'].isnull().all():
                tsamp_s_main = config.getfloat('ParametresPIDBase', 'tsamp_pid_sim_ms') / 1000.0
                metriques = calculate_performance_metrics(df_resultats, tsamp_s_main)
                metriques['nom_jeu'] = jeu_pid['nom']
                metriques['Kp'] = jeu_pid['Kp']
                metriques['Ti'] = jeu_pid['Ti']
                metriques['Td'] = jeu_pid['Td']
                resume_toutes_metriques.append(metriques)
                
                if config.getboolean('Sortie', 'plot_individual_runs', fallback=False):
                    plot_individual_run(df_resultats, jeu_pid['nom'], config['Sortie'])
            else:
                logger.warning(f"La simulation pour {jeu_pid['nom']} a retourné un DataFrame vide ou PV entièrement NaN. "
                               "Aucune métrique calculée.")
                # Ajouter une ligne avec NaNs pour ce jeu dans le résumé des métriques
                # pour maintenir la cohérence si un rapport CSV est généré.
                nan_metrics = {'nom_jeu': jeu_pid['nom'], 'Kp': jeu_pid['Kp'], 'Ti': jeu_pid['Ti'], 'Td': jeu_pid['Td'],
                               'IAE': np.nan, 'ISE': np.nan, 'ITAE': np.nan, 'Overshoot': np.nan, 
                               'TempsStabilisation': np.nan, 'TempsMontee': np.nan}
                resume_toutes_metriques.append(nan_metrics)


        plot_simulation_results(tous_dfs_resultats_simulation, noms_jeux_pour_plot_global, config['Sortie'])

        if resume_toutes_metriques:
            df_metriques = pd.DataFrame(resume_toutes_metriques)
            cols_metriques_ordre = ['nom_jeu', 'Kp', 'Ti', 'Td', 'IAE', 'ISE', 'ITAE', 'Overshoot', 'TempsMontee', 'TempsStabilisation']
            # S'assurer que les colonnes existent avant de réindexer pour éviter les KeyErrors
            cols_metriques_presentes = [col for col in cols_metriques_ordre if col in df_metriques.columns]
            df_metriques = df_metriques.reindex(columns=cols_metriques_presentes)


            path_csv_metriques_str = config.get('Sortie', 'results_metrics_csv')
            if path_csv_metriques_str:
                path_obj_csv = Path(path_csv_metriques_str).resolve()
                path_obj_csv.parent.mkdir(parents=True, exist_ok=True)
                try:
                    df_metriques.to_csv(path_obj_csv, index=False, float_format='%.3f', sep =';')
                    logger.info(f"Résumé des métriques sauvegardé : {path_obj_csv}")
                except Exception as e:
                    logger.error(f"Erreur sauvegarde CSV des métriques : {e}", exc_info=True)
            print("\nRésumé des Métriques :")
            print(df_metriques.to_string(float_format="%.3f"))

    except FileNotFoundError as e_fnf:
        logger.critical(f"Un fichier requis n'a pas été trouvé : {e_fnf}", exc_info=True)
    except ValueError as e_val:
        logger.critical(f"Une erreur de valeur s'est produite (souvent liée à la configuration) : {e_val}", exc_info=True)
    except configparser.Error as e_cfg_parser:
        logger.critical(f"Erreur lors de l'analyse du fichier de configuration : {e_cfg_parser}", exc_info=True)
    except Exception as e_globale: # Attraper les autres exceptions non prévues
        logger.critical(f"Une erreur inattendue est survenue dans l'exécution principale : {e_globale}", exc_info=True)
    finally:
        logger.info(f"--- {NOM_SCRIPT_SANS_EXTENSION}.py terminé ---")
        winsound.Beep(1000, 500)
if __name__ == "__main__":
    main()


# In[ ]:




