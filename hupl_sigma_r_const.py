#!/usr/local/bin/env python3

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


def preprocess_features_cold(task_dataframe):
  """Prepares input features of the cold dataset.

  Args:
    task_dataframe: A Pandas DataFrame expected to contain cold-started data.
  Returns:
    A DataFrame that contains the features to be used for the model.
  """

  # upper-limb information
  elbow_pos_features = task_dataframe[["elbow_x_mm","elbow_y_mm","elbow_z_mm"]]
  wrist_pos_features = task_dataframe[["wrist_x_mm", "wrist_y_mm", "wrist_z_mm"]]
  hand_pos_features = task_dataframe[["hand_x_mm", "hand_y_mm", "hand_z_mm"]]
  thumb_1_pos_features = task_dataframe[["thumb_1_x_mm", "thumb_1_y_mm", "thumb_1_z_mm"]]
  thumb_2_pos_features = task_dataframe[["thumb_2_x_mm", "thumb_2_y_mm", "thumb_2_z_mm"]]
  thumb_tip_pos_features = task_dataframe[["thumb_tip_x_mm", "thumb_tip_y_mm", "thumb_tip_z_mm"]]
  index_1_pos_features = task_dataframe[["index_1_x_mm", "index_1_y_mm", "index_1_z_mm"]]
  index_2_pos_features = task_dataframe[["index_2_x_mm", "index_2_y_mm", "index_2_z_mm"]]
  index_tip_pos_features = task_dataframe[["index_tip_x_mm", "index_tip_y_mm", "index_tip_z_mm"]]
  middle_1_pos_features = task_dataframe[["middle_1_x_mm", "middle_1_y_mm", "middle_1_z_mm"]]
  middle_2_pos_features = task_dataframe[["middle_2_x_mm", "middle_2_y_mm", "middle_2_z_mm"]]
  middle_tip_pos_features = task_dataframe[["middle_tip_x_mm", "middle_tip_y_mm", "middle_tip_z_mm"]]

  # target information
  target_pos_features = task_dataframe[["target_x_mm", "target_y_mm", "target_z_mm"]]
  target_or_features = task_dataframe[["target_roll_rad", "target_pitch_rad", "target_yaw_rad"]]

  # obstacle 1 information
  obstacle_1_pos_features = task_dataframe[["obstacle_1_x_mm", "obstacle_1_y_mm", "obstacle_1_z_mm"]]
  obstacle_1_or_features = task_dataframe[["obstacle_1_roll_rad", "obstacle_1_pitch_rad", "obstacle_1_yaw_rad"]]

  # --- elbow --- #
  # elbow - target
  elb_tar_diff = pd.DataFrame()
  elb_tar_diff['elb_tar_x'] = elbow_pos_features['elbow_x_mm'] - target_pos_features['target_x_mm']
  elb_tar_diff['elb_tar_y'] = elbow_pos_features['elbow_y_mm'] - target_pos_features['target_y_mm']
  elb_tar_diff['elb_tar_z'] = elbow_pos_features['elbow_z_mm'] - target_pos_features['target_z_mm']
  elb_tar_diff['elb_tar_d'] = np.sqrt(elb_tar_diff['elb_tar_x']**2+elb_tar_diff['elb_tar_y']**2+elb_tar_diff['elb_tar_z']**2)
  elb_tar_diff['elb_tar_psi_x'] = np.arccos(elb_tar_diff['elb_tar_x']/elb_tar_diff['elb_tar_d'])
  elb_tar_diff['elb_tar_psi_y'] = np.arccos(elb_tar_diff['elb_tar_y'] / elb_tar_diff['elb_tar_d'])
  elb_tar_diff['elb_tar_psi_z'] = np.arccos(elb_tar_diff['elb_tar_z'] / elb_tar_diff['elb_tar_d'])
  # elbow - obstacle 1
  elb_obst_1_diff = pd.DataFrame()
  elb_obst_1_diff['elb_obst_1_x'] = elbow_pos_features['elbow_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  elb_obst_1_diff['elb_obst_1_y'] = elbow_pos_features['elbow_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  elb_obst_1_diff['elb_obst_1_z'] = elbow_pos_features['elbow_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  elb_obst_1_diff['elb_obst_1_d'] = np.sqrt(elb_obst_1_diff['elb_obst_1_x']**2+elb_obst_1_diff['elb_obst_1_y']**2+elb_obst_1_diff['elb_obst_1_z']**2)
  elb_obst_1_diff['elb_obst_1_psi_x'] = np.arccos(elb_obst_1_diff['elb_obst_1_x'] / elb_obst_1_diff['elb_obst_1_d'])
  elb_obst_1_diff['elb_obst_1_psi_y'] = np.arccos(elb_obst_1_diff['elb_obst_1_y'] / elb_obst_1_diff['elb_obst_1_d'])
  elb_obst_1_diff['elb_obst_1_psi_z'] = np.arccos(elb_obst_1_diff['elb_obst_1_z'] / elb_obst_1_diff['elb_obst_1_d'])

  # --- wrist --- #
  # wrist - target
  wri_tar_diff = pd.DataFrame()
  wri_tar_diff['wri_tar_x'] = wrist_pos_features['wrist_x_mm'] - target_pos_features['target_x_mm']
  wri_tar_diff['wri_tar_y'] = wrist_pos_features['wrist_y_mm'] - target_pos_features['target_y_mm']
  wri_tar_diff['wri_tar_z'] = wrist_pos_features['wrist_z_mm'] - target_pos_features['target_z_mm']
  wri_tar_diff['wri_tar_d'] = np.sqrt(wri_tar_diff['wri_tar_x']**2+wri_tar_diff['wri_tar_y']**2+wri_tar_diff['wri_tar_z']**2)
  wri_tar_diff['wri_tar_psi_x'] = np.arccos(wri_tar_diff['wri_tar_x'] / wri_tar_diff['wri_tar_d'])
  wri_tar_diff['wri_tar_psi_y'] = np.arccos(wri_tar_diff['wri_tar_y'] / wri_tar_diff['wri_tar_d'])
  wri_tar_diff['wri_tar_psi_z'] = np.arccos(wri_tar_diff['wri_tar_z'] / wri_tar_diff['wri_tar_d'])
  # wrist - obstacle 1
  wri_obst_1_diff = pd.DataFrame()
  wri_obst_1_diff['wri_obst_1_x'] = wrist_pos_features['wrist_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  wri_obst_1_diff['wri_obst_1_y'] = wrist_pos_features['wrist_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  wri_obst_1_diff['wri_obst_1_z'] = wrist_pos_features['wrist_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  wri_obst_1_diff['wri_obst_1_d'] = np.sqrt(wri_obst_1_diff['wri_obst_1_x']**2+wri_obst_1_diff['wri_obst_1_y']**2+wri_obst_1_diff['wri_obst_1_z']**2)
  wri_obst_1_diff['wri_obst_1_psi_x'] = np.arccos(wri_obst_1_diff['wri_obst_1_x'] / wri_obst_1_diff['wri_obst_1_d'])
  wri_obst_1_diff['wri_obst_1_psi_y'] = np.arccos(wri_obst_1_diff['wri_obst_1_y'] / wri_obst_1_diff['wri_obst_1_d'])
  wri_obst_1_diff['wri_obst_1_psi_z'] = np.arccos(wri_obst_1_diff['wri_obst_1_z'] / wri_obst_1_diff['wri_obst_1_d'])

  # --- hand --- #
  # hand - target
  hand_tar_diff = pd.DataFrame()
  hand_tar_diff['hand_tar_x'] = hand_pos_features['hand_x_mm'] - target_pos_features['target_x_mm']
  hand_tar_diff['hand_tar_y'] = hand_pos_features['hand_y_mm'] - target_pos_features['target_y_mm']
  hand_tar_diff['hand_tar_z'] = hand_pos_features['hand_z_mm'] - target_pos_features['target_z_mm']
  hand_tar_diff['hand_tar_d'] = np.sqrt(hand_tar_diff['hand_tar_x']**2+hand_tar_diff['hand_tar_y']**2+hand_tar_diff['hand_tar_z']**2)
  hand_tar_diff['hand_tar_psi_x'] = np.arccos(hand_tar_diff['hand_tar_x'] / hand_tar_diff['hand_tar_d'])
  hand_tar_diff['hand_tar_psi_y'] = np.arccos(hand_tar_diff['hand_tar_y'] / hand_tar_diff['hand_tar_d'])
  hand_tar_diff['hand_tar_psi_z'] = np.arccos(hand_tar_diff['hand_tar_z'] / hand_tar_diff['hand_tar_d'])
  # hand - obstacle 1
  hand_obst_1_diff = pd.DataFrame()
  hand_obst_1_diff['hand_obst_1_x'] = hand_pos_features['hand_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  hand_obst_1_diff['hand_obst_1_y'] = hand_pos_features['hand_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  hand_obst_1_diff['hand_obst_1_z'] = hand_pos_features['hand_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  hand_obst_1_diff['hand_obst_1_d'] = np.sqrt(hand_obst_1_diff['hand_obst_1_x']**2+hand_obst_1_diff['hand_obst_1_y']**2+hand_obst_1_diff['hand_obst_1_z']**2)
  hand_obst_1_diff['hand_obst_1_psi_x'] = np.arccos(hand_obst_1_diff['hand_obst_1_x'] / hand_obst_1_diff['hand_obst_1_d'])
  hand_obst_1_diff['hand_obst_1_psi_y'] = np.arccos(hand_obst_1_diff['hand_obst_1_y'] / hand_obst_1_diff['hand_obst_1_d'])
  hand_obst_1_diff['hand_obst_1_psi_z'] = np.arccos(hand_obst_1_diff['hand_obst_1_z'] / hand_obst_1_diff['hand_obst_1_d'])

  # --- thumb 1 --- #
  # thumb 1 - target
  thumb_1_tar_diff = pd.DataFrame()
  thumb_1_tar_diff['thumb_1_tar_x'] = thumb_1_pos_features['thumb_1_x_mm'] - target_pos_features['target_x_mm']
  thumb_1_tar_diff['thumb_1_tar_y'] = thumb_1_pos_features['thumb_1_y_mm'] - target_pos_features['target_y_mm']
  thumb_1_tar_diff['thumb_1_tar_z'] = thumb_1_pos_features['thumb_1_z_mm'] - target_pos_features['target_z_mm']
  thumb_1_tar_diff['thumb_1_tar_d'] = np.sqrt(thumb_1_tar_diff['thumb_1_tar_x']**2+thumb_1_tar_diff['thumb_1_tar_y']**2+thumb_1_tar_diff['thumb_1_tar_z']**2)
  thumb_1_tar_diff['thumb_1_tar_psi_x'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_x'] / thumb_1_tar_diff['thumb_1_tar_d'])
  thumb_1_tar_diff['thumb_1_tar_psi_y'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_y'] / thumb_1_tar_diff['thumb_1_tar_d'])
  thumb_1_tar_diff['thumb_1_tar_psi_z'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_z'] / thumb_1_tar_diff['thumb_1_tar_d'])
  # thumb 1 - obstacle 1
  thumb_1_obst_1_diff = pd.DataFrame()
  thumb_1_obst_1_diff['thumb_1_obst_1_x'] = thumb_1_pos_features['thumb_1_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_y'] = thumb_1_pos_features['thumb_1_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_z'] = thumb_1_pos_features['thumb_1_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_d'] = np.sqrt(thumb_1_obst_1_diff['thumb_1_obst_1_x']**2+thumb_1_obst_1_diff['thumb_1_obst_1_y']**2+thumb_1_obst_1_diff['thumb_1_obst_1_z']**2)
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_x'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_x'] / thumb_1_obst_1_diff['thumb_1_obst_1_d'])
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_y'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_y'] / thumb_1_obst_1_diff['thumb_1_obst_1_d'])
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_z'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_z'] / thumb_1_obst_1_diff['thumb_1_obst_1_d'])

  # --- thumb 2 --- #
  # thumb 2 - target
  thumb_2_tar_diff = pd.DataFrame()
  thumb_2_tar_diff['thumb_2_tar_x'] = thumb_2_pos_features['thumb_2_x_mm'] - target_pos_features['target_x_mm']
  thumb_2_tar_diff['thumb_2_tar_y'] = thumb_2_pos_features['thumb_2_y_mm'] - target_pos_features['target_y_mm']
  thumb_2_tar_diff['thumb_2_tar_z'] = thumb_2_pos_features['thumb_2_z_mm'] - target_pos_features['target_z_mm']
  thumb_2_tar_diff['thumb_2_tar_d'] = np.sqrt(thumb_2_tar_diff['thumb_2_tar_x']**2+thumb_2_tar_diff['thumb_2_tar_y']**2+thumb_2_tar_diff['thumb_2_tar_z']**2)
  thumb_2_tar_diff['thumb_2_tar_psi_x'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_x'] / thumb_2_tar_diff['thumb_2_tar_d'])
  thumb_2_tar_diff['thumb_2_tar_psi_y'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_y'] / thumb_2_tar_diff['thumb_2_tar_d'])
  thumb_2_tar_diff['thumb_2_tar_psi_z'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_z'] / thumb_2_tar_diff['thumb_2_tar_d'])
  # thumb 2 - obstacle 1
  thumb_2_obst_1_diff = pd.DataFrame()
  thumb_2_obst_1_diff['thumb_2_obst_1_x'] = thumb_2_pos_features['thumb_2_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_y'] = thumb_2_pos_features['thumb_2_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_z'] = thumb_2_pos_features['thumb_2_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_d'] = np.sqrt(thumb_2_obst_1_diff['thumb_2_obst_1_x']**2+thumb_2_obst_1_diff['thumb_2_obst_1_y']**2+thumb_2_obst_1_diff['thumb_2_obst_1_z']**2)
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_x'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_x'] / thumb_2_obst_1_diff['thumb_2_obst_1_d'])
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_y'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_y'] / thumb_2_obst_1_diff['thumb_2_obst_1_d'])
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_z'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_z'] / thumb_2_obst_1_diff['thumb_2_obst_1_d'])

  # --- thumb tip --- #
  # thumb tip - target
  thumb_tip_tar_diff = pd.DataFrame()
  thumb_tip_tar_diff['thumb_tip_tar_x'] = thumb_tip_pos_features['thumb_tip_x_mm'] - target_pos_features['target_x_mm']
  thumb_tip_tar_diff['thumb_tip_tar_y'] = thumb_tip_pos_features['thumb_tip_y_mm'] - target_pos_features['target_y_mm']
  thumb_tip_tar_diff['thumb_tip_tar_z'] = thumb_tip_pos_features['thumb_tip_z_mm'] - target_pos_features['target_z_mm']
  thumb_tip_tar_diff['thumb_tip_tar_d'] = np.sqrt(thumb_tip_tar_diff['thumb_tip_tar_x']**2+thumb_tip_tar_diff['thumb_tip_tar_y']**2+thumb_tip_tar_diff['thumb_tip_tar_z']**2)
  thumb_tip_tar_diff['thumb_tip_tar_psi_x'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_x'] / thumb_tip_tar_diff['thumb_tip_tar_d'])
  thumb_tip_tar_diff['thumb_tip_tar_psi_y'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_y'] / thumb_tip_tar_diff['thumb_tip_tar_d'])
  thumb_tip_tar_diff['thumb_tip_tar_psi_z'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_z'] / thumb_tip_tar_diff['thumb_tip_tar_d'])
  # thumb tip - obstacle 1
  thumb_tip_obst_1_diff = pd.DataFrame()
  thumb_tip_obst_1_diff['thumb_tip_obst_1_x'] = thumb_tip_pos_features['thumb_tip_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_y'] = thumb_tip_pos_features['thumb_tip_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_z'] = thumb_tip_pos_features['thumb_tip_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_d'] = np.sqrt(thumb_tip_obst_1_diff['thumb_tip_obst_1_x']**2+thumb_tip_obst_1_diff['thumb_tip_obst_1_y']**2+thumb_tip_obst_1_diff['thumb_tip_obst_1_z']**2)
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_x'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_x'] / thumb_tip_obst_1_diff['thumb_tip_obst_1_d'])
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_y'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_y'] / thumb_tip_obst_1_diff['thumb_tip_obst_1_d'])
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_z'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_z'] / thumb_tip_obst_1_diff['thumb_tip_obst_1_d'])

  # --- index 1 --- #
  # index 1 - target
  index_1_tar_diff = pd.DataFrame()
  index_1_tar_diff['index_1_tar_x'] = index_1_pos_features['index_1_x_mm'] - target_pos_features['target_x_mm']
  index_1_tar_diff['index_1_tar_y'] = index_1_pos_features['index_1_y_mm'] - target_pos_features['target_y_mm']
  index_1_tar_diff['index_1_tar_z'] = index_1_pos_features['index_1_z_mm'] - target_pos_features['target_z_mm']
  index_1_tar_diff['index_1_tar_d'] = np.sqrt(index_1_tar_diff['index_1_tar_x']**2+index_1_tar_diff['index_1_tar_y']**2+index_1_tar_diff['index_1_tar_z']**2)
  index_1_tar_diff['index_1_tar_psi_x'] = np.arccos(index_1_tar_diff['index_1_tar_x'] / index_1_tar_diff['index_1_tar_d'])
  index_1_tar_diff['index_1_tar_psi_y'] = np.arccos(index_1_tar_diff['index_1_tar_y'] / index_1_tar_diff['index_1_tar_d'])
  index_1_tar_diff['index_1_tar_psi_z'] = np.arccos(index_1_tar_diff['index_1_tar_z'] / index_1_tar_diff['index_1_tar_d'])
  # index 1 - obstacle 1
  index_1_obst_1_diff = pd.DataFrame()
  index_1_obst_1_diff['index_1_obst_1_x'] = index_1_pos_features['index_1_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  index_1_obst_1_diff['index_1_obst_1_y'] = index_1_pos_features['index_1_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  index_1_obst_1_diff['index_1_obst_1_z'] = index_1_pos_features['index_1_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  index_1_obst_1_diff['index_1_obst_1_d'] = np.sqrt(index_1_obst_1_diff['index_1_obst_1_x']**2+index_1_obst_1_diff['index_1_obst_1_y']**2+index_1_obst_1_diff['index_1_obst_1_z']**2)
  index_1_obst_1_diff['index_1_obst_1_psi_x'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_x'] / index_1_obst_1_diff['index_1_obst_1_d'])
  index_1_obst_1_diff['index_1_obst_1_psi_y'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_y'] / index_1_obst_1_diff['index_1_obst_1_d'])
  index_1_obst_1_diff['index_1_obst_1_psi_z'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_z'] / index_1_obst_1_diff['index_1_obst_1_d'])

  # --- index 2 --- #
  # index 2 - target
  index_2_tar_diff = pd.DataFrame()
  index_2_tar_diff['index_2_tar_x'] = index_2_pos_features['index_2_x_mm'] - target_pos_features['target_x_mm']
  index_2_tar_diff['index_2_tar_y'] = index_2_pos_features['index_2_y_mm'] - target_pos_features['target_y_mm']
  index_2_tar_diff['index_2_tar_z'] = index_2_pos_features['index_2_z_mm'] - target_pos_features['target_z_mm']
  index_2_tar_diff['index_2_tar_d'] = np.sqrt(index_2_tar_diff['index_2_tar_x']**2+index_2_tar_diff['index_2_tar_y']**2+index_2_tar_diff['index_2_tar_z']**2)
  index_2_tar_diff['index_2_tar_psi_x'] = np.arccos(index_2_tar_diff['index_2_tar_x'] / index_2_tar_diff['index_2_tar_d'])
  index_2_tar_diff['index_2_tar_psi_y'] = np.arccos(index_2_tar_diff['index_2_tar_y'] / index_2_tar_diff['index_2_tar_d'])
  index_2_tar_diff['index_2_tar_psi_z'] = np.arccos(index_2_tar_diff['index_2_tar_z'] / index_2_tar_diff['index_2_tar_d'])
  # index 2 - obstacle 1
  index_2_obst_1_diff = pd.DataFrame()
  index_2_obst_1_diff['index_2_obst_1_x'] = index_2_pos_features['index_2_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  index_2_obst_1_diff['index_2_obst_1_y'] = index_2_pos_features['index_2_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  index_2_obst_1_diff['index_2_obst_1_z'] = index_2_pos_features['index_2_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  index_2_obst_1_diff['index_2_obst_1_d'] = np.sqrt(index_2_obst_1_diff['index_2_obst_1_x']**2+index_2_obst_1_diff['index_2_obst_1_y']**2+index_2_obst_1_diff['index_2_obst_1_z']**2)
  index_2_obst_1_diff['index_2_obst_1_psi_x'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_x'] / index_2_obst_1_diff['index_2_obst_1_d'])
  index_2_obst_1_diff['index_2_obst_1_psi_y'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_y'] / index_2_obst_1_diff['index_2_obst_1_d'])
  index_2_obst_1_diff['index_2_obst_1_psi_z'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_z'] / index_2_obst_1_diff['index_2_obst_1_d'])

  # --- index tip --- #
  # index tip - target
  index_tip_tar_diff = pd.DataFrame()
  index_tip_tar_diff['index_tip_tar_x'] = index_tip_pos_features['index_tip_x_mm'] - target_pos_features['target_x_mm']
  index_tip_tar_diff['index_tip_tar_y'] = index_tip_pos_features['index_tip_y_mm'] - target_pos_features['target_y_mm']
  index_tip_tar_diff['index_tip_tar_z'] = index_tip_pos_features['index_tip_z_mm'] - target_pos_features['target_z_mm']
  index_tip_tar_diff['index_tip_tar_d'] = np.sqrt(index_tip_tar_diff['index_tip_tar_x']**2+index_tip_tar_diff['index_tip_tar_y']**2+index_tip_tar_diff['index_tip_tar_z']**2)
  index_tip_tar_diff['index_tip_tar_psi_x'] = np.arccos(index_tip_tar_diff['index_tip_tar_x'] / index_tip_tar_diff['index_tip_tar_d'])
  index_tip_tar_diff['index_tip_tar_psi_y'] = np.arccos(index_tip_tar_diff['index_tip_tar_y'] / index_tip_tar_diff['index_tip_tar_d'])
  index_tip_tar_diff['index_tip_tar_psi_z'] = np.arccos(index_tip_tar_diff['index_tip_tar_z'] / index_tip_tar_diff['index_tip_tar_d'])
  # index tip - obstacle 1
  index_tip_obst_1_diff = pd.DataFrame()
  index_tip_obst_1_diff['index_tip_obst_1_x'] = index_tip_pos_features['index_tip_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  index_tip_obst_1_diff['index_tip_obst_1_y'] = index_tip_pos_features['index_tip_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  index_tip_obst_1_diff['index_tip_obst_1_z'] = index_tip_pos_features['index_tip_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  index_tip_obst_1_diff['index_tip_obst_1_d'] = np.sqrt(index_tip_obst_1_diff['index_tip_obst_1_x']**2+index_tip_obst_1_diff['index_tip_obst_1_y']**2+index_tip_obst_1_diff['index_tip_obst_1_z']**2)
  index_tip_obst_1_diff['index_tip_obst_1_psi_x'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_x'] / index_tip_obst_1_diff['index_tip_obst_1_d'])
  index_tip_obst_1_diff['index_tip_obst_1_psi_y'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_y'] / index_tip_obst_1_diff['index_tip_obst_1_d'])
  index_tip_obst_1_diff['index_tip_obst_1_psi_z'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_z'] / index_tip_obst_1_diff['index_tip_obst_1_d'])

  # --- middle 1 --- #
  # middle 1 - target
  middle_1_tar_diff = pd.DataFrame()
  middle_1_tar_diff['middle_1_tar_x'] = middle_1_pos_features['middle_1_x_mm'] - target_pos_features['target_x_mm']
  middle_1_tar_diff['middle_1_tar_y'] = middle_1_pos_features['middle_1_y_mm'] - target_pos_features['target_y_mm']
  middle_1_tar_diff['middle_1_tar_z'] = middle_1_pos_features['middle_1_z_mm'] - target_pos_features['target_z_mm']
  middle_1_tar_diff['middle_1_tar_d'] = np.sqrt(middle_1_tar_diff['middle_1_tar_x']**2+middle_1_tar_diff['middle_1_tar_y']**2+middle_1_tar_diff['middle_1_tar_z']**2)
  middle_1_tar_diff['middle_1_tar_psi_x'] = np.arccos(middle_1_tar_diff['middle_1_tar_x'] / middle_1_tar_diff['middle_1_tar_d'])
  middle_1_tar_diff['middle_1_tar_psi_y'] = np.arccos(middle_1_tar_diff['middle_1_tar_y'] / middle_1_tar_diff['middle_1_tar_d'])
  middle_1_tar_diff['middle_1_tar_psi_z'] = np.arccos(middle_1_tar_diff['middle_1_tar_z'] / middle_1_tar_diff['middle_1_tar_d'])
  # middle 1 - obstacle 1
  middle_1_obst_1_diff = pd.DataFrame()
  middle_1_obst_1_diff['middle_1_obst_1_x'] = middle_1_pos_features['middle_1_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  middle_1_obst_1_diff['middle_1_obst_1_y'] = middle_1_pos_features['middle_1_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  middle_1_obst_1_diff['middle_1_obst_1_z'] = middle_1_pos_features['middle_1_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  middle_1_obst_1_diff['middle_1_obst_1_d'] = np.sqrt(middle_1_obst_1_diff['middle_1_obst_1_x']**2+middle_1_obst_1_diff['middle_1_obst_1_y']**2+middle_1_obst_1_diff['middle_1_obst_1_z']**2)
  middle_1_obst_1_diff['middle_1_obst_1_psi_x'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_x'] / middle_1_obst_1_diff['middle_1_obst_1_d'])
  middle_1_obst_1_diff['middle_1_obst_1_psi_y'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_y'] / middle_1_obst_1_diff['middle_1_obst_1_d'])
  middle_1_obst_1_diff['middle_1_obst_1_psi_z'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_z'] / middle_1_obst_1_diff['middle_1_obst_1_d'])

  # --- middle 2 --- #
  # middle 2 - target
  middle_2_tar_diff = pd.DataFrame()
  middle_2_tar_diff['middle_2_tar_x'] = middle_2_pos_features['middle_2_x_mm'] - target_pos_features['target_x_mm']
  middle_2_tar_diff['middle_2_tar_y'] = middle_2_pos_features['middle_2_y_mm'] - target_pos_features['target_y_mm']
  middle_2_tar_diff['middle_2_tar_z'] = middle_2_pos_features['middle_2_z_mm'] - target_pos_features['target_z_mm']
  middle_2_tar_diff['middle_2_tar_d'] = np.sqrt(middle_2_tar_diff['middle_2_tar_x']**2+middle_2_tar_diff['middle_2_tar_y']**2+middle_2_tar_diff['middle_2_tar_z']**2)
  middle_2_tar_diff['middle_2_tar_psi_x'] = np.arccos(middle_2_tar_diff['middle_2_tar_x'] / middle_2_tar_diff['middle_2_tar_d'])
  middle_2_tar_diff['middle_2_tar_psi_y'] = np.arccos(middle_2_tar_diff['middle_2_tar_y'] / middle_2_tar_diff['middle_2_tar_d'])
  middle_2_tar_diff['middle_2_tar_psi_z'] = np.arccos(middle_2_tar_diff['middle_2_tar_z'] / middle_2_tar_diff['middle_2_tar_d'])
  # middle 2 - obstacle 1
  middle_2_obst_1_diff = pd.DataFrame()
  middle_2_obst_1_diff['middle_2_obst_1_x'] = middle_2_pos_features['middle_2_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  middle_2_obst_1_diff['middle_2_obst_1_y'] = middle_2_pos_features['middle_2_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  middle_2_obst_1_diff['middle_2_obst_1_z'] = middle_2_pos_features['middle_2_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  middle_2_obst_1_diff['middle_2_obst_1_d'] = np.sqrt(middle_2_obst_1_diff['middle_2_obst_1_x']**2+middle_2_obst_1_diff['middle_2_obst_1_y']**2+middle_2_obst_1_diff['middle_2_obst_1_z']**2)
  middle_2_obst_1_diff['middle_2_obst_1_psi_x'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_x'] / middle_2_obst_1_diff['middle_2_obst_1_d'])
  middle_2_obst_1_diff['middle_2_obst_1_psi_y'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_y'] / middle_2_obst_1_diff['middle_2_obst_1_d'])
  middle_2_obst_1_diff['middle_2_obst_1_psi_z'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_z'] / middle_2_obst_1_diff['middle_2_obst_1_d'])

  # --- middle tip --- #
  # middle tip - target
  middle_tip_tar_diff = pd.DataFrame()
  middle_tip_tar_diff['middle_tip_tar_x'] = middle_tip_pos_features['middle_tip_x_mm'] - target_pos_features['target_x_mm']
  middle_tip_tar_diff['middle_tip_tar_y'] = middle_tip_pos_features['middle_tip_y_mm'] - target_pos_features['target_y_mm']
  middle_tip_tar_diff['middle_tip_tar_z'] = middle_tip_pos_features['middle_tip_z_mm'] - target_pos_features['target_z_mm']
  middle_tip_tar_diff['middle_tip_tar_d'] = np.sqrt(middle_tip_tar_diff['middle_tip_tar_x']**2+middle_tip_tar_diff['middle_tip_tar_y']**2+middle_tip_tar_diff['middle_tip_tar_z']**2)
  middle_tip_tar_diff['middle_tip_tar_psi_x'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_x'] / middle_tip_tar_diff['middle_tip_tar_d'])
  middle_tip_tar_diff['middle_tip_tar_psi_y'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_y'] / middle_tip_tar_diff['middle_tip_tar_d'])
  middle_tip_tar_diff['middle_tip_tar_psi_z'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_z'] / middle_tip_tar_diff['middle_tip_tar_d'])
  # middle tip - obstacle 1
  middle_tip_obst_1_diff = pd.DataFrame()
  middle_tip_obst_1_diff['middle_tip_obst_1_x'] = middle_tip_pos_features['middle_tip_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_y'] = middle_tip_pos_features['middle_tip_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_z'] = middle_tip_pos_features['middle_tip_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_d'] = np.sqrt(middle_tip_obst_1_diff['middle_tip_obst_1_x']**2+middle_tip_obst_1_diff['middle_tip_obst_1_y']**2+middle_tip_obst_1_diff['middle_tip_obst_1_z']**2)
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_x'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_x'] / middle_tip_obst_1_diff['middle_tip_obst_1_d'])
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_y'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_y'] / middle_tip_obst_1_diff['middle_tip_obst_1_d'])
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_z'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_z'] / middle_tip_obst_1_diff['middle_tip_obst_1_d'])

  # --- target - obstacle 1 --- #
  tar_obst_1_diff = pd.DataFrame()
  tar_obst_1_diff['tar_obst_1_x'] = target_pos_features['target_x_mm'] - obstacle_1_pos_features['obstacle_1_x_mm']
  tar_obst_1_diff['tar_obst_1_y'] = target_pos_features['target_y_mm'] - obstacle_1_pos_features['obstacle_1_y_mm']
  tar_obst_1_diff['tar_obst_1_z'] = target_pos_features['target_z_mm'] - obstacle_1_pos_features['obstacle_1_z_mm']
  tar_obst_1_diff['tar_obst_1_d'] = np.sqrt(tar_obst_1_diff['tar_obst_1_x']**2+tar_obst_1_diff['tar_obst_1_y']**2+tar_obst_1_diff['tar_obst_1_z']**2)
  tar_obst_1_diff['tar_obst_1_psi_x'] = np.arccos(tar_obst_1_diff['tar_obst_1_x'] / tar_obst_1_diff['tar_obst_1_d'])
  tar_obst_1_diff['tar_obst_1_psi_y'] = np.arccos(tar_obst_1_diff['tar_obst_1_y'] / tar_obst_1_diff['tar_obst_1_d'])
  tar_obst_1_diff['tar_obst_1_psi_z'] = np.arccos(tar_obst_1_diff['tar_obst_1_z'] / tar_obst_1_diff['tar_obst_1_d'])


  # --- collect data --- #
  processed_features = elbow_pos_features.copy()
  # raw features
  processed_features = processed_features.join(wrist_pos_features)
  processed_features = processed_features.join(hand_pos_features)
  processed_features = processed_features.join(thumb_1_pos_features)
  processed_features = processed_features.join(thumb_2_pos_features)
  processed_features = processed_features.join(thumb_tip_pos_features)
  processed_features = processed_features.join(index_1_pos_features)
  processed_features = processed_features.join(index_2_pos_features)
  processed_features = processed_features.join(index_tip_pos_features)
  processed_features = processed_features.join(middle_1_pos_features)
  processed_features = processed_features.join(middle_2_pos_features)
  processed_features = processed_features.join(middle_tip_pos_features)
  processed_features = processed_features.join(target_pos_features)
  processed_features = processed_features.join(target_or_features)
  processed_features = processed_features.join(obstacle_1_pos_features)
  processed_features = processed_features.join(obstacle_1_or_features)
  # processed features
  processed_features = processed_features.join(elb_tar_diff)
  processed_features = processed_features.join(elb_obst_1_diff)
  processed_features = processed_features.join(wri_tar_diff)
  processed_features = processed_features.join(wri_obst_1_diff)
  processed_features = processed_features.join(hand_tar_diff)
  processed_features = processed_features.join(hand_obst_1_diff)
  processed_features = processed_features.join(thumb_1_tar_diff)
  processed_features = processed_features.join(thumb_1_obst_1_diff)
  processed_features = processed_features.join(thumb_2_tar_diff)
  processed_features = processed_features.join(thumb_2_obst_1_diff)
  processed_features = processed_features.join(thumb_tip_tar_diff)
  processed_features = processed_features.join(thumb_tip_obst_1_diff)
  processed_features = processed_features.join(index_1_tar_diff)
  processed_features = processed_features.join(index_1_obst_1_diff)
  processed_features = processed_features.join(index_2_tar_diff)
  processed_features = processed_features.join(index_2_obst_1_diff)
  processed_features = processed_features.join(index_tip_tar_diff)
  processed_features = processed_features.join(index_tip_obst_1_diff)
  processed_features = processed_features.join(middle_1_tar_diff)
  processed_features = processed_features.join(middle_1_obst_1_diff)
  processed_features = processed_features.join(middle_2_tar_diff)
  processed_features = processed_features.join(middle_2_obst_1_diff)
  processed_features = processed_features.join(middle_tip_tar_diff)
  processed_features = processed_features.join(middle_tip_obst_1_diff)
  processed_features = processed_features.join(tar_obst_1_diff)


  return processed_features



def preprocess_features_warm(task_dataframe,nn):
  """Prepares input features of the cold dataset.

  Args:
    task_dataframe: A Pandas DataFrame expected to contain warm-started data.
    nn: number of samples of the cold dataset
  Returns:
    A DataFrame that contains the features to be used for the model.
  """

  n_samples = round(len(task_dataframe) / nn) # number of samples of the output features

  # --- upper-limb information --- #
  # elbow
  elb_cols = ["elbow_x_warm_mm","elbow_y_warm_mm","elbow_z_warm_mm"]
  elbow_pos_features = pd.DataFrame(index = range(n_samples), columns = elb_cols)
  for ii in range(n_samples):
    elbow_pos_features.loc[ii] = task_dataframe[elb_cols].iloc[round(ii*nn)]
  #print(elbow_pos_features)
  # wrist
  wri_cols = ["wrist_x_warm_mm", "wrist_y_warm_mm", "wrist_z_warm_mm"]
  wrist_pos_features = pd.DataFrame(index = range(n_samples), columns = wri_cols)
  for ii in range(n_samples):
    wrist_pos_features.loc[ii] = task_dataframe[wri_cols].iloc[round(ii*nn)]
  #print(wrist_pos_features)
  # hand
  hand_cols = ["hand_x_warm_mm", "hand_y_warm_mm", "hand_z_warm_mm"]
  hand_pos_features = pd.DataFrame(index = range(n_samples), columns = hand_cols)
  for ii in range(n_samples):
    hand_pos_features.loc[ii] = task_dataframe[hand_cols].iloc[round(ii*nn)]
  #print(hand_pos_features)
  # thumb 1
  thumb_1_cols = ["thumb_1_x_warm_mm", "thumb_1_y_warm_mm", "thumb_1_z_warm_mm"]
  thumb_1_pos_features = pd.DataFrame(index = range(n_samples), columns = thumb_1_cols)
  for ii in range(n_samples):
    thumb_1_pos_features.loc[ii] = task_dataframe[thumb_1_cols].iloc[round(ii*nn)]
  #print(thumb_1_pos_features)
  # thumb 2
  thumb_2_cols = ["thumb_2_x_warm_mm", "thumb_2_y_warm_mm", "thumb_2_z_warm_mm"]
  thumb_2_pos_features = pd.DataFrame(index = range(n_samples), columns = thumb_2_cols)
  for ii in range(n_samples):
    thumb_2_pos_features.loc[ii] = task_dataframe[thumb_2_cols].iloc[round(ii*nn)]
  #print(thumb_2_pos_features)
  # thumb tip
  thumb_tip_cols = ["thumb_tip_x_warm_mm", "thumb_tip_y_warm_mm", "thumb_tip_z_warm_mm"]
  thumb_tip_pos_features = pd.DataFrame(index = range(n_samples), columns = thumb_tip_cols)
  for ii in range(n_samples):
    thumb_tip_pos_features.loc[ii] = task_dataframe[thumb_tip_cols].iloc[round(ii*nn)]
  #print(thumb_tip_pos_features)
  # index 1
  index_1_cols = ["index_1_x_warm_mm", "index_1_y_warm_mm", "index_1_z_warm_mm"]
  index_1_pos_features = pd.DataFrame(index = range(n_samples), columns = index_1_cols)
  for ii in range(n_samples):
    index_1_pos_features.loc[ii] = task_dataframe[index_1_cols].iloc[round(ii*nn)]
  #print(index_1_pos_features)
  # index 2
  index_2_cols = ["index_2_x_warm_mm", "index_2_y_warm_mm", "index_2_z_warm_mm"]
  index_2_pos_features = pd.DataFrame(index = range(n_samples), columns = index_2_cols)
  for ii in range(n_samples):
    index_2_pos_features.loc[ii] = task_dataframe[index_2_cols].iloc[round(ii*nn)]
  #print(index_2_pos_features)
  # index tip
  index_tip_cols = ["index_tip_x_warm_mm", "index_tip_y_warm_mm", "index_tip_z_warm_mm"]
  index_tip_pos_features = pd.DataFrame(index = range(n_samples), columns = index_tip_cols)
  for ii in range(n_samples):
    index_tip_pos_features.loc[ii] = task_dataframe[index_tip_cols].iloc[round(ii*nn)]
  #print(index_tip_pos_features)
  # middle 1
  middle_1_cols = ["middle_1_x_warm_mm", "middle_1_y_warm_mm", "middle_1_z_warm_mm"]
  middle_1_pos_features = pd.DataFrame(index = range(n_samples), columns = middle_1_cols)
  for ii in range(n_samples):
    middle_1_pos_features.loc[ii] = task_dataframe[middle_1_cols].iloc[round(ii*nn)]
  #print(middle_1_pos_features)
  # middle 2
  middle_2_cols = ["middle_2_x_warm_mm", "middle_2_y_warm_mm", "middle_2_z_warm_mm"]
  middle_2_pos_features = pd.DataFrame(index = range(n_samples), columns = middle_2_cols)
  for ii in range(n_samples):
    middle_2_pos_features.loc[ii] = task_dataframe[middle_2_cols].iloc[round(ii*nn)]
  #print(middle_2_pos_features)
  # middle tip
  middle_tip_cols = ["middle_tip_x_warm_mm", "middle_tip_y_warm_mm", "middle_tip_z_warm_mm"]
  middle_tip_pos_features = pd.DataFrame(index = range(n_samples), columns = middle_tip_cols)
  for ii in range(n_samples):
    middle_tip_pos_features.loc[ii] = task_dataframe[middle_tip_cols].iloc[round(ii*nn)]
  #print(index_tip_pos_features)

  # --- target information --- #
  tar_pos_cols = ["target_x_warm_mm", "target_y_warm_mm", "target_z_warm_mm"]
  target_pos_features = pd.DataFrame(index = range(n_samples), columns = tar_pos_cols)
  for ii in range(n_samples):
    target_pos_features.loc[ii] = task_dataframe[tar_pos_cols].iloc[round(ii*nn)]
  #print(target_pos_features)
  tar_or_cols = ["target_roll_warm_rad", "target_pitch_warm_rad", "target_yaw_warm_rad"]
  target_or_features = pd.DataFrame(index = range(n_samples), columns = tar_or_cols)
  for ii in range(n_samples):
    target_or_features.loc[ii] = task_dataframe[tar_or_cols].iloc[round(ii*nn)]
  #print(target_or_features)

  # --- obstacle 1 information --- #
  obst_1_pos_cols = ["obstacle_1_x_warm_mm", "obstacle_1_y_warm_mm", "obstacle_1_z_warm_mm"]
  obstacle_1_pos_features = pd.DataFrame(index = range(n_samples), columns = obst_1_pos_cols)
  for ii in range(n_samples):
    obstacle_1_pos_features.loc[ii] = task_dataframe[obst_1_pos_cols].iloc[round(ii*nn)]
  #print(obstacle_1_pos_features)
  obst_1_or_cols = ["obstacle_1_roll_warm_rad", "obstacle_1_pitch_warm_rad", "obstacle_1_yaw_warm_rad"]
  obstacle_1_or_features = pd.DataFrame(index = range(n_samples), columns = obst_1_or_cols)
  for ii in range(n_samples):
    obstacle_1_or_features.loc[ii] = task_dataframe[obst_1_or_cols].iloc[round(ii*nn)]
  #print(obstacle_1_or_features)

  # --- elbow --- #
  # elbow - target
  elb_tar_diff = pd.DataFrame()
  elb_tar_diff['elb_tar_x_warm'] = elbow_pos_features['elbow_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  elb_tar_diff['elb_tar_y_warm'] = elbow_pos_features['elbow_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  elb_tar_diff['elb_tar_z_warm'] = elbow_pos_features['elbow_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  elb_tar_diff['elb_tar_d_warm'] = np.sqrt(elb_tar_diff['elb_tar_x_warm'].astype('float')**2+elb_tar_diff['elb_tar_y_warm'].astype('float')**2+elb_tar_diff['elb_tar_z_warm'].astype('float')**2)
  elb_tar_diff['elb_tar_psi_x_warm'] = np.arccos(elb_tar_diff['elb_tar_x_warm'].astype('float')/elb_tar_diff['elb_tar_d_warm'].astype('float'))
  elb_tar_diff['elb_tar_psi_y_warm'] = np.arccos(elb_tar_diff['elb_tar_y_warm'].astype('float') / elb_tar_diff['elb_tar_d_warm'].astype('float'))
  elb_tar_diff['elb_tar_psi_z_warm'] = np.arccos(elb_tar_diff['elb_tar_z_warm'].astype('float') / elb_tar_diff['elb_tar_d_warm'].astype('float'))
  # elbow - obstacle 1
  elb_obst_1_diff = pd.DataFrame()
  elb_obst_1_diff['elb_obst_1_x_warm'] = elbow_pos_features['elbow_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  elb_obst_1_diff['elb_obst_1_y_warm'] = elbow_pos_features['elbow_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  elb_obst_1_diff['elb_obst_1_z_warm'] = elbow_pos_features['elbow_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  elb_obst_1_diff['elb_obst_1_d_warm'] = np.sqrt(elb_obst_1_diff['elb_obst_1_x_warm'].astype('float')**2+elb_obst_1_diff['elb_obst_1_y_warm'].astype('float')**2+elb_obst_1_diff['elb_obst_1_z_warm'].astype('float')**2)
  elb_obst_1_diff['elb_obst_1_psi_x_warm'] = np.arccos(elb_obst_1_diff['elb_obst_1_x_warm'].astype('float') / elb_obst_1_diff['elb_obst_1_d_warm'].astype('float'))
  elb_obst_1_diff['elb_obst_1_psi_y_warm'] = np.arccos(elb_obst_1_diff['elb_obst_1_y_warm'].astype('float') / elb_obst_1_diff['elb_obst_1_d_warm'].astype('float'))
  elb_obst_1_diff['elb_obst_1_psi_z_warm'] = np.arccos(elb_obst_1_diff['elb_obst_1_z_warm'].astype('float') / elb_obst_1_diff['elb_obst_1_d_warm'].astype('float'))

  # --- wrist --- #
  # wrist - target
  wri_tar_diff = pd.DataFrame()
  wri_tar_diff['wri_tar_x_warm'] = wrist_pos_features['wrist_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  wri_tar_diff['wri_tar_y_warm'] = wrist_pos_features['wrist_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  wri_tar_diff['wri_tar_z_warm'] = wrist_pos_features['wrist_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  wri_tar_diff['wri_tar_d_warm'] = np.sqrt(wri_tar_diff['wri_tar_x_warm'].astype('float')**2+wri_tar_diff['wri_tar_y_warm'].astype('float')**2+wri_tar_diff['wri_tar_z_warm'].astype('float')**2)
  wri_tar_diff['wri_tar_psi_x_warm'] = np.arccos(wri_tar_diff['wri_tar_x_warm'].astype('float') / wri_tar_diff['wri_tar_d_warm'].astype('float'))
  wri_tar_diff['wri_tar_psi_y_warm'] = np.arccos(wri_tar_diff['wri_tar_y_warm'].astype('float') / wri_tar_diff['wri_tar_d_warm'].astype('float'))
  wri_tar_diff['wri_tar_psi_z_warm'] = np.arccos(wri_tar_diff['wri_tar_z_warm'].astype('float') / wri_tar_diff['wri_tar_d_warm'].astype('float'))
  # wrist - obstacle 1
  wri_obst_1_diff = pd.DataFrame()
  wri_obst_1_diff['wri_obst_1_x_warm'] = wrist_pos_features['wrist_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  wri_obst_1_diff['wri_obst_1_y_warm'] = wrist_pos_features['wrist_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  wri_obst_1_diff['wri_obst_1_z_warm'] = wrist_pos_features['wrist_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  wri_obst_1_diff['wri_obst_1_d_warm'] = np.sqrt(wri_obst_1_diff['wri_obst_1_x_warm'].astype('float')**2+wri_obst_1_diff['wri_obst_1_y_warm'].astype('float')**2+wri_obst_1_diff['wri_obst_1_z_warm'].astype('float')**2)
  wri_obst_1_diff['wri_obst_1_psi_x_warm'] = np.arccos(wri_obst_1_diff['wri_obst_1_x_warm'].astype('float') / wri_obst_1_diff['wri_obst_1_d_warm'].astype('float'))
  wri_obst_1_diff['wri_obst_1_psi_y_warm'] = np.arccos(wri_obst_1_diff['wri_obst_1_y_warm'].astype('float') / wri_obst_1_diff['wri_obst_1_d_warm'].astype('float'))
  wri_obst_1_diff['wri_obst_1_psi_z_warm'] = np.arccos(wri_obst_1_diff['wri_obst_1_z_warm'].astype('float') / wri_obst_1_diff['wri_obst_1_d_warm'].astype('float'))

  # --- hand --- #
  # hand - target
  hand_tar_diff = pd.DataFrame()
  hand_tar_diff['hand_tar_x_warm'] = hand_pos_features['hand_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  hand_tar_diff['hand_tar_y_warm'] = hand_pos_features['hand_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  hand_tar_diff['hand_tar_z_warm'] = hand_pos_features['hand_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  hand_tar_diff['hand_tar_d_warm'] = np.sqrt(hand_tar_diff['hand_tar_x_warm'].astype('float')**2+hand_tar_diff['hand_tar_y_warm'].astype('float')**2+hand_tar_diff['hand_tar_z_warm'].astype('float')**2)
  hand_tar_diff['hand_tar_psi_x_warm'] = np.arccos(hand_tar_diff['hand_tar_x_warm'].astype('float') / hand_tar_diff['hand_tar_d_warm'].astype('float'))
  hand_tar_diff['hand_tar_psi_y_warm'] = np.arccos(hand_tar_diff['hand_tar_y_warm'].astype('float') / hand_tar_diff['hand_tar_d_warm'].astype('float'))
  hand_tar_diff['hand_tar_psi_z_warm'] = np.arccos(hand_tar_diff['hand_tar_z_warm'].astype('float') / hand_tar_diff['hand_tar_d_warm'].astype('float'))
  # hand - obstacle 1
  hand_obst_1_diff = pd.DataFrame()
  hand_obst_1_diff['hand_obst_1_x_warm'] = hand_pos_features['hand_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  hand_obst_1_diff['hand_obst_1_y_warm'] = hand_pos_features['hand_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  hand_obst_1_diff['hand_obst_1_z_warm'] = hand_pos_features['hand_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  hand_obst_1_diff['hand_obst_1_d_warm'] = np.sqrt(hand_obst_1_diff['hand_obst_1_x_warm'].astype('float')**2+hand_obst_1_diff['hand_obst_1_y_warm'].astype('float')**2+hand_obst_1_diff['hand_obst_1_z_warm'].astype('float')**2)
  hand_obst_1_diff['hand_obst_1_psi_x_warm'] = np.arccos(hand_obst_1_diff['hand_obst_1_x_warm'].astype('float') / hand_obst_1_diff['hand_obst_1_d_warm'].astype('float'))
  hand_obst_1_diff['hand_obst_1_psi_y_warm'] = np.arccos(hand_obst_1_diff['hand_obst_1_y_warm'].astype('float') / hand_obst_1_diff['hand_obst_1_d_warm'].astype('float'))
  hand_obst_1_diff['hand_obst_1_psi_z_warm'] = np.arccos(hand_obst_1_diff['hand_obst_1_z_warm'].astype('float') / hand_obst_1_diff['hand_obst_1_d_warm'].astype('float'))

  # --- thumb 1 --- #
  # thumb 1 - target
  thumb_1_tar_diff = pd.DataFrame()
  thumb_1_tar_diff['thumb_1_tar_x_warm'] = thumb_1_pos_features['thumb_1_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  thumb_1_tar_diff['thumb_1_tar_y_warm'] = thumb_1_pos_features['thumb_1_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  thumb_1_tar_diff['thumb_1_tar_z_warm'] = thumb_1_pos_features['thumb_1_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  thumb_1_tar_diff['thumb_1_tar_d_warm'] = np.sqrt(thumb_1_tar_diff['thumb_1_tar_x_warm'].astype('float')**2+thumb_1_tar_diff['thumb_1_tar_y_warm'].astype('float')**2+thumb_1_tar_diff['thumb_1_tar_z_warm'].astype('float')**2)
  thumb_1_tar_diff['thumb_1_tar_psi_x_warm'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_x_warm'].astype('float') / thumb_1_tar_diff['thumb_1_tar_d_warm'].astype('float'))
  thumb_1_tar_diff['thumb_1_tar_psi_y_warm'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_y_warm'].astype('float') / thumb_1_tar_diff['thumb_1_tar_d_warm'].astype('float'))
  thumb_1_tar_diff['thumb_1_tar_psi_z_warm'] = np.arccos(thumb_1_tar_diff['thumb_1_tar_z_warm'].astype('float') / thumb_1_tar_diff['thumb_1_tar_d_warm'].astype('float'))
  # thumb 1 - obstacle 1
  thumb_1_obst_1_diff = pd.DataFrame()
  thumb_1_obst_1_diff['thumb_1_obst_1_x_warm'] = thumb_1_pos_features['thumb_1_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_y_warm'] = thumb_1_pos_features['thumb_1_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_z_warm'] = thumb_1_pos_features['thumb_1_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  thumb_1_obst_1_diff['thumb_1_obst_1_d_warm'] = np.sqrt(thumb_1_obst_1_diff['thumb_1_obst_1_x_warm'].astype('float')**2+thumb_1_obst_1_diff['thumb_1_obst_1_y_warm'].astype('float')**2+thumb_1_obst_1_diff['thumb_1_obst_1_z_warm'].astype('float')**2)
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_x_warm'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_x_warm'].astype('float') / thumb_1_obst_1_diff['thumb_1_obst_1_d_warm'].astype('float'))
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_y_warm'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_y_warm'].astype('float') / thumb_1_obst_1_diff['thumb_1_obst_1_d_warm'].astype('float'))
  thumb_1_obst_1_diff['thumb_1_obst_1_psi_z_warm'] = np.arccos(thumb_1_obst_1_diff['thumb_1_obst_1_z_warm'].astype('float') / thumb_1_obst_1_diff['thumb_1_obst_1_d_warm'].astype('float'))

  # --- thumb 2 --- #
  # thumb 2 - target
  thumb_2_tar_diff = pd.DataFrame()
  thumb_2_tar_diff['thumb_2_tar_x_warm'] = thumb_2_pos_features['thumb_2_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  thumb_2_tar_diff['thumb_2_tar_y_warm'] = thumb_2_pos_features['thumb_2_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  thumb_2_tar_diff['thumb_2_tar_z_warm'] = thumb_2_pos_features['thumb_2_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  thumb_2_tar_diff['thumb_2_tar_d_warm'] = np.sqrt(thumb_2_tar_diff['thumb_2_tar_x_warm'].astype('float')**2+thumb_2_tar_diff['thumb_2_tar_y_warm'].astype('float')**2+thumb_2_tar_diff['thumb_2_tar_z_warm'].astype('float')**2)
  thumb_2_tar_diff['thumb_2_tar_psi_x_warm'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_x_warm'].astype('float') / thumb_2_tar_diff['thumb_2_tar_d_warm'].astype('float'))
  thumb_2_tar_diff['thumb_2_tar_psi_y_warm'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_y_warm'].astype('float') / thumb_2_tar_diff['thumb_2_tar_d_warm'].astype('float'))
  thumb_2_tar_diff['thumb_2_tar_psi_z_warm'] = np.arccos(thumb_2_tar_diff['thumb_2_tar_z_warm'].astype('float') / thumb_2_tar_diff['thumb_2_tar_d_warm'].astype('float'))
  # thumb 2 - obstacle 1
  thumb_2_obst_1_diff = pd.DataFrame()
  thumb_2_obst_1_diff['thumb_2_obst_1_x_warm'] = thumb_2_pos_features['thumb_2_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_y_warm'] = thumb_2_pos_features['thumb_2_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_z_warm'] = thumb_2_pos_features['thumb_2_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  thumb_2_obst_1_diff['thumb_2_obst_1_d_warm'] = np.sqrt(thumb_2_obst_1_diff['thumb_2_obst_1_x_warm'].astype('float')**2+thumb_2_obst_1_diff['thumb_2_obst_1_y_warm'].astype('float')**2+thumb_2_obst_1_diff['thumb_2_obst_1_z_warm'].astype('float')**2)
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_x_warm'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_x_warm'].astype('float') / thumb_2_obst_1_diff['thumb_2_obst_1_d_warm'].astype('float'))
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_y_warm'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_y_warm'].astype('float') / thumb_2_obst_1_diff['thumb_2_obst_1_d_warm'].astype('float'))
  thumb_2_obst_1_diff['thumb_2_obst_1_psi_z_warm'] = np.arccos(thumb_2_obst_1_diff['thumb_2_obst_1_z_warm'].astype('float') / thumb_2_obst_1_diff['thumb_2_obst_1_d_warm'].astype('float'))

  # --- thumb tip --- #
  # thumb tip - target
  thumb_tip_tar_diff = pd.DataFrame()
  thumb_tip_tar_diff['thumb_tip_tar_x_warm'] = thumb_tip_pos_features['thumb_tip_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  thumb_tip_tar_diff['thumb_tip_tar_y_warm'] = thumb_tip_pos_features['thumb_tip_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  thumb_tip_tar_diff['thumb_tip_tar_z_warm'] = thumb_tip_pos_features['thumb_tip_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  thumb_tip_tar_diff['thumb_tip_tar_d_warm'] = np.sqrt(thumb_tip_tar_diff['thumb_tip_tar_x_warm'].astype('float')**2+thumb_tip_tar_diff['thumb_tip_tar_y_warm'].astype('float')**2+thumb_tip_tar_diff['thumb_tip_tar_z_warm'].astype('float')**2)
  thumb_tip_tar_diff['thumb_tip_tar_psi_x_warm'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_x_warm'].astype('float') / thumb_tip_tar_diff['thumb_tip_tar_d_warm'].astype('float'))
  thumb_tip_tar_diff['thumb_tip_tar_psi_y_warm'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_y_warm'].astype('float') / thumb_tip_tar_diff['thumb_tip_tar_d_warm'].astype('float'))
  thumb_tip_tar_diff['thumb_tip_tar_psi_z_warm'] = np.arccos(thumb_tip_tar_diff['thumb_tip_tar_z_warm'].astype('float') / thumb_tip_tar_diff['thumb_tip_tar_d_warm'].astype('float'))
  # thumb tip - obstacle 1
  thumb_tip_obst_1_diff = pd.DataFrame()
  thumb_tip_obst_1_diff['thumb_tip_obst_1_x_warm'] = thumb_tip_pos_features['thumb_tip_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_y_warm'] = thumb_tip_pos_features['thumb_tip_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_z_warm'] = thumb_tip_pos_features['thumb_tip_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  thumb_tip_obst_1_diff['thumb_tip_obst_1_d_warm'] = np.sqrt(thumb_tip_obst_1_diff['thumb_tip_obst_1_x_warm'].astype('float')**2+thumb_tip_obst_1_diff['thumb_tip_obst_1_y_warm'].astype('float')**2+thumb_tip_obst_1_diff['thumb_tip_obst_1_z_warm'].astype('float')**2)
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_x_warm'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_x_warm'].astype('float') / thumb_tip_obst_1_diff['thumb_tip_obst_1_d_warm'].astype('float'))
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_y_warm'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_y_warm'].astype('float') / thumb_tip_obst_1_diff['thumb_tip_obst_1_d_warm'].astype('float'))
  thumb_tip_obst_1_diff['thumb_tip_obst_1_psi_z_warm'] = np.arccos(thumb_tip_obst_1_diff['thumb_tip_obst_1_z_warm'].astype('float') / thumb_tip_obst_1_diff['thumb_tip_obst_1_d_warm'].astype('float'))

  # --- index 1 --- #
  # index 1 - target
  index_1_tar_diff = pd.DataFrame()
  index_1_tar_diff['index_1_tar_x_warm'] = index_1_pos_features['index_1_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  index_1_tar_diff['index_1_tar_y_warm'] = index_1_pos_features['index_1_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  index_1_tar_diff['index_1_tar_z_warm'] = index_1_pos_features['index_1_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  index_1_tar_diff['index_1_tar_d_warm'] = np.sqrt(index_1_tar_diff['index_1_tar_x_warm'].astype('float')**2+index_1_tar_diff['index_1_tar_y_warm'].astype('float')**2+index_1_tar_diff['index_1_tar_z_warm'].astype('float')**2)
  index_1_tar_diff['index_1_tar_psi_x_warm'] = np.arccos(index_1_tar_diff['index_1_tar_x_warm'].astype('float') / index_1_tar_diff['index_1_tar_d_warm'].astype('float'))
  index_1_tar_diff['index_1_tar_psi_y_warm'] = np.arccos(index_1_tar_diff['index_1_tar_y_warm'].astype('float') / index_1_tar_diff['index_1_tar_d_warm'].astype('float'))
  index_1_tar_diff['index_1_tar_psi_z_warm'] = np.arccos(index_1_tar_diff['index_1_tar_z_warm'].astype('float') / index_1_tar_diff['index_1_tar_d_warm'].astype('float'))
  # index 1 - obstacle 1
  index_1_obst_1_diff = pd.DataFrame()
  index_1_obst_1_diff['index_1_obst_1_x_warm'] = index_1_pos_features['index_1_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  index_1_obst_1_diff['index_1_obst_1_y_warm'] = index_1_pos_features['index_1_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  index_1_obst_1_diff['index_1_obst_1_z_warm'] = index_1_pos_features['index_1_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  index_1_obst_1_diff['index_1_obst_1_d_warm'] = np.sqrt(index_1_obst_1_diff['index_1_obst_1_x_warm'].astype('float')**2+index_1_obst_1_diff['index_1_obst_1_y_warm'].astype('float')**2+index_1_obst_1_diff['index_1_obst_1_z_warm'].astype('float')**2)
  index_1_obst_1_diff['index_1_obst_1_psi_x_warm'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_x_warm'].astype('float') / index_1_obst_1_diff['index_1_obst_1_d_warm'].astype('float'))
  index_1_obst_1_diff['index_1_obst_1_psi_y_warm'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_y_warm'].astype('float') / index_1_obst_1_diff['index_1_obst_1_d_warm'].astype('float'))
  index_1_obst_1_diff['index_1_obst_1_psi_z_warm'] = np.arccos(index_1_obst_1_diff['index_1_obst_1_z_warm'].astype('float') / index_1_obst_1_diff['index_1_obst_1_d_warm'].astype('float'))

  # --- index 2 --- #
  # index 2 - target
  index_2_tar_diff = pd.DataFrame()
  index_2_tar_diff['index_2_tar_x_warm'] = index_2_pos_features['index_2_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  index_2_tar_diff['index_2_tar_y_warm'] = index_2_pos_features['index_2_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  index_2_tar_diff['index_2_tar_z_warm'] = index_2_pos_features['index_2_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  index_2_tar_diff['index_2_tar_d_warm'] = np.sqrt(index_2_tar_diff['index_2_tar_x_warm'].astype('float')**2+index_2_tar_diff['index_2_tar_y_warm'].astype('float')**2+index_2_tar_diff['index_2_tar_z_warm'].astype('float')**2)
  index_2_tar_diff['index_2_tar_psi_x_warm'] = np.arccos(index_2_tar_diff['index_2_tar_x_warm'].astype('float') / index_2_tar_diff['index_2_tar_d_warm'].astype('float'))
  index_2_tar_diff['index_2_tar_psi_y_warm'] = np.arccos(index_2_tar_diff['index_2_tar_y_warm'].astype('float') / index_2_tar_diff['index_2_tar_d_warm'].astype('float'))
  index_2_tar_diff['index_2_tar_psi_z_warm'] = np.arccos(index_2_tar_diff['index_2_tar_z_warm'].astype('float') / index_2_tar_diff['index_2_tar_d_warm'].astype('float'))
  # index 2 - obstacle 1
  index_2_obst_1_diff = pd.DataFrame()
  index_2_obst_1_diff['index_2_obst_1_x_warm'] = index_2_pos_features['index_2_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  index_2_obst_1_diff['index_2_obst_1_y_warm'] = index_2_pos_features['index_2_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  index_2_obst_1_diff['index_2_obst_1_z_warm'] = index_2_pos_features['index_2_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  index_2_obst_1_diff['index_2_obst_1_d_warm'] = np.sqrt(index_2_obst_1_diff['index_2_obst_1_x_warm'].astype('float')**2+index_2_obst_1_diff['index_2_obst_1_y_warm'].astype('float')**2+index_2_obst_1_diff['index_2_obst_1_z_warm'].astype('float')**2)
  index_2_obst_1_diff['index_2_obst_1_psi_x_warm'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_x_warm'].astype('float') / index_2_obst_1_diff['index_2_obst_1_d_warm'].astype('float'))
  index_2_obst_1_diff['index_2_obst_1_psi_y_warm'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_y_warm'].astype('float') / index_2_obst_1_diff['index_2_obst_1_d_warm'].astype('float'))
  index_2_obst_1_diff['index_2_obst_1_psi_z_warm'] = np.arccos(index_2_obst_1_diff['index_2_obst_1_z_warm'].astype('float') / index_2_obst_1_diff['index_2_obst_1_d_warm'].astype('float'))

  # --- index tip --- #
  # index tip - target
  index_tip_tar_diff = pd.DataFrame()
  index_tip_tar_diff['index_tip_tar_x_warm'] = index_tip_pos_features['index_tip_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  index_tip_tar_diff['index_tip_tar_y_warm'] = index_tip_pos_features['index_tip_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  index_tip_tar_diff['index_tip_tar_z_warm'] = index_tip_pos_features['index_tip_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  index_tip_tar_diff['index_tip_tar_d_warm'] = np.sqrt(index_tip_tar_diff['index_tip_tar_x_warm'].astype('float')**2+index_tip_tar_diff['index_tip_tar_y_warm'].astype('float')**2+index_tip_tar_diff['index_tip_tar_z_warm'].astype('float')**2)
  index_tip_tar_diff['index_tip_tar_psi_x_warm'] = np.arccos(index_tip_tar_diff['index_tip_tar_x_warm'].astype('float') / index_tip_tar_diff['index_tip_tar_d_warm'].astype('float'))
  index_tip_tar_diff['index_tip_tar_psi_y_warm'] = np.arccos(index_tip_tar_diff['index_tip_tar_y_warm'].astype('float') / index_tip_tar_diff['index_tip_tar_d_warm'].astype('float'))
  index_tip_tar_diff['index_tip_tar_psi_z_warm'] = np.arccos(index_tip_tar_diff['index_tip_tar_z_warm'].astype('float') / index_tip_tar_diff['index_tip_tar_d_warm'].astype('float'))
  # index tip - obstacle 1
  index_tip_obst_1_diff = pd.DataFrame()
  index_tip_obst_1_diff['index_tip_obst_1_x_warm'] = index_tip_pos_features['index_tip_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  index_tip_obst_1_diff['index_tip_obst_1_y_warm'] = index_tip_pos_features['index_tip_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  index_tip_obst_1_diff['index_tip_obst_1_z_warm'] = index_tip_pos_features['index_tip_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  index_tip_obst_1_diff['index_tip_obst_1_d_warm'] = np.sqrt(index_tip_obst_1_diff['index_tip_obst_1_x_warm'].astype('float')**2+index_tip_obst_1_diff['index_tip_obst_1_y_warm'].astype('float')**2+index_tip_obst_1_diff['index_tip_obst_1_z_warm'].astype('float')**2)
  index_tip_obst_1_diff['index_tip_obst_1_psi_x_warm'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_x_warm'].astype('float') / index_tip_obst_1_diff['index_tip_obst_1_d_warm'].astype('float'))
  index_tip_obst_1_diff['index_tip_obst_1_psi_y_warm'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_y_warm'].astype('float') / index_tip_obst_1_diff['index_tip_obst_1_d_warm'].astype('float'))
  index_tip_obst_1_diff['index_tip_obst_1_psi_z_warm'] = np.arccos(index_tip_obst_1_diff['index_tip_obst_1_z_warm'].astype('float') / index_tip_obst_1_diff['index_tip_obst_1_d_warm'].astype('float'))

  # --- middle 1 --- #
  # middle 1 - target
  middle_1_tar_diff = pd.DataFrame()
  middle_1_tar_diff['middle_1_tar_x_warm'] = middle_1_pos_features['middle_1_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  middle_1_tar_diff['middle_1_tar_y_warm'] = middle_1_pos_features['middle_1_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  middle_1_tar_diff['middle_1_tar_z_warm'] = middle_1_pos_features['middle_1_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  middle_1_tar_diff['middle_1_tar_d_warm'] = np.sqrt(middle_1_tar_diff['middle_1_tar_x_warm'].astype('float')**2+middle_1_tar_diff['middle_1_tar_y_warm'].astype('float')**2+middle_1_tar_diff['middle_1_tar_z_warm'].astype('float')**2)
  middle_1_tar_diff['middle_1_tar_psi_x_warm'] = np.arccos(middle_1_tar_diff['middle_1_tar_x_warm'].astype('float') / middle_1_tar_diff['middle_1_tar_d_warm'].astype('float'))
  middle_1_tar_diff['middle_1_tar_psi_y_warm'] = np.arccos(middle_1_tar_diff['middle_1_tar_y_warm'].astype('float') / middle_1_tar_diff['middle_1_tar_d_warm'].astype('float'))
  middle_1_tar_diff['middle_1_tar_psi_z_warm'] = np.arccos(middle_1_tar_diff['middle_1_tar_z_warm'].astype('float') / middle_1_tar_diff['middle_1_tar_d_warm'].astype('float'))
  # middle 1 - obstacle 1
  middle_1_obst_1_diff = pd.DataFrame()
  middle_1_obst_1_diff['middle_1_obst_1_x_warm'] = middle_1_pos_features['middle_1_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  middle_1_obst_1_diff['middle_1_obst_1_y_warm'] = middle_1_pos_features['middle_1_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  middle_1_obst_1_diff['middle_1_obst_1_z_warm'] = middle_1_pos_features['middle_1_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  middle_1_obst_1_diff['middle_1_obst_1_d_warm'] = np.sqrt(middle_1_obst_1_diff['middle_1_obst_1_x_warm'].astype('float')**2+middle_1_obst_1_diff['middle_1_obst_1_y_warm'].astype('float')**2+middle_1_obst_1_diff['middle_1_obst_1_z_warm'].astype('float')**2)
  middle_1_obst_1_diff['middle_1_obst_1_psi_x_warm'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_x_warm'].astype('float') / middle_1_obst_1_diff['middle_1_obst_1_d_warm'].astype('float'))
  middle_1_obst_1_diff['middle_1_obst_1_psi_y_warm'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_y_warm'].astype('float') / middle_1_obst_1_diff['middle_1_obst_1_d_warm'].astype('float'))
  middle_1_obst_1_diff['middle_1_obst_1_psi_z_warm'] = np.arccos(middle_1_obst_1_diff['middle_1_obst_1_z_warm'].astype('float') / middle_1_obst_1_diff['middle_1_obst_1_d_warm'].astype('float'))

  # --- middle 2 --- #
  # middle 2 - target
  middle_2_tar_diff = pd.DataFrame()
  middle_2_tar_diff['middle_2_tar_x_warm'] = middle_2_pos_features['middle_2_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  middle_2_tar_diff['middle_2_tar_y_warm'] = middle_2_pos_features['middle_2_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  middle_2_tar_diff['middle_2_tar_z_warm'] = middle_2_pos_features['middle_2_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  middle_2_tar_diff['middle_2_tar_d_warm'] = np.sqrt(middle_2_tar_diff['middle_2_tar_x_warm'].astype('float')**2+middle_2_tar_diff['middle_2_tar_y_warm'].astype('float')**2+middle_2_tar_diff['middle_2_tar_z_warm'].astype('float')**2)
  middle_2_tar_diff['middle_2_tar_psi_x_warm'] = np.arccos(middle_2_tar_diff['middle_2_tar_x_warm'].astype('float') / middle_2_tar_diff['middle_2_tar_d_warm'].astype('float'))
  middle_2_tar_diff['middle_2_tar_psi_y_warm'] = np.arccos(middle_2_tar_diff['middle_2_tar_y_warm'].astype('float') / middle_2_tar_diff['middle_2_tar_d_warm'].astype('float'))
  middle_2_tar_diff['middle_2_tar_psi_z_warm'] = np.arccos(middle_2_tar_diff['middle_2_tar_z_warm'].astype('float') / middle_2_tar_diff['middle_2_tar_d_warm'].astype('float'))
  # middle 2 - obstacle 1
  middle_2_obst_1_diff = pd.DataFrame()
  middle_2_obst_1_diff['middle_2_obst_1_x_warm'] = middle_2_pos_features['middle_2_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  middle_2_obst_1_diff['middle_2_obst_1_y_warm'] = middle_2_pos_features['middle_2_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  middle_2_obst_1_diff['middle_2_obst_1_z_warm'] = middle_2_pos_features['middle_2_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  middle_2_obst_1_diff['middle_2_obst_1_d_warm'] = np.sqrt(middle_2_obst_1_diff['middle_2_obst_1_x_warm'].astype('float')**2+middle_2_obst_1_diff['middle_2_obst_1_y_warm'].astype('float')**2+middle_2_obst_1_diff['middle_2_obst_1_z_warm'].astype('float')**2)
  middle_2_obst_1_diff['middle_2_obst_1_psi_x_warm'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_x_warm'].astype('float') / middle_2_obst_1_diff['middle_2_obst_1_d_warm'].astype('float'))
  middle_2_obst_1_diff['middle_2_obst_1_psi_y_warm'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_y_warm'].astype('float') / middle_2_obst_1_diff['middle_2_obst_1_d_warm'].astype('float'))
  middle_2_obst_1_diff['middle_2_obst_1_psi_z_warm'] = np.arccos(middle_2_obst_1_diff['middle_2_obst_1_z_warm'].astype('float') / middle_2_obst_1_diff['middle_2_obst_1_d_warm'].astype('float'))

  # --- middle tip --- #
  # middle tip - target
  middle_tip_tar_diff = pd.DataFrame()
  middle_tip_tar_diff['middle_tip_tar_x_warm'] = middle_tip_pos_features['middle_tip_x_warm_mm'] - target_pos_features['target_x_warm_mm']
  middle_tip_tar_diff['middle_tip_tar_y_warm'] = middle_tip_pos_features['middle_tip_y_warm_mm'] - target_pos_features['target_y_warm_mm']
  middle_tip_tar_diff['middle_tip_tar_z_warm'] = middle_tip_pos_features['middle_tip_z_warm_mm'] - target_pos_features['target_z_warm_mm']
  middle_tip_tar_diff['middle_tip_tar_d_warm'] = np.sqrt(middle_tip_tar_diff['middle_tip_tar_x_warm'].astype('float')**2+middle_tip_tar_diff['middle_tip_tar_y_warm'].astype('float')**2+middle_tip_tar_diff['middle_tip_tar_z_warm'].astype('float')**2)
  middle_tip_tar_diff['middle_tip_tar_psi_x_warm'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_x_warm'].astype('float') / middle_tip_tar_diff['middle_tip_tar_d_warm'].astype('float'))
  middle_tip_tar_diff['middle_tip_tar_psi_y_warm'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_y_warm'].astype('float') / middle_tip_tar_diff['middle_tip_tar_d_warm'].astype('float'))
  middle_tip_tar_diff['middle_tip_tar_psi_z_warm'] = np.arccos(middle_tip_tar_diff['middle_tip_tar_z_warm'].astype('float') / middle_tip_tar_diff['middle_tip_tar_d_warm'].astype('float'))
  # middle tip - obstacle 1
  middle_tip_obst_1_diff = pd.DataFrame()
  middle_tip_obst_1_diff['middle_tip_obst_1_x_warm'] = middle_tip_pos_features['middle_tip_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_y_warm'] = middle_tip_pos_features['middle_tip_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_z_warm'] = middle_tip_pos_features['middle_tip_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  middle_tip_obst_1_diff['middle_tip_obst_1_d_warm'] = np.sqrt(middle_tip_obst_1_diff['middle_tip_obst_1_x_warm'].astype('float')**2+middle_tip_obst_1_diff['middle_tip_obst_1_y_warm'].astype('float')**2+middle_tip_obst_1_diff['middle_tip_obst_1_z_warm'].astype('float')**2)
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_x_warm'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_x_warm'].astype('float') / middle_tip_obst_1_diff['middle_tip_obst_1_d_warm'].astype('float'))
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_y_warm'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_y_warm'].astype('float') / middle_tip_obst_1_diff['middle_tip_obst_1_d_warm'].astype('float'))
  middle_tip_obst_1_diff['middle_tip_obst_1_psi_z_warm'] = np.arccos(middle_tip_obst_1_diff['middle_tip_obst_1_z_warm'].astype('float') / middle_tip_obst_1_diff['middle_tip_obst_1_d_warm'].astype('float'))

  # --- target - obstacle 1 --- #
  tar_obst_1_diff = pd.DataFrame()
  tar_obst_1_diff['tar_obst_1_x_warm'] = target_pos_features['target_x_warm_mm'] - obstacle_1_pos_features['obstacle_1_x_warm_mm']
  tar_obst_1_diff['tar_obst_1_y_warm'] = target_pos_features['target_y_warm_mm'] - obstacle_1_pos_features['obstacle_1_y_warm_mm']
  tar_obst_1_diff['tar_obst_1_z_warm'] = target_pos_features['target_z_warm_mm'] - obstacle_1_pos_features['obstacle_1_z_warm_mm']
  tar_obst_1_diff['tar_obst_1_d_warm'] = np.sqrt(tar_obst_1_diff['tar_obst_1_x_warm'].astype('float')**2+tar_obst_1_diff['tar_obst_1_y_warm'].astype('float')**2+tar_obst_1_diff['tar_obst_1_z_warm'].astype('float')**2)
  tar_obst_1_diff['tar_obst_1_psi_x_warm'] = np.arccos(tar_obst_1_diff['tar_obst_1_x_warm'].astype('float') / tar_obst_1_diff['tar_obst_1_d_warm'].astype('float'))
  tar_obst_1_diff['tar_obst_1_psi_y_warm'] = np.arccos(tar_obst_1_diff['tar_obst_1_y_warm'].astype('float') / tar_obst_1_diff['tar_obst_1_d_warm'].astype('float'))
  tar_obst_1_diff['tar_obst_1_psi_z_warm'] = np.arccos(tar_obst_1_diff['tar_obst_1_z_warm'].astype('float') / tar_obst_1_diff['tar_obst_1_d_warm'].astype('float'))


  # --- collect data --- #
  processed_features = elbow_pos_features.copy()
  # raw features
  processed_features = processed_features.join(wrist_pos_features)
  processed_features = processed_features.join(hand_pos_features)
  processed_features = processed_features.join(thumb_1_pos_features)
  processed_features = processed_features.join(thumb_2_pos_features)
  processed_features = processed_features.join(thumb_tip_pos_features)
  processed_features = processed_features.join(index_1_pos_features)
  processed_features = processed_features.join(index_2_pos_features)
  processed_features = processed_features.join(index_tip_pos_features)
  processed_features = processed_features.join(middle_1_pos_features)
  processed_features = processed_features.join(middle_2_pos_features)
  processed_features = processed_features.join(middle_tip_pos_features)
  processed_features = processed_features.join(target_pos_features)
  processed_features = processed_features.join(target_or_features)
  processed_features = processed_features.join(obstacle_1_pos_features)
  processed_features = processed_features.join(obstacle_1_or_features)
  # processed features
  processed_features = processed_features.join(elb_tar_diff)
  processed_features = processed_features.join(elb_obst_1_diff)
  processed_features = processed_features.join(wri_tar_diff)
  processed_features = processed_features.join(wri_obst_1_diff)
  processed_features = processed_features.join(hand_tar_diff)
  processed_features = processed_features.join(hand_obst_1_diff)
  processed_features = processed_features.join(thumb_1_tar_diff)
  processed_features = processed_features.join(thumb_1_obst_1_diff)
  processed_features = processed_features.join(thumb_2_tar_diff)
  processed_features = processed_features.join(thumb_2_obst_1_diff)
  processed_features = processed_features.join(thumb_tip_tar_diff)
  processed_features = processed_features.join(thumb_tip_obst_1_diff)
  processed_features = processed_features.join(index_1_tar_diff)
  processed_features = processed_features.join(index_1_obst_1_diff)
  processed_features = processed_features.join(index_2_tar_diff)
  processed_features = processed_features.join(index_2_obst_1_diff)
  processed_features = processed_features.join(index_tip_tar_diff)
  processed_features = processed_features.join(index_tip_obst_1_diff)
  processed_features = processed_features.join(middle_1_tar_diff)
  processed_features = processed_features.join(middle_1_obst_1_diff)
  processed_features = processed_features.join(middle_2_tar_diff)
  processed_features = processed_features.join(middle_2_obst_1_diff)
  processed_features = processed_features.join(middle_tip_tar_diff)
  processed_features = processed_features.join(middle_tip_obst_1_diff)
  processed_features = processed_features.join(tar_obst_1_diff)


  return processed_features


def prepocess_features_complete(D_dataframe,Dx_dataframe,cost_dataframe):
  """Prepares the complete dataframe for the loss function
  Args:
    D_dataframe: dataframe of training situations
    Dx_dataframe: dataframe of novel situations
    cost_dataframe: dataframe of costs
  Returns:
    A D_prime dataframe with complete data
  """
  assert len(cost_dataframe) == len(D_dataframe)*len(Dx_dataframe)

  cols = Dx_dataframe.columns.values.tolist() + D_dataframe.columns.values.tolist() + cost_dataframe.columns.values.tolist()
  processed_features = pd.DataFrame(index = range(len(cost_dataframe)), columns = cols)
  for jj in range(len(Dx_dataframe)):
    for ii in range(len(D_dataframe)):
      processed_features.iloc[ii+jj*len(D_dataframe)] = pd.concat([Dx_dataframe.iloc[jj], D_dataframe.iloc[ii], cost_dataframe.iloc[ii+jj*len(D_dataframe)]])

  processed_features = processed_features.astype(float)
  return processed_features


def preprocess_targets_cold(task_dataframe):
  """Prepares target features of the cold dataset.

  Args:
    task_dataframe: A Pandas DataFrame expected to contain cold-started data.
  Returns:
    A DataFrame that contains the target features.
  """

  selected_targets = task_dataframe[["error_plan","error_bounce"]]
  output_targets = selected_targets.copy()

  return (output_targets)

def preprocess_init_cold(task_dataframe):
  """Prepares target features of the cold dataset.

  Args:
    task_dataframe: A Pandas DataFrame expected to contain cold-started data.
  Returns:
    A DataFrame that contains the target features.
  """

  x_f_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('xf_plan')]]
  z_f_L_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('zf_L_plan')]]
  z_f_U_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('zf_U_plan')]]
  dual_f_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('dual_f_plan')]]

  output_targets = x_f_df.copy()
  output_targets = output_targets.join(z_f_L_df)
  output_targets = output_targets.join(z_f_U_df)
  output_targets = output_targets.join(dual_f_df)

  return (output_targets)

def preprocess_init_b_cold(task_dataframe):
    """Prepares target features of the cold dataset.

    Args:
      task_dataframe: A Pandas DataFrame expected to contain cold-started data.
    Returns:
      A DataFrame that contains the target features.
    """

    x_b_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('x_bounce')]]
    z_b_L_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('zb_L')]]
    z_b_U_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('zb_U')]]
    dual_b_df = task_dataframe[task_dataframe.columns[task_dataframe.columns.to_series().str.contains('dual_bounce')]]

    output_targets = x_b_df.copy()
    output_targets = output_targets.join(z_b_L_df)
    output_targets = output_targets.join(z_b_U_df)
    output_targets = output_targets.join(dual_b_df)

    return (output_targets)


def preprocess_targets_warm(task_dataframe):
  """Prepares target features of the warm dataset.

  Args:
    task_dataframe: A Pandas DataFrame expected to contain warm-started data.
  Returns:
    A DataFrame that contains the target features.
  """

  selected_targets = task_dataframe[["error_plan_warm","mean_der_error_plan_warm","iterations_plan_warm"]]
  output_targets = selected_targets.copy()

  return (output_targets)


def k_sim(X1,X2,W):
  """ Weighted similarity function between two situations

  Args:
    X1: First situation vector (dim=s)
    X2: Second situation vector (dim=s)
    W: vector of the weights (dim=s)
  Returns:
    A coefficient of similarity
  """
  X1 = np.array(X1)
  X2 = np.array(X2)
  W2 = np.square(W)
  assert len(X1)==len(X2)

  X_diff_2 = np.square(X1-X2)

  return np.exp((-1/2)*np.sum(X_diff_2*W2))

def k_sim_sigma(X1,X2,W,sigma):
  """ Weighted similarity function between two situations

  Args:
    X1: First situation vector (dim=s)
    X2: Second situation vector (dim=s)
    W: vector of the weights (dim=s)
    sigma: kernel width
  Returns:
    A coefficient of similarity
  """
  X1c = X1.copy()
  X2c = X2.copy()
  Wc = W.copy()
  X1_arr = np.array(X1c)
  X2_arr = np.array(X2c)
  W2_arr = np.square(Wc)
  assert len(X1_arr)==len(X2_arr) and len(X1_arr)==len(W2_arr)

  X_diff_2 = np.square(X1_arr-X2_arr)
  sigma2 = np.square(sigma)
  d2 = np.sum(X_diff_2 * W2_arr)
  return np.exp(-d2/(2*sigma2))


def NNopt_loss_function_err_sigma(X_df,costs_df,n_D,M,W):
  """

  :param X_df:
  :param costs_df:
  :param n_D:
  :param M:
  :param W:
  :return:
  """
  f_dim = W.shape[0]
  xj_df = X_df.iloc[:,:f_dim]
  xi_df = X_df.iloc[0:n_D, f_dim:(2*f_dim)]
  n_samples = len(costs_df)
  r = 0.6

  #print(costs_df.head())
  #print(xj_df.iloc[:,:])
  #print(xi_df.iloc[:,:])

  L = 0
  n_Dx = round(n_samples/n_D)
  mm = M # number of nearest neighbors for sparsity
  nn = int(M*2)  # considered number of nearest neighbors
  W2 = np.square(W.copy())
  for jj in range(n_Dx):
    xj = xj_df.iloc[jj*n_D]
    dm_vect = []
    xdiff2_m = []
    ratio_m = []
    for ii in range(n_D):
        xii = xi_df.iloc[ii]
        xdiff2m = np.square(np.array(xj) - np.array(xii))
        xdiff2_m.append(xdiff2m)
        dm = np.sqrt(np.sum(xdiff2m * W2))
        dm_vect.append(dm)
        ratio = xdiff2m/dm
        ratio_m.append(ratio)
    dm_df = pd.DataFrame({'distance_m': dm_vect, 'xdiff2_m': xdiff2_m, 'ratio_m': ratio_m})
    dm_sorted_df = dm_df.sort(['distance_m'], ascending=True)
    dm_sorted_df = dm_sorted_df.reset_index(drop=True)
    dist_m_sorted_df = dm_sorted_df['distance_m']
    xdiff2_m_sorted_df = dm_sorted_df['xdiff2_m']
    ratio_m_sorted_df = dm_sorted_df['ratio_m']
    dm_mean = (1 / mm) * np.sum(dist_m_sorted_df.iloc[0:mm])
    sigma = r * dm_mean
    #sigma = 1

    k_sim_vect = []
    for ii in range(n_D):
        xii = xi_df.iloc[ii]
        k_sim_vect.append(k_sim_sigma(xj, xii, W, sigma))
    k_sim_df = pd.DataFrame({'k_sim': k_sim_vect})
    k_sim_sorted_df = k_sim_df.sort(['k_sim'], ascending=False)
    k_sim_sorted_df = k_sim_sorted_df.reset_index(drop=True)
    k_sim_sored_values = k_sim_sorted_df['k_sim']
    Z = np.sum(k_sim_sored_values.iloc[0:nn])
    if ((np.abs(Z)) < 1e-60):
        reg = 1e+60
    else:
        reg = (1 / Z)

    E_vect = []
    for ii in range(n_D):
        xii = xi_df.iloc[ii]
        cc = costs_df.iloc[(jj * n_D)+ii, 0]  # total cost
        E_vect.append(cc * reg * (k_sim_sigma(xj, xii, W, sigma)))
    E_df = pd.DataFrame({'k_sim': k_sim_vect,'E_value': E_vect})
    E_sorted_df = E_df.sort(['k_sim'], ascending=False)
    E_sorted_df = E_sorted_df.reset_index(drop=True)
    E_value_sorted_df = E_sorted_df['E_value']
    E = np.sum(E_value_sorted_df.iloc[0:nn])

    L += E
  return (L/n_Dx)

def jac_NNopt_loss_function_err_sigma(X_df,costs_df,n_D,M,W):
    """

    :param X_df:
    :param costs_df:
    :param n_D:
    :param W:
    :param M:
    :return:
    """

    f_dim = len(W)
    xj_df = X_df.copy().iloc[:, :f_dim]
    xi_df = X_df.copy().iloc[0:n_D, f_dim:(2 * f_dim)]
    n_samples = len(costs_df)
    n_Dx = round(n_samples/n_D)
    mm = M  # number of nearest neighbors for sparsity
    nn = int(M*2) # considered number of nearest neighbors
    W2 = np.square(W.copy())
    r = 0.6

    dLw = 0
    #dLr = 0
    for jj in range(n_Dx):
        xj = xj_df.iloc[jj*n_D]
        dm_vect = []
        xdiff2_m = []
        ratio_m = []
        for ii in range(n_D):
            xii = xi_df.iloc[ii]
            xdiff2m = np.square(np.array(xj) - np.array(xii))
            xdiff2_m.append(xdiff2m)
            dm = np.sqrt(np.sum(xdiff2m * W2))
            dm_vect.append(dm)
            ratio = xdiff2m/dm
            ratio_m.append(ratio)
        dm_df = pd.DataFrame({'distance_m': dm_vect, 'xdiff2_m': xdiff2_m, 'ratio_m': ratio_m})
        dm_sorted_df = dm_df.sort(['distance_m'], ascending=True)
        dm_sorted_df = dm_sorted_df.reset_index(drop=True)
        dist_m_sorted_df = dm_sorted_df['distance_m']
        xdiff2_m_sorted_df = dm_sorted_df['xdiff2_m']
        ratio_m_sorted_df = dm_sorted_df['ratio_m']
        dm_mean = (1 / mm) * np.sum(dist_m_sorted_df.iloc[0:mm])
        sigma = r * dm_mean
        #sigma = 1

        k_sim_vect = []
        dk_sim_w_vect = []
        #dk_sim_r_vect = []
        for ii in range(n_D):
            xii = xi_df.iloc[ii]
            xdiff2 = np.square(np.array(xj) - np.array(xii))
            dji2 = np.sum(xdiff2*W2)
            k_sim_vect.append(k_sim_sigma(xj, xii, W, sigma))
            dk_sim_w_vect.append(-k_sim_sigma(xj, xii, W, sigma) * W * (1 / (sigma**2)) * (xdiff2 - ((r * dji2) / (mm * sigma)) * np.sum(ratio_m_sorted_df.iloc[0:mm])))
            #dk_sim_r_vect.append((k_sim_sigma(xj, xii, W, sigma) * dji2) / (r * (sigma**2)))
            #dk_sim_w_vect.append(-k_sim_sigma(xj, xii, W, sigma) * W * (1 / (sigma ** 2)) * (xdiff2))

        #k_df = pd.DataFrame({'k_sim': k_sim_vect,'dk_sim_w': dk_sim_w_vect,'dk_sim_r': dk_sim_r_vect})
        k_df = pd.DataFrame({'k_sim': k_sim_vect,'dk_sim_w': dk_sim_w_vect})
        k_sorted_df = k_df.sort(['k_sim'], ascending=False)
        k_sorted_df = k_sorted_df.reset_index(drop=True)
        k_sim_sorted_df = k_sorted_df['k_sim']
        dk_w_sim_sorted_df = k_sorted_df['dk_sim_w']
        #dk_r_sim_sorted_df = k_sorted_df['dk_sim_r']
        Z = np.sum(k_sim_sorted_df.iloc[0:nn])
        dZw = np.sum(dk_w_sim_sorted_df.iloc[0:nn])
        #dZr = np.sum(dk_r_sim_sorted_df.iloc[0:nn])

        if ((Z ** 2) < 1e-60):
            reg = 1e+60
        else:
            reg = (1 / (Z ** 2))


        dEw_vect = []
        #dEr_vect = []
        for ii in range(n_D):
            xii = xi_df.iloc[ii]
            cc = costs_df.iloc[(jj * n_D) + ii, 0]  # total cost
            xdiff2 = np.square(np.array(xj) - np.array(xii))
            dji2 = np.sum(xdiff2*W2)
            dE2w = (k_sim_sigma(xj, xii, W, sigma)) * dZw
            dE1w = Z * (-k_sim_sigma(xj, xii, W, sigma) * W * (1 / (sigma**2)) * (xdiff2 - ((r * dji2) / (mm * sigma)) * np.sum(ratio_m_sorted_df.iloc[0:mm])))
            #dE1w = Z * (-k_sim_sigma(xj, xii, W, sigma) * W * (1 / (sigma ** 2)) * (xdiff2))
            dEw_vect.append(cc * reg * (dE1w - dE2w))
            #dE2r = (k_sim_sigma(xj, xii, W, sigma)) * dZr
            #dE1r = Z * ((k_sim_sigma(xj, xii, W, sigma) * dji2) / (r * (dm_mean ** 2)))
            #dEr_vect.append(cc * reg * (dE1r - dE2r))
        #E_df = pd.DataFrame({'k_sim': k_sim_vect, 'dEw': dEw_vect, 'dEr': dEr_vect})
        E_df = pd.DataFrame({'k_sim': k_sim_vect, 'dEw': dEw_vect})
        E_sorted_df = E_df.sort(['k_sim'], ascending=False)
        E_sorted_df = E_sorted_df.reset_index(drop=True)
        dEw_sorted_df = E_sorted_df['dEw']
        #dEr_sorted_df = E_sorted_df['dEr']
        dEw = np.sum(dEw_sorted_df.iloc[0:nn])
        #dEr = np.sum(dEr_sorted_df.iloc[0:nn])


        dLw += dEw
        #dLr += dEr

    #dL = np.append((dLw/n_Dx),(dLr/n_Dx))
    #return dL
    return (dLw/n_Dx)


def jac_NNopt_loss_function_der_err(X_df,costs_df,W):
    '''

    :param X_df:
    :param costs_df:
    :param W:
    :return:
    '''

    f_dim = len(W)
    xj_df = X_df.iloc[:, :f_dim]
    xi_df = X_df.iloc[:, f_dim:(2 * f_dim)]
    #costs_df = D_prime_dataframe.iloc[:, -2:]
    n_samples = len(costs_df)
    #n_Dx = round(n_samples / n_D)

    dE = 0
    dE1 = 0
    dE2 = 0
    Z = 0
    dZ = 0
    n_Dx = 1
    for ii in range(n_samples):
        xj = xj_df.iloc[ii]
        xii = xi_df.iloc[ii]
        cc = costs_df.iloc[ii, 0] # total cost
        der_cc = costs_df.iloc[ii, 1]  # derivative total cost
        Z += k_sim(xj, xii, W)
        xdiff2 = np.square(np.array(xj) - np.array(xii))
        dZ -= k_sim(xj, xii, W) * xdiff2 * W
        #dE1 += (cc+der_cc)*(-k_sim(xj, xii, W)*xdiff2*W)
        #dE2 += (cc+der_cc)*(k_sim(xj,xii,W))
        dE1 += (der_cc)*(-k_sim(xj, xii, W)*xdiff2*W)
        dE2 += (der_cc)*(k_sim(xj,xii,W))
        if ((ii + 1) != n_samples):
            xjj = xj_df.iloc[ii + 1]
            if (np.sqrt(np.sum(np.square(np.array(xj) - np.array(xjj)))) > 0.001):
                # the next row contains a different sample of the dataset
                if ((Z**2)<1e-40):
                    reg = 1e+40
                else:
                    reg = (1 / (Z ** 2))
                dE += reg * (Z * dE1 - dE2 * dZ)
                Z = 0
                dZ = 0
                dE1 = 0
                dE2 = 0
                n_Dx += 1
        #print(Z)
        #print(dE)
        else:
            if ((Z ** 2) < 1e-40):
                reg = 1e+40
            else:
                reg = (1 / (Z ** 2))
            dE += reg * (Z * dE1 - dE2 * dZ)

        #for kk in range(n_D):
        #    xi = xi_df.iloc[kk]
        #    Z = Z + k_sim(xj,xi,W)
        #    xdiff2 = np.array(np.square(xj - xi),dtype=float)
        #    dZ = dZ -k_sim(xj, xi, W)*xdiff2*W

        #print("Z: {}".format(Z))
        #print("dZ: {}".format(dZ))
        #print(np.array(np.square(xj-xii)))
        #print((xj - xii).to_string())
        #xiidiff2 = np.array(np.square(xj - xii),dtype=float)
        #dE = dE + (cc+der_cc)*(1/Z**2)*(-Z*k_sim(xj, xii, W)*xiidiff2*W-k_sim(xj,xii,W)*dZ)

    #print("dE: {}".format(dE/n_Dx))
    return (dE/n_Dx)


def quantile_outliers(series):
  '''
  :param series:
  :return:
  '''
  q1 = 0.3
  q2 = 0.7
  removed_outliers = series.between(series.quantile(q1), series.quantile(q2))   # without outliers
  return removed_outliers

def remove_outliers(examples_dataframe):
  z_score = np.abs(stats.zscore(examples_dataframe))
  examples_dataframe_wo_outliers = examples_dataframe[(z_score < 3).all(axis=1)]
  #error_th = examples_dataframe_wo_outliers['error_plan_warm'].max()/100
  #der_error_th = examples_dataframe_wo_outliers['mean_der_error_plan_warm'].max()/100
  #examples_dataframe_wo_outliers_th = examples_dataframe_wo_outliers[(examples_dataframe_wo_outliers['error_plan_warm'] < error_th )]

  examples_dataframe_wo_outliers_th = examples_dataframe_wo_outliers
  return examples_dataframe_wo_outliers_th


def linear_scale(series):
  '''
  Scales the series on the range [0,1]
  :param series: a pandas series
  :return: a pandas series scaled on the range [0,1]
           the maximum of the series
           the minimum of the series
  '''
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val)
  return (series.apply(lambda x:((x - min_val) / scale)) , max_val, min_val)

def delinear_scale(series,max,min):
  '''
  Scales the series on the range [0,1]
  :param a pandas series scaled on the range [0,1]
         the maximum of the series
         the minimum of the series
  :return: series: a pandas series
  '''
  scale = (max - min)
  return series.apply(lambda x:((x+1)*scale)+min)

def linear_scale_1(series):
  '''
  Scales the series on the range [-1,1]
  :param series: a pandas series
  :return: a pandas series scaled on the range [-1,1]
           the maximum of the series
           the minimum of the series
  '''
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return (series.apply(lambda x:((x - min_val) / scale) - 1.0) , max_val, min_val)

def delinear_scale_1(series,max,min):
  '''
  Scales the series on the range [-1,1]
  :param a pandas series scaled on the range [-1,1]
         the maximum of the series
         the minimum of the series
  :return: series: a pandas series
  '''
  scale = (max - min) / 2.0
  return series.apply(lambda x:((x+1)*scale)+min)

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv), mean, std_dv

def robust_scale(series):
  median = series.median()
  iqr = series.quantile(0.75) - series.quantile(0.25)
  return series.apply(lambda x:(x - median) / iqr), median, iqr

def z_score_denormalize(series,mean,std):

  return series.apply(lambda x:(x*std)+mean)

def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param examples_dataframe: pandas dataframe
    :return: processed_features: a pandas dataframe normalized on the range [0,1]
             processed_features_max: a pandas series of maximum values
             processed_features_min: a pandas series of minimum values
  """
  processed_features = pd.DataFrame()
  processed_features_max = pd.Series()
  processed_features_min = pd.Series()
  for column in examples_dataframe:
      (processed_features[column],processed_features_max[column],processed_features_min[column]) = linear_scale(examples_dataframe[column])
  return processed_features,processed_features_max, processed_features_min

def denormalize_linear_scale(processed_features,processed_features_max, processed_features_min):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param processed_features: a pandas dataframe normalized on the range [0,1]
            processed_features_max: a pandas series of maximum values
            processed_features_min: a pandas series of minimum values

    :return: examples_dataframe: a pandas dataframe
  """
  examples_dataframe = pd.DataFrame()
  for column in processed_features:
      examples_dataframe[column] = delinear_scale(processed_features[column],processed_features_max[column],processed_features_min[column])
  return examples_dataframe


def normalize_linear_scale_1(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param examples_dataframe: pandas dataframe
    :return: processed_features: a pandas dataframe normalized on the range [-1,1]
             processed_features_max: a pandas series of maximum values
             processed_features_min: a pandas series of minimum values
  """
  processed_features = pd.DataFrame()
  processed_features_max = pd.Series()
  processed_features_min = pd.Series()
  for column in examples_dataframe:
      (processed_features[column],processed_features_max[column],processed_features_min[column]) = linear_scale_1(examples_dataframe[column])
  return processed_features,processed_features_max, processed_features_min

def denormalize_linear_scale_1(processed_features,processed_features_max, processed_features_min):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param processed_features: a pandas dataframe normalized on the range [-1,1]
            processed_features_max: a pandas series of maximum values
            processed_features_min: a pandas series of minimum values

    :return: examples_dataframe: a pandas dataframe
  """
  examples_dataframe = pd.DataFrame()
  for column in processed_features:
      examples_dataframe[column] = delinear_scale_1(processed_features[column],processed_features_max[column],processed_features_min[column])
  return examples_dataframe

def normalize_z_score(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param examples_dataframe: pandas dataframe
    :return: processed_features: a pandas dataframe normalized on its mean
             processed_features_max: a pandas series of maximum values
             processed_features_min: a pandas series of minimum values
  """
  processed_features = pd.DataFrame()
  processed_features_mean = pd.Series()
  processed_features_std = pd.Series()
  for column in examples_dataframe:
      processed_features[column], processed_features_mean[column], processed_features_std[column] = z_score_normalize(examples_dataframe[column])
  return processed_features,processed_features_mean, processed_features_std

def scale_robust(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param examples_dataframe: pandas dataframe
    :return: processed_features: a pandas dataframe normalized on its mean
             processed_features_max: a pandas series of maximum values
             processed_features_min: a pandas series of minimum values
  """
  processed_features = pd.DataFrame()
  processed_features_median = pd.Series()
  processed_features_iqr = pd.Series()
  for column in examples_dataframe:
      processed_features[column], processed_features_median[column], processed_features_iqr[column] = robust_scale(examples_dataframe[column])
  return processed_features,processed_features_median, processed_features_iqr

def denormalize_z_score(processed_features,processed_features_mean, processed_features_std):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly.
    :param processed_features: a pandas dataframe normalized on the range [-1,1]
            processed_features_mean: a pandas series of maximum values
            processed_features_std: a pandas series of minimum values

    :return: examples_dataframe: a pandas dataframe
  """
  examples_dataframe = pd.DataFrame()
  for column in processed_features:
      examples_dataframe[column] = z_score_denormalize(processed_features[column],processed_features_mean[column],processed_features_std[column])
  return examples_dataframe


# --- Class VSM Model --- #
class VSMModel:
  """
    Variable Similarity model: fit by minimizing the provided loss_function with L1 regularization
  """

  def __init__(self, X=None, Y=None, n_D=1,weights_init=None, nn=10,mm=5,reg_w=2,tol=1e-06):
    self.loss_function = NNopt_loss_function_err_sigma
    self.jac_loss_function = jac_NNopt_loss_function_err_sigma
    self.reg_w = reg_w
    self.weights_init = weights_init.copy()
    self.weights = weights_init.copy()
    #self.r = r_init
    #self.r_init = r_init
    self.X = X # D prime dataframe
    self.Y = Y # costs dataframe
    self.n_D = n_D # size of the D dataset
    self.N = nn # considered number of nearest neighbours
    self.M = mm # size for sparsity
    self.tol = tol # tolerance of the objective function

  def regularized_loss(self,params):
      '''
      :param params: weights
      :return: S regularized loss function
      '''
      self.weights = params.copy()
      #pp = params.copy()
      #self.weights = pp[:-1]
      #self.r = pp[-1]
      n_Dx = round(len(self.Y)/self.n_D)
      #print("parameter n_Dx: {}.".format(n_Dx))
      #loss_tot = self.loss_function(self.X, self.Y, self.n_D,self.M,self.weights,self.r)
      loss_tot = self.loss_function(self.X, self.Y, self.n_D,self.M,self.weights)


      # S regularization
      loss_tot_reg = loss_tot + ((self.reg_w ** 2)/(n_Dx ** 2)) * np.sum(np.square(np.log(np.square(self.weights / self.weights_init))))
      #print(loss_tot)
      #print(loss_tot_reg)
      return loss_tot_reg


  def jac_regularized_loss(self,params):
      '''
      :param params: weights
      :return: S regularized first derivative of the loss function
      '''

      self.weights = params.copy()
      #pp = params.copy()
      #self.weights = pp[:-1]
      #self.r = pp[-1]
      n_Dx = round(len(self.Y)/self.n_D)
      der_loss_tot = self.jac_loss_function(self.X, self.Y, self.n_D, self.M, self.weights)
      #der_loss_tot = self.jac_loss_function(self.X, self.Y, self.n_D, self.M, self.weights,self.r)

      # S regularization
      der_loss_tot_reg = der_loss_tot + 4 * (1/self.weights) * ((self.reg_w ** 2)/(n_Dx ** 2)) * np.log(np.square(self.weights/self.weights_init))
      #der_loss_tot_reg = der_loss_tot[:-1] + 4 * (1/self.weights) * ((self.reg_w ** 2)/(n_Dx ** 2)) * np.log(np.square(self.weights/self.weights_init))
      #der_loss_tot_reg = np.append(der_loss_tot_reg , der_loss_tot[-1])
      return der_loss_tot_reg

  def fit(self,maxiter=250,disp=False):
    # Initialize weights
    if type(self.weights_init) == type(None):
        self.weights_init = np.ones(round(len(self.X.iloc[0,:])/2))

    #if (not self.weights is None) and (self.weights_init == self.weights).all():
    #    print("Model already fit once. Continuing fit with more iterations...")

    params = self.weights_init.copy()
    #params = np.append(self.weights_init,self.r_init)
    print("Training...")
    #print(results)
    results = minimize(fun=self.regularized_loss, x0=params, jac=self.jac_regularized_loss,method='L-BFGS-B', options={'maxiter': maxiter,'gtol': self.tol,'disp' : disp})
    #results = minimize(fun=self.regularized_loss, x0=params, jac=self.jac_regularized_loss, method='SLSQP', options={'maxiter': maxiter, 'ftol':self.tol,'disp': disp})
    self.weights = results.x.copy()
    self.weights_init = self.weights.copy()
    #self.weights = results.x[:-1]
    #self.r = results.x[-1]
    #self.weights_init = self.weights.copy()
    #self.r_init = self.r
    return results

  def predict(self,id_new,x,cold_data_in,warm_data_out):
      '''
      Prediction of the index in the cold-started database for a smarter initialization in the new situation x
      :param self: the TP model
      :param: id_new: index of the test sample in the Dx dataframe
      :param x: new situation
      :param cold_data_in: memory. it must have a size of n_D
      :param: warm_data_out: outputs of the test samples
      :return: the index of the sample in the cold-started database (D_dataframe)
      '''
      c_data_in = cold_data_in.copy()
      w_out = warm_data_out.copy()
      if type(self.weights) == type(None):
          w = np.ones(round(len(self.X.iloc[0, :]) / 2))
      else:
          w = self.weights.copy()
      sims = []
      f_dim = len(w) # features dimensions
      W2 = np.square(w)
      mm = self.M
      nn = self.N
      n_D = self.n_D
      r = 0.6
      idd = (np.arange(n_D)).tolist()

      dd = []
      for ii in range(n_D):
          xii = c_data_in.iloc[ii, 0:f_dim]
          x_diff_2 = np.square(np.array(x) - np.array(xii))
          dd.append(np.sqrt(np.sum(x_diff_2 * W2)))
      dd_df = pd.DataFrame({'distance_m': dd})
      dd_sorted_df = dd_df.sort(['distance_m'], ascending=True)
      dd_sorted_df = dd_sorted_df.reset_index(drop=True)
      dm_sorted_df = dd_sorted_df['distance_m']
      sigma = r * (1 / mm) * np.sum(dm_sorted_df.iloc[0:mm])
      #sigma = 1

      for i in range(n_D):
        sims.append(k_sim_sigma(x, c_data_in.iloc[i,0:f_dim], w, sigma))
      sims_df = pd.DataFrame({'id': idd,'sims': sims})
      sims_sorted_df = sims_df.sort(['sims'], ascending=False)
      sims_sorted_df = sims_sorted_df.reset_index(drop=True)
      sims_id_sorted_df = sims_sorted_df['id']
      sims_s_sorted_df = sims_sorted_df['sims']
      nn_id = sims_id_sorted_df.iloc[0:nn] # indexes of the nn nearest neighbours
      nn_sims = sims_s_sorted_df.iloc[0:nn] # similarities of the nn nearest neighbours
      nn_sims_sum = np.sum(nn_sims)

      cc = 0
      der_cc = 0
      for i in range(nn):
        id_warm = (id_new*n_D) + nn_id.iloc[i]
        sim = nn_sims.iloc[i]
        cost = w_out.iloc[id_warm, 0]
        der_cost = w_out.iloc[id_warm, 1]
        cc += (sim * cost)
        der_cc += (sim * der_cost)

      if(nn_sims_sum < 1e-60):
          pred_cc = (cc * 1e+60)
          pred_der_cc = (der_cc * 1e+60)
      else:
          pred_cc = (cc/nn_sims_sum)
          pred_der_cc = (der_cc/nn_sims_sum)

      return [pred_cc,pred_der_cc]

  def predict_init(self,x,cold_data_features,cold_data_init):
      '''
      Prediction of the index in the cold-started database for a smarter initialization in the new situation x
      :param self: the TP model
      :param: id_new: index of the test sample in the Dx dataframe
      :param x: new situation
      :param cold_data_features: memory. it must have a size of n_D
      :param: cold_data_init: outputs of the test samples
      :return: the index of the sample in the cold-started database (D_dataframe)
      '''
      if type(self.weights) == type(None):
          w = np.ones(round(len(self.X.iloc[0, :]) / 2))
      else:
          w = self.weights
      sims = []
      f_dim = len(w) # features dimensions
      W2 = np.square(w)
      mm = self.M
      nn = self.N
      n_D = self.n_D
      r = 0.6
      idd = (np.arange(n_D)).tolist()

      dd = []
      for ii in range(n_D):
          xii = cold_data_features.iloc[ii, 0:f_dim]
          x_diff_2 = np.square(np.array(x) - np.array(xii))
          dd.append(np.sqrt(np.sum(x_diff_2 * W2)))
      dd_df = pd.DataFrame({'distance_m': dd})
      dd_sorted_df = dd_df.sort(['distance_m'], ascending=True)
      dd_sorted_df = dd_sorted_df.reset_index(drop=True)
      dm_sorted_df = dd_sorted_df['distance_m']
      sigma = r * (1 / mm) * np.sum(dm_sorted_df.iloc[0:mm])
      #sigma = 1

      for i in range(n_D):
        sims.append(k_sim_sigma(x, cold_data_features.iloc[i,0:f_dim], w, sigma))
      sims_df = pd.DataFrame({'id': idd,'sims': sims})
      sims_sorted_df = sims_df.sort(['sims'], ascending=False)
      cold_data_init_c = cold_data_init.loc[sims_sorted_df.index]
      sims_sorted_df = sims_sorted_df.reset_index(drop=True)
      cold_data_init_c = cold_data_init_c.reset_index(drop=True)
      sims_id_sorted_df = sims_sorted_df['id']
      sims_s_sorted_df = sims_sorted_df['sims']
      nn_id = sims_id_sorted_df.iloc[0:nn] # indexes of the nn nearest neighbours
      nn_sims = sims_s_sorted_df.iloc[0:nn] # similarities of the nn nearest neighbours
      nn_sims_sum = np.sum(nn_sims)

      qi_sum = np.zeros(cold_data_init_c.shape[1])
      for i in range(nn):
        sim = nn_sims.iloc[i]
        qi = cold_data_init_c.iloc[i,:]
        qi_sum = np.add(qi_sum,np.multiply(qi,sim))

      if(nn_sims_sum < 1e-60):
        qi_pred = np.divide(qi_sum, 1e+60)
      else:
        qi_pred = np.divide(qi_sum,nn_sims_sum)

      return qi_pred
