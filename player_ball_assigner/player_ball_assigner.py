import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_foot_position, measure_distance

class PlayerBallAssigner:
    def __init__(self, max_dist_m=2.5, max_dist_px=70):
        """
        Si existen posiciones transformadas (metros), usa max_dist_m.
        Si no, usa distancia en píxeles (fallback).
        """
        self.max_dist_m = max_dist_m
        self.max_dist_px = max_dist_px

    def assign_ball_to_player(self, players, ball_dict):
        """
        players: dict de {player_id: {'bbox':..., 'position_transformed':... opcional}}
        ball_dict: {'bbox':..., 'position_transformed':... opcional}
        """
        if ball_dict is None or 'bbox' not in ball_dict:
            return -1

        ball_bbox = ball_dict['bbox']
        ball_pos_px = get_center_of_bbox(ball_bbox)
        ball_pos_m = ball_dict.get('position_transformed', None)

        best_id, best_d = -1, 1e9

        for pid, p in players.items():
            p_pos_m = p.get('position_transformed', None)
            if ball_pos_m is not None and p_pos_m is not None:

                d = measure_distance(p_pos_m, ball_pos_m)
                if d < self.max_dist_m and d < best_d:
                    best_id, best_d = pid, d
            else:

                foot = get_foot_position(p['bbox'])
                d = measure_distance(foot, ball_pos_px)
                if d < self.max_dist_px and d < best_d:
                    best_id, best_d = pid, d

        return best_id
