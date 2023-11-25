from player import Player


class ControlledPlayer(Player):
    def __init__(self):
        super().__init__()
        self.player_type = "controlled"

    def get_action(self, scorecard, turn_state):
        pass
