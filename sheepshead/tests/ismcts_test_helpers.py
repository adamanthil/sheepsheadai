"""Shared fixtures for the ISMCTS suites: an unseeded default agent and a
deal driver that stops at the second-bury root."""

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import is_private_decision


def fresh_agent():
    return PPOAgent(len(ACTIONS))


def drive_to_second_bury(game):
    """Force a JD pick, partner choice, and one bury; return the second-bury root."""
    fp = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                names = [ACTIONS[a - 1] for a in valid]
                if is_private_decision(valid) and len(game.bury) == 1:
                    return player.position, list(fp)
                if "PICK" in names:
                    a = ACTIONS.index("PICK") + 1
                elif "JD PARTNER" in names:
                    a = ACTIONS.index("JD PARTNER") + 1
                elif is_private_decision(valid):
                    a = sorted(valid)[0]
                else:
                    a = sorted(valid)[0]
                if not is_private_decision(valid):
                    fp.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
    return None
