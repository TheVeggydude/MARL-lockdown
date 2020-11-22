
__actions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


def get_n_actions():
    return len(__actions)


def action_level(level):
    if level < 0 or level >= len(__actions):
        raise ValueError("Containment level should be > 0 and < " + str(len(__actions)))

    return __actions[level]