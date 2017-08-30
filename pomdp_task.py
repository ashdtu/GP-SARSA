from pybrain.rl.environments.task import Task
from menu_model import Click,Quit
class SearchTask(Task):

    reward_success = 10000
    reward_failure = -10000

    def __init__(self, env, max_number_of_actions_per_session):
        super(Task, self).__init__()
        self.env=env
        self.reward_success = 10000
        self.reward_failure = -10000
        self.max_number_of_actions_per_session = max_number_of_actions_per_session

    def to_dict(self):
        return {
                "max_number_of_actions_per_session": self.max_number_of_actions_per_session
                }

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        if self.env.click_status != Click.NOT_CLICKED:
            if self.env.clicked_item.item_relevance == 1.0:

                return self.reward_success
            else:
                # penalty for clicking the wrong item
                return self.reward_failure
        elif self.env.quit_status == Quit.HAS_QUIT:
            if self.env.target_present is False:
                # reward for quitting when target is absent
                return self.reward_success
            else:
                # penalty for quitting when target is present
                return self.reward_failure
        # default penalty for spending time
        return int(-1 * self.env.action_duration)

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.click_status != Click.NOT_CLICKED:
            # click ends task
            return True
        elif self.env.quit_status == Quit.HAS_QUIT:
            # quit ends task
            return True
        return False

    def performAction(self, action):
        self.env.performAction(action)

    def reset(self):
        self.env.reset()

    def getObservation(self):
        return self.env.getbelief()
