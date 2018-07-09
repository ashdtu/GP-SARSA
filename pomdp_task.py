from pybrain.rl.environments.task import Task
from menu_model_short import Click,Quit,Action,Focus,MenuItem
import numpy as np
from scipy.stats import beta
import copy
class SearchTask():

    reward_success = 10000
    reward_failure = -10000

    def __init__(self, env, max_number_of_actions_per_session):
        self.env=env
        self.reward_success = 10000
        self.reward_failure = -10000
        self.max_number_of_actions_per_session = max_number_of_actions_per_session
        self.belief_state=np.ones(self.env.n_items+1)
        self.belief_state=self.belief_state/(self.env.n_items+1)
        self.menu=None

    def to_dict(self):
        return {
                "max_number_of_actions_per_session": self.max_number_of_actions_per_session
                }

    def getReward(self):

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
        self.action = Action(int(action))
        self.prev_state = self.belief_state.copy()
        #print('Focus before',self.env.Focus)
        self.env.duration_focus_ms, self.env.duration_saccade_ms = self.do_transition(self.prev_state,self.action)
        #print('focus after',self.env.Focus)
        #print(self.env.click_status)
        #print(self.env.quit_status)
        self.env.action_duration = self.env.duration_focus_ms + self.env.duration_saccade_ms
        self.env.gaze_location = int(self.env.Focus)
        self.env.n_actions += 1

    def do_transition(self, init_belief, action):

        len_obs = []

        if action != Action.CLICK and action != Action.QUIT:

            if self.env.Focus != Focus.ABOVE_MENU:
                amplitude = abs(self.env.item_locations[int(self.env.Focus) + 1] - self.env.item_locations[int(action) + 1])
            else:
                amplitude = abs(self.env.item_locations[0] - self.env.item_locations[int(action) + 1])
            saccade_duration = int(37 + 2.7 * amplitude)
            self.env.Focus = Focus(int(action))
            #print('focus point', self.env.Focus)
            focus_duration = 400
            semantic_obs = self.menu[int(self.env.Focus)].item_relevance

            loc = []

            # possible length observations with peripheral vision

            if self.env.len_observations is True:
                a=np.random.rand()
                if int(self.env.Focus) > 0 and a < self.env.p_obs_len_adj:
                    len_obs.append(self.menu[int(self.env.Focus) - 1].item_length)
                    loc.append(int(self.env.Focus) - 1)
                if a < self.env.p_obs_len_cur:
                    len_obs.append(self.menu[int(self.env.Focus)].item_length)
                    loc.append(int(self.env.Focus))
                if int(self.env.Focus) < self.env.n_items - 1 and a < self.env.p_obs_len_adj:
                    len_obs.append(self.menu[int(self.env.Focus) + 1].item_length)
                    loc.append(int(self.env.Focus) + 1)

            # belief update , only in Focus actions
            self.belief_state=self.belief_update(init_belief, semantic_obs, len_obs, loc, int(self.env.Focus))


        elif action == Action.CLICK:
            if self.env.Focus != Focus.ABOVE_MENU:
                self.env.click_status = Click(int(self.env.Focus))  # assume these match
            else:
                self.env.quit_status = Quit.HAS_QUIT

            focus_duration = 0
            saccade_duration = 0

        # quit without choosing any item
        elif action == Action.QUIT:
            self.env.quit_status = Quit.HAS_QUIT
            focus_duration = 0
            saccade_duration = 0


        else:
            raise ValueError("Unknown action: {}".format(action))

        return focus_duration, saccade_duration

    def belief_update(self, prev_belief, semantic_obs, len_obs, loc, focus_position):
        t_pm = [5.0, 1.0]
        non_pm = [2.0, 5.0]
        absent = [1, 5]
        belief = prev_belief.copy()

        for i in range(0, self.env.n_items + 1):
            if (i == focus_position):
                belief[i] = beta.pdf(semantic_obs, t_pm[0], t_pm[1])*belief[i]
            elif (i == self.env.n_items):
                belief[i] = beta.pdf(semantic_obs, absent[0], absent[1])*belief[i]
            else:
                belief[i] = beta.pdf(semantic_obs, non_pm[0], non_pm[1])*belief[i]

        norm = sum(belief)
        belief = belief / norm
        #belief = np.reshape(belief,(1, self.env.n_items + 1))[0]
        '''    
        if len(len_obs) == 0:
            norm = sum(belief)
            belief = belief/norm
            return belief

        else:
            for i in range(self.env.n_items + 1):
                if i in loc:
                    for j, k in enumerate(loc):  # loc contains location of length observations
                        if (i == k):
                            belief[i] = belief[i] * beta.pdf(len_obs[j], t_pm[0], t_pm[1])
                        else:
                            belief[i] = belief[i] * beta.pdf(len_obs[j], non_pm[0], non_pm[1])
                elif (i == self.env.n_items):
                    for j in range(len(len_obs)):
                        belief[i] = belief[i] * beta.pdf(len_obs[j], absent[0], absent[1])
                else:
                    for j in range(len(len_obs)):
                        belief[i] = belief[i] * beta.pdf(len_obs[j], non_pm[0], non_pm[1])
            norm = sum(belief)
            belief = belief/norm

        #belief[belief<10**(-4)]=0

        #dump=[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11]
        '''
        return belief


    def reset(self):
        self.env.reset()
        self.menu=self.env.getSensors()
        #print('menu',self.menu)
        self.belief_state=np.ones(self.env.n_items+1)
        self.belief_state=self.belief_state/(self.env.n_items+1)

    def getObservation(self):
        #print('belief state',self.belief_state)
        return self.belief_state


