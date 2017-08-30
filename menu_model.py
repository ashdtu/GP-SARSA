import numpy as np
import math
from enum import IntEnum
from scipy.stats import beta

class Quit(IntEnum):
    NOT_QUIT = 0
    HAS_QUIT = 1


class Focus(IntEnum):  # assume 8 items in menu
    ITEM_1 = 0
    ITEM_2 = 1
    ITEM_3 = 2
    ITEM_4 = 3
    ITEM_5 = 4
    ITEM_6 = 5
    ITEM_7 = 6
    ITEM_8 = 7
    ABOVE_MENU = 8

class Click(IntEnum):  # assume 8 items in menu
    CLICK_1 = 0
    CLICK_2 = 1
    CLICK_3 = 2
    CLICK_4 = 3
    CLICK_5 = 4
    CLICK_6 = 5
    CLICK_7 = 6
    CLICK_8 = 7
    NOT_CLICKED = 8

class Action(IntEnum):  # assume 8 items in menu
    LOOK_1 = 0
    LOOK_2 = 1
    LOOK_3 = 2
    LOOK_4 = 3
    LOOK_5 = 4
    LOOK_6 = 5
    LOOK_7 = 6
    LOOK_8 = 7
    CLICK = 8
    QUIT = 9

class MenuItem():
    """

    Parameters
    ----------
    item_relevance : ItemRelevance
    item_length : ItemLength
    """

    def __init__(self, item_relevance, item_length):
        self.item_relevance = item_relevance
        self.item_length = item_length

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (int(self.item_relevance), int(self.item_length)).__hash__()

    def __repr__(self):
        return "({},{})".format(self.item_relevance, self.item_length)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return MenuItem(self.item_relevance, self.item_length)

class SearchEnvironment():
    actions=range(0,10)
    def __init__(self,
            menu_type="semantic",
            menu_groups=2,
            menu_items_per_group=4,
            semantic_levels=3,
            gap_between_items=0.75,
            prop_target_abs=0.1,
            len_observations=True,
            p_obs_len_cur=0.95,
            p_obs_len_adj=0.89,
            n_training_menus=10000):


        #self.v = None # set with setup
        #self.random_state = None # set with setup
        self.task = None # set by Task

        self.menu_type = menu_type
        self.menu_groups = menu_groups
        self.menu_items_per_group = menu_items_per_group
        self.n_items = self.menu_groups * self.menu_items_per_group
        assert self.n_items == 8
        self.semantic_levels = semantic_levels
        self.gap_between_items = gap_between_items
        self.prop_target_abs = prop_target_abs
        self.len_observations = len_observations
        self.p_obs_len_cur = p_obs_len_cur
        self.p_obs_len_adj = p_obs_len_adj
        self.n_training_menus = n_training_menus
        self.training_menus = list()
        self.training = True
        self.n_item_lengths = 3
        #self.log_session_variables = ["items", "target_present", "target_idx"]       #Todo: use pybrain sequential dataset
        #self.log_step_variables = ["duration_focus_ms",
        #                           "duration_saccade_ms",
        #                           "action_duration",
        #                           "action",
        #                           "gaze_location"]

        # technical variables
        self.discreteStates = True
        self.outdim = 1
        self.indim = 1
        self.discreteActions = True
        self.numActions = self.n_items + 2 # look + click + quit
        self.click_status=Click.NOT_CLICKED
        self.Focus=Focus.ABOVE_MENU
        self.quit_status=Quit.NOT_QUIT
        self.belief_state=np.ones(self.n_items+1)
        self.belief_state=self.belief_state/(self.n_items+1)


    def clean(self):
        self.training_menus = list()

    def to_dict(self):
        return {
                "menu_type": self.menu_type,
                "menu_groups": self.menu_groups,
                "menu_items_per_group": self.menu_items_per_group,
                "semantic_levels": self.semantic_levels,
                "gap_between_items": self.gap_between_items,
                "prop_target_abs": self.prop_target_abs,
                "len_observations": self.len_observations,
                "n_training_menus": self.n_training_menus,
                }

    def _get_menu(self):
        if self.training is True and len(self.training_menus) >= self.n_training_menus:
            idx = np.random.randint(self.n_training_menus)
            return self.training_menus[idx]
        # generate menu item semantic relevances and lengths
        final_menu = [MenuItem(0,0) for i in range(self.n_items)]

        if self.menu_type == "semantic":
            items, target_idx = self._semantic(self.menu_groups,
                        self.menu_items_per_group,
                        self.prop_target_abs)
        elif self.menu_type == "unordered":
            items, target_idx = self._get_unordered_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.semantic_levels,
                        self.prop_target_abs)
        else:
            raise ValueError("Unknown menu type: {}".format(self.menu_type))
        target_present=(target_idx!=None)
        length_relevances= np.random.beta(0.3,0.3,len(items)).tolist()          #Length relevances, sampled as either relevant or non relevant
        target_len=1
        for i in range(len(length_relevances)):

            final_menu[i].item_relevance=items[i]
            final_menu[i].item_length=length_relevances[i]
        if target_present:
            final_menu[target_idx].item_length=target_len
        menu = (tuple(final_menu), target_present, target_idx)

        if self.training is True:
            self.training_menus.append(menu)
        print('get menu',menu)
        return menu

    def reset(self):
        """ Called by the library to reset the state
        """
        # state hidden from agent
        self.final_menu, self.target_present, self.target_idx = self._get_menu()

        print('Target location',self.target_idx)
        # state observed by agent
        #obs_items = [MenuItem(ItemRelevance.NOT_OBSERVED, ItemLength.NOT_OBSERVED) for i in range(self.n_items)]
        self.belief_state=np.ones(self.n_items+1)
        self.belief_state=self.belief_state/(self.n_items+1)
        self.Focus = Focus.ABOVE_MENU
        self.click_status = Click.NOT_CLICKED
        self.quit_status = Quit.NOT_QUIT


        # misc environment state variables
        self.action_duration = None
        self.duration_focus_ms = None
        self.duration_saccade_ms = None
        self.action = None
        self.gaze_location = None
        self.n_actions = 0
        self.item_locations = np.arange(self.gap_between_items, self.gap_between_items*(self.n_items+2), self.gap_between_items)
        #self._start_log_for_new_session()

    def performAction(self, action):
        """ Changes the state of the environment based on agent action """
        self.action = Action(int(action))
        print(self.action)
        self.prev_state = self.belief_state
        self.belief_state, self.duration_focus_ms, self.duration_saccade_ms = self.do_transition(self.prev_state,self.action)
        self.action_duration = self.duration_focus_ms + self.duration_saccade_ms
        self.gaze_location = int(self.Focus)
        self.n_actions += 1
        #self._log_transition()


    def do_transition(self,init_belief,action):

        #Todo:Menu recall probability without taking action
        #state = state.copy()
        # menu recall event may happen at first action
        #if self.n_actions == 0:
        #    if "menu_recall_probability" in self.v and np.random.rand() < float(self.v["menu_recall_probability"]):
        #        state.obs_items = [item.copy() for item in self.items]

        new_belief=[]
        len_obs=[]
        # observe items's state
        if action != Action.CLICK and action != Action.QUIT:
            # saccade
            # item_locations are off-by-one to other lists
            if self.Focus != Focus.ABOVE_MENU:
                amplitude = abs(self.item_locations[int(self.Focus)+1] - self.item_locations[int(action)+1])
            else:
                amplitude = abs(self.item_locations[0] - self.item_locations[int(action)+1])
            saccade_duration = int(37 + 2.7 * amplitude)
            self.Focus = Focus(int(action))
            print('focus point',self.Focus)# assume these match
            #Todo:Focus duration with Action=click
            # fixation
            #if "focus_duration_100ms" in self.v:
            #    focus_duration = int(self.v["focus_duration_100ms"] * 100)
            #else:
            focus_duration = 400
            # semantic observation at focus
            #state = self._observe_relevance_at(state, int(state.focus))

            

            semantic_obs=[self.final_menu[int(self.Focus)].item_relevance]
            loc=[]

            # possible length observations with peripheral vision
            if self.len_observations is True:
                if int(self.Focus) > 0 and np.random.rand() < self.p_obs_len_adj:
                    len_obs.append(self.final_menu[int(self.Focus)-1].item_length)
                    loc.append(int(self.Focus)-1)
                if np.random.rand() < self.p_obs_len_cur:
                    len_obs.append(self.final_menu[int(self.Focus)].item_length)
                    loc.append(int(self.Focus))

                if int(self.Focus) < self.n_items-1 and np.random.rand() < self.p_obs_len_adj:
                    len_obs.append(self.final_menu[int(self.Focus)+1].item_length)
                    loc.append(int(self.Focus)+1)

            #belief update , only in Focus actions
            new_belief = self.belief_update(init_belief, semantic_obs, len_obs,loc,int(self.Focus))

            #Todo: peripheral vison in semantics
            '''    
            # possible semantic peripheral observations
            if "prob_obs_adjacent" in self.v:
                if int(state.focus) > 0 and np.random.rand() < float(self.v["prob_obs_adjacent"]):
                    state = self._observe_relevance_at(state, int(state.focus)-1)
                if int(state.focus) < self.n_items-1 and np.random.rand() < float(self.v["prob_obs_adjacent"]):
                    state = self._observe_relevance_at(state, int(state.focus)+1)
            '''
        # choose item
        elif action == Action.CLICK:
            if self.Focus != Focus.ABOVE_MENU:
                self.click_status = Click(int(self.Focus))  # assume these match
            else:
                # trying to select an item when not focusing on any item equals to quitting
                self.quit_status = Quit.HAS_QUIT
            #if "selection_delay_s" in self.v:
            #    focus_duration = int(self.v["selection_delay_s"] * 1000)
            #else:
            focus_duration = 0
            saccade_duration = 0
            new_belief=init_belief

        # quit without choosing any item
        elif action == Action.QUIT:
            self.quit_status = Quit.HAS_QUIT
            focus_duration = 0
            saccade_duration = 0
            new_belief=init_belief

        else:
            raise ValueError("Unknown action: {}".format(action))


        return new_belief, focus_duration, saccade_duration

    def belief_update(self,prev_belief,semantic_obs,len_obs,loc,focus_position):
        t_pm=[5.0,1.0]
        non_pm=[2.0,5.0]
        absent=[1,5]
        new_belief=prev_belief
        #todo:length observation model: Done! (beta distribution same?)

        for i in range(0,self.n_items+1):
            if(i==focus_position):
                new_belief[i] = beta.pdf(semantic_obs, t_pm[0], t_pm[1]) * prev_belief[i]
            elif(i==self.n_items):
                new_belief[i]=beta.pdf(semantic_obs,absent[0],absent[1])*prev_belief[i]
            else:
                new_belief[i]=beta.pdf(semantic_obs,non_pm[0],non_pm[1])*prev_belief[i]

        return new_belief
        '''
    
        if len(len_obs)==0:
            norm = sum(new_belief)
            new_belief = [new_belief[x] / norm for x in range(len(new_belief))]


        else:
            for i in range(self.n_items+1):
                if i in loc:
                    for j,k in enumerate(loc):   #loc contains location of length observations
                        if(i==k):
                            new_belief[i]=new_belief[i]*beta.pdf(len_obs[j],t_pm[0],t_pm[1])
                        else:
                            new_belief[i]=new_belief[i]*beta.pdf(len_obs[j],non_pm[0],non_pm[1])
                elif(i==self.n_items):
                    for j in range(len(len_obs)):
                        new_belief[i]=new_belief[i]*beta.pdf(len_obs[j],absent[0],absent[1])

                else:
                    for j in range(len(len_obs)):
                        new_belief[i]=new_belief[i]*beta.pdf(len_obs[j],non_pm[0],non_pm[1])

            norm = sum(new_belief)
            c=new_belief
            new_belief = [(new_belief[x]/norm) for x in range(len(new_belief))]


        print('withoutnorm',c)
        print('finalsum',np.sum(new_belief))
        print('withnorm',new_belief)
        return new_belief
        '''
    @property
    def clicked_item(self):
        if self.click_status == Click.NOT_CLICKED:
            return None
        return self.final_menu[int(self.click_status)]  # assume indexes aligned

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return self.final_menu,self.click_status,self.quit_status  # returns raw sensor observations

    def getbelief(self):
        return self.belief_state

    def _semantic(self, n_groups, n_each_group, p_abs):
        n_items = n_groups * n_each_group
        target_value = 1

        """alpha and beta parameters for the menus with no target"""
        abs_menu_parameters = [2.1422, 13.4426]

        """alpha and beta for non-target/irrelevant menu items"""
        non_pm_group_paremeters = [5.3665, 18.8826]

        """alpha and beta for target/relevant menu items"""
        target_group_parameters = [3.1625, 1.2766]

        semantic_menu = np.array([0] * n_items)[np.newaxis]

        """randomly select whether the target is present or abscent"""
        target_type = np.random.rand()
        target_location = np.random.randint(0, n_items)

        if target_type > p_abs:
            target_group_samples = np.random.beta(target_group_parameters[0], target_group_parameters[1], (1, n_each_group))[0]
            """sample distractors from the Distractor group distribution"""
            distractor_group_samples = np.random.beta(non_pm_group_paremeters[0], non_pm_group_paremeters[1], (1, n_items))[0]

            """ step 3 using the samples above to create Organised Menu and Random Menu
                and then add the target group
                the menu is created with all distractors first
            """
            menu1 = distractor_group_samples
            target_in_group = math.ceil((target_location + 1) / float(n_each_group))
            begin = (target_in_group - 1) * n_each_group
            end = (target_in_group - 1) * n_each_group + n_each_group

            menu1[begin:end] = target_group_samples
            menu1[target_location] = target_value
        else:
            target_location = None
            menu1 = np.random.beta(abs_menu_parameters[0],abs_menu_parameters[1],(1, n_items))

        semantic_menu = menu1

        return semantic_menu, target_location

    def _get_unordered_menu(self, n_groups, n_each_group, n_grids, p_abs):
        assert(n_groups > 1)
        assert(n_each_group > 1)
        assert(n_grids > 0)
        semantic_menu, target = self._semantic(n_groups, n_each_group, p_abs)
        unordered_menu = np.random.permutation(semantic_menu)
        gridded_menu = self._griding(unordered_menu, target, n_grids)
        menu_length = n_each_group * n_groups
        coded_menu = [MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED) for i in range(menu_length)]
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[1]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.MED_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[2]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.HIGH_RELEVANCE, ItemLength.NOT_OBSERVED)
                count += 1
        return coded_menu, target

    def _griding(self, menu, target, n_levels):
        start = 1 / float(2 * n_levels)
        stop = 1
        step = 1 / float(n_levels)
        np_menu = np.array(menu)[np.newaxis]
        griding_semantic_levels = np.arange(start, stop, step)
        temp_levels = abs(griding_semantic_levels - np_menu.T)
        if target != None:
            min_index = temp_levels.argmin(axis=1)
            gridded_menu = griding_semantic_levels[min_index]
            gridded_menu[target] = 1
        else:
            min_index = temp_levels.argmin(axis=2)
            gridded_menu = griding_semantic_levels[min_index]
        return gridded_menu.T

    def _get_semantic_menu(self, n_groups, n_each_group, n_grids, p_abs):
        assert(n_groups > 0)
        assert(n_each_group > 0)
        assert(n_grids > 0)
        menu, target = self._semantic(n_groups, n_each_group, p_abs)
        gridded_menu = self._griding(menu, target, n_grids)
        menu_length = n_each_group*n_groups
        coded_menu = [MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED) for i in range(menu_length)]
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[1]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.MED_RELEVANCE, ItemLength.NOT_OBSERVED)
                elif False == (item - grids[2]).any():
                    coded_menu[count] = MenuItem(ItemRelevance.HIGH_RELEVANCE, ItemLength.NOT_OBSERVED)
                count += 1

        return coded_menu, target

'''
a=SearchEnvironment()

a.reset()
a.performAction(5)
print(Focus(a.Focus))
'''


        
