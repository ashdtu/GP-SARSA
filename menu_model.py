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
    actions=range(0,10)   # No of actions= n_items + 2
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

        #self.log_session_variables = ["items", "target_present", "target_idx"]
        #self.log_step_variables = ["duration_focus_ms",
        #                           "duration_saccade_ms",
        #                           "action_duration",
        #                           "action",
        #                           "gaze_location"]

        # technical variables
        self.discreteStates = False
        self.outdim = 1
        self.indim = 9
        self.discreteActions = True
        self.numActions = self.n_items + 2 # look + click + quit
        self.click_status=Click.NOT_CLICKED
        self.Focus=Focus.ABOVE_MENU
        self.quit_status=Quit.NOT_QUIT


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
        #if self.training is True and len(self.training_menus) >= self.n_training_menus:
        #    idx = np.random.randint(self.n_training_menus)

        #    return self.training_menus[idx]
        # generate menu item semantic relevances and lengths


        new_menu = [MenuItem(0,0) for i in range(self.n_items)]

        if self.menu_type == "semantic":
            items, target_idx = self._semantic(self.menu_groups,
                        self.menu_items_per_group,
                        self.prop_target_abs)
        elif self.menu_type == "unordered":
            items, target_idx = self._get_unordered_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.prop_target_abs)
        else:
            raise ValueError("Unknown menu type: {}".format(self.menu_type))
        target_present=(target_idx!=None)

        length_relevances= np.random.beta(0.3,0.3,len(items))          #Length relevances, sampled as either relevant or non relevant
        target_len=1

        for i in range(len(length_relevances)):
            new_menu[i].item_relevance=items[i]
            new_menu[i].item_length=length_relevances[i]

        if target_present:
            new_menu[target_idx].item_length=target_len

        menu = (list(new_menu),target_present,target_idx)

        if self.training is True:
            self.training_menus.append(menu)
        #print('get menu',menu)
        return menu

    def reset(self):
        """ Called by the library to reset the state
        """
        self.final_menu=None
        # state hidden from agent
        self.final_menu, self.target_present, self.target_idx = self._get_menu()

        #print('Target location',self.target_idx)

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

    @property
    def clicked_item(self):
        if self.click_status == Click.NOT_CLICKED:
            return None
        return self.final_menu[int(self.click_status)]  # assume indexes aligned

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return self.final_menu  # returns raw sensor observations


    def _semantic(self, n_groups, n_each_group, p_abs):
        n_items = n_groups * n_each_group
        target_value = 1

        """alpha and beta parameters for the menus with no target"""
        abs_menu_parameters = [2.1422, 13.4426]

        """alpha and beta for non-target/irrelevant menu items"""
        non_pm_group_paremeters = [5.3665, 18.8826]

        """alpha and beta for target/relevant menu items"""
        target_group_parameters = [3.1625, 1.2766]


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
            menu1 = np.random.beta(abs_menu_parameters[0],abs_menu_parameters[1],(1, n_items))[0]

        semantic_menu = menu1

        return semantic_menu, target_location

    def _get_unordered_menu(self, n_groups, n_each_group,p_abs):
        assert(n_groups > 1)
        assert(n_each_group > 1)
        semantic_menu, target = self._semantic(n_groups, n_each_group, p_abs)
        unordered_menu = np.random.permutation(semantic_menu)
        return unordered_menu, target


