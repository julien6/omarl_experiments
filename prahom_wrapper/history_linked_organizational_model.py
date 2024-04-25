from dataclasses import dataclass, field
import dataclasses
from enum import Enum
import json
from typing import Any, Callable, Dict, List, Tuple, Union
import re


class action (int):
    pass


class observation (object):
    pass

# A joint history is a n-tuple (n, number of agent).
# Each component is a the i-th agent's history

# From a set of joint history we would like to infer a set of organizational specifications
#
# For instance, 3 trained agents in three independent MCY environments, produce 3 joint history with 3 trained agents in MCY are mapped:
#
#  - Structural Specifications:
#     - Individual level: to the roles = "downward_bringer", "left_bringer", "upward_bringer"
#     - Social level: intra_links = .'link("downward_bringer", "leftward_bringer", acq)' and 'link("leftward_bringer", "upward_bringer", acq)'
#     - Collective level: no compatibilities between roles -> gr = (roles, {}, intra_links, {}, {}, {}, {"downward_bringer": [1,1], "left_bringer": [1,1], "upward_bringer": [1,1]}, {})
#
#  - Functional Specifications:
#     - Social Schemes:
#        - sch1 = (goals, missions, plans, mission_to_goals, mission_to_agent_cardinality)
#               - goals = {"package in bottom-left drop zone", "package in bottom-right drop zone", "package in upper-right drop zone"}
#               - missions = {"m1", "m2", "m3"}
#               - plans = {"package in upper-right drop zone" = '"package in bottom-left drop zone", "package in bottom-right drop zone"'}
#               - mission_to_goals = {"m1": "package in bottom-left drop zone", "m2": "package in bottom-right drop zone", "m3": "package in upper-right drop zone"}
#               - mission_to_agent_cardinality = {"m1": [1,1], "m2": [1,1], "m3": [1,1]}
#     - Social Preferences = {}
#
#  - Deontic Specifications:
#     - permissions = [("downward_bringer", "m1", "Any"), ("left_bringer", "m2", "Any"), ("upward_bringer", "m3", "Any")]
#     - obligations = [("downward_bringer", "m1", "Any"), ("left_bringer", "m2", "Any"), ("upward_bringer", "m3", "Any")]
#
# TODO: a role = a set of policies
#
# Partial Relation between Agents' Histories and Organizational Model (PRAHOM) is the algorithm for infering these specifications
# from trained agents' histories.
#
# The PRAHOM process workflow is roughly described below:
#
#  I) INFERRING STRUCTURAL SPECIFICATIONS
#
#      1) Inferring agents' roles:
#           For joint_history in joint_histories, for each agent's history in joint_history, infer by matching known roles or generalizing new ones
#
#           Definition of a role regarding a joint-history: a role is associated with a cluster of a dendogram produced by measuring by the length of
#           the longest common sequence between two histories, once associated, the role is the policy obtained after settifiying the longest common
#           sequence which is the root of the cluster
#
#           Example of results:
#                jhrs = {"jh1": {"agent_0": "role_0", "agent_1": "role_1", "agent_2": "role_2"},
#                  "jh2": {"agent_0": "role_1", "agent_1": "role_0", "agent_2": "role_2"},
#                 "jh3": {"agent_0": "role_2", "agent_1": "role_1", "agent_2": "role_0"}}
#
#      2) Inferring roles compatibilities:
#
#           Definition of a compatibility regarding several joint-histories
#
#           roles_compatibilities = {agent: {} for agent in agents}
#           For agent in agents
#               role_compatibilities[agent].add([jhr[agent] for jhr in jhrs])
#
#      3) Inferring roles links:
#
#           A link(role1, role2, role_type)
#
#           Definition of the non-existence of a link:
#               -> There is no link(r1, r2, rt) if generate_history(agent(r1)) == generate_history(agent(r1),agent(r2)))
#
#           Definition of a link(r1,r2,rt) regarding a joint-history
#               - rt:
#                   - acq: generate_history(agent(r1)) != generate_history(agent(r1),agent(r2, inactive=True)) => l'agent r1 peut voir et représenter
#                          l'agent r2 (qui ne fait rien mais doit être visible pour l'autre agent) car ses historiques ne sont pas les mêmes s'il n'est pas là => la sous-séquence de différence
#                          entre les deux correspond au lien link(r1,r2,rt) <-> delta_social = diff(generate_history(agent(r1),agent(r2, inactive=True)),
#                          generate_history(agent(r1)))
#                   - com: dans une relation avec acquaintance, il y a une action dans agent r1 (delta social) qui associe une observation particulière systématiquement chez r2 après plus ou moins de temps =>
#                   - aut: dans une relation avec communication, il y a une réaction (sous-séquence d'historique) qui apparait systématiquement après plus ou moins de temps de la reception de l'observation
#
#           zero_socially_impacted_history = {}
#
#           For each inferred_role in inferred_roles
#               non_socially_impacted_history = generate_history(env([agent(role=inferred_role)]))
#
#           one_socially_impacted_history = {}
#           For each inferred_role1 in inferred_roles
#               For each inferred_role2 in inferred_roles
#                 one_socially_impacted_history = generate_history(env([agent(role=inferred_role1), agent(role=inferred_role2)]))
#                 impact_of
#
#     4) Inferring sub-groups: on représente les liens entre les agents jouant en mesurant les impacts qu'ils ont les uns sur les autres, si un role joué par un agent a un impact uniquement sur son cluster (groupe), alors le lien est intra, sinon le lien est inter
#
#     5) Inferring role cardinality
#
#     6) Inferring sub-groups cardinality
#
#
# II) INFERRING FUNCTIONAL SPECIFICATIONS
#
#     1) Inferring goals:
#           - on regarde la fonction de transition des états qui mène au succès, on chosit les états seuils, l'état seuil est l'état à partir duquel
#             tous les autres états menant au succès découlent. Si tous les états sont des états seuils.
#           - si tous les états sont des états seuils, on échantillone n états pris équitablement dans la fonction de transition
#
#     3) Inferring missions
#           - pour un objectif donné, à quel point un agent y contribue
#
#     4) Inferring plans
#           - en même temps que inferring goals
#
#     5) Inferring missions to goals
#
#     6) Inferring mission to agent cardinality
#


class history:
    """
    A convenient joint history representation to link organizational specifications

    Example:
        h1 = history("^(\'14\',\'8\').*?(\'17\',\'1\')$") # all histories starting with ('14','8') and ending with ('17','1')
        h2 = history({"14": ["8", "12"], "0": ["1"]}) # all histories where all of th observations '14' and '0' are coupled respectively with either '8' or '12', and '1'
        h3 = history([[('0','1'),('8','19'),('21','3')],[('87','0'),('9','8'),('0', '5')]]) # all histories that are equal to those
    """

    history_representations: List[Union[str, List[Tuple[observation,
                                                        action]], Dict[observation, List[action]]]] = [["^.*$"], [], {}]

    observation_space: List[observation]
    action_space: List[action]

    regex_representations: List[str] = []
    list_representations: List[List[Tuple[observation, action]]] = []
    dict_representations: List[Dict[observation, List[action]]] = []

    def __init__(self, observation_space: List[observation], action_space: List[action], history_representations: List[Union[str, List[Tuple[observation, action]], Dict[observation, List[action]]]] = [["^.*$"], [], {}]):
        self.history_representations = history_representations
        """
        Represents all of the expected histories linked to organizational specification by combining three ways:
         - lists: describing exhaustively all of the possible histories
         - regex: describing a subset of histories in a short expression
         - dict: describing a history as a relation between an observation and some actions

        Parameters:
        history_representations (List[Union[List[str], List[Tuple[observation, action]], Dict[observation, List[action]]]]): The expected history representations

        Example:
        >>> linked_history([[("5", "4"),("9", "17")], {"7": ["14", "28"]}, "^("41", "5").*?("6", "23")$", [("5", "1"),("9", "12")]])
        """

        observation_space = observation_space
        action_space = action_space

        for histories_representation in self.history_representations:
            if isinstance(histories_representation, str):
                self.regex_representations += [histories_representation]
            if isinstance(histories_representation, List):
                self.list_representations += [histories_representation]
            if isinstance(histories_representation, Dict):
                self.dict_representations += [histories_representation]

    def match(self, last_history: List) -> bool:
        """
        Check an history is compliant with the expected histories

        Parameters:
        last_history (List[Tuple[observation, action]]): The last played history.

        Returns:
        bool: The compliance to the expected histories

        Example:
        >>> history([{"5": "1", "9": "12", "42": ["43", "6"]}])
                .match([("5", "1"),("9", "12")])
        True
        >>> history([{"5": "1", "8": "12", "47": ["43", "6"]}])
                .match([("5", "4"),("9", "14")])
        False
        """

        # check last_history is compliant the regex
        for regex_representation in self.regex_representations:
            if (re.fullmatch(regex_representation, str(last_history)) == None):
                return False

        # check last_history is indeed already described
        if not last_history in self.list_representations:
            return False

        # check last_history is compliant with dict representations
        for dict_representation in self.dict_representations:
            for lh_obs, lh_act in last_history:
                if lh_obs in dict_representation.keys():
                    if dict_representation[lh_obs] != lh_act:
                        return False
        
        return True

    def next(self, last_history: List[Tuple[observation, action]], last_observation: observation) -> List[action]:
        """
        Infer the possible next actions after an history has been played and a last observation is received
        so next actions are compliant with the expected histories

        Parameters:
        last_history (List[Tuple[observation, action]]): The last played history.
        last_observation (observation): The last observation received

        Returns:
        List[action]: The possible actions to be chosen

        Example:
        >>> history(history_dict={"5": "1", "9": "12", "42": ["43", "6"]})
                .next([("5", "1"),("9", "12")], "42")
        ["43", "6"]
        """
        if self.match(last_history):

            # check the next action according to list representations
            list_next_action = None
            for list_representation in self.list_representations:
                if last_history in list_representation:
                    if list_representation[len(last_history)][0] == last_observation:
                        if list_next_action is None:
                            list_next_action = list_representation[len(last_history)][1]
                        elif list_next_action != list_representation[len(last_history)][1]:
                            return False

            # check the next action according to dict representations
            dict_next_action = None
            for dict_representation in self.dict_representations:
                if last_observation in dict_representation.keys():
                    if dict_next_action is None:
                        dict_next_action = self.dict_representations[last_observation]
                    elif dict_next_action != self.dict_representations[last_observation]:
                        return False
            
            regex_next_action = None
            for regex_representation in self.regex_representations:
                # finding the subpart of a regex_representation that matches last_history
                if(re.findall(regex_representation, str(last_history))[0] == str(last_history)):
                    next_regex_representation = regex_representation[len(str(last_history)):]
                    possible_actions = []
                    for action in self.action_space:
                        if(re.findall(next_regex_representation, str(action))[0] == str(action)):
                            possible_actions += [action]

        return False


INFINITY = 'INFINITY'


class role(str):
    pass


class group_tag(str):
    pass


class link_type(str, Enum):
    ACQ = 'ACQ'
    COM = 'COM'
    AUT = 'AUT'


@dataclass
class link:
    source: role
    destination: role
    type: link_type

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'link':
        return link(
            source=d['source'],
            destination=d['destination'],
            type=link_type(d['type'])
        )


@dataclass
class compatibility:
    source: role
    destination: role

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'compatibility':
        return compatibility(
            source=d['source'],
            destination=d['destination']
        )


@dataclass
class cardinality:
    lower_bound: int | str
    upper_bound: int | str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'cardinality':
        return cardinality(
            lower_bound=d['lower_bound'],
            upper_bound=d['upper_bound']
        )


@dataclass
class group_specifications:
    roles: List[role]
    sub_groups: Dict[group_tag, 'group_specifications']
    intra_links: List[link]
    inter_links: List[link]
    intra_compatibilities: List[compatibility]
    inter_compatibilities: List[compatibility]
    # by default: cardinality(0, INFINITE)
    role_cardinalities: Dict[role, cardinality]
    # by default: cardinality(0, INFINITE)
    sub_group_cardinalities: Dict[group_tag, cardinality]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'group_specifications':
        return group_specifications(
            roles=[role(r) for r in d['roles']],
            sub_groups={group_tag(k): group_specifications.from_dict(v)
                        for k, v in d['sub_groups'].items()},
            intra_links=[link.from_dict(l) for l in d['intra_links']],
            inter_links=[link.from_dict(l) for l in d['inter_links']],
            intra_compatibilities=[compatibility.from_dict(
                c) for c in d['intra_compatibilities']],
            inter_compatibilities=[compatibility.from_dict(
                c) for c in d['inter_compatibilities']],
            role_cardinalities={role(k): cardinality.from_dict(v)
                                for k, v in d['role_cardinalities'].items()},
            sub_group_cardinalities={group_tag(k): cardinality.from_dict(
                v) for k, v in d['sub_group_cardinalities'].items()}
        )


@dataclass
class structural_specifications:
    roles: List[role]
    role_inheritance_relations: Dict[role, List[role]]
    root_groups: Dict[group_tag, group_specifications]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'structural_specifications':
        return structural_specifications(
            roles=[role(r) for r in d['roles']],
            role_inheritance_relations={role(
                k): [role(r) for r in v] for k, v in d['role_inheritance_relations'].items()},
            root_groups={group_tag(k): group_specifications.from_dict(v)
                         for k, v in d['root_groups'].items()}
        )


class goal(str):
    pass


class mission(str):
    pass


class plan_operator(str, Enum):
    SEQUENCE = 'SEQUENCE'
    CHOICE = 'CHOICE'
    PARALLEL = 'PARALLEL'


@dataclass
class plan:
    goal: goal
    sub_goals: List['goal']
    operator: plan_operator
    probability: float = 1.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'plan':
        return plan(
            goal=d['goal'],
            sub_goals=[goal(g) for g in d['sub_goals']],
            operator=plan_operator(d['operator']),
            probability=d['probability']
        )


@dataclass
class social_scheme:
    goals: List[goal]
    missions: List[mission]
    goals_structure: plan
    mission_to_goals: Dict[mission, List[goal]]
    # by default: cardinality(1, INFINITE)
    mission_to_agent_cardinality: Dict[mission, cardinality]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'social_scheme':
        return social_scheme(
            goals=[goal(g) for g in d['goals']],
            missions=[mission(m) for m in d['missions']],
            goals_structure=plan.from_dict(d['goals_structure']),
            mission_to_goals={k: [goal(g) for g in v]
                              for k, v in d['mission_to_goals'].items()},
            mission_to_agent_cardinality={k: cardinality.from_dict(
                v) for k, v in d['mission_to_agent_cardinality'].items()}
        )


class social_scheme_tag(str):
    pass


@dataclass
class social_preference:
    preferred_social_scheme: social_scheme_tag
    disfavored_scheme: social_scheme_tag

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'social_preference':
        return social_preference(
            preferred_social_scheme=d['preferred_social_scheme'],
            disfavored_scheme=d['disfavored_scheme']
        )


@dataclass
class functional_specifications:
    social_scheme: Dict[social_scheme_tag, social_scheme]
    social_preferences: List[social_preference]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'functional_specifications':
        return functional_specifications(
            social_scheme={k: social_scheme.from_dict(v)
                           for k, v in d['social_scheme'].items()},
            social_preferences=[social_preference.from_dict(
                p) for p in d['social_preferences']]
        )


class time_constraint_type(str, Enum):
    ANY = 'ANY'


@dataclass
class deontic_specification:
    role: role
    mission: mission
    time_constraint: time_constraint_type | str = time_constraint_type.ANY


@dataclass
class permission(deontic_specification):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'permission':
        return permission(
            role=d['role'],
            mission=d['mission'],
            time_constraint=time_constraint_type.ANY if 'ANY' in d.get('time_constraints', d.get(
                'time_constraints', time_constraint_type.ANY)) else time_constraint_type.ANY
        )


@dataclass
class obligation(deontic_specification):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'obligation':
        return obligation(
            role=d['role'],
            mission=d['mission'],
            time_constraint=time_constraint_type.ANY if 'ANY' in d.get('time_constraints', d.get(
                'time_constraints', time_constraint_type.ANY)) else time_constraint_type.ANY
        )


@dataclass
class deontic_specifications:
    permissions: List[permission]
    obligations: List[obligation]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'deontic_specifications':
        return deontic_specifications(
            permissions=[permission.from_dict(p) for p in d['permissions']],
            obligations=[obligation.from_dict(o) for o in d['obligations']]
        )


@dataclass
class organizational_model:
    structural_specifications: structural_specifications
    functional_specifications: functional_specifications
    deontic_specifications: deontic_specifications

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'organizational_model':
        return organizational_model(
            structural_specifications=structural_specifications.from_dict(
                d['structural_specifications']),
            functional_specifications=functional_specifications.from_dict(
                d['functional_specifications']),
            deontic_specifications=deontic_specifications.from_dict(
                d['deontic_specifications'])
        )


if __name__ == "__main__":

    print("Example: Creating an organizational model")

    ##############################################
    # Instantiate the structural specifications
    ##############################################

    # --------------------------------------------
    # Define all the roles
    roles = ["role1", "role2", "role3"]
    # --------------------------------------------

    # --------------------------------------------
    # Define the roles inheritance relations
    role_inheritance_relations = {"role2": ["role1"], "role3": ["role1"]}
    # --------------------------------------------

    # --------------------------------------------
    # Define the groups

    #  - Group 1
    intra_links = [link("role1", "role2", link_type.AUT),
                   link("role2", "role3", link_type.ACQ)]
    inter_links = [link("role1", "role3", link_type.ACQ)]
    intra_compatibilities = [compatibility("role1", "role3")]
    inter_compatibilities = [compatibility("role2", "role3")]
    role_cardinalities = {
        'role1': cardinality(1, 4),
        'role2': cardinality(0, INFINITY),
    }
    sub_group_cardinalities = {
        'group1': cardinality(1, INFINITY),
        'group2': cardinality(0, INFINITY),
    }
    group2 = group_specifications(["role1", "role2", "role3"], {}, intra_links, inter_links,
                                  intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)

    #  - Group 2
    intra_links = [link("role1", "role2", link_type.AUT),
                   link("role2", "role3", link_type.ACQ)]
    inter_links = [link("role1", "role3", link_type.ACQ)]
    intra_compatibilities = [compatibility("role1", "role3")]
    inter_compatibilities = [compatibility("role2", "role3")]
    role_cardinalities = {
        'role1': cardinality(1, 4),
        'role2': cardinality(0, INFINITY),
    }
    sub_group_cardinalities = {
        'group1': cardinality(1, INFINITY),
        'group2': cardinality(0, INFINITY),
    }
    group1 = group_specifications(roles, {"group2": group2}, intra_links, inter_links,
                                  intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)
    # --------------------------------------------

    structural_specs = structural_specifications(
        roles=['role1', 'role2'],
        role_inheritance_relations=role_inheritance_relations,
        root_groups={"group1": group1}
    )

    ##############################################
    # Instantiate the functional specifications
    ##############################################

    # --------------------------------------------
    # Instantiate the social schemes
    goals = ['goal1', 'goal2', 'goal3']
    missions = ['mission1', 'mission2']
    goals_structure = plan(
        goal='goal1',
        sub_goals=['goal2', 'goal3'],
        operator=plan_operator.SEQUENCE,
        probability=1.0
    )
    mission_to_goals = {
        'mission1': ['goal1', 'goal2'],
        'mission2': ['goal1', 'goal3'],
    }
    mission_to_agent_cardinality = {
        'mission1': cardinality(1, 1),
        'mission2': cardinality(1, 1),
    }

    social_schemes = {
        'scheme1': social_scheme(
            goals=goals,
            missions=missions,
            goals_structure=goals_structure,
            mission_to_goals=mission_to_goals,
            mission_to_agent_cardinality=mission_to_agent_cardinality
        ),
        'scheme2': social_scheme(
            goals=goals,
            missions=missions,
            goals_structure=goals_structure,
            mission_to_goals=mission_to_goals,
            mission_to_agent_cardinality={
                'mission1': cardinality(1, 9),
                'mission2': cardinality(1, 1),
            }
        )
    }
    # --------------------------------------------

    # --------------------------------------------
    # Instantiate the social preferences
    social_preferences = [
        social_preference(
            preferred_social_scheme='scheme1',
            disfavored_scheme='scheme2'
        )
    ]
    # --------------------------------------------

    functional_specs = functional_specifications(
        social_scheme=social_schemes,
        social_preferences=social_preferences
    )

    ##############################################
    # Instantiate the deontic specifications
    ##############################################

    # --------------------------------------------
    # Define the permissions
    permissions = [
        permission(
            role='role1',
            mission='mission1',
            time_constraint=time_constraint_type.ANY
        ),
        permission(
            role='role3',
            mission='mission1',
            time_constraint=time_constraint_type.ANY
        )
    ]
    # --------------------------------------------
    # --------------------------------------------
    # Define the obligations
    obligations = [
        obligation(
            role='role2',
            mission='mission2',
            time_constraint=time_constraint_type.ANY
        )
    ]
    # --------------------------------------------

    deontic_specs = deontic_specifications(
        permissions=permissions,
        obligations=obligations
    )

    ############################################
    # Instantiate the organizational model
    org_model = organizational_model(
        structural_specifications=structural_specs,
        functional_specifications=functional_specs,
        deontic_specifications=deontic_specs
    )
    ############################################

    print(org_model)

    print("="*30)

    class os_encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, dict):
                return
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    json.dump(dataclasses.asdict(org_model), open("organizational_model.json", "w"),
              indent=4, cls=os_encoder)

    os1 = organizational_model.from_dict(
        json.load(open("organizational_model.json")))

    json.dump(dataclasses.asdict(org_model), open("organizational_model.json", "w"),
              indent=4, cls=os_encoder)

    os2 = organizational_model.from_dict(
        json.load(open("organizational_model.json")))

    print(os1 == os2)
