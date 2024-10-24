from dataclasses import dataclass, field
import dataclasses
from enum import Enum
import json
from typing import Any, Callable, Dict, List

INFINITY = 'INFINITY'

class os_encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            return
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

class organizational_specification:
    """The basic class
    """
    pass


class role(str, organizational_specification):
    pass


class group_tag(str):
    pass


class link_type(str, Enum):
    ACQ = 'ACQ'
    COM = 'COM'
    AUT = 'AUT'


@dataclass
class link(organizational_specification):
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
class compatibility(organizational_specification):
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

    def __str__(self) -> str:
        return f'({self.lower_bound},{self.upper_bound})'

    def __repr__(self) -> str:
        return f'({self.lower_bound},{self.upper_bound})'

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'cardinality':
        return cardinality(
            lower_bound=d['lower_bound'],
            upper_bound=d['upper_bound']
        )


@dataclass
class group_specifications(organizational_specification):
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
class structural_specifications(organizational_specification):
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


class goal(str, organizational_specification):
    pass


class mission(str, organizational_specification):
    pass


class plan_operator(str, Enum):
    SEQUENCE = 'SEQUENCE'
    CHOICE = 'CHOICE'
    PARALLEL = 'PARALLEL'


@dataclass
class plan(organizational_specification):
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
class social_scheme(organizational_specification):
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
class social_preference(organizational_specification):
    preferred_social_scheme: social_scheme_tag
    disfavored_scheme: social_scheme_tag

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'social_preference':
        return social_preference(
            preferred_social_scheme=d['preferred_social_scheme'],
            disfavored_scheme=d['disfavored_scheme']
        )


@dataclass
class functional_specifications(organizational_specification):
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
class deontic_specification(organizational_specification):
    role: role
    mission: mission
    time_constraint: time_constraint_type | str = time_constraint_type.ANY


@dataclass
class permission(deontic_specification, organizational_specification):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'permission':
        return permission(
            role=d['role'],
            mission=d['mission'],
            time_constraint=time_constraint_type.ANY if 'ANY' in d.get('time_constraints', d.get(
                'time_constraints', time_constraint_type.ANY)) else time_constraint_type.ANY
        )


@dataclass
class obligation(deontic_specification, organizational_specification):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'obligation':
        return obligation(
            role=d['role'],
            mission=d['mission'],
            time_constraint=time_constraint_type.ANY if 'ANY' in d.get('time_constraints', d.get(
                'time_constraints', time_constraint_type.ANY)) else time_constraint_type.ANY
        )


@dataclass
class deontic_specifications(organizational_specification):
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

    def convert_to_label(self) -> str:
        json.dumps(dataclasses.asdict(self), indent=0, cls=os_encoder) 

    def convert_to_obj(self, os_label: str) -> 'organizational_model':
        return organizational_model.from_dict(json.loads(os_label))

if __name__ == "__main__":

    print("Example: Creating an organizational model")

    ##############################################
    # Instantiate the structural specifications
    ##############################################

    # --------------------------------------------
    # Define all the roles
    roles = ["role_1", "role_2", "role_3"]
    # --------------------------------------------

    # --------------------------------------------
    # Define the roles inheritance relations
    role_inheritance_relations = {"role_2": ["role_1"], "role_3": ["role_1"]}
    # --------------------------------------------

    # --------------------------------------------
    # Define the groups

    #  - Group 1
    intra_links = [link("role_1", "role_2", link_type.ACQ),
                   link("role_2", "role_3", link_type.ACQ)]
    inter_links = [link("role_1", "role_3", link_type.ACQ)]
    intra_compatibilities = []
    inter_compatibilities = []
    role_cardinalities = {
        'role_0': cardinality(1, 1),
        'role_1': cardinality(1, 1),
        'role_2': cardinality(1, 1),
    }
    sub_group_cardinalities = {}
    group2 = group_specifications(["role_1", "role_2", "role_3"], {}, intra_links, inter_links,
                                  intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)

    #  - Group 2
    intra_links = [link("role_1", "role_2", link_type.ACQ),
                   link("role_2", "role_3", link_type.ACQ)]
    inter_links = []
    intra_compatibilities = []
    inter_compatibilities = []
    role_cardinalities = {
        'role_0': cardinality(1, 1),
        'role_1': cardinality(1, 1),
        'role_2': cardinality(1, 1)
    }
    sub_group_cardinalities = {
    }
    group1 = group_specifications(roles, {}, intra_links, inter_links,
                                  intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)
    # --------------------------------------------

    structural_specs = structural_specifications(
        roles=['role_1', 'role_2'],
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
            role='role_1',
            mission='mission1',
            time_constraint=time_constraint_type.ANY
        ),
        permission(
            role='role_3',
            mission='mission1',
            time_constraint=time_constraint_type.ANY
        )
    ]
    # --------------------------------------------
    # --------------------------------------------
    # Define the obligations
    obligations = [
        obligation(
            role='role_2',
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

    json.dump(dataclasses.asdict(org_model), open("organizational_model.json", "w"),
              indent=4, cls=os_encoder)

    os1 = organizational_model.from_dict(
        json.load(open("organizational_model.json")))

    json.dump(dataclasses.asdict(org_model), open("organizational_model.json", "w"),
              indent=4, cls=os_encoder)

    os2 = organizational_model.from_dict(
        json.load(open("organizational_model.json")))

    print(os1 == os2)
