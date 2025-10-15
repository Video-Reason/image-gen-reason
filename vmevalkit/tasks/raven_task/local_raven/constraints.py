# -*- coding: utf-8 -*-
# Constraint generation for RAVEN rules

from .const import (
    ANGLE_MAX, ANGLE_MIN, NUM_MAX,
    NUM_MIN, SIZE_MAX, SIZE_MIN, TYPE_MAX, TYPE_MIN, UNI_MAX, UNI_MIN
)


def gen_layout_constraint(pos_type, pos_list, 
                          num_min=NUM_MIN, num_max=NUM_MAX,
                          uni_min=UNI_MIN, uni_max=UNI_MAX):
    constraint = {"Number": [num_min, num_max],
                  "Position": [pos_type, pos_list[:]],
                  "Uni": [uni_min, uni_max]}
    return constraint


def gen_entity_constraint(type_min=TYPE_MIN, type_max=TYPE_MAX, 
                          size_min=SIZE_MIN, size_max=SIZE_MAX, 
                          angle_min=ANGLE_MIN, angle_max=ANGLE_MAX):
    constraint = {"Type": [type_min, type_max],
                  "Size": [size_min, size_max],
                  "Angle": [angle_min, angle_max]}
    return constraint


def rule_constraint(rule_list, num_min, num_max, 
                               uni_min, uni_max,
                               type_min, type_max,
                               size_min, size_max):
    """Generate constraints given the rules and the original constraints 
    from layout and entity. Note that each attribute has at most one rule
    applied on it.
    Arguments:
        rule_list(ordered list of Rule): all rules applied to this layout
        others (int): boundary levels for each attribute in a layout; note that
            num_max + 1 == len(layout.position.values)
    Returns:
        layout_constraint(dict): a new layout constraint
        entity_constraint(dict): a new entity constraint
    """
    assert len(rule_list) > 0
    for rule in rule_list:
        if rule.name == "Progression":
            # rule.value: add/sub how many levels
            if rule.attr == "Number":
                if rule.value > 0:
                    num_max = num_max - rule.value * 2
                else:
                    num_min = num_min - rule.value * 2
            if rule.attr == "Position":
                # Progression here means moving in Layout slots in order
                abs_value = abs(rule.value)
                num_max = num_max - abs_value * 2
            if rule.attr == "Type":
                if rule.value > 0:
                    type_max = type_max - rule.value * 2
                else:
                    type_min = type_min - rule.value * 2
            if rule.attr == "Size":
                if rule.value > 0:
                    size_max = size_max - rule.value * 2
                else:
                    size_min = size_min - rule.value * 2
        
    return gen_layout_constraint(None, [], 
                                 num_min, num_max, 
                                 uni_min, uni_max), \
           gen_entity_constraint(type_min, type_max, 
                                 size_min, size_max)
