import random
from typing import List

from .rules import Rule_Wrapper


def sample_rules(max_components: int = 1, configuration: str = None) -> List[list]:
    """Simple rule sampler focused on visual attribute progressions.
    Returns list[component][rules]. Prioritizes Type and Size changes for clear patterns.
    """
    component_idx = 0
    
    # Focus on visual attributes that create clear progressions for all configurations
    # Avoid Number/Position rules that can cause empty panels with single entities
    main_attr = random.choice(["Type", "Size"])
    rules = [Rule_Wrapper("Progression", main_attr, 1, component_idx)]
    
    # Add a second attribute for variety
    remaining_attrs = [attr for attr in ["Type", "Size"] if attr != main_attr]
    if remaining_attrs:
        extra_attr = random.choice(remaining_attrs)
        rules.append(Rule_Wrapper("Progression", extra_attr, 1, component_idx))
    
    return [rules]


