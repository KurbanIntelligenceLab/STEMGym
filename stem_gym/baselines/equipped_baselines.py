from stem_gym.agents.analyst import Analyst

def _assemble_findings(analyst: Analyst, grid_shape: tuple[int, int]):
    return analyst.get_findings(grid_shape)
