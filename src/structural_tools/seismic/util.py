def generate_levels_list(number_of_levels: int, add_roof: bool = True, reverse_list: bool = False) -> list[str]:
    """Generates a list of strings representing all the level names based on how many levels are specified.

    Args:
        number_of_levels (int): The number of levels including ground floor (Level 1).
        add_roof (bool, optional): Whether to add a "Roof" level. Defaults to True.

    Returns:
        list[str]: A list of strings representing all the levels in increasing order (Level 1, Level 2, etc.).
    """
    levels_list = [f"Level {i}" for i in range(1, number_of_levels + 1)]
    if add_roof:
        levels_list.append("Roof")

    if reverse_list:
        levels_list.reverse()

    return levels_list
