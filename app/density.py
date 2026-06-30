def get_density_level(person_count):
    if person_count <= 1:
        return "Low"
    elif person_count <= 3:
        return "Medium"
    elif person_count <= 5:
        return "High"
    else:
        return "Critical"