BIGEARTHNET_CLASSES = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Transitional woodland-shrub",
    "Water courses",
    "Water bodies",
    "Coastal wetlands"
]

def get_domain_prompt(task_type, user_query):
    """Enriches standard user queries with remote-sensing domain vocabulary from BigEarthNet."""
    query_lower = user_query.lower()
    
    if "land-cover" in query_lower or "classify" in query_lower or "describe" in query_lower:
        return f"{user_query} Focus on Sentinel-2 land cover categories such as urban fabric, arable land, forest, or water bodies."
    
    return user_query
