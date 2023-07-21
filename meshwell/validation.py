from collections import OrderedDict

def validate_dimtags(dimtags):
    dims = [dim for dim, tag in dimtags]
    if len(set(dims)) != 1:
        raise ValueError(
            "All the entities corresponding to a mesh label must be of the same dimension."
        )
    else:
        return dims[0]


def unpack_dimtags(dimtags):
    dim = [dim for dim, tag in dimtags][0]
    tags = [tag for dim, tag in dimtags]
    if any(isinstance(el, list) for el in tags):
        tags = [item for sublist in tags for item in sublist]
    return [(dim, tag) for tag in tags]


def parse_entities(entities_dict):

    entities_3D = OrderedDict()
    entities_2D = OrderedDict()
    entities_1D = OrderedDict() 
    entities_0D = OrderedDict()

    for key, value in entities_dict:
        if value.dim == 3:
            entities_3D[key] = value
        elif value.dim == 2:
            entities_2D[key] = value
        elif value.dim == 1:
            entities_1D[key] = value
        elif value.dim == 0:
            entities_0D[key] = value
    
    return entities_3D, entities_2D, entities_1D, entities_0D