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
