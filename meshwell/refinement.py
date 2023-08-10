def constant_refinement(final_entity_list, refinement_field_index, model):
    # Refine
    n = refinement_field_index
    refinement_fields = []
    for entities in final_entity_list:
        if mesh_resolution := entities.base_resolution:
            model.mesh.field.add("MathEval", n)
            model.mesh.field.setString(n, "F", f"{mesh_resolution}")
            model.mesh.field.add("Restrict", n + 1)
            model.mesh.field.setNumber(n + 1, "InField", n)
            model.mesh.field.setNumbers(
                n + 1,
                "RegionsList",
                entities.get_tags(),
            )
            refinement_fields.extend((n + 1,))

            if entities.resolution and entities.resolution.keys() >= {
                "DistMax",
                "SizeMax",
            }:
                model.mesh.field.add("Distance", n + 2)
                model.mesh.field.setNumbers(n + 2, "SurfacesList", entities.boundaries)
                model.mesh.field.setNumber(n + 2, "Sampling", 100)
                model.mesh.field.add("Threshold", n + 3)
                model.mesh.field.setNumber(n + 3, "InField", n + 2)
                model.mesh.field.setNumber(
                    n + 3,
                    "SizeMin",
                    entities.resolution.get("SizeMin", mesh_resolution),
                )
                model.mesh.field.setNumber(
                    n + 3, "SizeMax", entities.resolution["SizeMax"]
                )
                model.mesh.field.setNumber(
                    n + 3, "DistMin", entities.resolution.get("DistMin", 0)
                )
                model.mesh.field.setNumber(
                    n + 3, "DistMax", entities.resolution["DistMax"]
                )
                model.mesh.field.setNumber(n + 3, "StopAtDistMax", 1)
                refinement_fields.extend((n + 3,))
                n += 2
            n += 2

    return refinement_fields, n
