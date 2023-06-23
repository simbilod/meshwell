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
            n += 2

    return refinement_fields, n
