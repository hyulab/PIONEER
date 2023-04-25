def models_to_use_cleaning(models_to_use_ori):
    models_to_use = {}
    for k, v in models_to_use_ori.items():
        p1, p2 = k[0], k[1]
        protein_in_v = v.keys()

        if len(v) == 0:
            continue

        if p1 in protein_in_v and p2 in protein_in_v:
            if len(v[p1]) > 0 and len(v[p2]) > 0:
                models_to_use[k] = v 
            elif len(v[p1]) > 0 and len(v[p2]) == 0:
                models_to_use[k] = {p1:v[p1]}
            elif len(v[p1]) == 0 and len(v[p2]) > 0:
                models_to_use[k] = {p2:v[p2]}
            else:
                pass
        elif p1 in protein_in_v and p2 not in protein_in_v:
            if len(v[p1]) > 0:
                models_to_use[k] = {p1:v[p1]}
            else:
                continue
        elif p1 not in protein_in_v and p2 in protein_in_v:
            if len(v[p2]) > 0:
                models_to_use[k] = {p2:v[p2]}
            else:
                continue
        else:
            continue
    return models_to_use
