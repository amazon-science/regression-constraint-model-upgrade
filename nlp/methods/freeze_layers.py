

def validate_argument(ft_layers, name):
    if ft_layers is None or ft_layers == 'all':
        name += '_fullFT'
        ft_layers = None
    else:
        name += '_pe_'
        ft_layers = set(ft_layers)
        for layer in ft_layers:
            assert layer in {'embeddings', 'classifier', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'bias', 'classifier.out_proj'}

        # generate consistent names
        for layer in ['embeddings', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'classifier', 'bias']:
            if layer in ft_layers:
                if layer == 'embeddings':
                    name += 'emb'
                if layer == 'classifier':
                    name += 'class'
                else:
                    name += str(layer)
                name += '-'
        name = name[:-1]

        ft_layers = [f'layer.{l-1}.' if isinstance(l, int) else l for l in ft_layers]
    return ft_layers, name


def apply_freeze_layers(model, ft_layers):
    if ft_layers is None:
        return model

    for name, param in model.named_parameters():
        if not any(ft_layer in name for ft_layer in ft_layers):
            param.requires_grad = False
    return model
