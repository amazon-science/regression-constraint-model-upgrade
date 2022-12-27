import copy

import torch

from model.evaluate import get_metrics, get_distance


def interpolate_weights(old_model, new_models, alphas, dataset, config, metrics_func, new_label_ids=None, weighted=None):
    """
    Interpolates between the weights of the old model and new model. When more than one new model is provided then the
     soup ensemble od new models will be formed first. When  new_label_ids is provided, the interpolation ignores
     the classifier weights of new classes and the new weights will be copied instead. When a weight matrix
     (usually diagonal fisher information matrix) is provided, the old model weights are multiplied with it before
     interpolation.
     @param old_model: The old model as pytorch module
     @param new_models: List of new models, each as pytorch module
     @param alphas: List of different interpolation parameters
     @param dataset: Dataset object that hold train, dev and test set
     @param config: Model config
     @param metrics_func: The function to be run during evaluation
     @param new_label_ids: Label ids used to identify the classifier weights of new classes
     @param weighted: Weighting parameter for each weight in the old model.
     @return: The metrics on train, dev and test set for each interpolation parameter alpha
    """
    all_metrics = dict()

    # Form soup ensemble of new models
    new_state_dicts = [new_model.state_dict() for new_model in new_models]
    new_model_state_dict = dict()
    for key in new_models[0].state_dict():
        if not (key.endswith('bias') or key.endswith('weight')):
            continue

        new_model_state_dict[key] = torch.mean(torch.stack([s[key] for s in new_state_dicts]), dim=0)

    for alpha in alphas:
        print('alpha', alpha)
        metrics = dict()

        # Use the old model as the basis of the interpolated model weights
        model = copy.deepcopy(old_model)
        # All weights of a model can be accessed by its state_dict
        state_dict = model.state_dict()
        for key in state_dict:
            # Be sure to only interpolate weight matrices; includes e.g. layer norm matrices
            if not (key.endswith('bias') or key.endswith('weight')):
                continue

            if weighted is not None:

                # when alpha = 1.0, there can be NaN values due to numerical instabilities when values in the weight
                # matrix are too small. In this case we replace the NaNs with the weights of the old model.
                if alpha == 1.0:
                    c = state_dict[key].detach().clone()

                # Inplace operations to modify the weights of the model.
                # State_dict initially holds the weights of the old model.
                state_dict[key] *= (alpha * weighted[key])
                state_dict[key] += ((1-alpha) * new_model_state_dict[key])
                state_dict[key] /= (alpha * weighted[key] + (1-alpha))

                # Three lines above as one-liner
                #state_dict[key].data.copy_(((alpha * weighted[key] * state_dict[key]) + ((1-alpha) * new_model_state_dict[key])) / (alpha * weighted[key] + (1-alpha)))


                if alpha == 1.0:
                    nans = state_dict[key] != state_dict[key]
                    state_dict[key][nans] = c[nans]
            else:
                # Simple linear interpolation with parameter alpha.
                # State_dict initially holds the weights of the old model.
                state_dict[key] *= alpha
                state_dict[key] += ((1 - alpha) * new_model_state_dict[key])

            # Copy classifier weights of new classes from the new model. The old model was not trained on those classes.
            if new_label_ids:
                if key == 'classifier.out_proj.weight':
                    state_dict[key][new_label_ids, :] = new_model_state_dict[key][new_label_ids, :]
                if key == 'classifier.out_proj.bias':
                    state_dict[key][new_label_ids] = new_model_state_dict[key][new_label_ids]

        # Get the L2 distance between old and interpolated model
        distance = get_distance(old_model, model)
        metrics['distance'] = distance

        # Run evaluation on the different splits
        train_metrics, _ = get_metrics(['train'], model, dataset, config, metrics_func, silent=True)
        metrics.update(train_metrics)

        dev_metrics, _ = get_metrics(['dev'], model, dataset, config, metrics_func, silent=True)
        metrics.update(dev_metrics)

        test_metrics, _ = get_metrics(['test'], model, dataset, config, metrics_func, silent=True)
        metrics.update(test_metrics)

        print('distance', distance)
        print(train_metrics)
        print(dev_metrics)
        print(test_metrics)
        print('')

        all_metrics[alpha] = metrics
    return all_metrics
