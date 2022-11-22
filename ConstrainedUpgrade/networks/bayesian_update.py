import torch


def shannon_entropy(prob):
    return -(torch.log(prob) * prob).sum()


def bayesian_combination(output_list, est_confu_mat_list, strategy='MB', cost_ratio=2.5):
    """
    Backward-Compatible Prediction Updates, based on
    `"Backward-Compatible Prediction Updates: A Probabilistic Approach" <https://arxiv.org/abs/2107.01057>`_.
    Args:
        est_confu_mat_list (list): List of estimation confusion matrix
        output_list (list): List of outputs from different version of models
        strategy (string): strategy of update posterior, MaxBelief (MB), MaxBeliefMinEntropy (MBME), Cost Ratio (CR)
        cost_ratio (float): cost ratio for flip updates
    return:
         max_belief_out (tensor): tensor of prediction that can maxium posterior belief
         max_belief_prob_output (list): List of samples' posterior belief vector
    """
    K = est_confu_mat_list[0].shape[0]
    max_belief_output = []
    max_belief_prob_output = []
    sample_amount = output_list[0].shape[0]
    preds_list = [torch.argmax(output, dim=1) for output in output_list]
    est_confu_mat_list = [i.cuda(torch.cuda.current_device()) for i in est_confu_mat_list]

    for index in range(sample_amount):
        prior = torch.ones(K).cuda(torch.cuda.current_device()) * 1. / K
        for est_mat, preds in zip(est_confu_mat_list, preds_list):
            normi = est_mat[preds[index]] * prior
            denormi = torch.dot(est_mat[preds[index]], prior)
            temp = normi / denormi
            if strategy == 'MB':
                prior = temp
            elif strategy == 'MBME':
                if shannon_entropy(temp) < shannon_entropy(prior):
                    prior = temp
            elif strategy == 'CR':
                if ((temp.argmax() != prior.argmax()) and (temp.max() / temp[prior.argmax()] > cost_ratio)) \
                        or (temp.argmax() == prior.argmax()):
                    update_pred = temp.argmax()
                else:
                    update_pred = prior.argmax()
                prior = temp
        if strategy == 'MB' or strategy == 'MBME':
            max_belief_output.append(torch.argmax(prior))
        else:
            max_belief_output.append(update_pred)
        max_belief_prob_output.append(prior)
    max_belief_output = torch.tensor(max_belief_output)
    max_belief_prob_output = torch.stack(max_belief_prob_output)
    return max_belief_output, max_belief_prob_output