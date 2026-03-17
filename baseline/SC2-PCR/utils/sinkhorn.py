import torch

'''
Reference:
https://github.com/magicleap/SuperGluePretrainedNetwork/blob/c0626d58c843ee0464b0fa1dd4de4059bfae0ab4/models/superglue.py#L150
'''


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    '''
    Perform Sinkhorn Normalization in Log-space for stability
    :param Z:
    :param log_mu:
    :param log_nu:
    :param iters:
    :return:
    '''
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bins0=None, bins1=None, alpha=None, iters=100):
    '''
    Perform Differentiable Optimal Transport in Log-space for stability
    :param scores:
    :param alpha:
    :param iters:
    :return:
    '''

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    if bins0 is None:
        bins0 = alpha.expand(b, m, 1)
    if bins1 is None:
        bins1 = alpha.expand(b, 1, n)
    
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm #multiply probabilities by M + N
    return Z


def rpmnet_sinkhorn(log_score, bins0, bins1, iters: int):
    b, m, n = log_score.shape
    alpha = torch.zeros(size=(b, 1, 1)).cuda()
    log_score_padded = torch.cat([torch.cat([log_score, bins0], -1),
                                  torch.cat([bins1, alpha], -1)], 1)
    
    for i in range(iters):
        #Row Normalization
        log_score_padded = torch.cat((
            log_score_padded[:, :-1, :] - (torch.logsumexp(log_score_padded[:, :-1, :], dim=2, keepdim=True)),
            log_score_padded[:, -1, None, :]),
            dim=1)

        #Column Normalization
        log_score_padded = torch.cat((
            log_score_padded[:, :, :-1] - (torch.logsumexp(log_score_padded[:, :, :-1], dim=1, keepdim=True)),
            log_score_padded[:, :, -1, None]),
            dim=2)


    return log_score_padded

