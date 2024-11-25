def calc_disruption_index(net, batch_size=None):
    """Calculate the Disruption index given by
    DI = (NF - NB) / (NR + NB + NF),
    where
    - NF: Number of papers citing a focal paper BUT NOT citing any of the reference of the focal paper
    - NB: Number of papers citing a focal paper AND citing at least one reference of the focal paper
    - NR: Number of papers not citing a focal paper but citing at least one reference of the focal paper
    net: sparse scipy matrix of a citation network. net[i,j] = 1 if i cites j.
    params: sparse.csr_matrix
    batch_size: batch size, default to None. Setting a larger batch_size makes computation faster, at the expense of memory. If None, the batch_size is set to the maximum.
    params: None or int
    Reference:
    - Funk, R. J. & Owen-Smith, J. A dynamic network measure of technological change. Manage. Sci. 63, 791–817 (2017).
    - Wu, L., Wang, D. & Evans, J.A. Large teams develop and small teams disrupt science and technology. Nature 566, 378–382 (2019). https://doi.org/10.1038/s41586-019-0941-9
    http://russellfunk.org/cdindex/static/funk_ms_2016.pdf
    """

    if batch_size is None:
        return _calc_disruption_index(net)

    # Homogenize the input data type
    net = sparse.csr_matrix(net)

    n_nodes = net.shape[0]
    n_chunks = int(n_nodes / batch_size)
    chunks = np.array_split(np.arange(n_nodes).astype(int), n_chunks)
    DI = np.zeros(n_nodes)
    netT = sparse.csr_matrix(net.T)
    for focal_node_ids in tqdm(chunks):

        is_relevant = (
            np.array(net[focal_node_ids, :].sum(axis=0)).reshape(-1)
            + np.array(net[:, focal_node_ids].sum(axis=1)).reshape(-1)
            + np.array((net[focal_node_ids, :] @ netT).sum(axis=0)).reshape(-1)
        )

        is_relevant[focal_node_ids] = -1
        supp_node_ids = np.where(is_relevant > 0)[0]
        node_ids = np.concatenate([focal_node_ids, supp_node_ids])
        subnet = net[node_ids, :][:, node_ids].copy()
        subnet.sort_indices()
        dindex = _calc_disruption_index(subnet)
        DI[focal_node_ids] = dindex[: len(focal_node_ids)]
    return DI


def _calc_disruption_index(net):
    """Calculate the Disruption index given by
    DI = (NF - NB) / (NR + NB + NF),
    where
    - NF: Number of papers citing a focal paper BUT NOT citing any of the reference of the focal paper
    - NB: Number of papers citing a focal paper AND citing at least one reference of the focal paper
    - NR: Number of papers not citing a focal paper but citing at least one reference of the focal paper
    net: sparse scipy matrix of a citation network. net[i,j] = 1 if i cites j.
    params: sparse.csr_matrix
    Reference:
    - Funk, R. J. & Owen-Smith, J. A dynamic network measure of technological change. Manage. Sci. 63, 791–817 (2017).
    - Wu, L., Wang, D. & Evans, J.A. Large teams develop and small teams disrupt science and technology. Nature 566, 378–382 (2019). https://doi.org/10.1038/s41586-019-0941-9
    http://russellfunk.org/cdindex/static/funk_ms_2016.pdf
    """

    # Homogenize the input data type
    net = sparse.csr_matrix(net)
    net.data = net.data * 0 + 1


    AAT = net @ net.T
    AAT.data = np.ones_like(AAT.data)
    AAT.setdiag(0)
    AAT.eliminate_zeros()


    AT = sparse.csr_matrix(net.copy().T)
    AT.data = np.ones_like(AT.data)

    NB = AT.multiply(AAT)

    NF = AT - NB

    NR = AAT - NB

    # Calculate the disruption
    DI = (NF.sum(axis=1) - NB.sum(axis=1)) / np.maximum(
        NR.sum(axis=1) + NB.sum(axis=1) + NF.sum(axis=1), 1
    )
    DI = np.array(DI).reshape(-1)
    return DI