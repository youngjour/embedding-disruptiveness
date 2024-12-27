import re
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

def parse_bibtex(bibtex_file):
    """Parse BibTeX file and extract citations."""
    with open(bibtex_file, 'r') as file:
        content = file.read()

    entries = content.split('@article')
    paper_ids = {}
    cited_refs = defaultdict(list)

    for entry in entries[1:]:  # Skip the first split (empty)
        unique_id = re.search(r'{(WOS:[^,]+)', entry)
        if not unique_id:
            continue
        unique_id = unique_id.group(1)
        paper_ids[unique_id] = len(paper_ids)  # Assign unique numerical ID
        
        cited = re.search(r'Cited-References = {(.*?)}', entry, re.S)
        if cited:
            references = cited.group(1).split('\n')
            for ref in references:
                ref = ref.strip()
                if ref:
                    cited_refs[unique_id].append(ref)

    return paper_ids, cited_refs

def build_citation_network(paper_ids, cited_refs):
    """Build adjacency matrix for the citation network."""
    row, col = [], []
    for citing, references in cited_refs.items():
        citing_id = paper_ids.get(citing)
        if citing_id is None:
            continue
        for ref in references:
            cited_id = paper_ids.get(ref)
            if cited_id is not None:
                row.append(citing_id)
                col.append(cited_id)
    
    n = len(paper_ids)
    data = np.ones(len(row), dtype=int)
    adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))
    return adj_matrix

def save_to_npz(adj_matrix, output_file):
    """Save the adjacency matrix to an NPZ file."""
    np.savez(output_file, data=adj_matrix.data, indices=adj_matrix.indices,
             indptr=adj_matrix.indptr, shape=adj_matrix.shape)

# Example usage:
bibtex_file = 'your_dataset.bib'  # Path to your BibTeX file
output_file = 'citation_network.npz'  # Desired output .npz file

paper_ids, cited_refs = parse_bibtex(bibtex_file)
adj_matrix = build_citation_network(paper_ids, cited_refs)
save_to_npz(adj_matrix, output_file)
print(f"Citation network saved to {output_file}")
