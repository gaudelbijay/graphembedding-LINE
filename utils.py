def partition_num(num,workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers+[num%workers]

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    for i,node in enumerate(graph.nodes()):
        node2idx[node]=i
        idx2node.append(node)
    return idx2node,node2idx


        
