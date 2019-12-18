import networkx as nx
from line  import Line  
def main():
    G = nx.read_edgelist('./data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])#read graph

    model = Line(G,embedding_size=128,order='second') #init model,order can be ['first','second','all']
    model.train(batch_size=1024,epochs=50,verbose=2)# train model
    embeddings = model.get_embeddings()
    print(embeddings)

if __name__ == '__main__':
    main()