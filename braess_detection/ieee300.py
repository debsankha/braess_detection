import requests
import networkx as nx


IEEE300_FILEPATH = "ieee_300.gpkl"
IEEE300_URL = "http://labs.ece.uw.edu/pstca/pf300/ieee300cdf.txt"

def fetch_ieee300():
    try:
        ieee300 = nx.read_gpickle(IEEE300_FILEPATH)
    except:
        r = requests.get(IEEE300_URL)
        cdf = r.text

        edges = []
        is_edges = False
        for line in cdf.split('\n'):
            if line.startswith("BRANCH DATA FOLLOWS"):
                is_edges = True
                continue
            if is_edges:
                if line.startswith("-999"):
                    break
                # this line contains an edge
                u,v,*_ = line.split()
                u = int(u)
                v = int(v)
                edges.append((u,v))

        ieee300 = nx.Graph()
        ieee300.add_edges_from(edges)
        nx.write_gpickle(ieee300, IEEE300_FILEPATH)

    assert ieee300.number_of_edges() == 409
    assert ieee300.number_of_nodes() == 300
    return ieee300

