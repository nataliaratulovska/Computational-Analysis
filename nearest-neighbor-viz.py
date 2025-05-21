import click
from gensim.models import fasttext
from itertools import combinations
import pickle
import graphviz as gv
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from matplotlib import cm
from matplotlib.colors import rgb2hex

config: 'Config'
cmap = [rgb2hex(color) for color in cm.plasma((range(256)))]
negatives = []

class Config(BaseModel):
    """
    Configuration class for the FastText model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_file: str = Field(default='cc.en.300.bin', description='Model to use. Filename in models/ directory.')
    positives: list = Field(default=[], description='List of positive words to use.')
    negatives: list = Field(default=[], description='List of negative words to use.')
    topn: int = Field(default=10, description='Number of nearest neighbors to find.')
    subt_topn: int = Field(default=3, description='Graph type to use.')
    combine: bool = Field(default=False, description='Combine the positives to a conglomerate.')
    vocab_restriction: int = Field(default=100000, description='Graph type to use.')
    round_count: int = Field(default=3, description='How much the similarity measure for the labels gets rounded.')
    depth: int = Field(default=2, description='How deep the graph should be.')
    model: fasttext.FastText = Field(default=fasttext.FastText, description='Model to use.')
    single_occurrence: bool = Field(default=True, description='Dont allow multiple occurrences of the same word.')


# Dont use matplotlib > 3.9.X
# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2411

def add_sub_nodes(graph, word, parent_node_id, hierarchy=1):
    '''
    Add sub nodes to the graph.
    word: Word to use.
    parent_node_id: Node id to use.
    hierarchy: Current depth in the graph.
    visited: Set of already visited words.
    '''
    if config is None:
        raise ValueError("Config is not initialized.")
    if hierarchy > config.depth:
        return
    # Get neighbors
    neighbors = config.model.wv.most_similar(
        word,
        topn=config.subt_topn if hierarchy > 1 else config.topn,
        restrict_vocab=config.vocab_restriction
    )
    for neighbor in neighbors:
        current_node_id = str(uuid4())
        graph.node(
            current_node_id,
            neighbor[0],
            fillcolor=cmap[len(cmap) // config.depth * (hierarchy-1)],
        )
        graph.edge(
            parent_node_id,
            current_node_id,
            label=str(round(neighbor[1], config.round_count)),
        )
        # Recursively add sub-nodes
        add_sub_nodes(
            graph,
            neighbor[0],
            current_node_id,
            hierarchy=hierarchy + 1
        )

def create_graph():
    '''
    Create a base nodes from userinput.
    '''
    # Create a graph, default styles are set here
    graph = gv.Graph(comment='Graph for: ' + "-".join(config.positives),
                     graph_attr={
                        'bgcolor': '#2f4f4f',
                        'label': 'Graph for: ' + "-".join(config.positives),
                        'overlap': 'false',
                     },
                     node_attr={
                         'shape': 'circle',
                         'style': 'filled',
                         'fontcolor': '#ffffff',
                         'fontsize': '12',
                     },
                     edge_attr={
                         'color': '#FBE7C6',
                         'label': 'weight',
                         'fontsize': '10',
                         'fontcolor': '#ffffff'
                     },
                     engine='neato'
                     )
    if not config.combine:
        node_mapping = []
        for word in config.positives:
            node_id = str(uuid4())
            graph.node(node_id, word, color='#a6335f')
            add_sub_nodes(graph, word, node_id)
            node_mapping.append((node_id, word))
        for combination in combinations(node_mapping, 2):
            similarity = config.model.wv.similarity(combination[0][1], combination[1][1])
            click.echo(f"Similarity between {combination[0][1]} and {combination[1][1]}: {similarity}")
            graph.edge(combination[0][0], combination[1][0], label=str(round(similarity, config.round_count)))
    else:
        node_id = str(uuid4())
        graph.node(node_id, str(config.positives), color=cmap[0])
    return graph



@click.command()
@click.option('--positive', prompt='Single or multiple words seperated by a whitespace.'
                                                 'These words are used to find the nearest neighbors. If used with '
                                                 'combine the words act as a conglomerate.', required=True)
@click.option('--negative', default='', help='Single or multiple words seperated by a whitespace.'
                                                 'These words will be excluded from the nearest neighbors.')
@click.option('--topn', default=10, help='Number of nearest neighbors to find.')
@click.option('--subttopn', default=3, help='Graph type to use.')
@click.option('--modelfile', default='cc.en.300.bin', help='Model to use. Filename in models/ directory.')
@click.option('--combine', is_flag=True, default=False, help='Combine the positives to a conglomerate.')
@click.option('--vocabres', default=30000, help='Graph type to use.')
@click.option('--roundcount', '-r', default=3, help='How much the similiarity measure for the labels gets rounded.')
@click.option('--depth', '-d', default=2, help='How deep the graph should be.')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Verbose output.')
@click.option('--singleoccurrence', '-so', is_flag=True, default=True, help='Dont allow multiple occurrences of the same word.')
def main(positive, negative, topn, subttopn, modelfile, combine, vocabres, roundcount, depth, verbose, singleoccurrence):
    '''
    Main function to load the pre-trained word vectors and find the most similar words.
    Binary fasttext model has some drawbacks. Checkout https://radimrehurek.com/gensim/models/_fasttext_bin.html
    '''
    global config
    # Set the config
    config = Config(
        model_file=modelfile,
        positives=positive.split(" "),
        negatives=negative.split(" ") if negative else [],
        topn=topn,
        subt_topn=subttopn,
        combine=combine,
        vocab_restriction=vocabres,
        round_count=roundcount,
        depth=depth,
        single_occurrence=singleoccurrence
    )
    # Print the config
    if verbose:
        click.echo('Using model: ' + modelfile + '\n')
        click.echo('Using positives: ' + positive + '\n')
        click.echo('Using negatives: ' + negative + '\n')
        click.echo('Using topn: ' + str(topn) + '\n')
        click.echo('Using subt_topn: ' + str(subttopn) + '\n')
        click.echo('Using combine: ' + str(combine) + '\n')
        click.echo('Using vocabres: ' + str(vocabres) + '\n')
        click.echo('Using roundcount: ' + str(roundcount) + '\n')
        click.echo('Using depth: ' + str(depth) + '\n')
    else:
        click.echo('Using model: ' + modelfile + '\n')

    # Load the model
    click.echo('Loading model...\n')
    config.model = fasttext.load_facebook_model('models/' + modelfile)

    # Create the graph
    click.echo('Creating Graph...\n')
    graph = create_graph()
    graph.render(view=True, filename=f'graph-{"-".join(config.positives)}', format='png')

    # Uncomment below to create new test graph data
    # with open('test_graph_data.pickle', 'wb') as f:
    #     pickle.dump(graph, f)


def test_design():
    # For testing without the need to run the whole script
    with open('test_graph_data.pickle', 'rb') as f:
        graph = pickle.load(f)
    graph.render(view=True)



def test_design():
    # For testing without the need to run the whole script
    with open('test_graph_data.pickle', 'rb') as f:
        graph = pickle.load(f)
    pos = nx.nx_pydot.graphviz_layout(graph)
    fig = plt.figure()
    nx.draw(graph, pos, with_labels=True, font_size=8)
    edge_labels = dict([((n1, n2), round(d['weight'], 3))
                        for n1, n2, d in graph.edges(data=True)])
    print(edge_labels)
    ATTRIBUTE_NAME = 'type'
    COLOR_SCHEME = {
        'base_word': '#a6335f',
        'sub_word_1': '#e194bc',
        'sub_word_2': '#86aba7'
    }
    colors = [COLOR_SCHEME[graph.nodes[node][ATTRIBUTE_NAME]] for node in list(graph.nodes())]
    print([graph.nodes[node][ATTRIBUTE_NAME] for node in list(graph.nodes())])
    nx.draw_networkx_nodes(graph, pos, node_color=colors, cmap=COLOR_SCHEME.values(), node_size=500,
                           edgecolors='#ffffff'
                           , alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color='#86aba7', width=4, alpha=0.8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_color='#ffffff', font_size=8, rotate=False, bbox=dict(alpha=0))

    fig.set_facecolor('#2F4F4F')
    plt.savefig('test_graph.png')


#TODO: Add your names and rearrange them alphabetically
if __name__ == "__main__":
    print("-----------------------------------\n"
          "Nearest Neighbor Visualization Tool\n"
          "Created by: Yannik Herbst, Natalia Ratulovska, ...\n"
          "-----------------------------------\n")

    main()
    #test_design()

