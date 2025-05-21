import click
from gensim.models import fasttext
from itertools import combinations
import pickle
import graphviz as gv
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from matplotlib import cm
from matplotlib.colors import rgb2hex

# Global configuration object, initialized later in the main function
config: 'Config'

# Generate a colormap using the plasma color scheme
cmap = [rgb2hex(color) for color in cm.plasma((range(256)))]

# List to store negative words
negatives = []


class Config(BaseModel):
    """
    Configuration class for the FastText model.

    Attributes:
        model_file (str): Path to the FastText model file.
        positives (list): List of positive words to use for similarity search.
        negatives (list): List of negative words to exclude from similarity search.
        topn (int): Number of nearest neighbors to find.
        subt_topn (int): Number of neighbors for sub-nodes.
        combine (bool): Whether to combine positive words into a conglomerate.
        vocab_restriction (int): Restrict vocabulary size for similarity search.
        round_count (int): Number of decimal places to round similarity scores.
        depth (int): Depth of the graph hierarchy.
        model (fasttext.FastText): Loaded FastText model instance.
        single_occurrence (bool): Whether to allow multiple occurrences of the same word.
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


def add_sub_nodes(graph, word, parent_node_id, hierarchy=1):
    """
    Recursively add sub-nodes to the graph.

    Args:
        graph (graphviz.Graph): The graph object to which nodes and edges are added.
        word (str): The word for which neighbors are added as sub-nodes.
        parent_node_id (str): The ID of the parent node.
        hierarchy (int): Current depth in the graph hierarchy.

    Raises:
        ValueError: If the global `config` object is not initialized.
    """
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
    """
    Create a graph based on user input and configuration.

    Returns:
        graphviz.Graph: The generated graph object.
    """
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
    """
    Main function to load the pre-trained word vectors and find the most similar words.

    Args:
        positive (str): Positive words separated by whitespace.
        negative (str): Negative words separated by whitespace.
        topn (int): Number of nearest neighbors to find.
        subttopn (int): Number of neighbors for sub-nodes.
        modelfile (str): Path to the FastText model file.
        combine (bool): Whether to combine positive words into a conglomerate.
        vocabres (int): Restrict vocabulary size for similarity search.
        roundcount (int): Number of decimal places to round similarity scores.
        depth (int): Depth of the graph hierarchy.
        verbose (bool): Whether to enable verbose output.
        singleoccurrence (bool): Whether to allow multiple occurrences of the same word.
    """
    global config
    # Set the config
    config = Config(
        model_file=modelfile,
        positives=positive.split(" ") if not combine else [positive],
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
    run = True
    # Main loop: To not reload the model every time, we keep it in memory, so we can create multiple graphs.
    while run:
        # Create the graph
        click.echo('Creating Graph...\n')
        graph = create_graph()
        # Save the graph
        graph.render(view=True, filename=f'graph-{"-".join(config.positives)}', format='png')
        click.echo(f'Graph created. Saved as: graph-{"-".join(config.positives)}.png\n')
        # Ask if the user wants to create a new graph
        run = click.confirm('Do you want to create a new graph?', default=False)
        if run:
            config.positives = click.prompt('Single or multiple words seperated by a whitespace.'
                                                 'These words are used to find the nearest neighbors. If used with '
                                                 'combine the words act as a conglomerate.', default=None).split(" ")


#TODO: Add your names and rearrange them alphabetically
if __name__ == "__main__":
    print("-----------------------------------\n"
          "Nearest Neighbor Visualization Tool\n"
          "Created by: Yannik Herbst, Natalia Ratulovska, ...\n"
          "-----------------------------------\n")
    main()
