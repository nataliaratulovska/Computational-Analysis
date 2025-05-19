import random

import click
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import fasttext
from itertools import combinations
import math

# Dont use matplotlib > 3.9.X
# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2411
#TODO: Add more documentation to the functions eg param descriptions etc

def create_base_word_graph(words, model, combine):
    '''
    Create a graph of the base words.
    '''
    graph = nx.Graph()
    if not combine:
        for word in words:
            graph.add_node(word, node_color='pink', type='base_word')
        for combination in combinations(words, 2):
            similarity = model.wv.similarity(combination[0], combination[1])
            click.echo(f"Similarity between {combination[0]} and {combination[1]}: {similarity}")
            click.echo('\n')
            graph.add_edge(combination[0], combination[1], weight=similarity, color='#FBE7C6', label=similarity)
    else:
        graph.add_node(str(words), node_color='#FFAEBC')
    return graph

def create_subgraphs(base_graph, positives, model, topn, subttopn, vocabres, negatives, combine):
    '''
    Create subgraphs of the nearest neighbors of the base words.
    '''
    #TODO: Naive implementation, should be improved.
    #TODO: No functionality for negatives. Needs to be added.
    if not combine:
        for word in positives:
            neighbors = model.wv.most_similar(word, topn=topn, restrict_vocab=vocabres)
            for neighbor in neighbors:
                base_graph.add_node(neighbor[0], node_color='#FBE7C6', type='sub_word_1')
                base_graph.add_edge(word, neighbor[0], weight=neighbor[1])
                sub_neighbors = model.wv.most_similar(neighbor[0], topn=subttopn, restrict_vocab=vocabres,
                                                      )
                for sub_neighbor in sub_neighbors:
                    base_graph.add_node(sub_neighbor[0], node_color='#FBE7C6', type='sub_word_2')
                    base_graph.add_edge(neighbor[0], sub_neighbor[0], weight=sub_neighbor[1])
    else:
        neighbors = model.wv.most_similar(positives, topn=topn, restrict_vocab=vocabres)
        for neighbor in neighbors:
            base_graph.add_node(neighbor[0], node_color='#FBE7C6', type='sub_word_1')
            base_graph.add_edge(str(positives), neighbor[0], weight=neighbor[1])
            sub_neighbors = model.wv.most_similar(neighbor[0], topn=subttopn, restrict_vocab=vocabres)
            for sub_neighbor in sub_neighbors:
                base_graph.add_node(sub_neighbor[0], node_color='#FBE7C6', type='sub_word_2')
                base_graph.add_edge(neighbor[0], sub_neighbor[0], weight=sub_neighbor[1])
    return base_graph


@click.command()
@click.option('--positive', prompt='Single or multiple words seperated by a whitespace.'
                                                 'These words are used to find the nearest neighbors. If used with '
                                                 'combine the words act as a conglomerate.', required=True)
@click.option('--negative', default=None, help='Single or multiple words seperated by a whitespace.'
                                                 'These words will be excluded from the nearest neighbors.')
@click.option('--topn', default=10, help='Number of nearest neighbors to find.')
@click.option('--subttopn', default=3, help='Graph type to use.')
@click.option('--model', default='cc.en.300.bin', help='Model to use. Filename in models/ directory.')
@click.option('--combine', is_flag=False, help='Combine the positives to a conglomerate.')
#TODO: Maybe find a better default for this?
@click.option('--vocabres', default=100000, help='Graph type to use.')
@click.option('--roundcount', default=3, help='How much the similiarity measure for the labels gets rounded.')
def main(positive, negative, topn, subttopn, model, combine, vocabres, roundcount):
    '''
    Main function to load the pre-trained word vectors and find the most similar words.
    Binary fasttext model has some drawbacks. Checkout https://radimrehurek.com/gensim/models/_fasttext_bin.html
    '''
    positives = positive.split(" ")
    negatives = negative.split(" ") if negative else None

    click.echo('Using model: ' + model + '\n')
    click.echo('Loading model...\n')
    model = fasttext.load_facebook_model('models/' + model)

    click.echo('Creating Basegraph...\n')
    base_graph = create_base_word_graph(positives, model, combine=combine)

    click.echo('Creating Subgraphs...\n')
    graph = create_subgraphs(base_graph, positives, model, topn, subttopn, vocabres, negatives, combine=combine)
    #TODO: The springlayout seems to overwrite nodecolors?
    #pos = nx.spring_layout(graph, k=0.12, iterations=30, scale=2)
    pos = nx.nx_pydot.graphviz_layout(graph)
    fig = plt.figure()
    nx.draw(graph, pos, with_labels=True, font_size=8)
    edge_labels = dict([((n1, n2), round(d['weight'], roundcount))
                        for n1, n2, d in graph.edges(data=True)])
    print(edge_labels)
    #color_map = {'base_word': '#a6335f', 'sub_word_1': '#e194bc', 'sub_word_2': '#86aba7'}
    #node_colors = dict([(n, color_map[typee]) for n, typee in nx.get_node_attributes(base_graph, 'type'))
    ATTRIBUTE_NAME = 'type'
    COLOR_SCHEME = {
        'base_word': '#a6335f',
        'sub_word_1': '#e194bc',
        'sub_word_2': '#86aba7'
    }
    colors = [COLOR_SCHEME[graph.nodes[node][ATTRIBUTE_NAME]] for node in list(graph.nodes())]
    print(colors)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, cmap=COLOR_SCHEME.values(), node_size=500, edgecolors='#ffffff'
                           , alpha=0.9)
    nx.draw_networkx_edges(base_graph, pos, edge_color='#86aba7', width=4, alpha=0.8)
    nx.draw_networkx_edge_labels(base_graph, pos, edge_labels=edge_labels,
                                 font_color='#ffffff', font_size=8, rotate=False, bbox=dict(alpha=0))

    fig.set_facecolor('#2F4F4F')

    fig.set_label('Nearest Neighbor Graph for: ' + str(positives))
    plt.savefig('graph.png')



#TODO: Add your names and rearrange them alphabetically
if __name__ == "__main__":
    print("-----------------------------------\n"
          "Nearest Neighbor Visualization Tool\n"
          "Created by: Yannik Herbst, ..., ...\n"
          "-----------------------------------\n")

    main()

