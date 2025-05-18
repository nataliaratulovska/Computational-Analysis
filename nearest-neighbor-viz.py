import click
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import fasttext
from itertools import combinations

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
            graph.add_node(word, node_color='#FFAEBC')
        for combination in combinations(words, 2):
            similarity = model.wv.similarity(combination[0], combination[1])
            click.echo(f"Similarity between {combination[0]} and {combination[1]}: {similarity}")
            click.echo('\n')
            graph.add_edge(combination[0], combination[1], weight=similarity, color='#FBE7C6', label=similarity)
    else:
        graph.add_node(str(words), node_color='#FFAEBC')
    return graph

def create_subgraphs(base_graph, positives, model, topn, subttopn, vocabres, combine):
    '''
    Create subgraphs of the nearest neighbors of the base words.
    '''
    #TODO: Naive implementation, should be improved.
    #TODO: No functionality for negatives. Needs to be added.
    if not combine:
        for word in positives:
            neighbors = model.wv.most_similar(word, topn=topn, restrict_vocab=vocabres)
            for neighbor in neighbors:
                base_graph.add_node(neighbor[0], node_color='#FBE7C6')
                base_graph.add_edge(word, neighbor[0], weight=neighbor[1])
                sub_neighbors = model.wv.most_similar(neighbor[0], topn=subttopn, restrict_vocab=vocabres)
                for sub_neighbor in sub_neighbors:
                    base_graph.add_node(sub_neighbor[0], node_color='#FBE7C6')
                    base_graph.add_edge(neighbor[0], sub_neighbor[0], weight=sub_neighbor[1])
    else:
        neighbors = model.wv.most_similar(positives, topn=topn, restrict_vocab=vocabres)
        for neighbor in neighbors:
            base_graph.add_node(neighbor[0], node_color='#FBE7C6')
            base_graph.add_edge(str(positives), neighbor[0], weight=neighbor[1])
            sub_neighbors = model.wv.most_similar(neighbor[0], topn=subttopn, restrict_vocab=vocabres)
            for sub_neighbor in sub_neighbors:
                base_graph.add_node(sub_neighbor[0], node_color='#FBE7C6')
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
def main(positive, negative, topn, subttopn, model, combine, vocabres):
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
    base_graph = create_base_word_graph(positives, model)

    click.echo('Creating Subgraphs...\n')
    graph = create_subgraphs(base_graph, positives, model, topn, subttopn, vocabres, negatives)
    #TODO: The springlayout seems to overwrite nodecolors?
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, font_size=10)
    edge_labels = dict([((n1, n2), d['weight'])
                        for n1, n2, d in graph.edges(data=True)])

    nx.draw_networkx_edge_labels(base_graph, pos, edge_labels=edge_labels,
                                 font_color='#B4F8C8', font_size=8)
    plt.savefig('graph.png')


#TODO: Add your names and rearrange them alphabetically
if __name__ == "__main__":
    print("-----------------------------------\n"
          "Nearest Neighbor Visualization Tool\n"
          "Created by: Yannik Herbst, ..., ...\n"
          "-----------------------------------\n")
    main()

