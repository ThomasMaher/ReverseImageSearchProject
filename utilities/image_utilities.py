from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utilities import image_to_vec

embedding_generator = image_to_vec.ImgageVectorizer()

def show_image(image_path, title=None):
    """Renders image given the file path.

    Args:
        image_path (str): path-like string to an image in the file system.
        title (None, optional): Used by other internal functions. Defaults to None.

    Returns:
        None
    """    
    image = Image.open(image_path)
    imshow(np.asarray(image))
    if title is not None:
        plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    show()

    return None


def show_query_results(inventory, search_results, new=False, save=False, saving_path='result.png'):
    """Plots the first 6 similar images according to the query results. This implementation
    only supports single vector queries.

    The inventory argument must be a pandas DataFrame and must at least contain two features named
    image_path and milvus_id. The former representing the path to the images in the filesystem and 
    the latter representing the assigned milvus ids.

    Args:
        inventory (pd.DataFrame): Dataframe containing the image inventory. Read above for more information.
        search_results (milvus.client.abstract.TopKQueryResult): resulting object from milvus query.
        new (bool, optional): Used by internal functions. Defaults to False.
        save (boo, optional): If True, the image will be saved. Defaults to False.
        saving_path (str, optional): Path on where to save the generated subplot. Defaults to 'result.png'.

    Returns:
        None
    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18), (ax19, ax20, ax21)) = plt.subplots(7, 3, figsize=(14,18))

    for ax, i in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21], search_results.id_array[0][:]):
        resulting_df = inventory[inventory.milvus_ids == i]
        image_path = resulting_df.image_path.values[0]
        image = Image.open(image_path)
        ax.imshow(np.asarray(image))
        ax.set_xticks([]) 
        ax.set_yticks([]) 
    if new:
        pass
    else:
        ax1.set_title("Original Image")
    #if save:
    #    plt.savefig(saving_path, bbox_inches="tight", dpi=600)
    return None


def get_similar_images(client, collection_name, image, top_k=21, nprobe=16, plot=False, inventory=None, save=False, saving_path="results.png"):
    """Gets similar images from the milvus server. Plotting option is avaliable.

    The inventory argument must be a pandas DataFrame and must at least contain two features named
    image_path and milvus_id. The former representing the path to the images in the filesystem and 
    the latter representing the assigned milvus ids.

    Args:
        client (object): milvus client.
        collection_name (str): milvus collection name to query.
        image (str): path-like string to an image for similarity retrieval.
        top_k (int, optional): top number of neighbors to return. Defaults to 6.
        nprobe (int, optional): nprobe. Defaults to 16.
        plot (bool, optional): If True, a plot of the nearest images will be rendered. Only works if top_k > 6.
        inventory (pd.DataFrame, optional): Dataframe containing the image inventory. Read above for more information.
        save (bool, optional): If True, the plotted image will be saved.
        saving_path (str, optional): Name for the image file to be saved.

    Returns:
        milvus.client.abstract.TopKQueryResult: resulting object from milvus query.
    """    
    if isinstance(image, str):
        new_vector = embedding_generator.get_embedding(image)
        new_vector = new_vector.reshape(1, -1)
    else:
        new_vector = image

    param = {
        'collection_name': collection_name,
        'query_records': new_vector,
        'top_k': top_k,
        'params': {"nprobe": nprobe},
    }
    status, results = client.search(**param)
    if status.OK():
        if plot:
            if isinstance(image, str):
                if top_k >= 6:
                    print("Plotting original image:")
                    show_image(image, title=True)
                    print("Plotting similar images:")
                    show_query_results(inventory, results, new=True, save=save, saving_path=saving_path)
                else:
                    print("Plotting not supported for top_k < 6.")
            else:
                print("Plotting not supported for embedding vectors.")
        return results
    else:
        return "Query failed."