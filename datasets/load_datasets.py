import os
import pandas as pd

import skimage.io as io

SKETCH_STATS_PATH = os.path.join('info', 'stats.csv')
PHOTO_PATH = os.path.join('rendered_256x256', '256x256', 'photo', 'tx_000000000000')
SKETCH_PATH = os.path.join('rendered_256x256', '256x256', 'sketch', 'tx_000000000000')

def load_sketch_stats():
    return pd.read_csv(SKETCH_STATS_PATH, index_col=None)


def load_sketch_images(df_sketch):

    print(df_sketch.head())
    print(df_sketch)

    
    for index, row in df_sketch.iterrows():
        category_id = row['CategoryID']
        category = row['Category']
        image_id = row['ImageNetID']
        sketch_id = row['SketchID']

        photo_path = os.path.join(PHOTO_PATH, category, image_id + '.jpg')
        sketch_path = os.path.join(SKETCH_PATH, category, image_id + '-' + str(sketch_id) + '.png')

        photo = io.imread(photo_path)
        sketch = io.imread(sketch_path)

        # preview photo and sketch
        io.imshow(photo)
        io.show()
        io.imshow(sketch)
        io.show()

        break


def main():
    df_sketch = load_sketch_stats()
    load_sketch_images(df_sketch)


if __name__ == '__main__':
    main()
