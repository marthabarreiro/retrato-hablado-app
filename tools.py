import pandas as pd
import os


def get_image_path(image_name):
    base_image_dir = "images/"
    for directory in os.listdir(base_image_dir):
        image_path = os.path.join(base_image_dir, directory, f"{image_name}.jpg")
        if os.path.exists(image_path):
            return image_path
        else:
            image_path = os.path.join(base_image_dir, directory, f"{image_name}.png")
            if os.path.exists(image_path):
                return image_path
    return None


def load_data():
    catalog_df = pd.read_csv("catalog.csv")
    faceparts_df = pd.read_csv("face_parts.csv")
    faceparts_df["codigo"] = faceparts_df["codigo"].astype(str).str.zfill(3)

    merged_df = pd.merge(
        catalog_df, faceparts_df, left_on="codigo", right_on="codigo_catalogo"
    )

    merged_df["image_name"] = merged_df["codigo_x"] + merged_df["codigo_y"]

    merged_df["image_path"] = merged_df["image_name"].apply(get_image_path)

    return merged_df


def load_catalog():
    catalog_df = pd.read_csv("catalog.csv")
    return catalog_df.parte.unique()
