from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
import pandas as pd

import tools
from custom_widgets import ImageTextButton


class MainWindow(BoxLayout):
    data = ObjectProperty(
        None
    )  # Contiene toso los datos en un dataframe para hacer consultas

    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        # self.data = tools.load_data()
        pass

    def apply_selection(self):
        print("Aqui se aplica las herramientas matematicas")


class MainApp(App):
    gruoped_features = BoxLayout(orientation="vertical", size_hint_y=2.0)
    parts_catalog = ObjectProperty(None)
    catalog = ObjectProperty(None)
    index = 0
    face_images = ObjectProperty(None)

    def on_start(self):
        self.root.ids.cb_hombre.active = True
        if self.parts_catalog is None:
            self.load_data()
        self.root.ids.prev_button.disabled = True

    def load_data(
        self, genre: str = "hombres", face_code="tp001", part: str = "tipo_rostro"
    ):
        # Cargar datos desde rostro_catalogo.json usando pandas
        if self.parts_catalog is None:
            self.parts_catalog = tools.read_file()
            print(self.parts_catalog)
            self.catalog = self.parts_catalog.parte.unique()
        filtro_partes = (
            (self.parts_catalog["genero"] == genre)
            | (self.parts_catalog["genero"] == "unisex")
        ) & (self.parts_catalog["parte"] == part)
        part_data = self.parts_catalog[filtro_partes]
        self.root.ids.scrollview.clear_widgets()
        self.gruoped_features.clear_widgets()
        self.root.ids.catalog_label.text = f"Tipo de {part}"
        for i in part_data.index:
            itb = ImageTextButton(
                image_source=part_data["imagen"][i],
                label_text=part_data["descripcion"][i],
                button_text="Aplicar",
                id_button=part_data["codigo"][i],
                app_instance=self,  # Pasar la referencia de la aplicación
            )
            self.gruoped_features.add_widget(itb)
        self.root.ids.scrollview.add_widget(self.gruoped_features)

    def next_section(self):
        genre = "hombres" if self.root.ids.cb_hombre.active else "mujeres"
        self.index += 1
        if self.index < len(self.catalog):
            self.load_data(genre=genre, part=self.catalog[self.index])
            self.root.ids.prev_button.disabled = False
        else:
            self.root.ids.next_button.disabled = True

    def prev_section(self):
        genre = "hombres" if self.root.ids.cb_hombre.active else "mujeres"
        self.index -= 1
        print(f"{self.index}, {len(self.catalog)}")
        self.root.ids.next_button.disabled = False
        if self.index >= 0:
            self.load_data(genre=genre, part=self.catalog[self.index])
        else:
            self.root.ids.prev_button.disabled = True

    def change_genre(self):
        genre = "hombres" if self.root.ids.cb_hombre.active else "mujeres"
        if self.parts_catalog is None:
            self.load_data()
        else:
            self.load_data(genre=genre)
        print(f"Cambio a {genre}")

    def quit(self):
        self.stop()


if __name__ == "__main__":
    MainApp().run()
# end main
