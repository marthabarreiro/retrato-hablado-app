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
    catalog = ObjectProperty(None)
    index = 0

    def on_start(self):
        self.root.ids.cb_hombre.active = True
        self.load_data(part=self.catalog[self.index])
        self.root.ids.prev_button.disabled = True

    def read_df(self, genre: str = "hombre", part: str = "ojos"):
        # Cargar datos desde rostro_catalogo.json usando pandas
        df = pd.read_json("rostro_catalogo.json")
        data = df.query("genero==@genre and parte_x==@part")
        self.root.ids.scrollview.clear_widgets()
        self.gruoped_features.clear_widgets()
        self.root.ids.catalog_label.text = f"Tipo de {part}"
        for i in data.index:
            itb = ImageTextButton(
                image_source=data["image_path"][i],
                label_text=data["descripcion"][i],
                button_text="Aplicar",
                id_button=data["image_name"][i],
            )
            self.gruoped_features.add_widget(itb)
        self.root.ids.scrollview.add_widget(self.gruoped_features)

    def next_section(self):
        genre = "hombre" if self.root.ids.cb_hombre.active else "mujer"
        self.index += 1
        if self.index < len(self.catalog):
            self.load_data(genre=genre, part=self.catalog[self.index])
            self.root.ids.prev_button.disabled = False
        else:
            self.root.ids.next_button.disabled = True

    def prev_section(self):
        genre = "hombre" if self.root.ids.cb_hombre.active else "mujer"
        self.index -= 1
        print(f"{self.index}, {len(self.catalog)}")
        self.root.ids.next_button.disabled = False
        if self.index >= 0:
            self.load_data(genre=genre, part=self.catalog[self.index])
        else:
            self.root.ids.prev_button.disabled = True

    def change_genre(self):
        genre = "hombre" if self.root.ids.cb_hombre.active else "mujer"
        if self.catalog is None:
            self.catalog = tools.load_catalog()
        self.read_df(genre=genre, part=self.catalog[self.index])

    def quit(self):
        self.stop()


if __name__ == "__main__":
    MainApp().run()
# end main
