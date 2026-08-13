import datetime
from tkinter import Tk, filedialog
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
import cv2

import tools
from custom_widgets import ImageTextButton


class MainWindow(BoxLayout):
    pass


class MainApp(App):
    gruoped_features = BoxLayout(orientation="vertical", size_hint_y=2.0)
    parts_catalog = ObjectProperty(None)
    catalog = ListProperty([])
    index = 0
    face_images = ListProperty([])

    def on_face_image_added(self, code):
        part = (
            self.parts_catalog.loc[
                self.parts_catalog["codigo"] == code, "parte"
            ].values[0]
            if not self.parts_catalog is None
            else None
        )

        if (
            self.face_images is None
            or len(self.face_images) == 0
            and part != "tipo_rostro"
        ):
            self.root.ids.messages.text = f"Error: Primero debes escoger un tipo de rostro antes de seleccionar otros rasgos."
        elif part in [f["part"] for f in self.face_images]:
            for i, f in enumerate(self.face_images):
                if f["part"] == part:
                    self.face_images[i] = {"part": part, "code": code}
                    break
            self.show_description()
        else:
            self.face_images.append({"part": part, "code": code})
            self.show_description()

    def show_description(self):
        self.root.ids.messages.text = "Rasgos seleccionados:\n"
        for f in self.face_images:
            desc = self.parts_catalog.loc[
                (self.parts_catalog["parte"] == f["part"])
                & (self.parts_catalog["codigo"] == f["code"]),
                "descripcion",
            ].values[0]
            self.root.ids.messages.text += f"- {desc} "
        self.create_face_sketch()

    def create_face_sketch(self):
        face_parts_data = self.get_parts_data()
        if len(face_parts_data) > 1:
            boceto, _ = tools.suma_imagenes_dominio_frecuencial(face_parts_data)

            # Voltear verticalmente la imagen para corregir el sistema de coordenadas
            # OpenCV usa origen arriba-izquierda, Kivy/OpenGL usa origen abajo-izquierda
            boceto = tools.flip_image(boceto)

            height, width = boceto.shape
            boceto_texture = Texture.create(size=(width, height), colorfmt="luminance")
            boceto_texture.blit_buffer(
                boceto.tobytes(), colorfmt="luminance", bufferfmt="ubyte"
            )
            self.root.ids.build_image.texture = boceto_texture

    def get_images_paths(self):
        paths = []
        for f in self.face_images:
            img_path = self.parts_catalog.loc[
                (self.parts_catalog["parte"] == f["part"])
                & (self.parts_catalog["codigo"] == f["code"]),
                "imagen",
            ].values[0]
            if img_path is not None:
                paths.append(img_path)
        return paths

    def get_parts_data(self):
        parts_data = []
        for f in self.face_images:
            part_info = self.parts_catalog.loc[
                (self.parts_catalog["parte"] == f["part"])
                & (self.parts_catalog["codigo"] == f["code"])
            ].to_dict(orient="records")[0]
            parts_data.append(part_info)
        return parts_data

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
        self.reset_app()
        genre = "hombres" if self.root.ids.cb_hombre.active else "mujeres"
        if self.parts_catalog is None:
            self.load_data()
        else:
            self.load_data(genre=genre)
        print(f"Cambio a {genre}")

    def reset_app(self):
        self.index = 0
        self.face_images = []
        self.root.ids.build_image.texture = None
        self.load_data()
        self.root.ids.prev_button.disabled = True
        self.root.ids.next_button.disabled = False
        self.root.ids.messages.text = (
            "Selecciona los rasgos faciales para crear el retrato hablado."
        )

    def save_image(self):
        if self.root.ids.build_image.texture is None:
            self.root.ids.messages.text = "Error: No hay imagen para guardar."
            return

        # Crear ventana invisible de tkinter
        root = Tk()
        root.withdraw()  # Ocultar la ventana principal
        root.attributes("-topmost", True)  # Traer al frente

        # Generar nombre sugerido con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"retrato_hablado_{timestamp}.png"

        # Abrir diálogo para guardar archivo
        filepath = filedialog.asksaveasfilename(
            title="Guardar retrato hablado",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()  # Cerrar ventana de tkinter

        # Si el usuario canceló, no hacer nada
        if not filepath:
            self.root.ids.messages.text = "Guardado cancelado."
            return

        # Generar boceto y guardar
        face_parts_data = self.get_parts_data()
        boceto, recortes = tools.suma_imagenes_dominio_frecuencial(face_parts_data)

        cv2.imwrite(filepath, boceto)
        # cv2.imwrite(filepath, real)

        self.root.ids.messages.text = f"Imagen guardada en: {filepath}"

    def quit(self):
        self.stop()


if __name__ == "__main__":
    MainApp().run()
# end main
