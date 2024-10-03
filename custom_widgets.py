from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button


class ImageTextButton(BoxLayout):
    def __init__(self, image_source, label_text, button_text, id_button, **kwargs):
        super(ImageTextButton, self).__init__(**kwargs)
        self.orientation = "horizontal"  # Arrange widgets vertically
        self.id_button = id_button
        # Add Image
        self.img = Image(source=image_source)
        self.add_widget(self.img)

        # Add Text
        self.label = Label(text=label_text)
        self.add_widget(self.label)

        # Add Button
        self.button = Button(text=button_text, size_hint=(0.5, 0.25))
        self.button.bind(on_press=self.on_button_click)
        self.add_widget(self.button)

    def on_button_click(self, instance):
        print(f"Button clicked: {self.id_button}")
