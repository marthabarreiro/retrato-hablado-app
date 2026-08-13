"""
GUI para aplicar filtros frecuenciales a imagenes en escala de grises.

Los filtros se descubren automaticamente desde filters.py mediante inspeccion
de firmas y type hints. Agregar una funcion publica a filters.py genera una
nueva pestana en la GUI sin modificar este archivo.

Uso:
    python freq_filter_gui.py
"""

import sys
import os
import inspect
import importlib.util

import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Asegura que el directorio de este archivo este en el path para importar filters
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filters as fil_module


# ---------------------------------------------------------------------------
# Descubrimiento automatico de filtros
# ---------------------------------------------------------------------------


def discover_filters(module):
    """
    Retorna lista de dicts con la informacion de cada filtro publico del modulo.

    Cada dict tiene:
        name   : str          - nombre de la funcion
        func   : callable     - referencia a la funcion
        label  : str          - nombre formateado para la UI
        params : list[dict]   - lista de parametros (excluye 'image')
            cada param: {name, type, default, min, max}
    """
    results = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        # Solo funciones definidas en este modulo (evita importadas como np.*)
        if getattr(func, "__module__", None) != module.__name__:
            continue

        sig = inspect.signature(func)
        fp = getattr(func, "__filter_params__", {})

        params = []
        for pname, param in sig.parameters.items():
            if pname == "image":
                continue
            annotation = param.annotation
            ptype = annotation if annotation != inspect.Parameter.empty else int
            default = param.default if param.default != inspect.Parameter.empty else 0
            range_hint = fp.get(pname, {})
            params.append(
                {
                    "name": pname,
                    "type": ptype,
                    "default": default,
                    "min": range_hint.get("min", 1),
                    "max": range_hint.get("max", 500),
                }
            )

        results.append(
            {
                "name": name,
                "func": func,
                "label": name.replace("_", " ").title(),
                "params": params,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Aplicacion principal
# ---------------------------------------------------------------------------

CANVAS_SIZE = 290
HISTORY_HEIGHT = 5
BG_CANVAS = "#1a1a2e"


class FilterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Filtros en Dominio Frecuencial")
        self.root.resizable(True, True)

        # Estado
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.filter_history: list[dict] = []

        # Referencias a PhotoImage (evita garbage collection)
        self._photo_original = None
        self._photo_filter = None
        self._photo_result = None

        # Vars de los sliders por filtro: {filter_name: {param_name: tk.Variable}}
        self.filter_vars: dict[str, dict] = {}
        self.filter_funcs: dict[str, callable] = {}

        self._build_ui()
        self._build_filter_tabs()

    # ------------------------------------------------------------------
    # Construccion de la UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Barra superior ----
        top = tk.Frame(self.root, pady=4)
        top.pack(fill=tk.X, padx=8)

        tk.Button(
            top,
            text="Cargar imagen",
            command=self.load_image,
            bg="#2563eb",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT)

        tk.Button(
            top,
            text="Reset",
            command=self.reset,
            bg="#dc2626",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Sin imagen cargada")
        tk.Label(top, textvariable=self.status_var, fg="gray").pack(
            side=tk.LEFT, padx=10
        )

        # ---- Paneles de imagen ----
        img_frame = tk.Frame(self.root)
        img_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        panels = [
            ("Original", "canvas_original"),
            ("Filtro\n(Dominio Frecuencial)", "canvas_filter"),
            ("Resultado\n(Dominio Espacial)", "canvas_result"),
        ]
        for label, attr in panels:
            frame = tk.LabelFrame(img_frame, text=label, font=("Helvetica", 9, "bold"))
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            canvas = tk.Canvas(
                frame,
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                bg=BG_CANVAS,
                highlightthickness=0,
            )
            canvas.pack(padx=4, pady=4)
            setattr(self, attr, canvas)

        # ---- Notebook de filtros ----
        nb_frame = tk.LabelFrame(
            self.root, text="Filtros disponibles", font=("Helvetica", 9, "bold")
        )
        nb_frame.pack(fill=tk.X, padx=8, pady=4)

        self.notebook = ttk.Notebook(nb_frame)
        self.notebook.pack(fill=tk.X, padx=4, pady=4)

        # ---- Historial ----
        hist_frame = tk.LabelFrame(
            self.root,
            text="Historial de filtros aplicados",
            font=("Helvetica", 9, "bold"),
        )
        hist_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL)
        self.history_list = tk.Listbox(
            hist_frame,
            height=HISTORY_HEIGHT,
            yscrollcommand=scrollbar.set,
            font=("Courier", 9),
        )
        scrollbar.config(command=self.history_list.yview)
        self.history_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

    def _build_filter_tabs(self):
        """Auto-genera una pestana por cada filtro descubierto en filters.py."""
        filters_info = discover_filters(fil_module)

        for info in filters_info:
            name = info["name"]
            func = info["func"]
            self.filter_funcs[name] = func
            self.filter_vars[name] = {}

            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=info["label"])

            # Descripcion del filtro (docstring)
            doc = (func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else ""
            if doc:
                tk.Label(tab, text=doc, fg="gray", font=("Helvetica", 8)).pack(
                    anchor="w", padx=12, pady=(6, 2)
                )

            # Sliders por parametro
            params_frame = tk.Frame(tab)
            params_frame.pack(fill=tk.X, padx=8, pady=4)

            for p in info["params"]:
                pname = p["name"]
                ptype = p["type"]
                default = p["default"]
                pmin = p["min"]
                pmax = p["max"]

                row = tk.Frame(params_frame)
                row.pack(fill=tk.X, pady=2)

                tk.Label(
                    row, text=pname, width=14, anchor="w", font=("Helvetica", 9)
                ).pack(side=tk.LEFT)

                if ptype == float:
                    var = tk.DoubleVar(value=float(default))
                    scale = tk.Scale(
                        row,
                        from_=pmin,
                        to=pmax,
                        orient=tk.HORIZONTAL,
                        variable=var,
                        resolution=0.1,
                        length=320,
                        showvalue=False,
                    )
                else:
                    var = tk.IntVar(value=int(default))
                    scale = tk.Scale(
                        row,
                        from_=pmin,
                        to=pmax,
                        orient=tk.HORIZONTAL,
                        variable=var,
                        resolution=1,
                        length=320,
                        showvalue=False,
                    )
                scale.pack(side=tk.LEFT)

                spin = ttk.Spinbox(row, from_=pmin, to=pmax, textvariable=var, width=7)
                spin.pack(side=tk.LEFT, padx=6)

                self.filter_vars[name][pname] = var

            # Boton aplicar
            btn_row = tk.Frame(tab)
            btn_row.pack(fill=tk.X, padx=8, pady=(2, 8))
            tk.Button(
                btn_row,
                text="Aplicar filtro",
                bg="#16a34a",
                fg="white",
                relief=tk.FLAT,
                padx=12,
                pady=4,
                command=lambda n=name, f=func: self.apply_filter(n, f),
            ).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if not path:
            return

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{path}")
            return

        self.original_image = img
        self.current_image = img.copy()
        self.last_mask = None
        self._clear_history()
        self.status_var.set(
            f"{os.path.basename(path)}  |  {img.shape[1]}x{img.shape[0]} px"
        )
        self.update_views()

    def apply_filter(self, name: str, func: callable):
        if self.current_image is None:
            messagebox.showwarning("Sin imagen", "Primero carga una imagen.")
            return

        params = {pname: var.get() for pname, var in self.filter_vars[name].items()}

        try:
            result, mask = func(self.current_image, **params)
        except ValueError as e:
            messagebox.showerror("Parametros invalidos", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error al aplicar filtro", str(e))
            return

        self.current_image = result
        self.last_mask = mask

        # Guardar en historial
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        entry = f"[{len(self.filter_history) + 1:02d}] {name}({param_str})"
        self.filter_history.append({"name": name, "params": params.copy()})
        self.history_list.insert(tk.END, entry)
        self.history_list.see(tk.END)

        self.update_views()

    def reset(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.last_mask = None
        self._clear_history()
        self.update_views()

    # ------------------------------------------------------------------
    # Vistas
    # ------------------------------------------------------------------

    def update_views(self):
        if self.original_image is not None:
            self._photo_original = self._to_photo(self.original_image)
            self._draw_on_canvas(self.canvas_original, self._photo_original)

        if self.last_mask is not None:
            # Mascara: float64 en [0,1] -> visualizar como imagen
            mask_vis = (self.last_mask * 255).astype(np.uint8)
            self._photo_filter = self._to_photo(mask_vis)
            self._draw_on_canvas(self.canvas_filter, self._photo_filter)
        else:
            # Sin filtro aplicado aun: limpiar canvas
            self.canvas_filter.delete("all")

        if self.current_image is not None:
            self._photo_result = self._to_photo(self.current_image)
            self._draw_on_canvas(self.canvas_result, self._photo_result)

    def _to_photo(self, arr: np.ndarray) -> ImageTk.PhotoImage:
        """Convierte un ndarray uint8 2D a PhotoImage escalada al CANVAS_SIZE."""
        img = Image.fromarray(arr)
        img.thumbnail((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _draw_on_canvas(self, canvas: tk.Canvas, photo: ImageTk.PhotoImage):
        canvas.delete("all")
        cx = CANVAS_SIZE // 2
        cy = CANVAS_SIZE // 2
        canvas.create_image(cx, cy, anchor=tk.CENTER, image=photo)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _clear_history(self):
        self.filter_history.clear()
        self.history_list.delete(0, tk.END)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = FilterApp(root)
    root.mainloop()
